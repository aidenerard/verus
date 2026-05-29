"""
server/inference.py
Batch inference + B-scan extraction + predictions helper.
"""

import base64
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from scipy.signal import hilbert as _hilbert

from model import THRESHOLD, INFER_BATCH, DEVICE

_REBAR_DIELECTRIC = {400: 8.0, 900: 7.0, 1600: 6.0, 2000: 5.5, 2600: 5.0}
_infer_logged = False


def _make_patch_batch(signals: np.ndarray, start: int, end: int, k: int) -> torch.Tensor:
    n, half = len(signals), k // 2
    patches = [
        signals[np.clip(np.arange(i - half, i - half + k), 0, n - 1)]
        for i in range(start, end)
    ]
    return torch.tensor(np.stack(patches), dtype=torch.float32)


def run_inference(
    model: nn.Module,
    signals: np.ndarray,
    threshold: float = THRESHOLD,
    model_config: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    global _infer_logged
    cfg        = model_config or {}
    patch_k    = cfg.get("patch_k", 1)
    in_ch      = cfg.get("in_channels", 1)
    crop_start = cfg.get("crop_start", 200)
    crop_end   = cfg.get("crop_end", 450)
    n_crop     = crop_end - crop_start

    cropped = signals[:, crop_start:crop_end]

    if in_ch == 2:
        envelope = np.abs(_hilbert(cropped, axis=1)).astype(np.float32)
        dual     = np.stack([cropped, envelope], axis=1)  # (n, 2, n_crop)

    if not _infer_logged:
        extra = f" → hilbert-stack={dual.shape}" if in_ch == 2 else ""
        print(
            f"[inference] preprocess: raw={signals.shape} "
            f"→ crop[{crop_start}:{crop_end}]={cropped.shape}{extra} "
            f"patch_k={patch_k} in_channels={in_ch}",
            flush=True,
        )
        _infer_logged = True

    probs_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cropped), INFER_BATCH):
            batch_np = cropped[start : start + INFER_BATCH]
            end      = start + len(batch_np)
            if in_ch == 2:
                batch_t = torch.tensor(dual[start:end], dtype=torch.float32)
            elif patch_k > 1:
                batch_t = _make_patch_batch(cropped, start, end, patch_k)
            else:
                batch_t = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1)
            if tuple(batch_t.shape) != (len(batch_np), in_ch, n_crop):
                raise ValueError(
                    f"[inference] tensor shape {tuple(batch_t.shape)} != "
                    f"expected ({len(batch_np)}, {in_ch}, {n_crop})"
                )
            out = model(batch_t.to(DEVICE)).sigmoid().cpu().numpy()
            probs_list.append(out)
            del batch_t, out
            if progress_callback:
                progress_callback(min(start + INFER_BATCH, len(cropped)), len(cropped))

    probs = np.concatenate(probs_list)
    preds = (probs >= threshold).astype(int)
    confs = np.where(preds == 1, probs, 1.0 - probs)
    return preds, confs


def make_predictions_list(
    file_names: list[str],
    file_preds: list[np.ndarray],
    file_confs: list[np.ndarray],
) -> list[tuple[str, int, float]]:
    """Flat list of (filename, signal_index, confidence) across all files."""
    results: list[tuple[str, int, float]] = []
    for fname, preds, confs in zip(file_names, file_preds, file_confs):
        for idx, (_, conf) in enumerate(zip(preds, confs)):
            results.append((fname, idx, float(conf)))
    return results


def run_rebar_inference(
    model: Optional[nn.Module],
    signals: np.ndarray,
    frequency_mhz: int = 1600,
    model_config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input: raw signals (n_signals, 512).
    Returns (depth_inches, twt_ns, peak_samples) all (n_signals,).
    Falls back to physics peak-time estimate when model is None or fails.
    model_config may carry max_depth_mm to match the training constant.
    """
    cfg      = model_config or {}
    er       = _REBAR_DIELECTRIC.get(frequency_mhz, 6.0)
    velocity = 0.3 / np.sqrt(er)  # m/ns in concrete

    def _physics(sigs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        peak_samples = np.argmax(np.abs(sigs), axis=1).astype(np.int32)
        twt          = peak_samples.astype(np.float32) * (15.0 / 512)
        depth        = (velocity * twt / 2.0) * 39.3701
        return depth.astype(np.float32), twt, peak_samples

    if model is None:
        return _physics(signals)

    try:
        # DC-remove + max-abs normalise — matches HorizonCNN training exactly
        raw = signals.astype(np.float32)
        raw = raw - raw.mean(axis=1, keepdims=True)
        mx  = np.abs(raw).max(axis=1, keepdims=True)
        mx  = np.where(mx == 0, 1.0, mx)
        raw = raw / mx
        X   = raw[:, np.newaxis, :]  # (n, 1, 512)

        preds: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X), INFER_BATCH):
                batch = torch.tensor(X[start:start + INFER_BATCH], dtype=torch.float32)
                out   = model(batch.to(DEVICE)).cpu().numpy()
                preds.append(out)

        norm_depth   = np.clip(np.concatenate(preds), 0.0, 1.0)
        depth_mm     = norm_depth * float(cfg.get("max_depth_mm", 300.0))
        depth_in     = (depth_mm / 25.4).astype(np.float32)
        twt          = (2.0 * (depth_mm / 1000.0) / velocity).astype(np.float32)
        peak_samples = np.argmax(np.abs(signals), axis=1).astype(np.int32)
        return depth_in, twt, peak_samples

    except Exception:
        return _physics(signals)


def extract_bscan_b64(
    signals: np.ndarray,
    max_traces: int = 256,
    max_samples: int = 128,
) -> dict:
    """
    Downsample (n_signals, 512) to (max_traces, max_samples) and return
    as base64 float32 blob for browser B-scan rendering.
    """
    n      = signals.shape[0]
    zoom_t = min(1.0, max_traces  / n)
    zoom_s = min(1.0, max_samples / signals.shape[1])

    if zoom_t < 1.0 or zoom_s < 1.0:
        arr = zoom(signals.astype(np.float32), [zoom_t, zoom_s], order=1)
    else:
        arr = signals.astype(np.float32)

    arr = np.ascontiguousarray(arr)
    return {
        'data':      base64.b64encode(arr.tobytes()).decode('ascii'),
        'n_traces':  arr.shape[0],
        'n_samples': arr.shape[1],
    }
