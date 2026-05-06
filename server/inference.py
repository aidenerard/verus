"""
server/inference.py
Batch inference + B-scan extraction + predictions helper.
"""

import base64
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from scipy.signal import hilbert as _hilbert

from model import THRESHOLD, INFER_BATCH, DEVICE

_REBAR_DIELECTRIC = {400: 8.0, 900: 7.0, 1600: 6.0, 2000: 5.5, 2600: 5.0}


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
) -> tuple[np.ndarray, np.ndarray]:
    patch_k = (model_config or {}).get("patch_k", 1)
    probs_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(signals), INFER_BATCH):
            batch_np = signals[start : start + INFER_BATCH]
            end = start + len(batch_np)
            if patch_k > 1:
                batch_t = _make_patch_batch(signals, start, end, patch_k)
            else:
                batch_t = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1)
            out = model(batch_t.to(DEVICE)).sigmoid().cpu().numpy()
            probs_list.append(out)
            del batch_t, out

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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Input: z-score normalised signals (n_signals, 512).
    Returns (depth_inches, twt_ns) both float32 (n_signals,).
    Falls back to physics peak-time estimate when model is None or fails.
    """
    er       = _REBAR_DIELECTRIC.get(frequency_mhz, 6.0)
    velocity = 0.3 / np.sqrt(er)  # m/ns in concrete

    def _physics(sigs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        peak   = np.argmax(np.abs(sigs), axis=1).astype(np.float32)
        twt    = peak * 0.023  # 0.023 ns/sample assumed
        depth  = (velocity * twt / 2.0) * 39.3701
        return depth.astype(np.float32), twt

    if model is None:
        return _physics(signals)

    try:
        # Downsample 512 → 256 (every other sample matches training data)
        raw = signals[:, ::2].astype(np.float32)
        # Re-normalise to max-abs (training convention, not z-score)
        mx  = np.abs(raw).max(axis=1, keepdims=True)
        mx  = np.where(mx == 0, 1.0, mx)
        raw = raw / mx
        # Hilbert envelope
        env = np.abs(_hilbert(raw, axis=1)).astype(np.float32)
        # Stack → (n, 2, 256)
        X   = np.stack([raw, env], axis=1)

        preds: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X), INFER_BATCH):
                batch = torch.tensor(X[start:start + INFER_BATCH], dtype=torch.float32)
                out   = model(batch.to(DEVICE)).cpu().numpy()
                preds.append(out)

        depth_in = np.clip(np.concatenate(preds), 0.5, 12.0).astype(np.float32)
        twt      = (2.0 * (depth_in / 39.3701) / velocity).astype(np.float32)
        return depth_in, twt

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
