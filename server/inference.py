"""
server/inference.py
Batch inference + B-scan extraction + predictions helper.
"""

import base64

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

from model import THRESHOLD, INFER_BATCH, DEVICE


def run_inference(
    model: nn.Module,
    signals: np.ndarray,
    threshold: float = THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model on (n_signals, 512) in batches of INFER_BATCH.

    Returns
    -------
    preds : int array (n,) — 1=sound / 0=delaminated
    confs : float array (n,) — confidence in predicted class
    """
    probs_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(signals), INFER_BATCH):
            batch_np = signals[start : start + INFER_BATCH]
            batch_t  = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1)
            out      = model(batch_t.to(DEVICE)).sigmoid().cpu().numpy()
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
