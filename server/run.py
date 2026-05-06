"""
run.py — GPR bridge deck delamination inference module + CLI.

Importable by server.py:
    from run import (CNN1D, TemporalAttention,
                     load_csv, run_inference,
                     render_cscan_b64, make_predictions_list)

CLI usage:
    python run.py --input /path/to/csvs --model model.pth [--output results.json]
"""

import argparse
import base64
import gc
import io
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

# ── Constants ─────────────────────────────────────────────────────────────────
THRESHOLD     = 0.65       # P(sound) < THRESHOLD → delaminated
DC_OFFSET     = 32768
N_SAMPLES     = 512
INFER_BATCH   = 256        # max signals per torch forward pass.
                           # Conv2 intermediate: (B,128,256)×4 bytes.
                           # B=1000 → 131 MB spike; B=256 → 33 MB. Fits 512 MB.
MAX_GRID_ROWS = 200        # C-scan Y axis (signals), downsampled if larger
MAX_GRID_COLS = 500        # C-scan X axis (files),   downsampled if larger
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLI-only defaults (not used when imported as a module)
_DEFAULT_MODEL = Path("models/model_v13.pth")
_DEFAULT_INPUT = Path(".")
_CSCAN_OUT     = Path("bridge_cscan.png")


# ── Model architecture — must exactly match cnn.py ───────────────────────────

class TemporalAttention(nn.Module):
    """Weighted sum over time steps. Input: (B, C, T) → Output: (B, C)."""
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x.permute(0, 2, 1)), dim=1)
        return (x * weights.permute(0, 2, 1)).sum(dim=2)


class CNN1D(nn.Module):
    """
    Architecture must exactly match the saved model.pth weights:
      net.0  Conv1d(1,  32, 7, padding=3)
      net.1  ReLU
      net.2  MaxPool1d(2)
      net.3  Conv1d(32, 64, 5, padding=2)
      net.4  ReLU
      net.5  MaxPool1d(2)
      net.6  Flatten  → 64 * 128 = 8192
      net.7  Linear(8192, 64)
      net.8  ReLU
      net.9  Dropout(0.3)
      net.10 Linear(64, 1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=7, padding=3),   # 0
            nn.ReLU(),                                       # 1
            nn.MaxPool1d(2),                                 # 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),   # 3
            nn.ReLU(),                                       # 4
            nn.MaxPool1d(2),                                 # 5
            nn.Flatten(),                                    # 6
            nn.Linear(8192, 64),                            # 7
            nn.ReLU(),                                       # 8
            nn.Dropout(0.3),                                 # 9
            nn.Linear(64, 1),                               # 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ── CSV parsing ───────────────────────────────────────────────────────────────

def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _normalise_key(s: str) -> str:
    """Lower-case, strip parens/spaces/underscores for loose matching."""
    return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")


# Pre-normalised so comparison against _normalise_key(field) always works.
_TIME_AXIS_KEYS = {
    _normalise_key(k) for k in (
        "time_ns", "time", "time(ns)", "t(ns)", "depth_m", "depth",
        "sample", "sample_no", "twt", "twt(ns)",
    )
}


def _sniff_csv(fpath: Path) -> tuple[str, int]:
    """
    Scan the file line by line (up to 300 lines) to find delimiter and data start row.

    Strategy A — keyword: if a row's first field matches a known time-axis
    label (Time_ns, Depth, Sample, …), data starts on the NEXT line.

    Strategy B — numeric: first row with ≥10 fields, no alpha chars, ≥80%
    parseable as float.
    """
    delimiter  = ","
    best_count = 0

    with open(fpath, "r", errors="replace") as f:
        lines = []
        for _ in range(300):
            line = f.readline()
            if not line:
                break
            lines.append(line)

    for line in lines:
        for d in (",", "\t", ";"):
            c = line.count(d)
            if c > best_count:
                best_count = c
                delimiter = d

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(delimiter)
        first = _normalise_key(parts[0].strip())

        # Strategy A: known time-axis header → data on next line
        if first in _TIME_AXIS_KEYS and len(parts) > 1:
            print(f"[sniff_csv] time-axis header '{parts[0].strip()}' "
                  f"at row {i} → data starts at row {i + 1}", flush=True)
            return delimiter, i + 1

        # Strategy B: large row, no letters, mostly numeric
        if len(parts) < 10:
            continue
        has_alpha = any(
            any(c.isalpha() for c in p.strip())
            for p in parts if p.strip()
        )
        if has_alpha:
            continue
        numeric = sum(1 for p in parts if _is_float(p.strip()))
        if numeric >= 0.8 * len(parts):
            print(f"[sniff_csv] numeric data at row {i} "
                  f"({len(parts)} fields, {numeric} numeric)", flush=True)
            return delimiter, i

    print("[sniff_csv] WARNING: no data row in first 300 lines → skiprows=0", flush=True)
    return delimiter, 0


def load_csv(fpath: Path) -> np.ndarray:
    """
    Load one CSV file and return normalised signals as (n_signals, 512).

    Accepts multiple GPR CSV formats:
      - SDNET2021: metadata header + Time_ns row + 512×N amplitude block
      - Simple: rows of amplitude samples (one A-scan per row)
      - Transposed: columns are A-scans

    Normalisation: per-signal z-score (each row gets its own mean/std).
    """
    delimiter, skiprows = _sniff_csv(fpath)
    print(f"[load_csv] delimiter={repr(delimiter)} skiprows={skiprows}", flush=True)

    # Read directly as float32 — no Python string-object overhead
    try:
        df = pd.read_csv(
            fpath, header=None, sep=delimiter,
            skiprows=skiprows, dtype=np.float32,
            on_bad_lines="skip",
        )
    except Exception as exc:
        raise ValueError(f"pd.read_csv failed: {exc}")

    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    if df.empty:
        raise ValueError("No numeric data found in CSV")

    data_array = df.to_numpy(dtype=np.float32, na_value=0.0)
    del df
    gc.collect()

    print(f"[load_csv] raw shape: {data_array.shape}", flush=True)

    # ── Auto-detect orientation ───────────────────────────────────────────────
    rows, cols = data_array.shape

    if 400 <= rows <= 600:
        # Rows ≈ N_SAMPLES → rows are time-samples, columns are A-scans.
        # Column 0 may be a Time_ns axis (small floats) — drop if so.
        col0 = data_array[:, 0]
        if np.abs(col0).max() < 500:
            print("[load_csv] Dropping time/index column 0", flush=True)
            data_array = np.ascontiguousarray(data_array[:, 1:])
        amps = np.ascontiguousarray(data_array.T)   # (n_scans, 512)
        del data_array
    elif 400 <= cols <= 600:
        col0 = data_array[:, 0]
        if np.abs(col0).max() < rows + 2:
            print("[load_csv] Dropping row-index column 0", flush=True)
            amps = np.ascontiguousarray(data_array[:, 1:])
            del data_array
        else:
            amps = data_array
    elif rows > cols:
        amps = np.ascontiguousarray(data_array.T)
        del data_array
    else:
        amps = data_array
    gc.collect()

    n_signals, n_samples = amps.shape
    if n_signals == 0:
        raise ValueError("No A-scan signals found in CSV")

    # ── Pad / truncate to N_SAMPLES ───────────────────────────────────────────
    if n_samples > N_SAMPLES:
        amps = np.ascontiguousarray(amps[:, :N_SAMPLES])
    elif n_samples < N_SAMPLES:
        amps = np.pad(amps, ((0, 0), (0, N_SAMPLES - n_samples)), mode="constant")

    # ── DC offset correction (in-place) ──────────────────────────────────────
    if np.abs(amps.mean()) > 1000:
        amps -= DC_OFFSET

    # ── Per-signal z-score normalisation (each A-scan independently) ─────────
    mean = amps.mean(axis=1, keepdims=True)
    std  = amps.std(axis=1,  keepdims=True) + 1e-8
    amps -= mean
    amps /= std

    print(f"[load_csv] done: {n_signals} signals normalised", flush=True)
    return amps   # (n_signals, 512)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    signals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model on (n_signals, 512) array in batches of INFER_BATCH signals.
    Never creates a tensor larger than INFER_BATCH × 512 × 4 bytes (~2 MB).

    Returns:
        preds — int array, 1=sound / 0=delaminated, shape (n,)
        confs — float array, confidence in predicted class, shape (n,)
    """
    probs_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(signals), INFER_BATCH):
            batch_np = signals[start : start + INFER_BATCH]          # view
            batch_t  = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1)
            out      = model(batch_t.to(DEVICE)).sigmoid().cpu().numpy()
            probs_list.append(out)
            del batch_t, out

    probs = np.concatenate(probs_list)
    preds = (probs >= THRESHOLD).astype(int)
    confs = np.where(preds == 1, probs, 1.0 - probs)
    return preds, confs


# ── B-scan extraction ─────────────────────────────────────────────────────────

def extract_bscan_b64(
    signals: np.ndarray,
    max_traces: int = 256,
    max_samples: int = 128,
) -> dict:
    """
    Downsample (n_signals, 512) amplitude array to (max_traces, max_samples)
    and return as a base64-encoded float32 blob for browser B-scan rendering.

    Returns: {'data': str, 'n_traces': int, 'n_samples': int}
    """
    n = signals.shape[0]
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


# ── Results helper ────────────────────────────────────────────────────────────

def make_predictions_list(
    file_names: list[str],
    file_preds: list[np.ndarray],
    file_confs: list[np.ndarray],
) -> list[tuple[str, int, float]]:
    """
    Return predictions as a flat list of (filename, signal_index, confidence_score)
    tuples, one entry per A-scan signal across all files.

    confidence_score is the model's confidence in its predicted class:
      - For sound signals (pred=1):       confidence = P(sound)
      - For delaminated signals (pred=0): confidence = 1 - P(sound)
    """
    results: list[tuple[str, int, float]] = []
    for fname, preds, confs in zip(file_names, file_preds, file_confs):
        for idx, (pred, conf) in enumerate(zip(preds, confs)):
            results.append((fname, idx, float(conf)))
    return results


# ── Optimal threshold selection ───────────────────────────────────────────────

def _otsu_threshold(probs: np.ndarray, bins: int = 256) -> float:
    """
    Otsu's method: find the P(sound) threshold that maximises the
    between-class variance of the probability distribution.

    Works without ground-truth labels — it finds the natural valley
    between the sound and delaminated populations in the P(sound)
    histogram.  Clipped to [0.30, 0.85] to stay physically reasonable
    if one class dominates and the histogram is nearly unimodal.
    """
    counts, edges = np.histogram(probs, bins=bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(counts.sum())
    if total == 0:
        return 0.5

    w0  = np.cumsum(counts) / total                           # weight ≤ t
    mu0 = np.cumsum(centers * counts) / np.maximum(np.cumsum(counts), 1)
    mu_all = float(np.sum(centers * counts) / total)
    w1  = 1.0 - w0
    mu1 = np.where(w1 > 0, (mu_all - w0 * mu0) / w1, mu_all)

    between_var = w0 * w1 * (mu0 - mu1) ** 2
    t = float(centers[np.argmax(between_var)])
    return float(np.clip(t, 0.30, 0.85))


# ── Peak extraction + extra grids ─────────────────────────────────────────────

_FREQ_TIME_WINDOW_NS: dict[int, float] = {
    400:  40.0,
    900:  20.0,
    1600: 15.0,
    2000: 10.0,
    2600:  8.0,
}

_FREQ_ER: dict[int, float] = {
    400:  8.0,
    900:  7.0,
    1600: 6.0,
    2000: 6.0,
    2600: 5.0,
}


def extract_peak_info(
    signals: np.ndarray,
    skip_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each A-scan in *signals* (n, 512), find the sample index of the peak
    absolute amplitude (skipping the first *skip_samples* = air-wave zone).

    Returns
    -------
    peak_idx  : int32 array (n,) — absolute sample index of peak in each trace
    peak_amp  : float32 array (n,) — absolute amplitude at that sample
    """
    window = np.abs(signals[:, skip_samples:])       # (n, 512-skip)
    local_idx = np.argmax(window, axis=1)             # (n,) index within window
    peak_idx  = (local_idx + skip_samples).astype(np.int32)
    peak_amp  = window[np.arange(len(signals)), local_idx].astype(np.float32)
    return peak_idx, peak_amp


def build_prob_grid(
    file_preds: list[np.ndarray],
    file_confs: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """
    Build the downsampled P(sound) grid and Otsu threshold used by the C-scan.

    Returns
    -------
    prob_grid : float32 (rows, cols) — P(sound) values 0-1, NaN for padding
    otsu_T    : float — Otsu threshold
    """
    n_files  = len(file_preds)
    max_sigs = max(len(p) for p in file_preds)

    grid = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    for row, (preds, confs) in enumerate(zip(file_preds, file_confs)):
        for col, (pred, conf) in enumerate(zip(preds, confs)):
            grid[row, col] = conf if pred == 1 else 1.0 - conf

    # Downsample
    if n_files > MAX_GRID_ROWS:
        idx  = np.linspace(0, n_files - 1, MAX_GRID_ROWS, dtype=int)
        grid = grid[idx, :]
    if max_sigs > MAX_GRID_COLS:
        idx  = np.linspace(0, max_sigs - 1, MAX_GRID_COLS, dtype=int)
        grid = grid[:, idx]

    # Fill trailing NaN
    for i in range(grid.shape[0]):
        valid = np.where(~np.isnan(grid[i, :]))[0]
        if len(valid) and valid[-1] < grid.shape[1] - 1:
            grid[i, valid[-1] + 1:] = grid[i, valid[-1]]

    if grid.shape[1] < 50:
        scale = max(1, 50 // grid.shape[1])
        grid  = zoom(grid, (1, scale), order=1)

    all_preds  = np.concatenate(file_preds)
    all_confs  = np.concatenate(file_confs)
    all_psound = np.where(all_preds == 1, all_confs, 1.0 - all_confs)
    T = _otsu_threshold(all_psound)
    del all_psound, all_confs, all_preds

    return grid.astype(np.float32), T


def build_extra_grids(
    file_peak_idxs: list[np.ndarray],
    file_peak_amps: list[np.ndarray],
    frequency_mhz: int = 1600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build depth, amplitude, and TWT grids from per-file peak info.

    Returns (depth_grid_in, amplitude_grid, twt_grid_ns) — all float32,
    shape (n_files, max_sigs), downsampled to ≤ MAX_GRID_ROWS × MAX_GRID_COLS.
    NaN for padding cells.
    """
    time_window = _FREQ_TIME_WINDOW_NS.get(frequency_mhz, 15.0)
    er          = _FREQ_ER.get(frequency_mhz, 6.0)
    velocity    = 0.3 / np.sqrt(er)   # m/ns

    n_files  = len(file_peak_idxs)
    max_sigs = max(len(a) for a in file_peak_idxs)

    depth_grid = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    amp_grid   = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    twt_grid   = np.full((n_files, max_sigs), np.nan, dtype=np.float32)

    for row, (peak_idx, peak_amp) in enumerate(zip(file_peak_idxs, file_peak_amps)):
        n = len(peak_idx)
        twt_ns   = (peak_idx.astype(np.float32) / 512.0) * time_window
        depth_m  = velocity * twt_ns / 2.0
        depth_in = depth_m * 39.3701

        max_amp  = peak_amp.max() if peak_amp.max() > 0 else 1.0
        norm_amp = peak_amp / max_amp

        depth_grid[row, :n] = depth_in
        amp_grid[row, :n]   = norm_amp
        twt_grid[row, :n]   = twt_ns

    # Downsample to same shape as prob_grid
    if n_files > MAX_GRID_ROWS:
        idx        = np.linspace(0, n_files - 1, MAX_GRID_ROWS, dtype=int)
        depth_grid = depth_grid[idx, :]
        amp_grid   = amp_grid[idx, :]
        twt_grid   = twt_grid[idx, :]
    if max_sigs > MAX_GRID_COLS:
        idx        = np.linspace(0, max_sigs - 1, MAX_GRID_COLS, dtype=int)
        depth_grid = depth_grid[:, idx]
        amp_grid   = amp_grid[:, idx]
        twt_grid   = twt_grid[:, idx]

    # Extend trailing NaN same as prob_grid
    for g in (depth_grid, amp_grid, twt_grid):
        for i in range(g.shape[0]):
            valid = np.where(~np.isnan(g[i, :]))[0]
            if len(valid) and valid[-1] < g.shape[1] - 1:
                g[i, valid[-1] + 1:] = g[i, valid[-1]]

    return depth_grid, amp_grid, twt_grid


def render_rebar_depth_b64(
    depth_grid: np.ndarray,
    dpi: int = 100,
) -> str:
    """
    Render rebar depth grid as a heatmap PNG.
    Color scale: blue (shallow ≤1") → green → yellow → red (deep ≥4").
    """
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "depth", ['#2563EB', '#10B981', '#FBBF24', '#EF4444'], N=256,
    )
    cmap_obj.set_bad(color='#F0EFEC')

    nan_mask  = np.isnan(depth_grid)
    valid     = depth_grid[~nan_mask]
    vmin, vmax = (1.0, 4.0)
    if len(valid):
        vmin = max(0.0, np.percentile(valid, 5))
        vmax = max(vmin + 0.5, np.percentile(valid, 95))

    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    masked = np.ma.array(depth_grid, mask=nan_mask)

    smooth = gaussian_filter(
        np.where(nan_mask, (vmin + vmax) / 2, depth_grid), sigma=(1.5, 3.0)
    )
    masked_s = np.ma.array(smooth, mask=nan_mask)

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22)
    ax.imshow(masked_s, cmap=cmap_obj, norm=norm, aspect='auto', origin='upper',
              interpolation='bilinear')

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#CCCCCC')
    ax.set_xlabel('Scan line (longitudinal →)', fontsize=8, color='#555555', labelpad=6)
    ax.set_ylabel('Swath', fontsize=8, color='#555555', labelpad=6)
    ax.tick_params(colors='#777777', length=3, width=0.6, labelsize=7)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm),
        ax=ax, orientation='horizontal', pad=0.18, fraction=0.04, aspect=50,
    )
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_ticklabels([f'Shallow ({vmin:.1f}")', f'{(vmin+vmax)/2:.1f}"', f'Deep ({vmax:.1f}")'],
                        fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#CCCCCC')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def render_amplitude_b64(
    amplitude_grid: np.ndarray,
    dpi: int = 100,
) -> str:
    """
    Render amplitude grid as a heatmap PNG.
    Color: low amplitude (red/deteriorated) → high amplitude (blue/sound).
    """
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "amp", ['#C0392B', '#E67E22', '#F1C40F', '#27AE60', '#2980B9'], N=256,
    )
    cmap_obj.set_bad(color='#F0EFEC')

    nan_mask  = np.isnan(amplitude_grid)
    norm      = mcolors.Normalize(vmin=0.0, vmax=1.0)

    smooth = gaussian_filter(
        np.where(nan_mask, 0.5, amplitude_grid), sigma=(1.5, 3.0)
    )
    masked = np.ma.array(smooth, mask=nan_mask)

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22)
    ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect='auto', origin='upper',
              interpolation='bilinear')

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#CCCCCC')
    ax.set_xlabel('Scan line (longitudinal →)', fontsize=8, color='#555555', labelpad=6)
    ax.set_ylabel('Swath', fontsize=8, color='#555555', labelpad=6)
    ax.tick_params(colors='#777777', length=3, width=0.6, labelsize=7)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm),
        ax=ax, orientation='horizontal', pad=0.18, fraction=0.04, aspect=50,
    )
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['Low Amplitude (Deteriorated)', 'Boundary', 'High Amplitude (Sound)'],
                        fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#CCCCCC')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_confidence_metrics(
    all_confs: np.ndarray,
    amplitude_grid: np.ndarray,
    frequency_mhz: int = 1600,
) -> tuple[float, float, str]:
    """
    Returns (model_confidence_pct, depth_accuracy_in, signal_quality).

    model_confidence_pct : mean of max(p, 1-p) across all signals, 0-100
    depth_accuracy_in    : ±inches based on antenna frequency
    signal_quality       : 'Good', 'Fair', or 'Poor' from amplitude SNR
    """
    conf_pct = float(np.mean(all_confs) * 100) if len(all_confs) else 50.0

    depth_acc = {400: 1.0, 900: 0.5, 1600: 0.25, 2000: 0.25, 2600: 0.125}.get(
        frequency_mhz, 0.5
    )

    valid_amp = amplitude_grid[~np.isnan(amplitude_grid)]
    mean_amp  = float(np.mean(valid_amp)) if len(valid_amp) else 0.5
    quality   = 'Good' if mean_amp >= 0.65 else ('Fair' if mean_amp >= 0.40 else 'Poor')

    return round(conf_pct, 1), depth_acc, quality


# ── C-scan rendering ──────────────────────────────────────────────────────────

def render_cscan_b64(
    file_preds:  list[np.ndarray],
    file_confs:  list[np.ndarray],
    file_names:  list[str],
    bridge_name: str = "Bridge Deck",
    dpi:         int = 100,
) -> str:
    """
    Render a clean GPR condition heatmap.
    Colormap: red (deteriorated) → yellow (boundary) → blue (sound).
    """
    prob_grid, T = build_prob_grid(file_preds, file_confs)
    print(f"[render] Otsu threshold: {T:.4f}", flush=True)

    # Smooth and rescale so T maps to colormap midpoint (0.5)
    nan_mask = np.isnan(prob_grid)
    p = gaussian_filter(np.where(nan_mask, T, prob_grid), sigma=(1.5, 3.0))
    display = np.where(
        p <= T,
        0.5 * p / T,
        0.5 + 0.5 * (p - T) / (1.0 - T),
    )
    masked = np.ma.array(display, mask=nan_mask)
    del prob_grid, p, display

    # Colormap: red → orange → yellow → green → cyan → blue
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "gpr", ['#C0392B', '#E67E22', '#F1C40F', '#27AE60', '#2980B9'], N=256,
    )
    cmap_obj.set_bad(color='#F0EFEC')
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22)

    ax.imshow(
        masked, cmap=cmap_obj, norm=norm,
        aspect='auto', origin='upper',
        interpolation='bilinear',
    )

    # Clean border
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#CCCCCC')

    # Minimal axis labels — scan line index only
    ax.set_xlabel('Scan line (longitudinal →)', fontsize=8, color='#555555', labelpad=6)
    ax.set_ylabel('Swath', fontsize=8, color='#555555', labelpad=6)
    ax.tick_params(colors='#777777', length=3, width=0.6, labelsize=7)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm),
        ax=ax, orientation='horizontal', pad=0.18, fraction=0.04, aspect=50,
    )
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['Deteriorated', 'Boundary', 'Sound'], fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#CCCCCC')

    # ── Render ────────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── CLI ───────────────────────────────────────────────────────────────────────

def _resolve_inputs(argv: list[str]) -> list[Path]:
    """Resolve CLI arguments to a list of CSV file paths."""
    if not argv:
        found = sorted(_DEFAULT_INPUT.rglob("FILE____*.csv"))
        if not found:
            sys.exit(f"No FILE____*.csv files found in {_DEFAULT_INPUT}")
        return found
    paths = [Path(p) for p in argv]
    if len(paths) == 1 and paths[0].is_dir():
        found = sorted(paths[0].rglob("FILE____*.csv"))
        if not found:
            sys.exit(f"No FILE____*.csv files found in {paths[0]}")
        return found
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"File(s) not found: {missing}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Verus GPR Inference")
    parser.add_argument("--input",     help="Input folder or CSV file(s)")
    parser.add_argument("--model",     help="Path to model .pth file")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output",    help="Write JSON results to this path")
    parser.add_argument("--dpi",       type=int, default=72,
                        help="C-scan PNG DPI (default 72 for web)")
    parser.add_argument("inputs",      nargs="*",
                        help="Positional CSV files / folder")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else _DEFAULT_MODEL
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    global THRESHOLD
    if args.threshold is not None:
        THRESHOLD = args.threshold

    csv_files = _resolve_inputs([args.input] if args.input else args.inputs)

    print("=" * 60, flush=True)
    print("Verus GPR Bridge Deck Inference", flush=True)
    print(f"  Model      : {model_path}", flush=True)
    print(f"  Device     : {DEVICE}", flush=True)
    print(f"  Threshold  : {THRESHOLD}", flush=True)
    print(f"  Input files: {len(csv_files)}", flush=True)
    print("=" * 60, flush=True)

    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Loaded model — {n_params:,} trainable parameters", flush=True)

    t0         = time.perf_counter()
    file_preds: list[np.ndarray] = []
    file_confs: list[np.ndarray] = []
    file_names: list[str]        = []
    per_file_summary: list[dict] = []
    total_sigs = 0

    print(f"\n  {'File':36} {'Signals':>8} {'Sound%':>8} {'Delam%':>8}", flush=True)
    print(f"  {'-'*63}", flush=True)

    for fpath in csv_files:
        try:
            signals = load_csv(fpath)
        except Exception as e:
            print(f"  WARNING  {fpath.name}: {e}", flush=True)
            continue

        preds, confs = run_inference(model, signals)
        del signals
        gc.collect()

        n       = len(preds)
        n_snd   = int(preds.sum())
        pct_del = (n - n_snd) / n * 100
        pct_snd = n_snd / n * 100

        tag = f"{fpath.parent.name}/{fpath.name}"
        print(f"  {tag:36} {n:>8,} {pct_snd:>7.1f}% {pct_del:>7.1f}%", flush=True)

        file_preds.append(preds)
        file_confs.append(confs)
        file_names.append(str(fpath))
        per_file_summary.append({
            "filename":  fpath.name,
            "signals":   n,
            "delam_pct": round(pct_del, 2),
        })
        total_sigs += n

    elapsed = time.perf_counter() - t0
    all_preds   = np.concatenate(file_preds)
    delam_pct   = round(int((all_preds == 0).sum()) / total_sigs * 100, 2)
    sound_pct   = round(100.0 - delam_pct, 2)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Files: {len(file_preds)}  |  Signals: {total_sigs:,}  |  "
          f"Sound: {sound_pct}%  |  Delam: {delam_pct}%  |  "
          f"Time: {elapsed:.2f}s", flush=True)

    # C-scan + predictions list
    print("\n  Rendering C-scan …", flush=True)
    cscan_b64    = render_cscan_b64(file_preds, file_confs, file_names, dpi=args.dpi)
    predictions  = make_predictions_list(file_names, file_preds, file_confs)

    output = {
        "signals_analyzed":  total_sigs,
        "delamination_pct":  delam_pct,
        "sound_pct":         sound_pct,
        "analysis_time_sec": round(elapsed, 2),
        "cscan_image":       cscan_b64,
        "per_file_summary":  per_file_summary,
        "predictions_count": len(predictions),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"  Results JSON → {out_path.resolve()}", flush=True)
    else:
        print("\n" + json.dumps({k: v for k, v in output.items()
                                 if k != "cscan_image"}, indent=2))
        print(f'  "cscan_image": "<{len(cscan_b64)} chars base64>"')


if __name__ == "__main__":
    main()
