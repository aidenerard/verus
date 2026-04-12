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
import datetime
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
from matplotlib.patches import FancyBboxPatch, Rectangle

# ── Constants ─────────────────────────────────────────────────────────────────
THRESHOLD     = 0.65       # P(sound) < THRESHOLD → delaminated
DC_OFFSET     = 32768
N_SAMPLES     = 512
INFER_BATCH   = 1000       # max signals per torch forward pass (~2 MB each)
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
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,  128, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.attn = TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.attn(x)
        return self.head(x).squeeze(1)


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


# ── C-scan rendering ──────────────────────────────────────────────────────────

def render_cscan_b64(
    file_preds:  list[np.ndarray],
    file_confs:  list[np.ndarray],
    file_names:  list[str],
    bridge_name: str = "Bridge Deck",
    dpi:         int = 150,
) -> str:
    """
    Render a professional ASTM D6087 / FHWA LTBP-style GPR deterioration map.

    Colormap: deep red (high attenuation = deteriorated) → orange → yellow →
              green → cyan → blue → violet (low attenuation = sound), matching
              the spectral scale used by Sensoft/IRIS and the FHWA LTBP program.
    Layout: 24 × 6 in landscape, 300 DPI.  X = Longitudinal Distance (ft),
            Y = Lateral Distance (ft).  Horizontal dB colorbar below.
    """
    from matplotlib.patches import Polygon as MplPolygon

    n_files  = len(file_preds)
    max_sigs = max(len(p) for p in file_preds)

    # ── Grid: rows = scan lines (lateral), cols = signals (longitudinal) ───────
    # Store raw P(sound) directly from the model sigmoid output.
    # confs convention (from run_inference):
    #   pred=1 (sound):       conf = P(sound)
    #   pred=0 (delaminated): conf = 1 - P(sound)  [= P(delam)]
    # So P(sound) = conf if pred==1 else 1.0 - conf.
    prob_grid = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    for row, (preds, confs) in enumerate(zip(file_preds, file_confs)):
        for col, (pred, conf) in enumerate(zip(preds, confs)):
            prob_grid[row, col] = conf if pred == 1 else 1.0 - conf  # P(sound)

    # Downsample if grid exceeds memory limits
    if n_files > MAX_GRID_ROWS:
        idx       = np.linspace(0, n_files - 1, MAX_GRID_ROWS, dtype=int)
        prob_grid = prob_grid[idx, :]
    if max_sigs > MAX_GRID_COLS:
        idx       = np.linspace(0, max_sigs - 1, MAX_GRID_COLS, dtype=int)
        prob_grid = prob_grid[:, idx]

    # Fill trailing NaN in each row (shorter scan lines) by extending last value.
    # Without this, scan lines shorter than max_sigs show as gray on the right.
    for i in range(prob_grid.shape[0]):
        valid = np.where(~np.isnan(prob_grid[i, :]))[0]
        if len(valid) and valid[-1] < prob_grid.shape[1] - 1:
            prob_grid[i, valid[-1] + 1 :] = prob_grid[i, valid[-1]]

    # Upsample longitudinally when too few signals (sparse test uploads)
    if prob_grid.shape[1] < 50:
        scale     = max(1, 50 // prob_grid.shape[1])
        prob_grid = zoom(prob_grid, (1, scale), order=1)

    grid_lat, grid_long = prob_grid.shape   # (lateral rows, longitudinal cols)

    # ── Stats + optimal threshold ─────────────────────────────────────────────
    all_preds  = np.concatenate(file_preds)
    all_confs  = np.concatenate(file_confs)
    total_sigs = len(all_preds)

    # Reconstruct raw P(sound) from (pred, conf) pairs across all files, then
    # find the Otsu-optimal threshold for THIS dataset.  This adapts to each
    # bridge — noise level, rebar depth, pavement overlay thickness all shift
    # the bimodal P(sound) distribution, and a fixed threshold would be wrong.
    all_psound = np.where(all_preds == 1, all_confs, 1.0 - all_confs)
    T = _otsu_threshold(all_psound)
    del all_psound, all_confs

    n_delam   = int((all_preds == 0).sum())
    pct_delam = n_delam / total_sigs * 100 if total_sigs else 0.0
    del all_preds

    print(f"[render] Otsu threshold: {T:.4f}  (fixed THRESHOLD={THRESHOLD})",
          flush=True)

    # ── Threshold-centred rescaling ───────────────────────────────────────────
    # Pin the Otsu boundary to 0.5 (colormap midpoint) so that every signal
    # the model labels delaminated always falls in the warm half (red/orange/
    # yellow) and every sound signal in the cool half (green/cyan/blue),
    # regardless of confidence magnitude.
    #
    #   P(sound) in [0, T] → display in [0.0, 0.5]
    #   P(sound) in [T, 1] → display in [0.5, 1.0]
    nan_mask = np.isnan(prob_grid)
    p        = np.where(nan_mask, T, prob_grid)   # fill NaN with boundary

    # Gaussian smoothing produces the continuous, organic blob appearance seen
    # in professional ASTM D6087 maps.  Smoothing happens on raw P(sound)
    # values BEFORE rescaling so that the threshold boundary stays sharp.
    # sigma=(lateral, longitudinal): more lateral blending blends between scan
    # lines; longitudinal blends along the drive direction.
    p = gaussian_filter(p, sigma=(1.5, 3.0))

    display  = np.where(
        p <= T,
        0.5 * p / T,                          # [0, T] → [0, 0.5]
        0.5 + 0.5 * (p - T) / (1.0 - T),     # [T, 1] → [0.5, 1.0]
    )
    masked = np.ma.array(display, mask=nan_mask)
    del prob_grid, p, display

    # ── Colormap: deep red → orange → yellow → green → cyan → blue → violet ───
    # Matches Sensoft/IRIS and FHWA LTBP spectral attenuation scale.
    # After threshold-centred rescaling:
    #   display=0.0 → dark red  → P(sound)=0    → highly deteriorated
    #   display=0.5 → yellow    → P(sound)=0.65 → decision boundary (threshold)
    #   display=1.0 → indigo    → P(sound)=1.0  → confidently sound
    cmap_colors = [
        '#8B0000', '#FF0000', '#FF4500', '#FF8C00', '#FFD700',
        '#ADFF2F', '#00FF7F', '#00CED1', '#1E90FF', '#4B0082',
    ]
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "gpr_attn", cmap_colors, N=256,
    )
    cmap_obj.set_bad(color='lightgray')
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 5), facecolor='white')
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.20)

    # Title — centred above everything
    fig.text(0.5, 0.97, "Bridge Deck Condition Assessment",
             ha='center', va='top', fontsize=11, fontweight='bold', color='#111111')

    # Header block — top-left, above the map, not overlaid on it
    survey_date = datetime.date.today().strftime('%B %d, %Y')
    hdr = (
        f"Structure: {bridge_name}    Survey: {survey_date}\n"
        f"Standard: ASTM D6087    Scan lines: {n_files}\n"
        f"Signals: {total_sigs}    Threshold: {T:.3f} (Otsu)    Delamination: {pct_delam:.1f}%"
    )
    fig.text(0.07, 0.91, hdr,
             ha='left', va='top', fontsize=7, color='#333333',
             fontfamily='monospace', linespacing=1.6)

    # Map axes
    ax = fig.add_axes([0.07, 0.20, 0.90, 0.60])

    ax.imshow(
        masked,
        cmap=cmap_obj, norm=norm,
        aspect='auto', origin='upper',
        extent=[0, grid_long, grid_lat, 0],
        interpolation='bilinear',
    )

    # Bridge outline rectangle
    ax.add_patch(Rectangle(
        (0, 0), grid_long, grid_lat,
        linewidth=1.5, edgecolor='black', facecolor='none', zorder=5,
    ))

    # Hatched abutment triangle — left edge (approach end)
    tri_w    = max(6, int(grid_long * 0.04))
    abutment = MplPolygon(
        [(0, 0), (0, grid_lat), (tri_w, grid_lat)],
        closed=True, hatch='///', facecolor='none',
        edgecolor='black', linewidth=0.8, zorder=6,
    )
    ax.add_patch(abutment)

    # Dashed scan-line reference markers (every ~25% of lateral span)
    for frac in (0.25, 0.50, 0.75):
        ax.axhline(grid_lat * frac, color='black', linewidth=0.5,
                   linestyle='--', alpha=0.45, zorder=4)

    # "Deterioration Map" label — top-right corner of map axes
    ax.text(0.99, 0.97, "Deterioration Map", transform=ax.transAxes,
            ha='right', va='top', fontsize=9, style='italic', color='#222222',
            zorder=7)

    # Axis labels and ticks
    ax.set_xlabel('Longitudinal Distance (ft.)', fontsize=8, labelpad=4)
    ax.set_ylabel('Lateral\nDistance\n(ft.)', fontsize=8, labelpad=4)

    n_xticks = 12
    x_pos    = np.linspace(0, grid_long, n_xticks + 1, dtype=int)
    est_len  = max_sigs * 0.03          # ~0.03 ft/signal at 400 MHz
    x_ft     = np.round(np.linspace(0, est_len, n_xticks + 1)).astype(int)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in x_ft], fontsize=7)

    n_yticks = min(grid_lat, 8)
    y_pos    = np.linspace(0, grid_lat, n_yticks + 1, dtype=int)
    y_ft     = np.round(np.linspace(0, n_files, n_yticks + 1)).astype(int)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(v) for v in y_ft], fontsize=7)
    ax.tick_params(direction='out', length=3, width=0.6)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.20, 0.05, 0.55, 0.055])
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    # Specific dB ticks: -38 -33 -30 -27 -24 -21 -18 -15 -14
    db_ticks = [-38, -33, -30, -27, -24, -21, -18, -15, -14]
    db_range = db_ticks[-1] - db_ticks[0]
    tick_pos = [(v - db_ticks[0]) / db_range for v in db_ticks]
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels([str(v) for v in db_ticks], fontsize=7)
    cbar.ax.tick_params(length=2, width=0.5)
    cbar.outline.set_linewidth(0.6)
    cbar_ax.set_title(
        "Attenuation at top rebar level (dB)  —  corrected for depth variation",
        fontsize=8, pad=3,
    )

    # Colored "More / Less" labels flanking the colorbar
    cbar_ax.text(-0.01, 0.5, "More\nDeterioration",
                 transform=cbar_ax.transAxes, ha='right', va='center',
                 fontsize=7, fontweight='bold', color='#8B0000')
    cbar_ax.text(1.01, 0.5, "Less Deteriorated /\nStronger Reflections",
                 transform=cbar_ax.transAxes, ha='left', va='center',
                 fontsize=7, fontweight='bold', color='#1E3A8A')

    # ── Render to in-memory PNG ────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    gc.collect()   # release matplotlib canvas memory immediately
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
