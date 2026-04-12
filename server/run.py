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
from scipy.signal import windows as sig_windows
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle

# ── Constants ─────────────────────────────────────────────────────────────────
THRESHOLD    = 0.65       # P(sound) < THRESHOLD → delaminated
DC_OFFSET    = 32768
N_SAMPLES    = 512
INFER_BATCH  = 1000       # max signals per torch forward pass (~2 MB each)
MAX_GRID_ROWS = 500       # C-scan grid rows, downsampled if larger
MAX_GRID_COLS = 100       # C-scan grid cols, downsampled if larger
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TAPER       = np.ones(N_SAMPLES, dtype=np.float32)
_TAPER[410:] = sig_windows.hann(204)[102:].astype(np.float32)

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

    Normalisation: subtract file mean, divide by file std.
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

    # ── DC offset + Hann taper (in-place) ────────────────────────────────────
    if np.abs(amps.mean()) > 1000:
        amps -= DC_OFFSET
    amps *= _TAPER[np.newaxis, :]

    # ── Spatial averaging radius=2 ────────────────────────────────────────────
    amps_avg = np.empty_like(amps)
    for i in range(n_signals):
        amps_avg[i] = amps[max(0, i - 2):i + 3].mean(axis=0)
    del amps
    gc.collect()

    # ── Per-file z-score normalisation (in-place) ─────────────────────────────
    std = amps_avg.std()
    if std < 1e-8:
        raise ValueError("Signal has no variation (constant values)")
    amps_avg -= amps_avg.mean()
    amps_avg /= std

    print(f"[load_csv] done: {n_signals} signals normalised", flush=True)
    return amps_avg   # (n_signals, 512)


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


# ── C-scan rendering ──────────────────────────────────────────────────────────

def render_cscan_b64(
    file_preds:  list[np.ndarray],
    file_confs:  list[np.ndarray],
    file_names:  list[str],
    bridge_name: str = "Bridge Deck",
    dpi:         int = 72,
) -> str:
    """
    Render an ASTM D6087-style bridge deck condition map.

    Layout:
      - 2D grid: X = file index (distance), Y = signal index (offset)
      - RdYlGn_r colormap (red = delaminated, green = sound)
      - gaussian_filter smoothing (sigma=2)
      - Bridge outline rectangle
      - Dashed lane markings at 20%, 40%, 60%, 80% of Y axis
      - ASTM header box top-right
      - Color legend / colorbar bottom
      - Axis labels: Distance (ft) and Offset (ft)
      - Figure size 24×10 inches, DPI=72 for web output

    Grid is downsampled to MAX_GRID_ROWS × MAX_GRID_COLS before rendering
    to cap memory usage regardless of input size.

    Returns base64-encoded PNG string (no disk I/O).
    """
    n_files  = len(file_preds)
    max_sigs = max(len(p) for p in file_preds)

    # ── Build P(delam) grid then downsample ───────────────────────────────────
    prob_grid = np.full((max_sigs, n_files), np.nan, dtype=np.float32)
    for col, (preds, confs) in enumerate(zip(file_preds, file_confs)):
        for row, (pred, conf) in enumerate(zip(preds, confs)):
            prob_grid[row, col] = conf if pred == 0 else 1.0 - conf

    if max_sigs > MAX_GRID_ROWS:
        row_idx   = np.linspace(0, max_sigs - 1, MAX_GRID_ROWS, dtype=int)
        prob_grid = prob_grid[row_idx, :]
    if n_files > MAX_GRID_COLS:
        col_idx   = np.linspace(0, n_files - 1, MAX_GRID_COLS, dtype=int)
        prob_grid = prob_grid[:, col_idx]

    grid_rows, grid_cols = prob_grid.shape
    nan_mask = np.isnan(prob_grid)
    filled   = np.where(nan_mask, 0.0, prob_grid)
    smoothed = gaussian_filter(filled, sigma=2.0)
    masked   = np.ma.array(smoothed, mask=nan_mask)
    del filled, prob_grid

    # ── Stats (use original counts for header text) ───────────────────────────
    all_preds  = np.concatenate(file_preds)
    total_sigs = len(all_preds)
    n_delam    = int((all_preds == 0).sum())
    pct_delam  = n_delam / total_sigs * 100 if total_sigs else 0.0
    deck_area  = n_files * max_sigs
    delam_area = deck_area * pct_delam / 100.0
    del all_preds

    # ── Figure ────────────────────────────────────────────────────────────────
    half  = grid_cols // 2
    spans = [(0, half, "Span 1"), (half, grid_cols, "Span 2")]

    fig = plt.figure(figsize=(24, 10), facecolor="white")
    ax  = fig.add_axes([0.06, 0.14, 0.70, 0.72])

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    ax.imshow(
        masked,
        cmap=cmap, norm=norm,
        aspect="auto", origin="upper",
        extent=[-0.5, grid_cols - 0.5, grid_rows - 0.5, -0.5],
        interpolation="bilinear",
    )

    # Bridge outline
    ax.add_patch(Rectangle(
        (-0.5, -0.5), grid_cols, grid_rows,
        linewidth=2.0, edgecolor="#1A1A1A", facecolor="none", zorder=5,
    ))

    # Dashed lane markings at 20%, 40%, 60%, 80% of Y axis
    for frac in (0.20, 0.40, 0.60, 0.80):
        ax.axhline(
            grid_rows * frac,
            color="#333333", linewidth=1.1,
            linestyle=(0, (8, 6)), alpha=0.75, zorder=4,
        )

    # Pier at midpoint
    ax.axvline(half, color="#222266", linewidth=1.8, linestyle="-", alpha=0.85, zorder=4)
    ax.text(half, -1.8, "PIER", ha="center", va="bottom",
            fontsize=6.5, color="#222266", fontweight="bold", clip_on=False)

    # Span bracket lines and labels
    for s_fi, e_fi, s_label in spans:
        mid = (s_fi + e_fi) / 2.0
        ax.text(mid, -3.5, s_label, ha="center", va="bottom",
                fontsize=9, color="#111111", fontweight="bold", clip_on=False)
        for xb in [s_fi, e_fi]:
            ax.plot([xb, xb], [-0.5, -2.8], color="#555555",
                    linewidth=0.8, clip_on=False, zorder=3)

    # Axis labels and ticks
    ax.set_xlabel("Distance (ft)", fontsize=10, labelpad=6)
    ax.set_ylabel("Offset (ft)",   fontsize=10, labelpad=6)

    x_ticks = np.linspace(0, grid_cols - 1, min(11, grid_cols), dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(v) for v in x_ticks], fontsize=8)

    y_ticks = np.linspace(0, grid_rows - 1, min(9, grid_rows), dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(v) for v in y_ticks], fontsize=8)
    ax.tick_params(direction="out", length=4, width=0.8)

    # Scale bar
    sb_len = max(5, round(grid_cols * 0.15 / 5) * 5)
    sb_x0  = grid_cols * 0.04
    sb_y   = grid_rows * 0.96
    ax.annotate("", xy=(sb_x0 + sb_len, sb_y), xytext=(sb_x0, sb_y),
                arrowprops=dict(arrowstyle="<->", color="#111111", lw=1.2), zorder=6)
    ax.text(sb_x0 + sb_len / 2, sb_y + grid_rows * 0.025, f"{sb_len} ft",
            ha="center", va="bottom", fontsize=7.5)

    # North arrow
    na_x, na_y = grid_cols * 0.96, grid_rows * 0.88
    ax.annotate("", xy=(na_x, na_y - grid_rows * 0.12), xytext=(na_x, na_y),
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.4,
                                mutation_scale=10), zorder=6)
    ax.text(na_x, na_y + grid_rows * 0.01, "N", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

    # ── ASTM header box (top-right) ───────────────────────────────────────────
    delam_color = "#C0392B" if pct_delam > 15 else ("#E67E22" if pct_delam > 5 else "#27AE60")
    hdr = fig.add_axes([0.775, 0.38, 0.205, 0.52])
    hdr.set_axis_off()
    hdr.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="#333333", facecolor="#F5F5F5",
        transform=hdr.transAxes,
    ))
    hdr.add_patch(Rectangle(
        (0, 0.82), 1, 0.18, transform=hdr.transAxes,
        facecolor="#1F3864", edgecolor="none",
    ))
    header_items = [
        ("BRIDGE DECK CONDITION ASSESSMENT", 11, "bold",   0.95, "white"),
        ("GPR – CONCRETE DELAMINATION",       9, "bold",   0.89, "white"),
        (f"Structure:   {bridge_name}",        8, "normal", 0.79, "#111111"),
        (f"Survey Date: {datetime.date.today().strftime('%B %d, %Y')}",
                                               8, "normal", 0.72, "#111111"),
        ("Standard:    ASTM D6087",            8, "normal", 0.65, "#111111"),
        (f"Scan lines:  {n_files}",            8, "normal", 0.58, "#111111"),
        (f"Signals:     {total_sigs:,}",       8, "normal", 0.51, "#111111"),
        (f"Deck Area:   {deck_area:,.0f} ft²", 8, "normal", 0.44, "#111111"),
        ("",                                   5, "normal", 0.37, "#111111"),
        (f"Total Delam: {pct_delam:.1f}%",     9, "bold",   0.30, delam_color),
        (f"Delam Area:  {delam_area:,.0f} ft²",8, "normal", 0.22, "#111111"),
    ]
    for text, fsize, fweight, y, color in header_items:
        hdr.text(0.05, y, text, transform=hdr.transAxes,
                 fontsize=fsize, fontweight=fweight, color=color, va="top")

    # ── Per-span summary (below header) ──────────────────────────────────────
    orig_half  = n_files // 2
    orig_spans = [(0, orig_half, "Span 1"), (orig_half, n_files, "Span 2")]
    span_ax = fig.add_axes([0.775, 0.075, 0.205, 0.28])
    span_ax.set_axis_off()
    span_ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02",
        linewidth=1.0, edgecolor="#AAAAAA", facecolor="#FAFAFA",
        transform=span_ax.transAxes,
    ))
    span_ax.text(0.5, 0.95, "SPAN SUMMARY", transform=span_ax.transAxes,
                 fontsize=8, fontweight="bold", color="#1F3864", ha="center", va="top")
    for col_x, col_lbl in [(0.04, "Span"), (0.32, "Delam%"),
                            (0.56, "Area ft²"), (0.78, "Delam ft²")]:
        span_ax.text(col_x, 0.84, col_lbl, transform=span_ax.transAxes,
                     fontsize=7, fontweight="bold", va="top")
    row_y = 0.72
    for s_fi, e_fi, s_label in orig_spans:
        sp_preds = np.concatenate(file_preds[s_fi:e_fi]) if s_fi < e_fi else np.array([])
        sp_n     = len(sp_preds)
        sp_del   = int((sp_preds == 0).sum()) if sp_n else 0
        sp_pct   = sp_del / sp_n * 100 if sp_n else 0.0
        sp_area  = (e_fi - s_fi) * max_sigs
        sp_da    = sp_area * sp_pct / 100.0
        col      = "#C0392B" if sp_pct > 25 else ("#E67E22" if sp_pct > 10 else "#27AE60")
        span_ax.text(0.04, row_y, s_label,           transform=span_ax.transAxes, fontsize=7, va="top")
        span_ax.text(0.32, row_y, f"{sp_pct:.1f}%",  transform=span_ax.transAxes, fontsize=7, va="top",
                     color=col, fontweight="bold")
        span_ax.text(0.56, row_y, f"{sp_area:,.0f}", transform=span_ax.transAxes, fontsize=7, va="top")
        span_ax.text(0.78, row_y, f"{sp_da:,.0f}",   transform=span_ax.transAxes, fontsize=7, va="top")
        row_y -= 0.20

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.06, 0.055, 0.70, 0.030])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Concrete Delamination Probability", fontsize=9, labelpad=4)
    cbar.set_ticks([0.0, 0.35, 0.50, 0.65, 1.0])
    cbar.set_ticklabels(["0%  (Sound)", "35%", "50%", "65%", "100%  (Delaminated)"],
                        fontsize=7.5)
    cbar.outline.set_linewidth(0.8)

    # Color legend swatches
    sw_ax = fig.add_axes([0.06, 0.005, 0.70, 0.040])
    sw_ax.set_axis_off()
    for sx, sc, sl in [
        (0.02, "#CCCCCC", "N/A"),
        (0.18, "#2ECC71", "Low  (Sound)"),
        (0.44, "#F39C12", "Medium (Uncertain)"),
        (0.68, "#E84040", "High  (Delaminated)"),
    ]:
        sw_ax.add_patch(Rectangle((sx, 0.35), 0.035, 0.55,
                                   transform=sw_ax.transAxes,
                                   facecolor=sc, edgecolor="#555555", linewidth=0.6))
        sw_ax.text(sx + 0.042, 0.62, sl, transform=sw_ax.transAxes,
                   fontsize=7.5, va="center")

    # Footnote
    fig.text(
        0.06, 0.002,
        f"Survey performed in accordance with ASTM D6087.  |  "
        f"Threshold: P(delamination) > {1.0 - THRESHOLD:.2f}  |  "
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=6, color="#555555", va="bottom",
    )

    # ── Render to in-memory PNG (no disk I/O) ─────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


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
