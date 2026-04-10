"""
run.py — GPR bridge deck delamination inference (no training).

Loads a trained CNN1D model and runs inference on all CSV files in
INPUT_PATH (a folder) or a list of CSV file paths supplied via the
command line.

Usage
-----
    # Scan a folder:
    python3 run.py /path/to/csv_folder

    # Scan specific files:
    python3 run.py file1.csv file2.csv file3.csv

    # Use defaults (INPUT_PATH variable below):
    python3 run.py

Outputs
-------
    - Console summary table per file + overall stats
    - bridge_cscan.png  — 2-D C-scan heat map (X=file, Y=signal)
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import datetime

import numpy as np
import pandas as pd
from scipy.signal import windows as sig_windows
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Configure these two paths ─────────────────────────────────────────────────
MODEL_PATH = Path("/content/drive/MyDrive/fluxspace_gpr_data/model.pth")
INPUT_PATH = Path("~/Desktop/verus/all_bridges_csv").expanduser()
CSCAN_OUT  = Path("bridge_cscan.png")
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD  = 0.65          # predict delaminated if sigmoid(logit) < THRESHOLD
DC_OFFSET  = 32768
N_SAMPLES  = 512
BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hann taper — same as cnn.py
_TAPER       = np.ones(N_SAMPLES, dtype=np.float32)
_TAPER[410:] = sig_windows.hann(204)[102:].astype(np.float32)


# ── Model architecture — must exactly match cnn.py ───────────────────────────

class TemporalAttention(nn.Module):
    """Weighted sum over time steps. Input: (B, C, T) → Output: (B, C)."""
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x.permute(0, 2, 1)), dim=1)  # (B, T, 1)
        return (x * weights.permute(0, 2, 1)).sum(dim=2)                 # (B, C)


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


# ── Data loading — identical normalisation to cnn.py ─────────────────────────

def _find_data_start(raw) -> int:
    for row in range(9, 14):
        try:
            val = float(raw[row, 0])
            if not np.isnan(val):
                return row
        except (ValueError, TypeError):
            continue
    raise ValueError("Could not locate amplitude data rows in file")


def load_csv(fpath: Path) -> np.ndarray:
    """
    Load one CSV file and return normalised signals as (n_signals, 512).
    Per-file normalisation: subtract file mean, divide by file std.
    Labels are not loaded — this is inference-only.
    """
    raw        = pd.read_csv(fpath, header=None).values
    n_signals  = int(raw[0, 4])
    data_start = _find_data_start(raw)
    amp_block  = raw[data_start:data_start + N_SAMPLES, 0:n_signals + 1].astype(np.float32)

    if amp_block.shape[0] < N_SAMPLES:
        pad = np.zeros((N_SAMPLES - amp_block.shape[0], amp_block.shape[1]), dtype=np.float32)
        amp_block = np.vstack([amp_block, pad])

    amps = (amp_block[:, 1:] - DC_OFFSET) * _TAPER[:, np.newaxis]

    # Spatial average (radius=2) — matches cnn.py pre-processing
    n = amps.shape[1]
    amps_avg = np.empty_like(amps)
    for i in range(n):
        amps_avg[:, i] = amps[:, max(0, i - 2):i + 3].mean(axis=1)

    # Per-file normalisation
    amps_avg = (amps_avg - amps_avg.mean()) / (amps_avg.std() + 1e-8)
    return amps_avg.T   # (n_signals, 512)


# ── Resolve input files ───────────────────────────────────────────────────────

def resolve_inputs(argv: list[str]) -> list[Path]:
    if argv:
        paths = [Path(p) for p in argv]
        # If a single directory was given, glob it
        if len(paths) == 1 and paths[0].is_dir():
            found = sorted(paths[0].rglob("FILE____*.csv"))
            if not found:
                sys.exit(f"No FILE____*.csv files found in {paths[0]}")
            return found
        # Otherwise treat each argument as a file
        missing = [p for p in paths if not p.exists()]
        if missing:
            sys.exit(f"File(s) not found: {missing}")
        return paths
    else:
        # Fall back to INPUT_PATH default
        found = sorted(INPUT_PATH.rglob("FILE____*.csv"))
        if not found:
            sys.exit(f"No FILE____*.csv files found in {INPUT_PATH}")
        return found


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model on (n_signals, 512) array.
    Returns:
        preds  — int array of 0 (delaminated) or 1 (sound), shape (n,)
        confs  — float array of confidence in predicted class, shape (n,)
    """
    tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)  # (n, 1, 512)
    ds     = TensorDataset(tensor)
    dl     = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    probs_list = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            logits = model(xb.to(DEVICE))
            probs_list.append(logits.sigmoid().cpu().numpy())

    probs = np.concatenate(probs_list)            # P(sound)
    preds = (probs >= THRESHOLD).astype(int)      # 1=sound, 0=delaminated
    confs = np.where(preds == 1, probs, 1.0 - probs)   # confidence in predicted class
    return preds, confs


# ── C-scan visualisation ──────────────────────────────────────────────────────

def save_cscan(file_preds: list[np.ndarray], file_confs: list[np.ndarray],
               file_names: list[str], out_path: Path) -> None:
    """
    2-D C-scan heat map.
      X axis = file index (scan line position along bridge)
      Y axis = signal index within file
      Green  = predicted sound  (intensity ∝ confidence)
      Red    = predicted delaminated (intensity ∝ confidence)
    """
    n_files  = len(file_preds)
    max_sigs = max(len(p) for p in file_preds)

    # RGB image: background neutral grey
    img = np.full((max_sigs, n_files, 3), 0.85, dtype=np.float32)

    for col, (preds, confs) in enumerate(zip(file_preds, file_confs)):
        for row, (pred, conf) in enumerate(zip(preds, confs)):
            if pred == 1:          # sound → green channel
                img[row, col] = [1 - conf * 0.9, 1.0, 1 - conf * 0.9]
            else:                  # delaminated → red channel
                img[row, col] = [1.0, 1 - conf * 0.9, 1 - conf * 0.9]

    fig, ax = plt.subplots(figsize=(max(10, n_files * 0.35), 8))
    ax.imshow(img, aspect="auto", origin="upper",
              extent=[-0.5, n_files - 0.5, max_sigs - 0.5, -0.5])

    ax.set_xlabel("Scan line (file index)", fontsize=11)
    ax.set_ylabel("Signal index within scan line", fontsize=11)
    ax.set_title("GPR Bridge Deck C-scan — Delamination Map", fontsize=13, fontweight="bold")

    # Colour legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=(0.1, 1.0, 0.1), label="Sound (confidence-weighted)"),
            Patch(facecolor=(1.0, 0.1, 0.1), label="Delaminated (confidence-weighted)"),
        ],
        loc="upper right", fontsize=9,
    )

    # X-tick labels: show file names sparsely if many files
    tick_step = max(1, n_files // 20)
    ax.set_xticks(range(0, n_files, tick_step))
    ax.set_xticklabels(
        [Path(file_names[i]).name[:12] for i in range(0, n_files, tick_step)],
        rotation=45, ha="right", fontsize=7,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  C-scan saved → {out_path.resolve()}", flush=True)


# ── ASTM D6087 condition map ──────────────────────────────────────────────────

def generate_cscan_map(
    file_preds:  list[np.ndarray],
    file_confs:  list[np.ndarray],
    file_names:  list[str],
    bridge_name: str = "Bridge Deck",
    ft_per_file: float = 1.0,     # longitudinal distance represented by one file (ft)
    ft_per_sig:  float = 1.0,     # transverse offset represented by one signal (ft)
    lane_offsets_ft: list[float] | None = None,  # Y positions of lane dividers
    pier_file_indices: list[int]  | None = None,  # X positions (file indices) of piers
    span_labels: list[tuple[int, int, str]] | None = None,  # [(start_fi, end_fi, label), ...]
    out_stem:    str = "bridge_deck_condition",
) -> None:
    """
    Generate a professional ASTM D6087-style bridge deck condition map.

    Parameters
    ----------
    file_preds        : per-file prediction arrays (1=sound, 0=delaminated)
    file_confs        : per-file confidence arrays (confidence in predicted class)
    file_names        : file path strings (used for axis labels)
    bridge_name       : title string embedded in the header box
    ft_per_file       : feet of bridge length per scan-line file
    ft_per_sig        : feet of bridge width per signal within a file
    lane_offsets_ft   : Y positions (ft) where lane-divider dashes are drawn
    pier_file_indices : file indices where pier lines are drawn
    span_labels       : list of (start_file_idx, end_file_idx, "Span N") tuples
    out_stem          : output filename stem — saves <stem>.png and <stem>.pdf
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
    from matplotlib.lines import Line2D
    import matplotlib.ticker as ticker
    import matplotlib.patheffects as pe

    n_files  = len(file_preds)
    max_sigs = max(len(p) for p in file_preds)

    # ── 1. Build delamination probability grid ────────────────────────────────
    # Grid shape: (max_sigs, n_files)  — rows=offset, cols=distance
    # Cell value: P(delaminated) = 1 - P(sound)
    # Where pred==1 (sound):   P(delam) = 1 - confidence  (low)
    # Where pred==0 (delam):   P(delam) = confidence       (high)
    prob_grid = np.full((max_sigs, n_files), np.nan)
    for col, (preds, confs) in enumerate(zip(file_preds, file_confs)):
        for row, (pred, conf) in enumerate(zip(preds, confs)):
            prob_grid[row, col] = conf if pred == 0 else 1.0 - conf

    # Fill NaN (padding for short files) with 0 before smoothing
    nan_mask = np.isnan(prob_grid)
    prob_grid_filled = np.where(nan_mask, 0.0, prob_grid)

    # Gaussian smoothing — creates natural blob-like delamination zones
    smoothed = gaussian_filter(prob_grid_filled, sigma=(1.8, 1.8))
    # Re-mask padding region so it shows as N/A
    smoothed_masked = np.ma.array(smoothed, mask=nan_mask)

    # ── 2. Compute statistics ─────────────────────────────────────────────────
    all_preds_flat = np.concatenate(file_preds)
    total_sigs     = len(all_preds_flat)
    n_delam        = int((all_preds_flat == 0).sum())
    pct_delam_total = n_delam / total_sigs * 100 if total_sigs else 0.0

    # Deck area in ft² (approximate)
    deck_length_ft = n_files  * ft_per_file
    deck_width_ft  = max_sigs * ft_per_sig
    deck_area_ft2  = deck_length_ft * deck_width_ft
    delam_area_ft2 = deck_area_ft2 * pct_delam_total / 100.0

    # Per-span stats
    span_stats: list[dict] = []
    if span_labels:
        for (s_fi, e_fi, s_label) in span_labels:
            s_fi = max(0, min(s_fi, n_files - 1))
            e_fi = max(s_fi, min(e_fi, n_files))
            span_preds = np.concatenate(file_preds[s_fi:e_fi]) if s_fi < e_fi else np.array([])
            span_n     = len(span_preds)
            span_del   = int((span_preds == 0).sum()) if span_n else 0
            span_pct   = span_del / span_n * 100 if span_n else 0.0
            span_area  = (e_fi - s_fi) * ft_per_file * deck_width_ft
            span_stats.append({
                "label":    s_label,
                "start_fi": s_fi,
                "end_fi":   e_fi,
                "pct_delam": span_pct,
                "area_ft2":  span_area,
                "delam_ft2": span_area * span_pct / 100.0,
            })

    # ── 3. Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 10), facecolor="white")

    # Main axes: leave room for header box on the right and legend below
    ax = fig.add_axes([0.06, 0.14, 0.70, 0.72])   # [left, bottom, width, height]

    # ── 4. Colour map — RdYlGn reversed (red=bad, green=good) ────────────────
    cmap = plt.cm.RdYlGn_r
    # Clamp colours to match the three-zone scheme:
    #   0.00–0.35 → green  (sound)
    #   0.35–0.65 → yellow (uncertain)
    #   0.65–1.00 → red    (delaminated)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    extent = [-0.5, n_files - 0.5, max_sigs - 0.5, -0.5]
    im = ax.imshow(
        smoothed_masked,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="upper",
        extent=extent,
        interpolation="bilinear",
    )

    # ── 5. Deck outline ───────────────────────────────────────────────────────
    deck_rect = Rectangle(
        (-0.5, -0.5), n_files, max_sigs,
        linewidth=2.0, edgecolor="#1A1A1A", facecolor="none", zorder=5,
    )
    ax.add_patch(deck_rect)

    # ── 6. Lane markings — dashed horizontal lines ────────────────────────────
    if lane_offsets_ft:
        lane_sig_indices = [y / ft_per_sig for y in lane_offsets_ft]
        for ly in lane_sig_indices:
            ax.axhline(
                ly, color="#333333", linewidth=1.1,
                linestyle=(0, (8, 6)), alpha=0.75, zorder=4,
            )

    # ── 7. Pier markers — solid vertical lines with label ─────────────────────
    if pier_file_indices:
        for px in pier_file_indices:
            ax.axvline(
                px, color="#222266", linewidth=1.8,
                linestyle="-", alpha=0.85, zorder=4,
            )
            ax.text(
                px, -1.8, "PIER", ha="center", va="bottom",
                fontsize=6.5, color="#222266", fontweight="bold",
                clip_on=False,
            )

    # ── 8. Span labels — above the deck ───────────────────────────────────────
    if span_labels:
        for (s_fi, e_fi, s_label) in span_labels:
            mid = (s_fi + e_fi) / 2.0
            ax.text(
                mid, -3.5, s_label, ha="center", va="bottom",
                fontsize=9, color="#111111", fontweight="bold",
                clip_on=False,
            )
            # Bracket lines
            for xb in [s_fi, e_fi]:
                ax.plot(
                    [xb, xb], [-0.5, -2.8], color="#555555",
                    linewidth=0.8, clip_on=False, zorder=3,
                )

    # ── 9. Axis labels and ticks ──────────────────────────────────────────────
    ax.set_xlabel("Distance (feet)", fontsize=10, labelpad=6)
    ax.set_ylabel("Offset (feet)",   fontsize=10, labelpad=6)

    # Convert file-index ticks → feet
    x_tick_fi  = np.linspace(0, n_files - 1, min(11, n_files), dtype=int)
    ax.set_xticks(x_tick_fi)
    ax.set_xticklabels([f"{v * ft_per_file:.0f}" for v in x_tick_fi], fontsize=8)

    y_tick_si  = np.linspace(0, max_sigs - 1, min(9, max_sigs), dtype=int)
    ax.set_yticks(y_tick_si)
    ax.set_yticklabels([f"{v * ft_per_sig:.0f}" for v in y_tick_si], fontsize=8)

    ax.tick_params(direction="out", length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # ── 10. Scale bar ─────────────────────────────────────────────────────────
    sb_len_ft  = max(10, round(deck_length_ft * 0.15 / 10) * 10)   # ~15% of deck
    sb_len_fi  = sb_len_ft / ft_per_file
    sb_x0      = n_files * 0.04
    sb_y       = max_sigs * 0.96
    ax.annotate(
        "", xy=(sb_x0 + sb_len_fi, sb_y), xytext=(sb_x0, sb_y),
        arrowprops=dict(arrowstyle="<->", color="#111111", lw=1.2),
        zorder=6,
    )
    ax.text(
        sb_x0 + sb_len_fi / 2, sb_y + max_sigs * 0.025,
        f"{sb_len_ft:.0f} ft", ha="center", va="bottom",
        fontsize=7.5, color="#111111",
    )

    # ── 11. North arrow ───────────────────────────────────────────────────────
    na_x = n_files * 0.96
    na_y = max_sigs * 0.88
    ax.annotate(
        "", xy=(na_x, na_y - max_sigs * 0.12), xytext=(na_x, na_y),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.4,
                        mutation_scale=10),
        zorder=6,
    )
    ax.text(na_x, na_y + max_sigs * 0.01, "N", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#111111")

    # ── 12. Header info box (top-right of figure) ─────────────────────────────
    header_ax = fig.add_axes([0.775, 0.38, 0.205, 0.52])
    header_ax.set_axis_off()

    header_lines = [
        ("BRIDGE DECK CONDITION ASSESSMENT", 11, "bold"),
        ("GPR – CONCRETE DELAMINATION", 9,  "bold"),
        ("", 6, "normal"),
        (f"Structure: {bridge_name}", 8, "normal"),
        (f"Survey Date: {datetime.date.today().strftime('%B %d, %Y')}", 8, "normal"),
        (f"Standard: ASTM D6087", 8, "normal"),
        ("", 5, "normal"),
        (f"Deck Length:   {deck_length_ft:.0f} ft", 8, "normal"),
        (f"Deck Width:    {deck_width_ft:.0f} ft", 8, "normal"),
        (f"Total Area:    {deck_area_ft2:,.0f} ft²", 8, "normal"),
        ("", 5, "normal"),
        (f"Total Delam:   {pct_delam_total:.1f}%", 9, "bold"),
        (f"Delam Area:    {delam_area_ft2:,.0f} ft²", 8, "normal"),
    ]

    header_ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="#333333", facecolor="#F5F5F5",
        transform=header_ax.transAxes, zorder=0,
    ))
    header_ax.add_patch(Rectangle(
        (0.0, 0.82), 1.0, 0.18,
        transform=header_ax.transAxes,
        facecolor="#1F3864", edgecolor="none", zorder=1,
    ))

    y_pos = 0.95
    for text, fsize, fweight in header_lines:
        color = "white" if y_pos > 0.82 else "#111111"
        header_ax.text(
            0.05, y_pos, text,
            transform=header_ax.transAxes,
            fontsize=fsize, fontweight=fweight, color=color,
            va="top", ha="left",
        )
        y_pos -= 0.06 if text == "" else 0.072

    # ── 13. Per-span summary boxes (below header) ─────────────────────────────
    if span_stats:
        n_spans    = len(span_stats)
        box_h      = 0.28 / n_spans
        span_top   = 0.355
        span_ax    = fig.add_axes([0.775, span_top - 0.28, 0.205, 0.30])
        span_ax.set_axis_off()

        span_ax.add_patch(FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0.02",
            linewidth=1.0, edgecolor="#AAAAAA", facecolor="#FAFAFA",
            transform=span_ax.transAxes,
        ))
        span_ax.text(
            0.5, 0.97, "SPAN SUMMARY",
            transform=span_ax.transAxes,
            fontsize=8, fontweight="bold", color="#1F3864",
            ha="center", va="top",
        )

        col_headers = ["Span", "Delam%", "Area (ft²)", "Delam (ft²)"]
        col_x       = [0.04, 0.30, 0.55, 0.78]
        span_ax.text(0.04, 0.88, "Span",      transform=span_ax.transAxes, fontsize=7, fontweight="bold", va="top")
        span_ax.text(0.30, 0.88, "Delam%",    transform=span_ax.transAxes, fontsize=7, fontweight="bold", va="top")
        span_ax.text(0.55, 0.88, "Area ft²",  transform=span_ax.transAxes, fontsize=7, fontweight="bold", va="top")
        span_ax.text(0.78, 0.88, "Delam ft²", transform=span_ax.transAxes, fontsize=7, fontweight="bold", va="top")

        row_y = 0.80
        for ss in span_stats:
            pct   = ss["pct_delam"]
            color = "#C0392B" if pct > 25 else ("#E67E22" if pct > 10 else "#27AE60")
            span_ax.text(0.04, row_y, ss["label"],              transform=span_ax.transAxes, fontsize=7, va="top")
            span_ax.text(0.30, row_y, f"{pct:.1f}%",            transform=span_ax.transAxes, fontsize=7, va="top", color=color, fontweight="bold")
            span_ax.text(0.55, row_y, f"{ss['area_ft2']:,.0f}", transform=span_ax.transAxes, fontsize=7, va="top")
            span_ax.text(0.78, row_y, f"{ss['delam_ft2']:,.0f}",transform=span_ax.transAxes, fontsize=7, va="top")
            row_y -= 0.13

    # ── 14. Colour legend bar at bottom ──────────────────────────────────────
    cbar_ax = fig.add_axes([0.06, 0.055, 0.70, 0.030])
    sm      = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar    = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Concrete Delamination Probability", fontsize=9, labelpad=4)
    cbar.set_ticks([0.0, 0.175, 0.35, 0.50, 0.65, 0.825, 1.0])
    cbar.set_ticklabels(["0%", "", "35%", "50%", "65%", "", "100%"], fontsize=7.5)
    cbar.outline.set_linewidth(0.8)

    # Swatch labels below the colorbar
    swatch_ax = fig.add_axes([0.06, 0.005, 0.70, 0.040])
    swatch_ax.set_axis_off()
    swatches = [
        (0.02,  "#CCCCCC", "N/A"),
        (0.18,  "#2ECC71", "Low  (Sound)"),
        (0.44,  "#F39C12", "Medium (Uncertain)"),
        (0.68,  "#E84040", "High  (Delaminated)"),
    ]
    for sx, sc, sl in swatches:
        swatch_ax.add_patch(Rectangle(
            (sx, 0.35), 0.035, 0.55,
            transform=swatch_ax.transAxes,
            facecolor=sc, edgecolor="#555555", linewidth=0.6,
        ))
        swatch_ax.text(
            sx + 0.042, 0.62, sl,
            transform=swatch_ax.transAxes,
            fontsize=7.5, va="center",
        )

    # ── 15. Footnotes ─────────────────────────────────────────────────────────
    footnotes = (
        f"Survey performed in accordance with ASTM D6087 – Standard Test Method for Evaluating "
        f"Asphalt-Covered Concrete Bridge Decks Using Ground Penetrating Radar.  |  "
        f"Model threshold: P(delamination) > {1.0 - THRESHOLD:.2f}  |  "
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"Total scan lines: {n_files}  |  Signals analysed: {total_sigs:,}"
    )
    fig.text(
        0.06, 0.002, footnotes,
        fontsize=6, color="#555555", va="bottom",
        wrap=True,
    )

    # ── 16. Save ──────────────────────────────────────────────────────────────
    png_path = Path(f"{out_stem}.png")
    pdf_path = Path(f"{out_stem}.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n  Condition map saved:", flush=True)
    print(f"    PNG → {png_path.resolve()}", flush=True)
    print(f"    PDF → {pdf_path.resolve()}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_files = resolve_inputs(sys.argv[1:])

    # Load model
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}\n"
                 "Train first with cnn.py before running inference.")

    print("=" * 60, flush=True)
    print("GPR Bridge Deck Inference", flush=True)
    print(f"  Model      : {MODEL_PATH}", flush=True)
    print(f"  Device     : {DEVICE}", flush=True)
    print(f"  Threshold  : {THRESHOLD}  (sound if P(sound) ≥ {THRESHOLD})", flush=True)
    print(f"  Input files: {len(csv_files)}", flush=True)
    print("=" * 60, flush=True)

    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Loaded model — {n_params:,} trainable parameters", flush=True)

    # Per-file inference
    print(f"\n  {'File':36} {'Signals':>8} {'Sound%':>8} {'Delam%':>8} {'Avg conf':>9}", flush=True)
    print(f"  {'-'*73}", flush=True)

    t0 = time.perf_counter()

    results     = {}          # file_path → dict
    file_preds  = []
    file_confs  = []
    file_names  = []
    total_sigs  = 0

    for fpath in csv_files:
        try:
            signals = load_csv(fpath)
        except Exception as e:
            print(f"  WARNING  {fpath.name}: {e}", flush=True)
            continue

        preds, confs = run_inference(model, signals)

        n      = len(preds)
        n_snd  = int(preds.sum())
        n_del  = n - n_snd
        pct_snd = n_snd / n * 100
        pct_del = n_del / n * 100
        avg_conf = float(confs.mean())

        tag = f"{fpath.parent.name}/{fpath.name}"
        print(f"  {tag:36} {n:>8,} {pct_snd:>7.1f}% {pct_del:>7.1f}% {avg_conf:>8.3f}", flush=True)

        results[str(fpath)] = {
            "n_signals":         n,
            "predictions":       preds.tolist(),
            "confidences":       confs.tolist(),
            "pct_sound":         round(pct_snd, 2),
            "pct_delaminated":   round(pct_del, 2),
            "avg_confidence":    round(avg_conf, 4),
        }

        file_preds.append(preds)
        file_confs.append(confs)
        file_names.append(str(fpath))
        total_sigs += n

    elapsed   = time.perf_counter() - t0
    sig_per_s = total_sigs / elapsed if elapsed > 0 else 0.0

    # Overall summary
    all_preds = np.concatenate(file_preds)
    all_confs = np.concatenate(file_confs)
    tot_snd   = int(all_preds.sum())
    tot_del   = total_sigs - tot_snd

    overall = {
        "total_signals":        total_sigs,
        "total_sound":          tot_snd,
        "total_delaminated":    tot_del,
        "pct_sound":            round(tot_snd / total_sigs * 100, 2) if total_sigs else 0,
        "pct_delaminated":      round(tot_del / total_sigs * 100, 2) if total_sigs else 0,
        "avg_confidence":       round(float(all_confs.mean()), 4),
        "elapsed_s":            round(elapsed, 2),
        "signals_per_second":   round(sig_per_s, 1),
    }
    results["__overall__"] = overall

    print(f"\n{'=' * 60}", flush=True)
    print("OVERALL SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Files analysed   : {len(file_preds)}", flush=True)
    print(f"  Total signals    : {total_sigs:,}", flush=True)
    print(f"  Sound            : {tot_snd:,}  ({overall['pct_sound']:.1f}%)", flush=True)
    print(f"  Delaminated      : {tot_del:,}  ({overall['pct_delaminated']:.1f}%)", flush=True)
    print(f"  Avg confidence   : {overall['avg_confidence']:.3f}", flush=True)
    print(f"  Time             : {elapsed:.2f} s  ({sig_per_s:,.0f} signals/s)", flush=True)

    # Simple C-scan (quick raster)
    save_cscan(file_preds, file_confs, file_names, CSCAN_OUT)

    # ASTM D6087 professional condition map
    # Span and pier geometry — adjust to match the actual bridge layout
    n_fi = len(file_preds)
    half = n_fi // 2
    generate_cscan_map(
        file_preds  = file_preds,
        file_confs  = file_confs,
        file_names  = file_names,
        bridge_name = INPUT_PATH.name,
        ft_per_file = 1.0,
        ft_per_sig  = 1.0,
        lane_offsets_ft   = [len(file_preds[0]) * 0.33, len(file_preds[0]) * 0.66]
                              if file_preds else None,
        pier_file_indices = [half],
        span_labels       = [(0, half, "Span 1"), (half, n_fi, "Span 2")],
        out_stem    = "bridge_deck_condition",
    )


if __name__ == "__main__":
    main()
