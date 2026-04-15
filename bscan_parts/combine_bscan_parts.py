"""
combine_bscan_parts.py
──────────────────────
Runs locally on Mac. Combines the 20 downloaded Kaggle CSV parts into
SDNET2021-format FILE____NNN.csv files loadable by cnn.py.

Usage:
    cd <folder containing this script and the synthetic_bscan_c*.csv files>
    python combine_bscan_parts.py

Input CSVs (no header, integer values):
    synthetic_bscan_c1_p01.csv … c1_p10.csv   (sound,       label col = 1)
    synthetic_bscan_c2_p01.csv … c2_p10.csv   (delaminated, label col = 2)
    Each row: 512 amplitude columns + 1 label column.

Output (SDNET2021 format, loadable by cnn.py without changes):
    ./combined_output/
        synthetic_sound/FILE____001.csv …
        synthetic_delam/FILE____001.csv …

    Each file layout — (521, n+1) integer CSV:
        Row 0 col 4   : n_signals
        Row 7 cols 1..n : labels (1=sound or 2=delaminated)
        Rows 9..520   : amplitude data  (col 0 = 0, cols 1..n = A-scans)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
HERE            = Path(__file__).parent
OUT_ROOT        = HERE / "combined_output"
MAX_PER_FILE    = 1000
N_SAMPLES       = 512
DC_OFFSET       = 32_768


# ── Load part CSVs ────────────────────────────────────────────────────────────

def load_parts(glob_pattern: str) -> tuple:
    """
    Load all CSV parts matching the pattern.
    Each CSV has no header: 512 amplitude cols + 1 label col.
    Returns (signals, labels) as numpy arrays.
    """
    paths = sorted(HERE.glob(glob_pattern))
    if not paths:
        return np.empty((0, N_SAMPLES), dtype=np.int32), np.empty(0, dtype=np.int32)

    chunks_x, chunks_y = [], []
    for p in paths:
        df  = pd.read_csv(p, header=None, dtype=np.int32)
        arr = df.values                                   # (n, 513)
        chunks_x.append(arr[:, :N_SAMPLES])              # (n, 512) amplitudes
        chunks_y.append(arr[:, N_SAMPLES])               # (n,)     labels
        print(f"  Loaded {p.name}  ({len(arr):,} signals)", flush=True)

    X = np.concatenate(chunks_x, axis=0)
    y = np.concatenate(chunks_y, axis=0)
    return X, y


# ── SDNET2021 file writer ─────────────────────────────────────────────────────

def save_sdnet(signals: np.ndarray, labels: np.ndarray, out_dir: Path) -> int:
    """
    Write signals as FILE____NNN.csv files in SDNET2021 format.

    Layout per file — (521, n+1) integer CSV:
        Row 0, col 4     : n_signals
        Row 7, cols 1..n : labels
        Row 8            : zeros
        Rows 9..520      : amplitude data (col 0 = 0, cols 1..n = A-scans)

    Returns number of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(signals)
    n_files = 0

    for start in range(0, N, MAX_PER_FILE):
        end      = min(start + MAX_PER_FILE, N)
        sigs_blk = signals[start:end]    # (n, 512)
        lbl_blk  = labels[start:end]     # (n,)
        n        = len(sigs_blk)

        grid = np.zeros((521, n + 1), dtype=np.int32)
        grid[0, 4]          = n
        grid[7, 1 : n + 1]  = lbl_blk
        grid[9 : 9 + N_SAMPLES, 1 : n + 1] = sigs_blk.T.astype(np.int32)

        n_files += 1
        fpath = out_dir / f"FILE____{n_files:03d}.csv"
        np.savetxt(fpath, grid, delimiter=",", fmt="%d")

    return n_files


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60, flush=True)
print("combine_bscan_parts.py", flush=True)
print(f"  Source dir : {HERE}", flush=True)
print(f"  Output dir : {OUT_ROOT}", flush=True)
print("=" * 60, flush=True)

# Load
print("\nLoading Class 1 (sound) parts …", flush=True)
c1_paths = sorted(HERE.glob("synthetic_bscan_c1_p*.csv"))
X1, y1   = load_parts("synthetic_bscan_c1_p*.csv")

print("\nLoading Class 2 (delaminated) parts …", flush=True)
c2_paths = sorted(HERE.glob("synthetic_bscan_c2_p*.csv"))
X2, y2   = load_parts("synthetic_bscan_c2_p*.csv")

print(f"\nClass 1 parts found: {len(c1_paths)}", flush=True)
print(f"Class 2 parts found: {len(c2_paths)}", flush=True)
print(f"Total Class 1 signals: {len(X1):,}", flush=True)
print(f"Total Class 2 signals: {len(X2):,}", flush=True)

if len(X1) == 0 and len(X2) == 0:
    sys.exit("No part CSVs found. Place synthetic_bscan_c*_p*.csv files in the same directory as this script.")

# Save
print("\nSaving SDNET2021 files …", flush=True)
n_sound = save_sdnet(X1, y1, OUT_ROOT / "synthetic_sound")
n_delam = save_sdnet(X2, y2, OUT_ROOT / "synthetic_delam")

print(f"\nSound files created: {n_sound}", flush=True)
print(f"Delam files created: {n_delam}", flush=True)
print(f"Output saved to {OUT_ROOT}/", flush=True)
print("=" * 60, flush=True)
