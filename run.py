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

import numpy as np
import pandas as pd
from scipy.signal import windows as sig_windows

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

    # C-scan
    save_cscan(file_preds, file_confs, file_names, CSCAN_OUT)


if __name__ == "__main__":
    main()
