#!/usr/bin/env python3
"""
scripts/eval_model.py  —  GPR delamination model evaluation

Run from the repo root (verus/):
    python scripts/eval_model.py --model server/model.pth --data data/csv/sdnet2021/
    python scripts/eval_model.py --model server/model.pth --data data/csv/ --threshold 0.65

Reads model_config.json from the same directory as model.pth when present.
Outputs to eval_results/<timestamp>/:
    metrics.json          overall metrics at requested + best-F1 thresholds
    per_bridge.csv        per-bridge precision/recall/F1/FNR
    confusion_matrix.png
    pr_curve.png
    threshold_sweep.png

Label convention (matches kaggle_push/cnn.py throughout):
    model output = P(sound) via sigmoid
    1 = sound,  0 = delaminated
    "positive class" for inspection metrics = delaminated (label 0)
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.signal import windows as sig_windows
from sklearn.metrics import average_precision_score, precision_recall_curve

import torch
import torch.nn as nn

# ── Constants — must match kaggle_push/cnn.py ─────────────────────────────────
DC_OFFSET  = 32768
N_SAMPLES  = 512
CROP_START = 200
CROP_END   = 450
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TAPER       = np.ones(N_SAMPLES, dtype=np.float32)
_TAPER[410:] = sig_windows.hann(204)[102:].astype(np.float32)


# ── Model (mirrors kaggle_push/cnn.py exactly) ────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x.permute(0, 2, 1)), dim=1)
        return (x * w.permute(0, 2, 1)).sum(dim=2)


class CNN1D(nn.Module):
    def __init__(self, in_channels=2, conv_channels=(32, 128, 128), head_hidden=128):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.attn = TemporalAttention(c3)
        self.head = nn.Sequential(
            nn.Linear(c3, head_hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.conv(x))).squeeze(1)


# ── Data loading (SDNET2021 CSV format) ───────────────────────────────────────

def _find_data_start(raw: np.ndarray) -> int:
    for row in range(9, 14):
        try:
            val = float(raw[row, 0])
            if not np.isnan(val):
                return row
        except (ValueError, TypeError):
            continue
    raise ValueError("Cannot locate amplitude rows (expected rows 9–13)")


def load_csv(fpath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        amps:   (n, 512) float32, per-signal z-score
        labels: (n,) int — 1=sound, 0=delaminated
    """
    raw       = pd.read_csv(fpath, header=None).values
    n_signals = int(raw[0, 4])
    raw_labels = raw[7, 1 : n_signals + 1].astype(int)
    data_start = _find_data_start(raw)
    amp_block  = raw[data_start : data_start + N_SAMPLES,
                     0 : n_signals + 1].astype(np.float32)
    if amp_block.shape[0] < N_SAMPLES:
        pad       = np.zeros((N_SAMPLES - amp_block.shape[0], amp_block.shape[1]),
                             dtype=np.float32)
        amp_block = np.vstack([amp_block, pad])
    amps = ((amp_block[:, 1:] - DC_OFFSET) * _TAPER[:, np.newaxis]).T
    mean = amps.mean(axis=1, keepdims=True)
    std  = amps.std(axis=1,  keepdims=True) + 1e-8
    amps = (amps - mean) / std
    labels = (raw_labels == 1).astype(int)
    return amps, labels


def build_tensor(fpath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (n, 2, 250) float32 input tensor and (n,) int labels."""
    amps, labels = load_csv(fpath)
    env = np.abs(hilbert(amps, axis=1)).astype(np.float32)
    x   = np.stack([amps, env], axis=1)[:, :, CROP_START:CROP_END]
    return x, labels


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            t = torch.tensor(X[s : s + batch], dtype=torch.float32).to(DEVICE)
            probs.append(model(t).sigmoid().cpu().numpy())
    return np.concatenate(probs)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Positive class = delaminated = label 0."""
    y_pred = (y_prob >= threshold).astype(int)
    n_d = int((y_true == 0).sum())
    n_s = int((y_true == 1).sum())
    TP  = int(((y_pred == 0) & (y_true == 0)).sum())
    FP  = int(((y_pred == 0) & (y_true == 1)).sum())
    FN  = int(((y_pred == 1) & (y_true == 0)).sum())
    TN  = int(((y_pred == 1) & (y_true == 1)).sum())
    sens = TP / n_d if n_d else 0.0
    spec = TN / n_s if n_s else 0.0
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    fnr  = FN / n_d if n_d else 0.0
    pr_auc = average_precision_score(1 - y_true, 1.0 - y_prob)
    return dict(
        threshold=round(float(threshold), 4),
        n_delam=n_d, n_sound=n_s,
        TP=TP, FP=FP, FN=FN, TN=TN,
        precision=round(prec, 6), recall=round(sens, 6),
        f1=round(f1, 6), fnr=round(fnr, 6),
        specificity=round(spec, 6), pr_auc=round(pr_auc, 6),
    )


def find_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, list[dict]]:
    """Sweep 0.10–0.90 in steps of 0.01, return best F1 threshold and full sweep."""
    sweep: list[dict] = []
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.10, 0.90, 81):
        m = compute_metrics(y_true, y_prob, float(t))
        sweep.append(m)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t, sweep


# ── Plots ──────────────────────────────────────────────────────────────────────

def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def save_confusion_matrix(m: dict, out_path: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    matrix = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Sound", "Pred Delam"])
    ax.set_yticklabels(["GT Sound", "GT Delam"])
    for (r, c), val in np.ndenumerate(matrix):
        ax.text(c, r, f"{val:,}", ha="center", va="center", fontsize=12,
                color="white" if val > matrix.max() * 0.6 else "black")
    ax.set_title(f"Confusion Matrix  (threshold={m['threshold']:.2f})\n"
                 f"Precision={m['precision']*100:.1f}%  Recall={m['recall']*100:.1f}%  "
                 f"F1={m['f1']:.4f}  FNR={m['fnr']*100:.1f}%")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, pr_auc: float, out_path: Path
) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    prec_pts, rec_pts, _ = precision_recall_curve(1 - y_true, 1.0 - y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_pts, prec_pts, lw=2, color="#E8601C")
    ax.fill_between(rec_pts, prec_pts, alpha=0.15, color="#E8601C")
    ax.set_xlabel("Recall (Delaminated)")
    ax.set_ylabel("Precision (Delaminated)")
    ax.set_title(f"Precision-Recall Curve — AUC={pr_auc:.4f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_threshold_sweep(sweep: list[dict], best_t: float, out_path: Path) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    ts  = [m["threshold"] for m in sweep]
    f1s = [m["f1"] for m in sweep]
    rec = [m["recall"] for m in sweep]
    pre = [m["precision"] for m in sweep]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ts, f1s, label="F1 (delam)",  lw=2,   color="#E8601C")
    ax.plot(ts, rec, label="Recall",       lw=1.5, color="steelblue", linestyle="--")
    ax.plot(ts, pre, label="Precision",    lw=1.5, color="seagreen",  linestyle="--")
    ax.axvline(best_t, color="red", linestyle=":", lw=1.5,
               label=f"best threshold={best_t:.2f}")
    ax.set_xlabel("Threshold (P(sound) cutoff)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep  (positive = delaminated)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Per-bridge breakdown ───────────────────────────────────────────────────────

def per_bridge_breakdown(file_results: list[dict], threshold: float) -> list[dict]:
    """Group files by parent directory name (bridge ID) and compute metrics for each."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for fr in file_results:
        groups[fr["bridge"]].append(fr)

    rows = []
    for bridge in sorted(groups):
        frs    = groups[bridge]
        y_true = np.concatenate([fr["y_true"] for fr in frs])
        y_prob = np.concatenate([fr["y_prob"] for fr in frs])
        m      = compute_metrics(y_true, y_prob, threshold)
        rows.append({
            "bridge":    bridge,
            "n_files":   len(frs),
            "n_delam":   m["n_delam"],
            "n_sound":   m["n_sound"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "fnr":       m["fnr"],
            "pr_auc":    m["pr_auc"],
        })
    return rows


# ── Printing helpers ───────────────────────────────────────────────────────────

def print_metrics_block(m: dict, label: str = "") -> None:
    bar = "=" * 60
    if label:
        print(f"\n{bar}\n{label}\n{bar}")
    print(f"\n  Confusion matrix  (positive = delaminated = label 0)")
    print(f"  {'':22}  Pred Sound  Pred Delam")
    print(f"  {'GT Sound':22}  {m['TN']:>10,}  {m['FP']:>10,}")
    print(f"  {'GT Delaminated':22}  {m['FN']:>10,}  {m['TP']:>10,}")
    print(f"\n  Precision (of flagged-delam) {m['precision']*100:>6.1f}%")
    print(f"  Recall / Sensitivity         {m['recall']*100:>6.1f}%")
    print(f"  F1 (delaminated class)        {m['f1']:>8.4f}")
    print(f"  False-negative rate (FNR)    {m['fnr']*100:>6.1f}%  ← missed delamination")
    print(f"  Specificity                  {m['specificity']*100:>6.1f}%")
    print(f"  PR-AUC (delam positive)       {m['pr_auc']:>8.4f}")


def print_per_bridge(rows: list[dict], threshold: float) -> None:
    if not rows:
        return
    print(f"\n  Per-bridge breakdown  (threshold={threshold:.2f})")
    hdr = f"  {'Bridge':>24}  {'Files':>5}  {'Delam':>7}  {'Sound':>7}"
    hdr += f"  {'Prec%':>6}  {'Rec%':>6}  {'F1':>6}  {'FNR%':>5}"
    print(hdr)
    print(f"  {'-'*80}")
    for r in rows:
        print(
            f"  {r['bridge']:>24}  {r['n_files']:>5}  {r['n_delam']:>7,}  {r['n_sound']:>7,}"
            f"  {r['precision']*100:>5.1f}%  {r['recall']*100:>5.1f}%"
            f"  {r['f1']:>6.4f}  {r['fnr']*100:>4.1f}%"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GPR delamination model — full metrics + plots"
    )
    parser.add_argument("--model",     required=True, help="Path to model.pth checkpoint")
    parser.add_argument("--data",      required=True, help="Root dir of CSV data files")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold to report at (default 0.5); best-F1 also reported")
    parser.add_argument("--out",       default="eval_results",
                        help="Output directory prefix (default: eval_results)")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_root  = Path(args.data)

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    if not data_root.exists():
        sys.exit(f"Data dir not found: {data_root}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Verus GPR Model Eval  —  {ts}")
    print(f"  model     : {model_path}")
    print(f"  data      : {data_root}")
    print(f"  threshold : {args.threshold}")
    print(f"  device    : {DEVICE}")
    print(f"  output    : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load model ──────────────────────────────────────────────────────────────
    config_path = model_path.parent / "model_config.json"
    cfg: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"  model_config.json: {cfg}")
    else:
        print("  No model_config.json found — using default CNN1D(in_channels=2)")

    in_ch   = cfg.get("in_channels", 2)
    conv_ch = cfg.get("conv_channels", [32, 128, 128])
    head_h  = cfg.get("head_hidden", 128)

    model = CNN1D(in_channels=in_ch, conv_channels=conv_ch, head_hidden=head_h).to(DEVICE)
    model.load_state_dict(
        torch.load(str(model_path), map_location=DEVICE, weights_only=False)
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  CNN1D  in_channels={in_ch}  conv={conv_ch}  head={head_h}  params={n_params:,}\n")

    # ── Discover CSV files ──────────────────────────────────────────────────────
    csv_files = sorted(data_root.rglob("FILE____*.csv"))
    if not csv_files:
        csv_files = sorted(data_root.rglob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV files found under {data_root}")
    print(f"  Found {len(csv_files)} CSV files\n")

    # ── Per-file inference ──────────────────────────────────────────────────────
    all_y_true: list[np.ndarray] = []
    all_y_prob: list[np.ndarray] = []
    file_results: list[dict]     = []

    for fpath in csv_files:
        try:
            X, labels = build_tensor(fpath)
            y_prob    = run_inference(model, X)
            all_y_true.append(labels)
            all_y_prob.append(y_prob)
            bridge = fpath.parent.name
            file_results.append({
                "file": fpath.name, "bridge": bridge,
                "y_true": labels, "y_prob": y_prob,
            })
            n_d = int((labels == 0).sum())
            print(f"  {bridge}/{fpath.name:<40}  n={len(labels):>7,}  delam={n_d:>7,}", flush=True)
        except Exception as exc:
            print(f"  SKIP {fpath.name}: {exc}", flush=True)

    if not all_y_true:
        sys.exit("No data loaded — check CSV format (expected SDNET2021 FILE____.csv).")

    y_true = np.concatenate(all_y_true)
    y_prob = np.concatenate(all_y_prob)
    n_d = int((y_true == 0).sum())
    n_s = int((y_true == 1).sum())
    print(f"\n  Total: {len(y_true):,}  delam={n_d:,}  sound={n_s:,}")

    # ── Compute metrics ─────────────────────────────────────────────────────────
    m_req        = compute_metrics(y_true, y_prob, args.threshold)
    best_t, sweep = find_best_threshold(y_true, y_prob)
    m_best       = compute_metrics(y_true, y_prob, best_t)
    bridge_rows  = per_bridge_breakdown(file_results, best_t)

    print_metrics_block(m_req,  label=f"At requested threshold={args.threshold:.2f}")
    print_metrics_block(m_best, label=f"At best-F1 threshold={best_t:.2f}")
    print_per_bridge(bridge_rows, best_t)

    # ── Threshold sweep table ───────────────────────────────────────────────────
    print(f"\n  Threshold sweep (every 0.10 step)")
    print(f"  {'Thresh':>7}  {'Sens%':>7}  {'Spec%':>7}  {'Prec%':>7}  {'F1':>7}  {'FNR%':>6}")
    print(f"  {'-'*50}")
    for m in sweep:
        t = m["threshold"]
        if round(t * 100) % 10 == 0:
            star = " ← best" if abs(t - best_t) < 1e-6 else ""
            print(f"  {t:>7.2f}  {m['recall']*100:>6.1f}%  {m['specificity']*100:>6.1f}%  "
                  f"{m['precision']*100:>6.1f}%  {m['f1']:>6.4f}  {m['fnr']*100:>5.1f}%{star}")

    # ── Save outputs ────────────────────────────────────────────────────────────
    print(f"\n  Writing outputs to {out_dir}/")

    results_doc = {
        "run":       ts,
        "model":     str(model_path),
        "data":      str(data_root),
        "n_total":   int(len(y_true)),
        "n_delam":   int(n_d),
        "n_sound":   int(n_s),
        "at_requested_threshold": m_req,
        "at_best_f1_threshold":   m_best,
        "per_bridge": bridge_rows,
        "threshold_sweep": sweep,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results_doc, f, indent=2)
    print(f"  Saved: {out_dir / 'metrics.json'}")

    if bridge_rows:
        with open(out_dir / "per_bridge.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=bridge_rows[0].keys())
            w.writeheader()
            w.writerows(bridge_rows)
        print(f"  Saved: {out_dir / 'per_bridge.csv'}")

    save_confusion_matrix(m_best, out_dir / "confusion_matrix.png")
    save_pr_curve(y_true, y_prob, m_best["pr_auc"], out_dir / "pr_curve.png")
    save_threshold_sweep(sweep, best_t, out_dir / "threshold_sweep.png")

    print(f"\nDone. All outputs in: {out_dir}/\n")


if __name__ == "__main__":
    main()
