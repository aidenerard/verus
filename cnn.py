"""
1D-CNN classifier on raw A-scan waveforms — GPR bridge deck delamination.
Loads from pre-converted CSVs in DATA_FOLDER.
Uses PyTorch (CPU).

Behaviour:
  - If MODEL_PATH does not exist: trains on all CSVs in DATA_FOLDER, saves model.
  - If MODEL_PATH exists:         skips training, loads saved model, evaluates on
                                   all CSVs in DATA_FOLDER (useful for a new bridge).

Usage:
    python3 cnn.py
"""

import warnings, sys
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import windows as sig_windows
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# ── Change these two paths to point at any folder of CSVs ─────────────────────
DATA_FOLDER = Path("~/Desktop/verus/all_bridges_csv").expanduser()
MODEL_PATH  = Path("/content/drive/MyDrive/fluxspace_gpr_data/model.pth")
# ──────────────────────────────────────────────────────────────────────────────

DC_OFFSET = 32768
N_SAMPLES = 512
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hann taper — falling half over samples 410–511
_TAPER       = np.ones(N_SAMPLES, dtype=np.float32)
_TAPER[410:] = sig_windows.hann(204)[102:].astype(np.float32)


# ─── Data loading ─────────────────────────────────────────────────────────────

def _find_data_start(raw) -> int:
    """Return the first row index (9–13) where col 0 is a non-NaN float (the time column)."""
    for row in range(9, 14):
        try:
            val = float(raw[row, 0])
            if not np.isnan(val):
                return row
        except (ValueError, TypeError):
            continue
    raise ValueError("Could not locate amplitude data rows in file")


def load_csv(fpath: Path) -> tuple[np.ndarray, np.ndarray]:
    raw        = pd.read_csv(fpath, header=None).values
    n_signals  = int(raw[0, 4])
    raw_labels = raw[7, 1:n_signals + 1].astype(int)
    data_start = _find_data_start(raw)
    amp_block  = raw[data_start:data_start + N_SAMPLES, 0:n_signals + 1].astype(np.float32)
    # Some files have one fewer sample — zero-pad to N_SAMPLES
    if amp_block.shape[0] < N_SAMPLES:
        pad = np.zeros((N_SAMPLES - amp_block.shape[0], amp_block.shape[1]), dtype=np.float32)
        amp_block = np.vstack([amp_block, pad])
    amps = (amp_block[:, 1:] - DC_OFFSET) * _TAPER[:, np.newaxis]
    # Per-file normalization: subtract file mean, divide by file std
    amps = (amps - amps.mean()) / (amps.std() + 1e-8)
    labels = (raw_labels == 1).astype(int)   # 1=sound, 0=delaminated
    return amps, labels


def spatial_average(amps: np.ndarray, radius: int = 2) -> np.ndarray:
    n, out = amps.shape[1], np.empty_like(amps)
    for i in range(n):
        out[:, i] = amps[:, max(0, i-radius):i+radius+1].mean(axis=1)
    return out


# ─── Model definition (must match architecture used during training) ───────────

class TemporalAttention(nn.Module):
    """Weighted sum over time steps. Input: (B, C, T) → Output: (B, C)."""
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → permute → (B, T, C)
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
            nn.Linear(128, 1),  # raw logit; sigmoid applied in loss / eval
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)        # (B, 128, T//8)
        x = self.attn(x)        # (B, 128)
        return self.head(x).squeeze(1)  # (B,)


def augment_batch(xb: torch.Tensor) -> torch.Tensor:
    """Random augmentations applied per-batch during training only."""
    # Gaussian noise: std = 1% of each signal's std
    noise_std = xb.std(dim=-1, keepdim=True) * 0.01
    xb = xb + torch.randn_like(xb) * noise_std

    # Amplitude scale: uniform in [0.95, 1.05]
    scale = torch.empty(xb.size(0), 1, 1, device=xb.device).uniform_(0.95, 1.05)
    xb = xb * scale

    # Time shift: ±5 samples, zero-pad edges
    shift = random.randint(-5, 5)
    if shift > 0:
        xb = torch.cat([torch.zeros_like(xb[:, :, :shift]),  xb[:, :, :-shift]], dim=2)
    elif shift < 0:
        s = -shift
        xb = torch.cat([xb[:, :, s:], torch.zeros_like(xb[:, :, :s])], dim=2)

    return xb


# ─── Shared: load all CSVs in DATA_FOLDER ─────────────────────────────────────

csv_files = sorted(DATA_FOLDER.rglob("FILE____*.csv"))
if not csv_files:
    sys.exit(f"No FILE____*.csv files found in {DATA_FOLDER}")

print(f"Step 1 — Loading CSV files from {DATA_FOLDER.name}", flush=True)
print(f"  {'File':36} {'Signals':>8} {'Sound':>8} {'Delam':>8}", flush=True)
print(f"  {'-'*64}", flush=True)

all_X, all_y = [], []
for fpath in csv_files:
    amps, labels = load_csv(fpath)
    amps_avg     = spatial_average(amps, radius=2)
    all_X.append(amps_avg.T)          # (n_signals, 512)
    all_y.append(labels)
    ns = int(labels.sum()); nd = int((labels == 0).sum())
    tag = f"{fpath.parent.name}/{fpath.name}"
    print(f"  {tag:36} {len(labels):>8,} {ns:>8,} {nd:>8,}", flush=True)

X = np.concatenate(all_X, axis=0)    # (N, 512)
y = np.concatenate(all_y, axis=0)    # (N,)
print(f"\n  Total: {len(y):,}  Sound: {int(y.sum()):,}  Delam: {int((y==0).sum()):,}", flush=True)



# ─── Branch: evaluate saved model  OR  train from scratch ─────────────────────

if MODEL_PATH.exists():
    # ── Evaluate saved model on all data in DATA_FOLDER ───────────────────────
    print(f"\nStep 2 — Loading saved model from {MODEL_PATH}", flush=True)

    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}", flush=True)

    eval_ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y, dtype=torch.float32),
    )
    eval_dl = DataLoader(eval_ds, batch_size=256, shuffle=False)

    y_prob_list, y_true_list = [], []
    with torch.no_grad():
        for xb, yb in eval_dl:
            logits = model(xb.to(DEVICE))
            y_prob_list.append(logits.sigmoid().cpu().numpy())
            y_true_list.append(yb.numpy())

    y_prob     = np.concatenate(y_prob_list)
    y_true_arr = np.concatenate(y_true_list).astype(int)
    y_pred     = (y_prob >= 0.5).astype(int)

    n_delam = int((y_true_arr == 0).sum())
    n_sound = int((y_true_arr == 1).sum())

    TP = int(((y_pred == 0) & (y_true_arr == 0)).sum())
    FP = int(((y_pred == 0) & (y_true_arr == 1)).sum())
    FN = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    TN = int(((y_pred == 1) & (y_true_arr == 1)).sum())

    fnr = FN / n_delam * 100
    fpr = FP / n_sound * 100
    acc = (TP + TN) / len(y_true_arr) * 100
    f1  = f1_score(y_true_arr, y_pred, average="weighted")

    print("\n" + "=" * 60, flush=True)
    print(f"EVALUATION RESULTS — {DATA_FOLDER.name}", flush=True)
    print("=" * 60, flush=True)
    print(f"\n  Model trained on:  forest_river_north_bound (files 050–055)", flush=True)
    print(f"  Evaluated on:      {DATA_FOLDER.name} ({len(csv_files)} files, {len(y):,} signals)", flush=True)

    print(f"\n  Confusion matrix (delamination = positive class):", flush=True)
    print(f"  {'':24} Pred Sound   Pred Delam", flush=True)
    print(f"  {'GT Sound':24} {TN:>10,}   {FP:>10,}", flush=True)
    print(f"  {'GT Delaminated':24} {FN:>10,}   {TP:>10,}", flush=True)
    print(f"\n  FNR          {fnr:.1f}%", flush=True)
    print(f"  FPR          {fpr:.1f}%", flush=True)
    print(f"  Accuracy     {acc:.1f}%", flush=True)
    print(f"  Weighted F1  {f1:.4f}", flush=True)
    print(f"\n  V3 baseline: FNR 60.2%  FPR 10.3%", flush=True)
    print(f"  Delta:       FNR {60.2-fnr:+.1f} pp  FPR {10.3-fpr:+.1f} pp", flush=True)

    print(f"\n  [threshold sweep — flag delam if prob < threshold]", flush=True)
    print(f"  {'Thresh':>8} {'TPR':>7} {'FPR':>7} {'FNR':>7} {'Acc':>7}", flush=True)
    print(f"  {'-'*38}", flush=True)
    for thresh in np.arange(0.10, 0.96, 0.05):
        yp   = (y_prob >= thresh).astype(int)
        tp_  = int(((yp == 0) & (y_true_arr == 0)).sum())
        fp_  = int(((yp == 0) & (y_true_arr == 1)).sum())
        fn_  = int(((yp == 1) & (y_true_arr == 0)).sum())
        tn_  = int(((yp == 1) & (y_true_arr == 1)).sum())
        tpr_ = tp_ / n_delam; fpr_ = fp_ / n_sound
        acc_ = (tp_ + tn_) / len(y_true_arr) * 100
        mark = " <-- (FPR≤10.3%)" if fpr_ <= 0.103 else ""
        print(f"  {thresh:>8.2f} {tpr_*100:>6.1f}% {fpr_*100:>6.1f}% {(1-tpr_)*100:>6.1f}% {acc_:>6.1f}%{mark}", flush=True)

else:
    # ── Train from scratch ────────────────────────────────────────────────────
    print("\nStep 2 — Normalising and splitting", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}", flush=True)

    n_snd      = int(y_train.sum())
    n_del      = len(y_train) - n_snd
    pos_weight = torch.tensor([n_snd / n_del], dtype=torch.float32)
    print(f"  pos_weight (sound/delam): {pos_weight.item():.3f}", flush=True)

    def to_tensor(arr_x, arr_y):
        tx = torch.tensor(arr_x, dtype=torch.float32).unsqueeze(1)
        ty = torch.tensor(arr_y, dtype=torch.float32)
        return TensorDataset(tx, ty)

    train_ds = to_tensor(X_train, y_train)
    test_ds  = to_tensor(X_test,  y_test)
    test_dl  = DataLoader(test_ds, batch_size=256, shuffle=False)

    print("\nStep 3 — Building model", flush=True)
    model    = CNN1D().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model, flush=True)
    print(f"  Trainable parameters: {n_params:,}", flush=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nStep 4 — Training (50 epochs, early stopping patience=6)", flush=True)
    print(f"  {'Epoch':>5} {'Train loss':>11} {'Val loss':>10} {'Val acc':>9}", flush=True)
    print(f"  {'-'*40}", flush=True)

    val_size  = int(0.10 * len(train_ds))
    tr_size   = len(train_ds) - val_size
    tr_ds, val_ds = torch.utils.data.random_split(
        train_ds, [tr_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    tr_dl  = DataLoader(tr_ds,  batch_size=32,  shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_loss = float("inf")
    patience_left = 6
    best_state    = None

    for epoch in range(1, 51):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = augment_batch(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_ds)

        model.eval()
        val_loss = 0.0; correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                correct  += ((logits.sigmoid() >= 0.5).float() == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc   = correct / len(val_ds) * 100

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = 6
            marker = " *"
        else:
            patience_left -= 1
            marker = f"  (patience {patience_left}/6)"

        print(f"  {epoch:>5}   {tr_loss:>10.4f}   {val_loss:>9.4f}   {val_acc:>7.2f}%{marker}", flush=True)

        if patience_left == 0:
            print(f"  Early stopping at epoch {epoch}.", flush=True)
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}", flush=True)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("TEST SET RESULTS", flush=True)
    print("=" * 60, flush=True)

    model.eval()
    y_prob_list, y_true_list = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb.to(DEVICE))
            y_prob_list.append(logits.sigmoid().cpu().numpy())
            y_true_list.append(yb.numpy())

    y_prob     = np.concatenate(y_prob_list)
    y_true_arr = np.concatenate(y_true_list).astype(int)
    y_pred     = (y_prob >= 0.5).astype(int)

    n_delam = int((y_true_arr == 0).sum())
    n_sound = int((y_true_arr == 1).sum())

    TP = int(((y_pred == 0) & (y_true_arr == 0)).sum())
    FP = int(((y_pred == 0) & (y_true_arr == 1)).sum())
    FN = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    TN = int(((y_pred == 1) & (y_true_arr == 1)).sum())

    fnr = FN / n_delam * 100
    fpr = FP / n_sound * 100
    acc = (TP + TN) / len(y_true_arr) * 100
    f1  = f1_score(y_true_arr, y_pred, average="weighted")

    print(f"\n  Confusion matrix (delamination = positive class):", flush=True)
    print(f"  {'':24} Pred Sound   Pred Delam", flush=True)
    print(f"  {'GT Sound':24} {TN:>10,}   {FP:>10,}", flush=True)
    print(f"  {'GT Delaminated':24} {FN:>10,}   {TP:>10,}", flush=True)
    print(f"\n  FNR          {fnr:.1f}%", flush=True)
    print(f"  FPR          {fpr:.1f}%", flush=True)
    print(f"  Accuracy     {acc:.1f}%", flush=True)
    print(f"  Weighted F1  {f1:.4f}", flush=True)
    print(f"\n  V3 baseline: FNR 60.2%  FPR 10.3%", flush=True)
    print(f"  Delta:       FNR {60.2-fnr:+.1f} pp  FPR {10.3-fpr:+.1f} pp", flush=True)

    print(f"\n  [threshold sweep — flag delam if prob < threshold]", flush=True)
    print(f"  {'Thresh':>8} {'TPR':>7} {'FPR':>7} {'FNR':>7} {'Acc':>7}", flush=True)
    print(f"  {'-'*38}", flush=True)
    for thresh in np.arange(0.10, 0.96, 0.05):
        yp   = (y_prob >= thresh).astype(int)
        tp_  = int(((yp == 0) & (y_true_arr == 0)).sum())
        fp_  = int(((yp == 0) & (y_true_arr == 1)).sum())
        fn_  = int(((yp == 1) & (y_true_arr == 0)).sum())
        tn_  = int(((yp == 1) & (y_true_arr == 1)).sum())
        tpr_ = tp_ / n_delam; fpr_ = fp_ / n_sound
        acc_ = (tp_ + tn_) / len(y_true_arr) * 100
        mark = " <-- (FPR≤10.3%)" if fpr_ <= 0.103 else ""
        print(f"  {thresh:>8.2f} {tpr_*100:>6.1f}% {fpr_*100:>6.1f}% {(1-tpr_)*100:>6.1f}% {acc_:>6.1f}%{mark}", flush=True)
