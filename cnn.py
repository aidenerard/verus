# V17 — Production architecture for Verus
# Focal loss (alpha=0.75) + balanced sampler + raw+envelope dual-channel input
# Crops to samples 200–450 (250 samples) to exclude pre-signal noise floor
"""
GPR bridge-deck delamination — 1D-CNN, dual-channel input.

Label convention (throughout this file and the server):
  model output = P(sound) via sigmoid
  1 = sound,  0 = delaminated
  Positive class for inspection metrics = delaminated (label 0).

Key design decisions
─────────────────────────────────────────────────────────────────────
FILE-LEVEL split  Each CSV goes entirely into one split. Adjacent traces
  within a scan line are spatially correlated; a random trace-level split
  leaks near-duplicate samples into both train and test.

Per-signal z-score  Each A-scan normalised over its own 512 time samples
  before envelope computation and cropping.

Dual-channel input  Channel 0 = z-scored raw waveform. Channel 1 = Hilbert
  envelope amplitude. Crops samples 200–450 (250 samples) to focus on the
  reflection window and exclude the pre-signal noise floor.

FocalLoss  alpha=0.75 up-weights the delaminated class (minority).
  gamma=2.0 down-weights easy examples. No pos_weight needed.

WeightedRandomSampler  Each training batch is class-balanced by sampling
  each class with probability 1/class_count. Replaces pos_weight tuning.

CosineAnnealingLR  Smooth LR decay from 3e-4 to 1e-6 over T_max epochs.
  Avoids the sharp LR drops of StepLR that can destabilise focal loss.
─────────────────────────────────────────────────────────────────────
"""

import warnings, sys
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import windows as sig_windows, hilbert
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.manual_seed(42)
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_FOLDERS = [
    Path("/kaggle/input/datasets/aidenerard/all-bridges-csv/all_bridges_csv"),
]
MODEL_PATH  = Path("/kaggle/working/model.pth")

DC_OFFSET  = 32768
N_SAMPLES  = 512          # raw samples loaded from CSV
CROP_START = 200          # crop window start (exclude pre-signal noise floor)
CROP_END   = 450          # crop window end  → 250 samples into model
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TAPER       = np.ones(N_SAMPLES, dtype=np.float32)
_TAPER[410:] = sig_windows.hann(204)[102:].astype(np.float32)


# ── Data loading ───────────────────────────────────────────────────────────────

def _find_data_start(raw) -> int:
    for row in range(9, 14):
        try:
            val = float(raw[row, 0])
            if not np.isnan(val):
                return row
        except (ValueError, TypeError):
            continue
    raise ValueError("Cannot locate amplitude rows in file (expected rows 9–13)")


def load_csv(fpath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        amps:   (n_signals, N_SAMPLES) float32, per-signal z-score
        labels: (n_signals,) int — 1=sound, 0=delaminated
    """
    raw        = pd.read_csv(fpath, header=None).values
    n_signals  = int(raw[0, 4])
    raw_labels = raw[7, 1 : n_signals + 1].astype(int)
    data_start = _find_data_start(raw)
    amp_block  = raw[data_start : data_start + N_SAMPLES,
                     0 : n_signals + 1].astype(np.float32)
    if amp_block.shape[0] < N_SAMPLES:
        pad       = np.zeros((N_SAMPLES - amp_block.shape[0], amp_block.shape[1]),
                             dtype=np.float32)
        amp_block = np.vstack([amp_block, pad])

    # (N_SAMPLES, n_signals) → DC-correct, taper, transpose → (n_signals, N_SAMPLES)
    amps = ((amp_block[:, 1:] - DC_OFFSET) * _TAPER[:, np.newaxis]).T

    # Per-signal z-score (matches server/run.py inference)
    mean = amps.mean(axis=1, keepdims=True)
    std  = amps.std(axis=1,  keepdims=True) + 1e-8
    amps = (amps - mean) / std

    labels = (raw_labels == 1).astype(int)   # 1=sound, 0=delaminated
    return amps, labels


def load_files(fpaths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load files and return X: (n, 2, 250), y: (n,).

    Channel 0: z-scored raw waveform, cropped to samples CROP_START:CROP_END.
    Channel 1: Hilbert envelope of the z-scored raw, same crop.
    """
    all_X, all_y = [], []
    for fpath in fpaths:
        amps, labels = load_csv(fpath)          # (n, 512) z-scored float32

        # Hilbert envelope on full 512-sample z-scored signal
        envelope = np.abs(hilbert(amps, axis=1)).astype(np.float32)  # (n, 512)

        # Stack channels then crop → (n, 2, 250)
        x = np.stack([amps, envelope], axis=1)           # (n, 2, 512)
        x = x[:, :, CROP_START:CROP_END]                 # (n, 2, 250)

        all_X.append(x)
        all_y.append(labels)
        nd  = int((labels == 0).sum())
        ns  = int(labels.sum())
        tag = f"{fpath.parent.name}/{fpath.name}"
        print(f"  {tag:40} {len(labels):>7,}  snd {ns:>7,}  del {nd:>7,}", flush=True)
    return np.concatenate(all_X), np.concatenate(all_y)


# ── Model ──────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x.permute(0, 2, 1)), dim=1)   # (B, T, 1)
        return (x * w.permute(0, 2, 1)).sum(dim=2)                  # (B, C)


class CNN1D(nn.Module):
    """
    1D-CNN on dual-channel 250-sample A-scan input.
    in_channels=2: channel 0 = raw z-scored waveform, channel 1 = Hilbert envelope.
    Output logit → sigmoid = P(sound).
    """
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32,  kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,          128, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128,         128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.attn = TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.conv(x))).squeeze(1)


# ── Loss ───────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss.
    alpha*(1-pt)^gamma*bce — down-weights easy examples, focuses on hard ones.
    alpha=0.75 up-weights the delaminated (minority) class.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment_batch(xb: torch.Tensor) -> torch.Tensor:
    """
    Strong per-sample augmentations (training only).
    Each transform applied independently with 50% probability per sample.
    Input: (B, C, T) — C=2: channel 0 = raw, channel 1 = envelope.
    """
    B, C, T = xb.shape
    dev = xb.device

    # Gaussian noise — both channels
    mask = torch.rand(B, device=dev) < 0.5
    if mask.any():
        xb = xb.clone()
        xb[mask] = xb[mask] + torch.randn_like(xb[mask]) * 0.05

    # Amplitude scale — both channels (preserves raw/envelope ratio)
    mask = torch.rand(B, device=dev) < 0.5
    if mask.any():
        scale = torch.empty(int(mask.sum()), 1, 1, device=dev).uniform_(0.7, 1.3)
        xb[mask] = xb[mask] * scale

    # Time shift (roll — wraps signal, avoids zero-padding artefacts)
    mask_idx = torch.where(torch.rand(B, device=dev) < 0.5)[0]
    if len(mask_idx):
        shifts = torch.randint(-20, 21, (len(mask_idx),)).tolist()
        for i, s in zip(mask_idx.tolist(), shifts):
            xb[i] = torch.roll(xb[i], s, dims=-1)

    # Polarity flip — raw channel only (channel 0); envelope is unsigned
    mask = torch.rand(B, device=dev) < 0.5
    if mask.any():
        xb[mask, 0:1] = -xb[mask, 0:1]

    return xb


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, dl: DataLoader,
             threshold: float = 0.5) -> dict:
    """
    Positive class = delaminated = label 0.
    Returns sensitivity, specificity, precision, F1, PR-AUC and raw arrays.
    """
    model.eval()
    probs_list, true_list = [], []
    with torch.no_grad():
        for xb, yb in dl:
            probs_list.append(model(xb.to(DEVICE)).sigmoid().cpu().numpy())
            true_list.append(yb.numpy())
    y_prob = np.concatenate(probs_list)
    y_true = np.concatenate(true_list).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    n_d = int((y_true == 0).sum())
    n_s = int((y_true == 1).sum())
    TP  = int(((y_pred == 0) & (y_true == 0)).sum())
    FP  = int(((y_pred == 0) & (y_true == 1)).sum())
    FN  = int(((y_pred == 1) & (y_true == 0)).sum())
    TN  = int(((y_pred == 1) & (y_true == 1)).sum())

    sens  = TP / n_d if n_d else 0.0
    spec  = TN / n_s if n_s else 0.0
    prec  = TP / (TP + FP) if (TP + FP) else 0.0
    f1    = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    # PR-AUC: positive class = delaminated (label 0)
    # P(delaminated) = 1 - P(sound);  delaminated label = 1 - y_true
    pr_auc = average_precision_score(1 - y_true, 1.0 - y_prob)

    return dict(y_true=y_true, y_prob=y_prob, y_pred=y_pred,
                TP=TP, FP=FP, FN=FN, TN=TN,
                sensitivity=sens, specificity=spec, precision=prec,
                f1_delam=f1, pr_auc=pr_auc,
                n_delam=n_d, n_sound=n_s)


def print_metrics(m: dict, label: str = "") -> None:
    if label:
        print(f"\n{'='*60}\n{label}\n{'='*60}", flush=True)
    print(f"\n  Confusion matrix  (positive = delaminated = label 0)", flush=True)
    print(f"  {'':22}  Pred Sound  Pred Delam", flush=True)
    print(f"  {'GT Sound':22}  {m['TN']:>10,}  {m['FP']:>10,}", flush=True)
    print(f"  {'GT Delaminated':22}  {m['FN']:>10,}  {m['TP']:>10,}", flush=True)
    print(f"\n  Sensitivity (recall) {m['sensitivity']*100:>7.1f}%", flush=True)
    print(f"  Specificity          {m['specificity']*100:>7.1f}%", flush=True)
    print(f"  Precision            {m['precision']*100:>7.1f}%", flush=True)
    print(f"  F1 (delaminated)     {m['f1_delam']:>8.4f}", flush=True)
    print(f"  PR-AUC (delam pos)   {m['pr_auc']:>8.4f}", flush=True)
    print(f"  FNR (miss rate)      {(1-m['sensitivity'])*100:>7.1f}%", flush=True)
    print(f"  FPR (false alarm)    {(1-m['specificity'])*100:>7.1f}%", flush=True)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                     label: str = "Threshold sweep") -> float:
    """
    Sweep thresholds 0.10–0.90.  Return the one that maximises
    F1 for the delaminated class.  Prints a summary table.
    """
    n_d = int((y_true == 0).sum())
    n_s = int((y_true == 1).sum())
    print(f"\n  {label}  (positive = delaminated = label 0)", flush=True)
    print(f"  {'Thresh':>7} {'Sens%':>7} {'Spec%':>7} {'Prec%':>7} "
          f"{'F1':>7} {'FPR%':>6}", flush=True)
    print(f"  {'-'*46}", flush=True)
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.10, 0.90, 81):
        yp   = (y_prob >= t).astype(int)
        tp_  = int(((yp == 0) & (y_true == 0)).sum())
        fp_  = int(((yp == 0) & (y_true == 1)).sum())
        fn_  = int(((yp == 1) & (y_true == 0)).sum())
        tn_  = int(((yp == 1) & (y_true == 1)).sum())
        sns_ = tp_ / n_d if n_d else 0.0
        spc_ = tn_ / n_s if n_s else 0.0
        prc_ = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        f1_  = 2 * prc_ * sns_ / (prc_ + sns_) if (prc_ + sns_) else 0.0
        if f1_ > best_f1:
            best_f1, best_t = f1_, t
        if round(t * 100) % 10 == 0:
            star = " *best" if abs(t - best_t) < 1e-6 else ""
            print(f"  {t:>7.2f} {sns_*100:>6.1f}% {spc_*100:>6.1f}% "
                  f"{prc_*100:>6.1f}% {f1_:>6.4f} {(1-spc_)*100:>5.1f}%{star}",
                  flush=True)
    print(f"\n  → Best threshold: {best_t:.2f}  (F1_delam={best_f1:.4f})", flush=True)
    return best_t


# ── Step 1: Discover files ─────────────────────────────────────────────────────

csv_files = []
for folder in DATA_FOLDERS:
    if folder.exists():
        found = sorted(folder.rglob("FILE____*.csv"))
        csv_files.extend(found)
        print(f"  {folder.name}: {len(found)} files")
    else:
        print(f"  WARNING: {folder} not found, skipping")

csv_files = sorted(set(csv_files))
print(f"  Total CSV files: {len(csv_files)}")

if not csv_files:
    sys.exit("No FILE____*.csv files found in any DATA_FOLDERS")
print(f"Found {len(csv_files)} CSV files  device={DEVICE}", flush=True)


# ── Step 2: File-level grouped split ──────────────────────────────────────────
# Entire files go to one split — prevents leakage from correlated traces.

rng    = np.random.default_rng(42)
order  = rng.permutation(len(csv_files)).tolist()
n_test = max(1, round(0.15 * len(csv_files)))
n_val  = max(1, round(0.15 * len(csv_files)))

test_files  = [csv_files[i] for i in order[:n_test]]
val_files   = [csv_files[i] for i in order[n_test : n_test + n_val]]
train_files = [csv_files[i] for i in order[n_test + n_val :]]

print(f"File split → train={len(train_files)}  val={len(val_files)}  "
      f"test={len(test_files)}", flush=True)


# ── Step 3: Load data ──────────────────────────────────────────────────────────

if MODEL_PATH.exists():
    print(f"\nModel found at {MODEL_PATH} — evaluation mode.", flush=True)
    print(f"  Loading all {len(csv_files)} files …", flush=True)
    X_all, y_all = load_files(csv_files)
    print(f"  Total  {len(y_all):,}  sound={int(y_all.sum()):,}  "
          f"delam={int((y_all==0).sum()):,}", flush=True)

else:
    for tag, fpaths in [("train", train_files),
                        ("val",   val_files),
                        ("test",  test_files)]:
        print(f"\nLoading {tag} ({len(fpaths)} files) …", flush=True)
        if tag == "train":
            X_train, y_train = load_files(fpaths)
        elif tag == "val":
            X_val, y_val = load_files(fpaths)
        else:
            X_test, y_test = load_files(fpaths)

    for split, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        pct = int((y == 0).sum()) / len(y) * 100
        print(f"  {split:5}: {len(y):>8,}  sound {int(y.sum()):>7,}  "
              f"delam {int((y==0).sum()):>7,}  ({pct:.1f}% delam)", flush=True)


# ── Step 4: Build or load model ────────────────────────────────────────────────

model    = CNN1D(in_channels=2).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nCNN1D  in_channels=2  crop={CROP_START}:{CROP_END}  params={n_params:,}", flush=True)


# ═══ Evaluation branch ════════════════════════════════════════════════════════

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                     weights_only=False))
    model.eval()
    eval_dl = DataLoader(
        TensorDataset(torch.tensor(X_all, dtype=torch.float32),
                      torch.tensor(y_all, dtype=torch.float32)),
        batch_size=256, shuffle=False,
    )
    m = evaluate(model, eval_dl, threshold=0.5)
    print_metrics(m, label=f"EVALUATION — combined data  (threshold=0.50)")
    best_t = select_threshold(m['y_true'], m['y_prob'])
    m2 = evaluate(model, eval_dl, threshold=best_t)
    print_metrics(m2, label=f"EVALUATION — threshold={best_t:.2f} (val-selected)")


# ═══ Training branch ══════════════════════════════════════════════════════════

else:
    # ── DataLoaders ────────────────────────────────────────────────────────────
    # WeightedRandomSampler: each batch is class-balanced regardless of the
    # natural class distribution in the training files.
    class_counts = np.bincount(y_train)
    sample_weights = 1.0 / class_counts[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )
    delam_frac = class_counts[0] / len(y_train)
    print(f"\nClass balance: {delam_frac*100:.1f}% delaminated  "
          f"class_counts={class_counts.tolist()}  "
          f"→ WeightedRandomSampler (50/50 batches)", flush=True)

    tr_dl = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=256,
        sampler=sampler,
        num_workers=2,
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val,  dtype=torch.float32)),
        batch_size=256, shuffle=False,
    )
    test_dl = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=256, shuffle=False,
    )

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────────
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

    print(f"\nFocalLoss(alpha=0.75, gamma=2.0)  "
          f"CosineAnnealingLR(T_max=60, eta_min=1e-6)  lr=3e-4", flush=True)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n{'Ep':>4} {'TR loss':>9} {'Val loss':>9} "
          f"{'Val F1':>8} {'PR-AUC':>8} {'Sens%':>7} {'FPR%':>6} {'LR':>9}",
          flush=True)
    print(f"  {'-'*66}", flush=True)

    best_val_loss = float("inf")
    patience_left = 20
    best_state    = None

    for epoch in range(1, 101):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = augment_batch(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(y_train)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        probs_v, trues_v = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                probs_v.append(logits.sigmoid().cpu().numpy())
                trues_v.append(yb.cpu().numpy())
        val_loss /= len(y_val)

        y_prob_v = np.concatenate(probs_v)
        y_true_v = np.concatenate(trues_v).astype(int)
        y_pred_v = (y_prob_v >= 0.5).astype(int)

        n_dv  = int((y_true_v == 0).sum())
        n_sv  = int((y_true_v == 1).sum())
        tp_v  = int(((y_pred_v == 0) & (y_true_v == 0)).sum())
        fp_v  = int(((y_pred_v == 0) & (y_true_v == 1)).sum())
        tn_v  = int(((y_pred_v == 1) & (y_true_v == 1)).sum())
        sens_v = tp_v / n_dv if n_dv else 0.0
        spec_v = tn_v / n_sv if n_sv else 0.0
        prc_v  = tp_v / (tp_v + fp_v) if (tp_v + fp_v) else 0.0
        f1_v   = 2 * prc_v * sens_v / (prc_v + sens_v) if (prc_v + sens_v) else 0.0
        pr_auc_v = average_precision_score(1 - y_true_v, 1.0 - y_prob_v)
        lr_now   = optimizer.param_groups[0]['lr']

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = 20
            marker = " *"
        else:
            patience_left -= 1

        print(f"  {epoch:>4}  {tr_loss:>8.4f}  {val_loss:>8.4f}  "
              f"{f1_v:>7.4f}  {pr_auc_v:>7.4f}  "
              f"{sens_v*100:>6.1f}%  {(1-spec_v)*100:>5.1f}%  "
              f"{lr_now:>8.2e}{marker}", flush=True)

        if patience_left == 0:
            print(f"  Early stopping at epoch {epoch}.", flush=True)
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}", flush=True)

    # ── Final evaluation ───────────────────────────────────────────────────────

    # 1. Select threshold on val set
    m_val  = evaluate(model, val_dl, threshold=0.5)
    best_t = select_threshold(m_val['y_true'], m_val['y_prob'],
                              label="Val-set threshold sweep")

    # 2. Report test set at default and selected thresholds
    m_t50 = evaluate(model, test_dl, threshold=0.5)
    print_metrics(m_t50, label="TEST SET — threshold=0.50")

    m_tbs = evaluate(model, test_dl, threshold=best_t)
    print_metrics(m_tbs, label=f"TEST SET — threshold={best_t:.2f} (val-selected F1-max)")

    print(f"\n─── Deployment note ───────────────────────────────────────────", flush=True)
    print(f"  V17 uses in_channels=2 — server/run.py must be updated:", flush=True)
    print(f"    1. CNN1D(in_channels=2) in run.py", flush=True)
    print(f"    2. Inference pipeline: compute Hilbert envelope, stack channels,", flush=True)
    print(f"       crop samples {CROP_START}:{CROP_END} → shape (n, 2, {CROP_END-CROP_START})", flush=True)
    print(f"    3. Set THRESHOLD = {best_t:.2f}", flush=True)
