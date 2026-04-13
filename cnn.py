"""
GPR bridge-deck delamination — 1D-CNN with lateral patch context.

Label convention (throughout this file and the server):
  model output = P(sound) via sigmoid
  1 = sound,  0 = delaminated
  Positive class for inspection metrics = delaminated (label 0).

Key design decisions
─────────────────────────────────────────────────────────────────────
FILE-LEVEL split  Each CSV goes entirely into one split.  Adjacent traces
  within a scan line are spatially correlated; a random trace-level split
  leaks near-duplicate samples into both train and test, producing
  unrealistically optimistic metrics.

Per-signal z-score  Normalise each A-scan independently over its 512 time
  samples.  Per-file normalisation (the old approach) erases the between-
  scan-line amplitude variation that encodes rebar attenuation — exactly
  the physical signal used to detect delamination.  This normalization
  also matches server/run.py inference exactly.

Patch input (PATCH_K=5)  Feed the central trace plus its 2 lateral
  neighbours on each side as K input channels.  This gives the model
  spatial context to separate coherent delamination signatures from
  single-trace noise without explicit spatial averaging (which was
  increasing inter-sample correlation).  Patches are built per-file so
  no context crosses file boundaries.

FocalLoss with per-class alpha  alpha is now applied per example
  (alpha for y=1, 1-alpha for y=0) so it actually up-weights the
  minority delaminated class instead of scaling everything uniformly.

Threshold selection  The operating threshold is chosen on the validation
  set by maximising F1 for the delaminated class.  0.5 is printed but
  never treated as sacred.
─────────────────────────────────────────────────────────────────────

NOTE — server compatibility:
  PATCH_K=5 changes the model's first Conv1d from in_channels=1 to
  in_channels=5.  After training, update CNN1D and run_inference in
  server/run.py to match (see run_inference patch notes at bottom).
  The existing model_v13.pth (PATCH_K=1) will not load into this
  architecture.
"""

import warnings, sys, random
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import windows as sig_windows
from sklearn.metrics import f1_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_FOLDER = Path("~/Desktop/verus/all_bridges_csv").expanduser()
MODEL_PATH  = Path("/content/drive/MyDrive/fluxspace_gpr_data/model.pth")

DC_OFFSET = 32768
N_SAMPLES = 512
PATCH_K   = 5       # Neighbouring traces as input channels. Must match
                    # CNN1D(in_channels=) in server/run.py when deploying.
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Debug flag ─────────────────────────────────────────────────────────────────
# Set USE_BCE=True to train with plain BCEWithLogitsLoss (+ pos_weight).
# This confirms the PATCH_K=5 architecture can learn before adding FocalLoss.
# If BCE learns and FocalLoss doesn't, the bug is definitively in loss config.
# Switch to False once BCE training succeeds.
USE_BCE = True

_TAPER        = np.ones(N_SAMPLES, dtype=np.float32)
_TAPER[410:]  = sig_windows.hann(204)[102:].astype(np.float32)


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

    # Per-signal z-score: each A-scan normalised over its own 512 time samples.
    # Preserves relative amplitude differences between scan lines (attenuation info).
    mean = amps.mean(axis=1, keepdims=True)
    std  = amps.std(axis=1,  keepdims=True) + 1e-8
    amps = (amps - mean) / std

    labels = (raw_labels == 1).astype(int)    # 1=sound, 0=delaminated
    return amps, labels


def build_patches(amps: np.ndarray, k: int = PATCH_K) -> np.ndarray:
    """
    amps: (n_signals, N_SAMPLES)
    Returns: (n_signals, k, N_SAMPLES)

    Each signal gets k-1 lateral neighbours as additional channels.
    Edge signals are padded by repeating the boundary trace.
    Patches are built per-file so no context crosses file boundaries.
    """
    if k == 1:
        return amps[:, np.newaxis, :]
    half   = k // 2
    padded = np.concatenate([
        np.repeat(amps[:1],  half, axis=0),
        amps,
        np.repeat(amps[-1:], half, axis=0),
    ], axis=0)                                         # (n + k-1, T)
    return np.stack([padded[i : i + len(amps)] for i in range(k)], axis=1)


def load_files(fpaths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    for fpath in fpaths:
        amps, labels = load_csv(fpath)
        all_X.append(build_patches(amps, PATCH_K))
        all_y.append(labels)
        nd = int((labels == 0).sum()); ns = int(labels.sum())
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
    1D-CNN. in_channels=PATCH_K so the first conv sees all neighbouring
    traces simultaneously.  Output logit → sigmoid = P(sound).
    """
    def __init__(self, in_channels: int = PATCH_K):
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
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.conv(x))).squeeze(1)


class FocalLoss(nn.Module):
    """
    Binary focal loss with correct per-class alpha.
    alpha = weight for y=1 (sound); (1-alpha) = weight for y=0 (delaminated).
    alpha=0.25 → delaminated:sound loss ratio = 3:1.
    The original code applied alpha as a global scale (no per-class effect).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt      = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment_batch(xb: torch.Tensor) -> torch.Tensor:
    """Per-sample augmentations (training only)."""
    B, K, T = xb.shape
    xb = xb + xb.std(dim=-1, keepdim=True) * 0.01 * torch.randn_like(xb)
    xb = xb + torch.randn_like(xb) * 0.02
    xb = xb * torch.empty(B, 1, 1, device=xb.device).uniform_(0.95, 1.05)
    out    = torch.zeros_like(xb)
    shifts = torch.randint(-5, 6, (B,)).tolist()
    for i, s in enumerate(shifts):
        if s == 0:
            out[i] = xb[i]
        elif s > 0:
            out[i, :, s:] = xb[i, :, :-s]
        else:
            out[i, :, :s] = xb[i, :, -s:]
    return out


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
    #   score for "is delaminated" = 1 - P(sound); label = 1 - y_true
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
    Sweep thresholds on a labelled set.  Return the threshold that maximises
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
        if round(t * 100) % 10 == 0:          # print every 10th step
            star = " *best" if abs(t - best_t) < 1e-6 else ""
            print(f"  {t:>7.2f} {sns_*100:>6.1f}% {spc_*100:>6.1f}% "
                  f"{prc_*100:>6.1f}% {f1_:>6.4f} {(1-spc_)*100:>5.1f}%{star}",
                  flush=True)
    print(f"\n  → Best threshold: {best_t:.2f}  (F1_delam={best_f1:.4f})", flush=True)
    return best_t


# ── Step 1: Discover files ─────────────────────────────────────────────────────

csv_files = sorted(DATA_FOLDER.rglob("FILE____*.csv"))
if not csv_files:
    sys.exit(f"No FILE____*.csv files found in {DATA_FOLDER}")
print(f"Found {len(csv_files)} CSV files in {DATA_FOLDER.name}  "
      f"PATCH_K={PATCH_K}  device={DEVICE}", flush=True)


# ── Step 2: File-level grouped split ──────────────────────────────────────────
# Shuffle file list with a fixed seed, then assign entire files to splits.
# This prevents leakage from correlated neighbouring traces.
#
# TODO: when bridge-ID metadata is available, group by bridge so that no
# bridge appears in both train and test — even stronger anti-leakage guarantee.

rng      = np.random.default_rng(42)
order    = rng.permutation(len(csv_files)).tolist()
n_test   = max(1, round(0.15 * len(csv_files)))
n_val    = max(1, round(0.15 * len(csv_files)))

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

model    = CNN1D(in_channels=PATCH_K).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nCNN1D  in_channels={PATCH_K}  params={n_params:,}", flush=True)


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
    print_metrics(m, label=f"EVALUATION — {DATA_FOLDER.name}  (threshold=0.50)")
    best_t = select_threshold(m['y_true'], m['y_prob'])
    m2 = evaluate(model, eval_dl, threshold=best_t)
    print_metrics(m2, label=f"EVALUATION — threshold={best_t:.2f} (Otsu/val-selected)")


# ═══ Training branch ══════════════════════════════════════════════════════════

else:
    def make_dl(X, y, shuffle: bool, bs: int = 256) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32)),
            batch_size=bs, shuffle=shuffle,
        )

    tr_dl   = make_dl(X_train, y_train, shuffle=True,  bs=32)
    val_dl  = make_dl(X_val,   y_val,   shuffle=False)
    test_dl = make_dl(X_test,  y_test,  shuffle=False)

    delam_frac = int((y_train == 0).sum()) / len(y_train)

    if USE_BCE:
        # BCEWithLogitsLoss with pos_weight to handle class imbalance.
        # pos_weight = n_neg / n_pos where positive class = sound (y=1).
        # If delam (y=0) is majority: pos_weight < 1 (down-weight sound loss).
        # If sound (y=1) is majority: pos_weight > 1 (up-weight sound loss).
        n_delam    = int((y_train == 0).sum())
        n_sound    = int((y_train == 1).sum())
        # V15: fixed pos_weight=2.0 to push FPR down toward 10% target.
        # Higher pos_weight penalises false positives (sound predicted as delam)
        # more heavily; data-derived ratio (~0.565) was too low to correct FPR.
        pos_weight = torch.tensor([2.0], dtype=torch.float32).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"\nClass balance: {delam_frac*100:.1f}% delaminated  "
              f"→ BCEWithLogitsLoss pos_weight={pos_weight.item():.3f}  [DEBUG]",
              flush=True)
    else:
        # FocalLoss: alpha is the weight for y=1 (sound) in the formula
        #   alpha_t = alpha * y + (1-alpha) * (1-y)
        # So weight for y=0 (delam) = 1-alpha.
        # User convention: alpha=0.75 → y=1 (sound) gets 0.75, y=0 (delam) gets 0.25.
        # This prevents collapse to all-delaminated by keeping strong gradient on sound.
        # For standard minority-upweighting (delam heavy), swap to alpha=0.25.
        alpha_fl  = 0.75 if delam_frac <= 0.5 else 0.25
        criterion = FocalLoss(alpha=alpha_fl, gamma=2.0)
        print(f"\nClass balance: {delam_frac*100:.1f}% delaminated  "
              f"→ FocalLoss alpha={alpha_fl}  gamma=2.0  "
              f"(sound weight={alpha_fl:.2f}, delam weight={1-alpha_fl:.2f})",
              flush=True)

    # Lower LR to 3e-4: FocalLoss collapses more easily at 1e-3 because
    # the focal term (gamma=2) dramatically shrinks gradients for easy examples,
    # making the effective learning rate feel much larger for hard examples.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # V15: StepLR halves every 15 epochs — slower decay gives the model more
    # time at each LR level before being forced to fine-tune.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.5,
    )

    # ── Sanity check: one forward pass before training ─────────────────────────
    print(f"\n[sanity] Running pre-training forward pass …", flush=True)
    model.eval()
    with torch.no_grad():
        _xb_s, _yb_s = next(iter(tr_dl))
        _xb_s = _xb_s.to(DEVICE)
        assert _xb_s.shape[1] == PATCH_K and _xb_s.shape[2] == N_SAMPLES, (
            f"[sanity] Bad input shape: {_xb_s.shape}  expected (B, {PATCH_K}, {N_SAMPLES})"
        )
        _logits_s = model(_xb_s)
        _probs_s  = _logits_s.sigmoid()
        print(f"[sanity] input shape : {tuple(_xb_s.shape)}", flush=True)
        print(f"[sanity] logit range : [{_logits_s.min():.3f}, {_logits_s.max():.3f}]  "
              f"mean={_logits_s.mean():.3f}  std={_logits_s.std():.3f}", flush=True)
        print(f"[sanity] sigmoid range: [{_probs_s.min():.3f}, {_probs_s.max():.3f}]  "
              f"mean={_probs_s.mean():.3f}", flush=True)
        _all_same = (_probs_s.max() - _probs_s.min()).item() < 1e-4
        if _all_same:
            print(f"[sanity] WARNING — all sigmoid outputs are nearly identical "
                  f"({_probs_s.mean():.4f}). Model may be in a degenerate state.",
                  flush=True)
        else:
            print(f"[sanity] OK — output values vary (model is not collapsed).", flush=True)
        del _xb_s, _logits_s, _probs_s
    model.train()

    print(f"\n{'Ep':>4} {'TR loss':>9} {'Val loss':>9} "
          f"{'Val F1':>8} {'Sens%':>7} {'FPR%':>6} {'LR':>9}", flush=True)
    print(f"  {'-'*56}", flush=True)

    best_val_loss = float("inf")
    patience_left = 10
    best_state    = None
    _first_batch  = True     # used for per-batch shape assertion on epoch 1

    for epoch in range(1, 101):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # Assert correct patch shape on very first batch of epoch 1
            if _first_batch:
                assert xb.shape[1] == PATCH_K and xb.shape[2] == N_SAMPLES, (
                    f"Bad patch shape: {tuple(xb.shape)}  "
                    f"expected (batch, {PATCH_K}, {N_SAMPLES})"
                )
                print(f"[assert] Epoch 1, batch 1 shape: {tuple(xb.shape)}  ✓",
                      flush=True)
                _first_batch = False
            xb     = augment_batch(xb)
            optimizer.zero_grad()
            loss   = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(y_train)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        probs_v, trues_v = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss  += criterion(logits, yb).item() * len(xb)
                probs_v.append(logits.sigmoid().cpu().numpy())
                trues_v.append(yb.cpu().numpy())
        val_loss /= len(y_val)
        scheduler.step()

        y_prob_v = np.concatenate(probs_v)
        y_true_v = np.concatenate(trues_v).astype(int)
        y_pred_v = (y_prob_v >= 0.5).astype(int)
        n_dv     = int((y_true_v == 0).sum())
        n_sv     = int((y_true_v == 1).sum())
        tp_v     = int(((y_pred_v == 0) & (y_true_v == 0)).sum())
        fp_v     = int(((y_pred_v == 0) & (y_true_v == 1)).sum())
        fn_v     = int(((y_pred_v == 1) & (y_true_v == 0)).sum())
        tn_v     = int(((y_pred_v == 1) & (y_true_v == 1)).sum())
        sens_v   = tp_v / n_dv if n_dv else 0.0
        spec_v   = tn_v / n_sv if n_sv else 0.0
        prc_v    = tp_v / (tp_v + fp_v) if (tp_v + fp_v) else 0.0
        f1_v     = 2 * prc_v * sens_v / (prc_v + sens_v) if (prc_v + sens_v) else 0.0
        lr_now   = optimizer.param_groups[0]['lr']

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = 10
            marker = " *"
        else:
            patience_left -= 1

        print(f"  {epoch:>4}  {tr_loss:>8.4f}  {val_loss:>8.4f}  "
              f"{f1_v:>7.4f}  {sens_v*100:>6.1f}%  {(1-spec_v)*100:>5.1f}%  "
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

    print(f"\n─── Deployment note ───────────────────────────────────────", flush=True)
    print(f"  Set THRESHOLD = {best_t:.2f} in server/run.py", flush=True)
    print(f"  Update CNN1D(in_channels={PATCH_K}) and run_inference patch logic.", flush=True)
    print(f"  See patch-inference note below.", flush=True)


# ── run_inference patch note (for server/run.py after retraining) ─────────────
#
# In server/run.py, CNN1D must be updated to in_channels=PATCH_K.
# run_inference must build K-trace patches before inference:
#
#   def run_inference(model, signals):   # signals: (n, 512)
#       half = PATCH_K // 2
#       padded = np.concatenate([
#           np.repeat(signals[:1],  half, axis=0),
#           signals,
#           np.repeat(signals[-1:], half, axis=0),
#       ], axis=0)
#       all_probs = []
#       for start in range(0, len(signals), INFER_BATCH):
#           end     = min(start + INFER_BATCH, len(signals))
#           patches = np.stack(
#               [padded[start+i : end+i] for i in range(PATCH_K)], axis=1
#           )  # (batch, PATCH_K, 512)
#           t = torch.tensor(patches, dtype=torch.float32).to(DEVICE)
#           all_probs.append(model(t).sigmoid().cpu().numpy())
#       probs = np.concatenate(all_probs)
#       preds = (probs >= THRESHOLD).astype(int)
#       confs = np.where(preds == 1, probs, 1.0 - probs)
#       return preds, confs
