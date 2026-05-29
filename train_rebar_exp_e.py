"""
Exp E — Infrasense training integration for HorizonCNN rebar depth.

Key changes vs colab_train_horizon.py:
  - Adds Infrasense B440029 DZT as training data (strong ground-truth labels from Ken's depth report)
  - B170020 is held out as the validation bridge (same bridge used for all prior baselines)
  - MAX_DEPTH_MM raised to 250 (covers B440029 max truth 9.22" = 234mm)
  - Infrasense samples weighted 3× in WeightedRandomSampler
  - Warm-start from gprMax pretrain checkpoint (horizon_model_pretrained.pth)
  - depth_head always re-initialized after loading pretrain (--reinit-head default True)

Run locally:
    python train_rebar_exp_e.py          # reinit-head=True by default
    python train_rebar_exp_e.py --no-reinit-head  # resume only; never for fresh pretrain load

Run on Kaggle:
  - Upload as script with horizon_model_pretrained.pth attached as dataset
  - Set TERRACON_DIR and INFRASENSE_DIR to Kaggle input paths
  - GPU runtime recommended (T4/P100)
"""
import argparse, os, glob, csv, sys, warnings, time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample as scipy_resample
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
TERRACON_DIR   = r"C:\Users\quack\Documents\Projects\Verus\Data\Stephen Terracon Cornbread\Data"
INFRASENSE_DIR = r"C:\Users\quack\Documents\Projects\Verus\Data\Ken Infrasense Data #1"
PRETRAIN_PATH  = Path(__file__).parent / "server" / "models" / "horizon_model_pretrained.pth"
MODEL_OUT      = Path(__file__).parent / "server" / "models" / "horizon_model_exp_e.pth"

# ── Hyper-parameters ───────────────────────────────────────────────────────────
MAX_DEPTH_MM    = 250.0   # covers B440029 max truth 9.22" = 234mm
TARGET_SAMPLES  = 512
BATCH_SIZE      = 256
EPOCHS          = 120
PATIENCE        = 20
LR              = 5e-4    # lower LR for fine-tune warm start
INFRASENSE_WEIGHT = 3.0   # strong-label upweight
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  MAX_DEPTH_MM: {MAX_DEPTH_MM}")

# ── Model ──────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.score = nn.Linear(c, 1)
    def forward(self, x):
        w = torch.softmax(self.score(x.transpose(1, 2)), dim=1)
        return (x.transpose(1, 2) * w).sum(dim=1)

class HorizonCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,   32,  7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,  64,  5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.attn = TemporalAttention(128)
        self.depth_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.depth_head(self.attn(self.conv(x))).squeeze(-1)


def reinit_head(model: "HorizonCNN") -> None:
    """Re-initialize depth_head to random weights.

    gprMax synthetic amplitudes are unreliable (no antenna coupling, no 3D
    spreading). The head must always be retrained on real labels; only conv
    feature extractors transfer.
    """
    for layer in model.depth_head:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    print("depth_head re-initialized — conv weights retained, head weights discarded")


# ── Preprocessing (matches run_rebar_inference exactly) ────────────────────────
def preprocess(arr: np.ndarray) -> np.ndarray:
    """arr: (n_traces, n_samples). Returns (n_traces, 512) DC-removed, max-abs normalized."""
    if arr.shape[1] != TARGET_SAMPLES:
        arr = scipy_resample(arr, TARGET_SAMPLES, axis=1).astype(np.float32)
    arr = arr - arr.mean(axis=1, keepdims=True)
    mx = np.abs(arr).max(axis=1, keepdims=True)
    mx[mx == 0] = 1.0
    return (arr / mx).astype(np.float32)

# ── Terracon Proceq loader ─────────────────────────────────────────────────────
_MAGIC   = b"VH01SW"
_D_START = 0x027C
_D_SIZE  = 0x040C
_D_HDR   = 16
_D_REF   = 16
_D_MARK  = b"D\x00"
_D_SAMP  = (_D_SIZE - _D_HDR) // 2

def read_proceq(path: str):
    raw = open(path, "rb").read()
    if raw[:6] != _MAGIC:
        return None, 0
    n = 0; pos = _D_START
    while pos + 2 <= len(raw) and raw[pos:pos+2] == _D_MARK:
        n += 1; pos += _D_SIZE
    nd = n - _D_REF
    if nd <= 0:
        return None, 0
    traces = np.zeros((nd, _D_SAMP), dtype=np.float32)
    with open(path, "rb") as f:
        f.seek(_D_START + _D_REF * _D_SIZE)
        for i in range(nd):
            blk = f.read(_D_SIZE)
            if len(blk) < _D_SIZE:
                traces = traces[:i]; break
            s = np.frombuffer(blk[_D_HDR:], dtype="<i2").astype(np.float32)
            s -= s.mean()
            traces[i] = s
    norms = np.abs(traces).max(axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    traces /= norms
    return traces, len(traces)

SKIP_SWATHS = {0, 4}

def load_terracon():
    scan_files = sorted(glob.glob(os.path.join(TERRACON_DIR, "PRC_*.scan")))
    odd_scans  = [f for f in scan_files
                  if int(os.path.basename(f).replace("PRC_","").replace(".scan","")) % 2 == 1]
    ts_files   = sorted(glob.glob(os.path.join(TERRACON_DIR, "TS_*_1.txt")))
    n_swaths   = min(len(odd_scans) // 4, len(ts_files))
    good       = [i for i in range(n_swaths) if i not in SKIP_SWATHS]

    X_list, y_list = [], []
    for sw in good:
        ts = np.loadtxt(ts_files[sw])
        for sp in odd_scans[sw * 4 : sw * 4 + 4]:
            raw, n = read_proceq(sp)
            if raw is None or n < 10:
                continue
            gt = np.interp(np.arange(n), np.linspace(0, n - 1, len(ts)), ts)
            X_list.append(preprocess(raw))
            y_list.append(np.clip(gt / MAX_DEPTH_MM, 0, 1).astype(np.float32))
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f"  Terracon: {len(X):,} traces  depth=[{(y*MAX_DEPTH_MM).min():.0f}, {(y*MAX_DEPTH_MM).max():.0f}]mm")
    return X, y

# ── Infrasense DZT loader ──────────────────────────────────────────────────────
def _load_gt_csv(csv_path: str) -> dict:
    """
    Mirrors test_rebar_validation.py load_csv_gt exactly.
    Returns dict: {(prep_fname_upper, scan_int): depth_in}
    """
    gt = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw_d = row.get("L2Depth_inches", "").strip()
            if not raw_d:
                continue
            try:
                fname = row["preProcessedFileName"].strip().upper()
                scan  = int(row["scanNumber"].strip())
                gt[(fname, scan)] = float(raw_d)
            except (ValueError, KeyError):
                continue
    return gt

def _dzt_key(file_num: int, ch_num: int) -> str:
    """Matches test_rebar_validation.py dzt_key()."""
    return f"WISDOT24_{file_num:03d} P_1_PREP_CH0{ch_num}.DZT"

def load_infrasense_bridge(bridge_name: str, layout: list, gt_csv: str):
    """
    Load Infrasense DZT traces with ground-truth depth labels.
    Mirrors test_rebar_validation.py run_bridge: uses readdzt, slices by scan
    index range, matches each trace's scan index to the GT CSV.

    layout: list of (file_num, ch_num, start_scan, end_scan, reversed_flag)
    Returns (X, y): (n, 512) preprocessed traces, (n,) normalised depth labels.
    """
    try:
        from readgssi import readgssi as rgssi
    except ImportError:
        raise ImportError("readgssi required: pip install readgssi")

    gt       = _load_gt_csv(gt_csv)
    data_dir = Path(INFRASENSE_DIR) / bridge_name / "raw data"
    from scipy.signal import resample as _resample

    X_list, y_list = [], []
    for file_num, ch_num, start_scan, end_scan, reversed_flag in layout:
        dzt_name = f"WISDOT24_{file_num:03d} P_1.DZT"
        dzt_path = data_dir / dzt_name
        if not dzt_path.exists():
            print(f"  WARN: {dzt_name} not found, skipping")
            continue

        try:
            header, data, _ = rgssi.readdzt(str(dzt_path))
        except Exception as e:
            print(f"  WARN: readdzt failed for {dzt_name}: {e}")
            continue

        if isinstance(data, dict):
            arr = list(data.values())[ch_num - 1]
        elif isinstance(data, (list, tuple)):
            arr = data[ch_num - 1]
        else:
            nchan = int(header.get("nchan", 1))
            arr   = data[:, ch_num - 1 :: nchan]

        arr          = np.asarray(arr, dtype=np.float32)
        n_samp_raw   = arr.shape[0]
        n_total_cols = arr.shape[1]

        # Clamp scan range to actual trace count
        s_start = min(start_scan, n_total_cols)
        s_end   = min(end_scan,   n_total_cols)
        if s_end <= s_start:
            print(f"  WARN: {dzt_name} scan range [{start_scan},{end_scan}] out of bounds ({n_total_cols} traces)")
            continue

        arr          = arr[:, s_start:s_end]
        scan_indices = np.arange(s_start, s_end)
        if reversed_flag:
            arr          = arr[:, ::-1].copy()
            scan_indices = scan_indices[::-1].copy()

        # Match each trace to the GT dict by its scan index
        gt_key_prefix = _dzt_key(file_num, ch_num)
        depths_in = np.full(len(scan_indices), np.nan, dtype=np.float32)
        for i, scan_idx in enumerate(scan_indices):
            k = (gt_key_prefix, int(scan_idx))
            if k in gt:
                depths_in[i] = gt[k]

        valid = np.isfinite(depths_in)
        if valid.sum() < 10:
            print(f"  WARN: {dzt_name} ch{ch_num}: only {valid.sum()} GT-matched traces")
            continue

        traces = arr[:, valid].T.copy()   # (n_valid, n_samp_raw)
        depths = depths_in[valid]

        if n_samp_raw != 512:
            traces = np.stack([
                _resample(traces[i].astype(np.float64), 512).astype(np.float32)
                for i in range(len(traces))
            ])

        X_list.append(preprocess(traces))
        y_list.append(np.clip(depths * 25.4 / MAX_DEPTH_MM, 0.0, 1.0))

    if not X_list:
        return np.empty((0, 512), np.float32), np.empty(0, np.float32)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list).astype(np.float32)
    d_min = y.min() * MAX_DEPTH_MM / 25.4
    d_max = y.max() * MAX_DEPTH_MM / 25.4
    print(f"  {bridge_name}: {len(X):,} GT-matched traces  depth=[{d_min:.2f}\", {d_max:.2f}\"]")
    return X, y

# ── Bridge layout (matches test_rebar_validation.py) ──────────────────────────
INFRASENSE_LAYOUT = {
    "B440029": {
        "gt_csv": str(Path(INFRASENSE_DIR) / "B440029 Rebar Depth Report.csv"),
        "layout": [
            # Scan ranges verified against B440029 Rebar Depth Report.csv.
            # Each DZT file uses a session-local scan counter; ranges are not
            # contiguous across files. Prior bug: all 9 entries used file-799's
            # range [3535,4163], so files 801/803/805/807 loaded wrong DZT columns
            # with zero GT matches (~959 traces vs ~3,875 expected).
            (799, 1, 3535, 4163, False),
            (799, 2, 3535, 4163, False),
            (801, 1, 4737, 5373, False),
            (801, 2, 4737, 5373, False),
            (803, 1, 4835, 5462, False),
            (803, 2, 4835, 5462, False),
            (805, 1, 4984, 5614, False),
            (805, 2, 4984, 5614, False),
            (807, 2, 4016, 4643, False),
        ],
    },
    "B170020": {
        "gt_csv": str(Path(INFRASENSE_DIR) / "B170020 Rebar Depth Report.csv"),
        "layout": [
            # Ranges match test_rebar_validation.py exactly.
            # Files 096/098 (_r suffix in GT CSV) omitted: _dzt_key cannot generate
            # the _r filename, so adding them without fixing key generation would
            # corrupt depth labels for reversed traversals.
            (95,  1,  610, 1802, False),
            (95,  2,  617, 1806, False),
            (97,  1,  437, 1623, False),
            (97,  2,  443, 1645, False),
        ],
    },
}

# Expected GT match counts (from Rebar Depth Report CSVs). Used by --dry-run.
_EXPECTED_GT: dict = {
    "B440029": {(799, 1): 429, (799, 2): 431, (801, 1): 430, (801, 2): 431,
                (803, 1): 431, (803, 2): 431, (805, 1): 430, (805, 2): 431,
                (807, 2): 431},
    "B170020": {(95, 1): 817, (95, 2): 818, (97, 1): 817, (97, 2): 816},
}


def _audit_layout(bridge_name: str, layout: list, gt_csv: str) -> bool:
    """Verify layout scan ranges against the GT CSV without loading DZTs.
    Returns True if every file-channel pair has ≥90% of expected GT matches."""
    gt       = _load_gt_csv(gt_csv)
    expected = _EXPECTED_GT.get(bridge_name, {})
    all_ok   = True
    print(f"\n  {bridge_name} layout audit (GT CSV scan-range check):")
    print(f"  {'file':>6} ch  expected  matched  rate   status")
    for file_num, ch_num, start_scan, end_scan, _ in layout:
        prefix    = _dzt_key(file_num, ch_num)
        n_matched = sum(1 for s in range(start_scan, end_scan) if (prefix, s) in gt)
        exp       = expected.get((file_num, ch_num), "?")
        thresh    = int(exp * 0.9) if isinstance(exp, int) else 1
        ok        = n_matched >= thresh
        if not ok:
            all_ok = False
        rate = f"{n_matched/exp:.0%}" if isinstance(exp, int) and exp > 0 else "?"
        print(f"  {file_num:>6}  {ch_num}  {str(exp):>8}  {n_matched:>7}  {rate:>5}  "
              f"{'OK' if ok else 'FAIL'}")
    total_exp     = sum(v for v in expected.values()) or 1
    total_matched = sum(
        sum(1 for s in range(s0, s1) if (_dzt_key(fn, cn), s) in gt)
        for fn, cn, s0, s1, _ in layout
    )
    print(f"  Total: {total_matched}/{total_exp} "
          f"({total_matched/total_exp:.0%})  "
          f"{'PASS' if all_ok else 'FAIL — fix scan ranges before training'}")
    return all_ok

# ── Dataset ────────────────────────────────────────────────────────────────────
class GPRDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, aug: bool = False):
        self.X   = torch.from_numpy(X).unsqueeze(1)
        self.y   = torch.from_numpy(y)
        self.aug = aug

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x, y = self.X[i].clone(), self.y[i]
        if self.aug:
            if torch.rand(1) < 0.5:
                x += torch.randn_like(x) * 0.01
            if torch.rand(1) < 0.5:
                x *= 0.9 + torch.rand(1) * 0.2
            if torch.rand(1) < 0.5:
                sh = torch.randint(-10, 11, (1,)).item()
                x = torch.roll(x, sh, -1)
                if sh > 0:  x[..., :sh]  = 0
                elif sh < 0: x[..., sh:] = 0
        return x, y

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reinit-head", action=argparse.BooleanOptionalAction, default=True,
        help="Re-initialize depth_head after loading pretrain checkpoint (default: True). "
             "Always True when loading from gprMax pretrained weights — set False only "
             "when resuming a fine-tune run mid-training.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Audit GT scan-range match counts and print data stats without training.",
    )
    args = ap.parse_args()

    if args.dry_run:
        print("\n=== Dry-run: layout audit (CSV-only, no DZT files needed) ===")
        b440_ok = _audit_layout("B440029",
                                INFRASENSE_LAYOUT["B440029"]["layout"],
                                INFRASENSE_LAYOUT["B440029"]["gt_csv"])
        b170_ok = _audit_layout("B170020",
                                INFRASENSE_LAYOUT["B170020"]["layout"],
                                INFRASENSE_LAYOUT["B170020"]["gt_csv"])
        status = "PASS" if b440_ok and b170_ok else "FAIL — fix layout before training"
        print(f"\nDry-run result: {status}")
        return

    print("\n=== Loading training data ===")
    sys.path.insert(0, str(Path(__file__).parent))

    X_tc, y_tc = load_terracon()

    cfg = INFRASENSE_LAYOUT["B440029"]
    X_if_tr, y_if_tr = load_infrasense_bridge("B440029", cfg["layout"], cfg["gt_csv"])

    cfg_val = INFRASENSE_LAYOUT["B170020"]
    X_val, y_val = load_infrasense_bridge("B170020", cfg_val["layout"], cfg_val["gt_csv"])

    if len(X_if_tr) == 0:
        print("ERROR: No Infrasense training data loaded. Check paths and DZT files.")
        return

    # Combine train: Terracon + Infrasense B440029
    X_tr = np.concatenate([X_tc, X_if_tr])
    y_tr = np.concatenate([y_tc, y_if_tr])

    # Weighted sampler: Infrasense traces get 3× weight
    n_tc = len(X_tc)
    n_if = len(X_if_tr)
    weights = np.ones(n_tc + n_if, dtype=np.float32)
    weights[n_tc:] = INFRASENSE_WEIGHT
    sampler = WeightedRandomSampler(weights=torch.from_numpy(weights),
                                    num_samples=len(X_tr),
                                    replacement=True)

    tr_loader = DataLoader(GPRDataset(X_tr, y_tr, aug=True),
                           batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    va_loader = DataLoader(GPRDataset(X_val, y_val, aug=False),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nTrain: {len(X_tr):,} ({n_tc} Terracon + {n_if} Infrasense B440029)")
    print(f"Val:   {len(X_val):,} (Infrasense B170020, held-out gold standard)")
    print(f"Train depth range: [{(y_tr*MAX_DEPTH_MM).min():.0f}, {(y_tr*MAX_DEPTH_MM).max():.0f}]mm")

    # ── Model init ─────────────────────────────────────────────────────────────
    model = HorizonCNN().to(DEVICE)
    if PRETRAIN_PATH.exists():
        state = torch.load(str(PRETRAIN_PATH), map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        print(f"Loaded pretrain checkpoint: {PRETRAIN_PATH.name}")
        if args.reinit_head:
            reinit_head(model)
        else:
            print("WARNING: --no-reinit-head set — depth_head weights from synthetic pretrain retained")
    else:
        print("WARNING: pretrain checkpoint not found, training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"HorizonCNN params: {n_params:,}")

    crit  = nn.SmoothL1Loss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_mae_mm = float("inf"); patience_ctr = 0; t0 = time.time()
    print(f"\n  Ep   tr_loss  val_loss  MAE_mm   MAE_in  best_in  elapsed")
    print(f"  {'-'*63}")

    for epoch in range(1, EPOCHS + 1):
        model.train(); tr_loss = 0.0
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(tr_loader.dataset)

        model.eval(); val_loss = 0.0; pv, tv = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                out = model(xb.to(DEVICE))
                val_loss += crit(out, yb.to(DEVICE)).item() * len(yb)
                pv.append(out.cpu().numpy()); tv.append(yb.numpy())
        val_loss /= len(va_loader.dataset)
        pv = np.concatenate(pv) * MAX_DEPTH_MM
        tv = np.concatenate(tv) * MAX_DEPTH_MM
        mae_mm = float(np.abs(pv - tv).mean())
        sched.step()

        if mae_mm < best_mae_mm:
            best_mae_mm = mae_mm; patience_ctr = 0
            torch.save(model.state_dict(), str(MODEL_OUT))

        else:
            patience_ctr += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  {epoch:3d}  {tr_loss:.5f}  {val_loss:.5f}  {mae_mm:7.2f}  "
                  f"{mae_mm/25.4:7.3f}  {best_mae_mm/25.4:7.3f}  {elapsed:.0f}s", flush=True)

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    best_in = best_mae_mm / 25.4
    target  = 12.0 / 25.4  # 0.472" floor
    status  = "PASS" if best_in < target else "NEEDS WORK"
    print(f"\nBest val MAE: {best_mae_mm:.2f}mm ({best_in:.3f}in)  target<0.472in  [{status}]")
    print(f"Saved: {MODEL_OUT}")
    print("\nNext: update test_rebar_validation.py MODEL_PATH to horizon_model_exp_e.pth and re-run.")

if __name__ == "__main__":
    main()
