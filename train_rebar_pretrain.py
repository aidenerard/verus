"""
Pretrain HorizonCNN conv layers on gprMax synthetic rebar traces.

Input is either:
  - a directory of .out (HDF5) files produced by gprMax + meta.csv (from generate_sweep.py)
  - a .npz file from the Kaggle rebar batch script (keys: X, depth_mm)

Output: server/models/horizon_model_pretrained.pth  (conv + attn + depth_head weights)

Depth labels normalized by MAX_DEPTH_MM=250 (matches train_rebar_exp_e.py). Critical:
depth_head from this checkpoint must be re-initialized before fine-tuning on real data.
Amplitude statistics in gprMax synthetics are physically unreliable (no antenna coupling,
no 3D spreading). Only the conv feature extractors are worth transferring.

Usage:
    python train_rebar_pretrain.py gprmax_pipeline/runs/rebar2mat_test
    python train_rebar_pretrain.py data/synthetic_rebar_gprmax.npz
    python train_rebar_pretrain.py <dir_or_npz> --reinit-head  # fine-tune safety check
"""
import argparse, csv, os, time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample as scipy_resample
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

MAX_DEPTH_MM   = 250.0   # matches train_rebar_exp_e.py and covers B440029 max truth (234mm)
TARGET_SAMPLES = 512
BATCH_SIZE     = 512
EPOCHS         = 50
PATIENCE       = 15
LR             = 1e-3
VAL_FRAC       = 0.1
MODEL_OUT      = Path(__file__).parent / "server" / "models" / "horizon_model_pretrained.pth"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_ez(out_path: str) -> np.ndarray:
    import h5py
    with h5py.File(out_path, "r") as f:
        return f["/rxs/rx1/Ez"][:].astype(np.float32)


def _process(ez: np.ndarray) -> np.ndarray:
    """DC-remove → resample to 512 → max-abs normalize.
    Order differs from colab_train_horizon.py (DC-remove before resample vs after);
    numerically equivalent for GPR A-scans.
    """
    ez = ez - ez.mean()
    ez = scipy_resample(ez, TARGET_SAMPLES).astype(np.float32)
    peak = float(np.abs(ez).max())
    if peak > 1e-20:
        ez /= peak
    return ez


def _load_from_dir(input_dir: str):
    """Load .out files + meta.csv from a gprMax run directory."""
    meta_path = os.path.join(input_dir, "meta.csv")
    with open(meta_path) as fh:
        rows = list(csv.DictReader(fh))
    depth_col = "top_depth_mm" if "top_depth_mm" in rows[0] else "depth_mm"

    X_list, y_list = [], []
    n_missing = 0
    for row in rows:
        out_name = os.path.splitext(row["filename"])[0] + ".out"
        out_path = os.path.join(input_dir, out_name)
        if not os.path.exists(out_path):
            n_missing += 1
            continue
        try:
            ez = _extract_ez(out_path)
            sig = _process(ez)
        except Exception:
            n_missing += 1
            continue
        X_list.append(sig)
        y_list.append(np.clip(float(row[depth_col]) / MAX_DEPTH_MM, 0.0, 1.0))

    if n_missing:
        print(f"  WARNING: {n_missing} .out files missing or unreadable (run gprMax first)")
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def _load_from_npz(npz_path: str):
    """Load from Kaggle rebar batch NPZ output (keys: X, depth_mm)."""
    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    if X.ndim == 3:
        X = X[:, 0, :]   # (N, 1, 512) → (N, 512)
    depth_mm = data["depth_mm"].astype(np.float32)
    y = np.clip(depth_mm / MAX_DEPTH_MM, 0.0, 1.0)
    return X, y


class _GPRDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, aug: bool = False):
        self.X   = torch.from_numpy(X).unsqueeze(1)   # (N, 1, 512)
        self.y   = torch.from_numpy(y)
        self.aug = aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x, y = self.X[i].clone(), self.y[i]
        if self.aug:
            if torch.rand(1) < 0.5:
                x += torch.randn_like(x) * 0.056   # ~25 dB SNR, matches real GSSI noise floor
            if torch.rand(1) < 0.5:
                x *= 0.85 + torch.rand(1) * 0.30
            # Extended ±30 shift to approximate timing uncertainty from asphalt and
            # antenna coupling absent in simulation.
            if torch.rand(1) < 0.6:
                sh = torch.randint(-30, 31, (1,)).item()
                x  = torch.roll(x, sh, -1)
                if sh > 0:  x[..., :sh]  = 0.0
                elif sh < 0: x[..., sh:] = 0.0
        return x, y


class _TemporalAttention(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.score = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.attn       = _TemporalAttention(128)
        self.depth_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depth_head(self.attn(self.conv(x))).squeeze(-1)


def reinit_head(model: HorizonCNN) -> None:
    """Re-initialize depth_head to random weights before fine-tuning on real data.

    gprMax amplitude statistics are physically unreliable. Only conv feature
    extractors should be transferred; the head must always be retrained on real labels.
    """
    for layer in model.depth_head:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    print("depth_head re-initialized — conv weights retained, head weights discarded")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretrain HorizonCNN on gprMax synthetic data")
    ap.add_argument("input", help=".out directory with meta.csv, or path to .npz")
    ap.add_argument("--out",   default=str(MODEL_OUT), help="Output checkpoint path")
    ap.add_argument("--load",  default=None,           help="Resume from this checkpoint")
    ap.add_argument(
        "--reinit-head", action="store_true",
        help="Re-initialize depth_head after loading --load checkpoint. "
             "Required in fine-tune mode; validates no amplitude leak from synthetic weights.",
    )
    args = ap.parse_args()

    print(f"Device: {DEVICE}  MAX_DEPTH_MM: {MAX_DEPTH_MM}")

    if args.input.endswith(".npz"):
        X, y = _load_from_npz(args.input)
    else:
        X, y = _load_from_dir(args.input)

    n = len(X)
    print(f"Loaded {n:,} traces  depth=[{(y*MAX_DEPTH_MM).min():.1f}, {(y*MAX_DEPTH_MM).max():.1f}]mm")

    rng   = np.random.RandomState(42)
    idx   = rng.permutation(n)
    n_val = max(int(n * VAL_FRAC), 1)
    X_va, y_va = X[idx[:n_val]], y[idx[:n_val]]
    X_tr, y_tr = X[idx[n_val:]], y[idx[n_val:]]

    tr_loader = DataLoader(_GPRDataset(X_tr, y_tr, aug=True),  BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(_GPRDataset(X_va, y_va, aug=False), BATCH_SIZE, shuffle=False)
    print(f"Train: {len(X_tr):,}  Val: {len(X_va):,}")

    model = HorizonCNN().to(DEVICE)

    if args.load:
        state = torch.load(args.load, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.load}")

    if args.reinit_head:
        before = {k: p.data.clone() for k, p in model.depth_head.named_parameters()}
        reinit_head(model)
        after = {k: p.data for k, p in model.depth_head.named_parameters()}
        assert any(not torch.equal(before[k], after[k]) for k in before), \
            "depth_head re-init failed — all parameters unchanged after reset_parameters()"

    crit  = nn.SmoothL1Loss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf"); pat = 0; t0 = time.time()
    print(f"\n  Ep   tr_loss  val_loss  MAE_mm   best_mm  elapsed")
    print(f"  {'-'*52}")

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
        mae = float(np.abs(pv - tv).mean())
        sched.step()

        if mae < best_mae:
            best_mae = mae; pat = 0
            torch.save(model.state_dict(), str(out_path))
        else:
            pat += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  {epoch:3d}  {tr_loss:.5f}  {val_loss:.5f}  {mae:7.2f}  "
                  f"{best_mae:7.2f}  {elapsed:.0f}s", flush=True)

        if pat >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"\nBest pretrain val MAE: {best_mae:.2f}mm ({best_mae/25.4:.3f}in)  [synthetic data only]")
    print(f"Saved: {out_path}")
    print("\nNext: run train_rebar_exp_e.py — it loads this checkpoint with --reinit-head=True "
          "to discard the synthetic head before fine-tuning on real data.")


if __name__ == "__main__":
    main()
