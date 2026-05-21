"""
Paste this entire file as a single Colab cell (after mounting Drive + unzipping data).
Trains HorizonCNN on Terracon Proceq .scan files, saves horizon_model.pth to Drive.

Setup cells to run first:
  Cell 1:
    from google.colab import drive
    drive.mount('/content/drive')

  Cell 2:
    import zipfile, os
    # terracon_data.zip must have a shortcut added to MyDrive first
    # (right-click in Drive -> Organize -> Add shortcut to Drive -> My Drive)
    with zipfile.ZipFile('/content/drive/MyDrive/terracon_data.zip', 'r') as z:
        z.extractall('/content/data')
    print('Files:', len(os.listdir('/content/data')))

  Cell 3 (this file):
    paste and run
"""
import os, glob, numpy as np, torch, torch.nn as nn, time
from scipy.signal import resample as scipy_resample
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = '/content/data'
MODEL_OUT = '/content/drive/MyDrive/horizon_model.pth'   # persists after session ends

MAX_DEPTH_MM   = 300.0
TARGET_SAMPLES = 512
BATCH_SIZE     = 512
EPOCHS         = 100
PATIENCE       = 15
LR             = 1e-3
SKIP_SWATHS    = {0, 4}   # swath 1 (bottom-mat depths) + swath 5 (corrupted TS)
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Proceq D-block reader ─────────────────────────────────────────────────────
_MAGIC   = b'VH01SW'
_D_START = 0x027C
_D_SIZE  = 0x040C
_D_HDR   = 16
_D_REF   = 16
_D_MARK  = b'D\x00'
_D_SAMP  = (_D_SIZE - _D_HDR) // 2   # 510 samples per trace

def read_proceq(path):
    raw = open(path, 'rb').read()
    if raw[:6] != _MAGIC:
        return None, 0
    n = 0; pos = _D_START
    while pos + 2 <= len(raw) and raw[pos:pos+2] == _D_MARK:
        n += 1; pos += _D_SIZE
    nd = n - _D_REF
    if nd <= 0:
        return None, 0
    traces = np.zeros((nd, _D_SAMP), dtype=np.float32)
    with open(path, 'rb') as f:
        f.seek(_D_START + _D_REF * _D_SIZE)
        for i in range(nd):
            blk = f.read(_D_SIZE)
            if len(blk) < _D_SIZE:
                traces = traces[:i]; break
            s = np.frombuffer(blk[_D_HDR:], dtype='<i2').astype(np.float32)
            s -= s.mean()
            traces[i] = s
    norms = np.abs(traces).max(axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    traces /= norms
    return traces, len(traces)

def preprocess(arr):
    if arr.shape[1] != TARGET_SAMPLES:
        arr = scipy_resample(arr, TARGET_SAMPLES, axis=1).astype(np.float32)
    arr = arr - arr.mean(axis=1, keepdims=True)
    norms = np.abs(arr).max(axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)

# ── Load dataset ──────────────────────────────────────────────────────────────
scan_files = sorted(glob.glob(os.path.join(DATA_DIR, 'PRC_*.scan')))
odd_scans  = [f for f in scan_files
              if int(os.path.basename(f).replace('PRC_','').replace('.scan','')) % 2 == 1]
ts_files   = sorted(glob.glob(os.path.join(DATA_DIR, 'TS_*_1.txt')))
n_swaths   = min(len(odd_scans) // 4, len(ts_files))
good       = [i for i in range(n_swaths) if i not in SKIP_SWATHS]
val_set    = set(good[-3:])
print(f'Swaths used: {[s+1 for s in good]}  |  val: {[s+1 for s in val_set]}')

all_X, all_y, all_sw = [], [], []
for sw in good:
    ts = np.loadtxt(ts_files[sw])
    for sp in odd_scans[sw*4 : sw*4+4]:
        raw, n = read_proceq(sp)
        if raw is None or n < 10:
            continue
        gt = np.interp(np.arange(n), np.linspace(0, n-1, len(ts)), ts)
        all_X.append(preprocess(raw))
        all_y.append(np.clip(gt / MAX_DEPTH_MM, 0, 1).astype(np.float32))
        all_sw.extend([sw] * n)
    print(f'  sw{sw+1:02d} loaded', flush=True)

X      = np.concatenate(all_X)
y      = np.concatenate(all_y)
sw_ids = np.array(all_sw)
tr_m   = np.array([s not in val_set for s in sw_ids])
X_tr, y_tr = X[tr_m],  y[tr_m]
X_va, y_va = X[~tr_m], y[~tr_m]
print(f'\nTrain: {len(X_tr):,}  Val: {len(X_va):,}')
print(f'Depth range: {(y*MAX_DEPTH_MM).min():.0f}-{(y*MAX_DEPTH_MM).max():.0f} mm')

# ── Dataset + DataLoader ──────────────────────────────────────────────────────
class GPRDataset(Dataset):
    def __init__(self, X, y, aug=False):
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y)
        self.aug = aug
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self.X[i].clone(); y = self.y[i]
        if self.aug:
            if torch.rand(1) < 0.5: x += torch.randn_like(x) * 0.01
            if torch.rand(1) < 0.5: x *= (0.9 + torch.rand(1) * 0.2)
            if torch.rand(1) < 0.5:
                sh = torch.randint(-10, 11, (1,)).item()
                x = torch.roll(x, sh, -1)
                if sh > 0:  x[..., :sh]  = 0
                elif sh < 0: x[..., sh:] = 0
        return x, y

tr_loader = DataLoader(GPRDataset(X_tr, y_tr, aug=True),  BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
va_loader = DataLoader(GPRDataset(X_va, y_va, aug=False), BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
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
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.head(self.attn(self.conv(x))).squeeze(-1)

model = HorizonCNN().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

crit  = nn.SmoothL1Loss()
opt   = torch.optim.Adam(model.parameters(), lr=LR)
sched = CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

# ── Training loop ─────────────────────────────────────────────────────────────
best_mae = float('inf'); pat = 0; history = []
print(f'\n  Ep   tr_loss  val_loss  MAE_mm   MAE_in   best_mm  elapsed')
print(f'  {"-"*60}')
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train(); tr_loss = 0
    for xb, yb in tr_loader:
        opt.zero_grad()
        loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
        loss.backward(); opt.step()
        tr_loss += loss.item() * len(yb)
    tr_loss /= len(tr_loader.dataset)

    model.eval(); val_loss = 0; pv, tv = [], []
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
    history.append({'epoch': epoch, 'tr_loss': tr_loss, 'val_loss': val_loss, 'mae_mm': mae})

    if mae < best_mae:
        best_mae = mae; pat = 0
        torch.save(model.state_dict(), MODEL_OUT)
    else:
        pat += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f'  {epoch:3d}  {tr_loss:.5f}  {val_loss:.5f}  {mae:7.2f}  '
              f'{mae/25.4:7.3f}  {best_mae:7.2f}  {time.time()-t0:.0f}s', flush=True)

    if pat >= PATIENCE:
        print(f'  Early stop at epoch {epoch}')
        break

print(f'\nBest val MAE: {best_mae:.2f} mm  ({best_mae/25.4:.3f} in)')
print(f'Saved: {MODEL_OUT}')
