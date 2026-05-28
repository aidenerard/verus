# Running the gprMax Pipeline on Kaggle

One session at a time. Each run generates as much data as possible within 7.5 hours,
then stops automatically and saves everything.

**Expected output per session:**
| Mode | Runs/hour | Per 7.5h session | Notes |
|---|---|---|---|
| T4 GPU (working) | ~2,400 | ~9,000 rebar + ~9,000 delam | Target |
| CPU fallback | ~180 | ~700 rebar + ~700 delam | Do not use — check GPU setup |

To reach 50,000 rebar + 50,000 delam: **~6 sessions on GPU** (~2 weeks at 30h/week budget).

---

## One-time Kaggle account setup

1. Go to **kaggle.com** and sign in (or create a free account).
2. Click your profile picture (top-right) → **Settings**.
3. Under **Phone Verification**, verify your phone number — this unlocks GPU access.

---

## Each session (repeat with RUN_ID = 0, 1, 2, ...)

### Step 1 — Create a new notebook

Go to **kaggle.com/code** → click **+ New Notebook**.

### Step 2 — Enable GPU (do this first, before pasting code)

1. Right sidebar → **Session options** (gear icon).
2. Under **Accelerator**, select **GPU T4 x 1**.
3. Click **Save**. The sidebar should show "GPU T4 x 1".

> If GPU T4 doesn't appear as an option, phone verification hasn't cleared yet — wait a few minutes and refresh.

### Step 3 — Paste the script

1. Click inside the code cell, select all (Ctrl+A), delete it.
2. Open `gprmax_pipeline/kaggle_combined.py` from this repo.
3. At the very top, set `RUN_ID` to the current session number:
   ```python
   RUN_ID = 0   # 0 for first run, 1 for second, 2 for third, ...
   ```
4. Copy the entire file and paste it into the Kaggle cell.

### Step 4 — Run using Save & Run All (NOT the plain Run button)

> **Important:** Use **Save & Run All (Commit)** — not the plain ▶ Run All button. Only committed runs save outputs to the Output tab. Interactive runs lose their files the moment you navigate away or a new session starts.

1. Click the **Save Version** button (top-right, looks like a floppy disk or says "Save").
2. Select **Save & Run All (Commit)**.
3. Click **Save**. Kaggle queues the notebook to run in the background — you can close the tab and come back later.
4. The run appears under **Your Work → Notebooks** with a spinning indicator while running.
5. When it finishes (green checkmark), click into it → **Output** tab → download your files.

After the install finishes (~3 min), look for:

```
GPU: Tesla T4 — GPU mode enabled
```

If you see `GPU setup failed — using CPU` instead, **stop the run** and see the GPU troubleshooting section below.

### Step 5 — Wait (you can close the tab)

Since you used Save & Run All, Kaggle runs this in the background. You'll get an email when it finishes. You can also check progress at **kaggle.com/code** — your notebook will show a spinning indicator.

The script saves a rebar checkpoint to `/kaggle/working/rebar_run{N}.npz` every 1,000 signals, so even if Kaggle cuts the session early, you won't lose everything.

Progress in the log every 200 runs:
```
--- PHASE 1: REBAR (budget 3.7h) ---
  [rebar] 200 runs | 198 ok, 2 failed | 1.4s/run | time left 415min | ~17,700 more possible  [####################]
  [rebar] 400 runs | 396 ok, 4 failed | 1.4s/run | time left 389min | ~16,650 more possible  [####################]
  ...
--- PHASE 2: DELAMINATION (remaining budget) ---
  ...
=== DONE ===
Total time: 447.2 min
Rebar signals:      9,341  -> /kaggle/working/rebar_run0.npz
Delam signals:      9,204  -> /kaggle/working/delam_run0/
Next session:       set RUN_ID = 1
```

The script prints `Next session: set RUN_ID = 1` at the end so you know what to set next time.

### Step 6 — Download outputs

Right sidebar → **Output** tab. You'll see:
- `rebar_run0.npz` — click to download
- `delam_run0/` folder — click the download icon (downloads as zip)

Save both. Repeat for each session, incrementing `RUN_ID`.

---

## GPU troubleshooting

**"GPU setup failed — using CPU":**

Run this diagnostic in a separate cell before pasting the main script:
```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pycuda"])
import pycuda.driver as cuda
cuda.init()
print(cuda.Device.count(), "GPU(s) found")
print(cuda.Device(0).name())
```

- If it prints `1 GPU(s) found` + `Tesla T4` → pycuda works. Restart kernel, then run the main script.
- If it errors → GPU accelerator wasn't properly enabled. Stop, go to Session options, re-select GPU T4, save, restart kernel, try again.

**Many "FAILED — no .out produced" errors:**

GPU flag may be crashing gprMax. Force CPU mode by adding this line right after the GPU detection block:
```python
USE_GPU = False   # force CPU
```
CPU is slow but reliable. Reduce `MAX_HOURS = 7.5` to `MAX_HOURS = 6.0` if using CPU to give yourself time to download before the session auto-closes.

---

## After all sessions — merge outputs locally

### Merge rebar NPZ files

```python
import numpy as np, glob, os

files = sorted(glob.glob("rebar_run*.npz"))
print(f"Found {len(files)} session files")

X    = np.concatenate([np.load(f)["X"]           for f in files])
dep  = np.concatenate([np.load(f)["depth_mm"]    for f in files])
eps  = np.concatenate([np.load(f)["epsr"]        for f in files])
diam = np.concatenate([np.load(f)["diameter_mm"] for f in files])

os.makedirs("data", exist_ok=True)
np.savez_compressed("data/synthetic_rebar_gprmax.npz",
                    X=X, depth_mm=dep, epsr=eps, diameter_mm=diam)
print(f"Merged: X={X.shape}")   # e.g. (55000, 1, 512) after 6 sessions
```

### Move delamination CSVs

Unzip each `delam_run{N}.zip` and move all `FILE____*.csv` files into:
```
data/csv/synthetic_gprmax/
```

Add that path to `DATA_FOLDERS` in `kaggle_push/cnn.py` and the training script picks them up automatically.

---

## Tracking your progress

After each session, note down how many signals you collected:

| RUN_ID | Rebar | Delam |
|---|---|---|
| 0 | | |
| 1 | | |
| 2 | | |
| ... | | |
| **Total** | **target: 50,000** | **target: 50,000** |
