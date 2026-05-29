"""
Self-contained Kaggle script — rebar depth regression synthetic data.

Paste the entire file as a single Kaggle code cell.
Set BATCH_ID to 0-9 before running; 10 notebooks in parallel cover 50,000 runs.

Output saved to /kaggle/working/rebar_batch{BATCH_ID}.npz with keys:
  X          (5000, 1, 512) float32   normalised Ez signal
  depth_mm   (5000,)        float32
  epsr       (5000,)        float32
  diameter_mm(5000,)        float32
"""

# -- CONFIG: change BATCH_ID per notebook -------------------------------------
BATCH_ID    = 0       # 0-9
N_PER_BATCH = 5000
GLOBAL_SEED = 1000    # each batch uses seed GLOBAL_SEED + BATCH_ID
# -----------------------------------------------------------------------------

import subprocess, sys, os, time

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("numpy", "scipy", "h5py")
pip("--no-build-isolation", "git+https://github.com/gprMax/gprMax.git")

# Try GPU support (pycuda). If install fails, fall back to CPU silently.
USE_GPU = False
try:
    pip("pycuda")
    import pycuda.driver as cuda
    cuda.init()
    if cuda.Device.count() > 0:
        USE_GPU = True
        print(f"GPU available: {cuda.Device(0).name()} — using GPU mode")
    else:
        print("pycuda installed but no GPU device found — using CPU")
except Exception as e:
    print(f"GPU setup failed ({e}) — using CPU")

import random, shutil, tempfile
import numpy as np
from scipy.signal import resample
import h5py

# -- Signal extraction --------------------------------------------------------

N_OUT = 512

def extract_ez(out_path):
    with h5py.File(out_path, "r") as f:
        return f["/rxs/rx1/Ez"][:].astype(np.float32)

def process_rebar(ez):
    ez = ez - ez.mean()
    ez = resample(ez, N_OUT).astype(np.float32)
    peak = np.abs(ez).max()
    if peak > 1e-20:
        ez /= peak
    return ez

# -- .in file generation ------------------------------------------------------

TEMPLATE = """\
#title: rebar depth={depth_mm:.1f}mm diam={diameter_mm:.1f}mm epsr={epsr:.2f}
#domain: 0.60 0.50 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 8e-9

#material: {epsr:.4f} {sigma:.4f} 1 0 concrete

#waveform: ricker 1 {freq_hz:.0f} src_wave
#hertzian_dipole: z {src_x:.3f} 0.450 0 src_wave
#rx: {rx_x:.3f} 0.450 0

#box: 0 0 0 0.60 0.40 0.002 concrete
#cylinder: 0.300 {depth_m:.4f} 0 0.300 {depth_m:.4f} 0.002 {radius_m:.4f} pec
"""

def snap(v, cell=0.002):
    return round(round(v / cell) * cell, 6)

def make_params(rng):
    epsr      = rng.uniform(4.0, 12.0)
    sigma     = rng.uniform(0.001, 0.05)
    depth_mm  = rng.uniform(40.0, 180.0)
    diam_mm   = rng.uniform(10.0, 32.0)
    freq_mhz  = rng.choice([900, 1500, 2000])
    offset_mm = rng.uniform(30.0, 60.0)
    depth_m   = snap(depth_mm / 1000)
    radius_m  = snap(diam_mm / 2000)
    src_x     = snap(0.300 - offset_mm / 2000)
    rx_x      = snap(0.300 + offset_mm / 2000)
    return dict(
        epsr=epsr, sigma=sigma,
        depth_mm=depth_mm, diameter_mm=diam_mm,
        depth_m=depth_m, radius_m=radius_m,
        freq_hz=freq_mhz * 1e6, src_x=src_x, rx_x=rx_x,
    )

def run_gprmax(in_path):
    cmd = [sys.executable, "-m", "gprMax", in_path]
    if USE_GPU:
        cmd += ["-gpu", "0"]
    return subprocess.run(cmd, capture_output=True, text=True)

# -- Main loop ----------------------------------------------------------------

seed = GLOBAL_SEED + BATCH_ID
rng  = random.Random(seed)

X_all     = np.zeros((N_PER_BATCH, 1, N_OUT), dtype=np.float32)
depth_all = np.zeros(N_PER_BATCH,             dtype=np.float32)
epsr_all  = np.zeros(N_PER_BATCH,             dtype=np.float32)
diam_all  = np.zeros(N_PER_BATCH,             dtype=np.float32)

work_dir = tempfile.mkdtemp(prefix="rebar_")
mode_str = "GPU" if USE_GPU else "CPU"
print(f"Batch {BATCH_ID} | seed={seed} | mode={mode_str} | workdir={work_dir}")

t_batch = time.time()
n_failed = 0

for i in range(N_PER_BATCH):
    p        = make_params(rng)
    in_path  = os.path.join(work_dir, f"r_{i:05d}.in")
    out_path = os.path.join(work_dir, f"r_{i:05d}.out")

    with open(in_path, "w") as f:
        f.write(TEMPLATE.format(**p))

    result = run_gprmax(in_path)

    if not os.path.exists(out_path):
        n_failed += 1
        if n_failed <= 5:
            print(f"  [{i}] FAILED — no .out produced")
            print((result.stderr or "")[-400:])
        continue

    try:
        ez          = extract_ez(out_path)
        sig         = process_rebar(ez)
        X_all[i, 0] = sig
        depth_all[i] = p["depth_mm"]
        epsr_all[i]  = p["epsr"]
        diam_all[i]  = p["diameter_mm"]
    except Exception as exc:
        print(f"  [{i}] extract error: {exc}")
        n_failed += 1

    os.remove(in_path)
    os.remove(out_path)

    if (i + 1) % 500 == 0:
        elapsed  = time.time() - t_batch
        per_run  = elapsed / (i + 1)
        remaining = per_run * (N_PER_BATCH - i - 1)
        eta_min  = remaining / 60
        print(f"  {i+1}/{N_PER_BATCH} | {per_run:.1f}s/run | ETA {eta_min:.0f} min")

shutil.rmtree(work_dir, ignore_errors=True)

out_npz = f"/kaggle/working/rebar_batch{BATCH_ID}.npz"
np.savez_compressed(
    out_npz,
    X=X_all,
    depth_mm=depth_all,
    epsr=epsr_all,
    diameter_mm=diam_all,
)
total_min = (time.time() - t_batch) / 60
print(f"\nSaved {out_npz}  shape X={X_all.shape}")
print(f"Total time: {total_min:.1f} min | failed: {n_failed}/{N_PER_BATCH}")
