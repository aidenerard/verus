"""
Self-contained Kaggle script — delamination classification synthetic data.

Paste the entire file as a single Kaggle code cell.
Set BATCH_ID to 0-9 before running; 10 notebooks in parallel cover 50,000 runs
(25,000 delaminated + 25,000 sound, balanced per batch).

Output saved to /kaggle/working/delam_batch{BATCH_ID}/ as SDNET2021 CSVs
(directly loadable by kaggle_push/cnn.py's load_csv).
"""

# -- CONFIG: change BATCH_ID per notebook -------------------------------------
BATCH_ID     = 0      # 0-9
N_PER_BATCH  = 5000   # 2500 delaminated + 2500 sound per notebook
GLOBAL_SEED  = 2000   # each batch uses seed GLOBAL_SEED + BATCH_ID
SIGS_PER_CSV = 1000   # signals written per output CSV file
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

N_OUT     = 512
DC_OFFSET = 32768

def extract_ez(out_path):
    with h5py.File(out_path, "r") as f:
        return f["/rxs/rx1/Ez"][:].astype(np.float32)

def process_delam(ez):
    """Resample -> scale to +-30000 -> add DC_OFFSET (matches cnn.py load_csv)."""
    ez = resample(ez, N_OUT).astype(np.float32)
    peak = np.abs(ez).max()
    if peak > 1e-20:
        ez = ez / peak * 30000.0
    return (ez + DC_OFFSET).astype(np.float32)

# -- .in file templates -------------------------------------------------------

SOUND_TEMPLATE = """\
#title: sound epsr={epsr:.2f} freq={freq_mhz:.0f}MHz
#domain: 0.60 0.50 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 8e-9

#material: {epsr:.4f} {sigma:.4f} 1 0 concrete

#waveform: ricker 1 {freq_hz:.0f} src_wave
#hertzian_dipole: z 0.300 0.450 0 src_wave
#rx: 0.340 0.450 0

#box: 0 0 0 0.60 0.50 0.002 concrete
"""

DELAM_TEMPLATE = """\
#title: delaminated gap_depth={gap_depth_mm:.1f}mm thick={gap_thick_mm:.1f}mm fill={fill}
#domain: 0.60 0.50 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 8e-9

#material: {epsr:.4f} {sigma:.4f} 1 0 concrete
#material: {fill_epsr:.4f} {fill_sigma:.4f} 1 0 gap_fill

#waveform: ricker 1 {freq_hz:.0f} src_wave
#hertzian_dipole: z 0.300 0.450 0 src_wave
#rx: 0.340 0.450 0

#box: 0 0 0 0.60 0.50 0.002 concrete
#box: 0 {gap_top_m:.4f} 0 0.60 {gap_bot_m:.4f} 0.002 gap_fill
"""

FILL_PROPS = {
    "air":    (1.0,  0.0),
    "water":  (81.0, 0.01),
    "debris": (4.0,  0.005),
}

def snap(v, cell=0.002):
    return round(round(v / cell) * cell, 6)

def make_sound(rng):
    epsr     = rng.uniform(4.0, 10.0)
    sigma    = rng.uniform(0.001, 0.05)
    freq_mhz = rng.choice([900, 1500, 2000])
    return dict(epsr=epsr, sigma=sigma, freq_mhz=freq_mhz, freq_hz=freq_mhz * 1e6)

def make_delam(rng):
    epsr         = rng.uniform(4.0, 10.0)
    sigma        = rng.uniform(0.001, 0.05)
    freq_mhz     = rng.choice([900, 1500, 2000])
    gap_depth_mm = rng.uniform(30.0, 120.0)
    gap_thick_mm = rng.uniform(2.0, 20.0)
    fill         = rng.choice(["air", "water", "debris"])
    fill_epsr, fill_sigma = FILL_PROPS[fill]
    gap_top_m    = snap((0.50 - gap_depth_mm / 1000) - gap_thick_mm / 1000)
    gap_bot_m    = snap(0.50 - gap_depth_mm / 1000)
    return dict(
        epsr=epsr, sigma=sigma,
        freq_mhz=freq_mhz, freq_hz=freq_mhz * 1e6,
        gap_depth_mm=gap_depth_mm, gap_thick_mm=gap_thick_mm,
        fill=fill, fill_epsr=fill_epsr, fill_sigma=fill_sigma,
        gap_top_m=gap_top_m, gap_bot_m=gap_bot_m,
    )

def run_gprmax(in_path):
    cmd = [sys.executable, "-m", "gprMax", in_path]
    if USE_GPU:
        cmd += ["-gpu", "0"]
    return subprocess.run(cmd, capture_output=True, text=True)

# -- SDNET2021 CSV writer -----------------------------------------------------

def write_sdnet_csv(path, signals, labels):
    """
    Write (n_signals, 512) signals + labels to SDNET2021 CSV.
    Row 0 col 4 = n_signals; row 7 cols 1..n = labels; rows 9+ = amplitude.
    """
    n = len(labels)
    rows = []
    r0 = [""] * max(n + 1, 5)
    r0[4] = str(n)
    rows.append(r0)
    for _ in range(6):
        rows.append([""] * max(n + 1, 5))
    rows.append([""] + [str(int(lbl)) for lbl in labels])
    rows.append([""] * max(n + 1, 5))
    for s in range(N_OUT):
        rows.append([str(s)] + [f"{signals[j, s]:.2f}" for j in range(n)])
    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(row) + "\n")

# -- Main loop ----------------------------------------------------------------

seed       = GLOBAL_SEED + BATCH_ID
rng        = random.Random(seed)
half       = N_PER_BATCH // 2
labels_seq = [0] * half + [1] * (N_PER_BATCH - half)
rng.shuffle(labels_seq)

signals_acc = []
labels_acc  = []
csv_index   = 0
out_dir     = f"/kaggle/working/delam_batch{BATCH_ID}"
os.makedirs(out_dir, exist_ok=True)

work_dir = tempfile.mkdtemp(prefix="delam_")
mode_str = "GPU" if USE_GPU else "CPU"
print(f"Batch {BATCH_ID} | seed={seed} | mode={mode_str} | workdir={work_dir}")

t_batch  = time.time()
n_failed = 0

def flush_csv():
    global signals_acc, labels_acc, csv_index
    if not signals_acc:
        return
    sigs = np.stack(signals_acc, axis=0)
    path = os.path.join(out_dir, f"FILE____{BATCH_ID:02d}_{csv_index:04d}.csv")
    write_sdnet_csv(path, sigs, labels_acc)
    print(f"  wrote {path}  ({len(labels_acc)} signals)")
    signals_acc = []
    labels_acc  = []
    csv_index  += 1

for i, label in enumerate(labels_seq):
    p        = make_delam(rng) if label == 0 else make_sound(rng)
    in_path  = os.path.join(work_dir, f"d_{i:05d}.in")
    out_path = os.path.join(work_dir, f"d_{i:05d}.out")

    tmpl = DELAM_TEMPLATE if label == 0 else SOUND_TEMPLATE
    with open(in_path, "w") as f:
        f.write(tmpl.format(**p))

    result = run_gprmax(in_path)

    if not os.path.exists(out_path):
        n_failed += 1
        if n_failed <= 5:
            print(f"  [{i}] FAILED — skipping")
            print((result.stderr or "")[-400:])
        continue

    try:
        ez  = extract_ez(out_path)
        sig = process_delam(ez)
        signals_acc.append(sig)
        labels_acc.append(label)
    except Exception as exc:
        print(f"  [{i}] extract error: {exc}")
        n_failed += 1

    os.remove(in_path)
    os.remove(out_path)

    if len(signals_acc) >= SIGS_PER_CSV:
        flush_csv()

    if (i + 1) % 500 == 0:
        elapsed   = time.time() - t_batch
        per_run   = elapsed / (i + 1)
        remaining = per_run * (N_PER_BATCH - i - 1)
        eta_min   = remaining / 60
        print(f"  {i+1}/{N_PER_BATCH} | {per_run:.1f}s/run | ETA {eta_min:.0f} min")

flush_csv()
shutil.rmtree(work_dir, ignore_errors=True)
total_min = (time.time() - t_batch) / 60
print(f"\nDone. CSVs saved to {out_dir}/")
print(f"Total time: {total_min:.1f} min | failed: {n_failed}/{N_PER_BATCH}")
