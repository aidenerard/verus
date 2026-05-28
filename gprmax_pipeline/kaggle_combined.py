"""
Self-contained Kaggle script — rebar + delamination synthetic data in one session.

HOW TO USE:
  1. Change RUN_ID below (0 for first session, 1 for second, etc.)
  2. Paste this entire file into a Kaggle code cell
  3. Enable GPU T4 x 1 in Session options before running
  4. Run — it stops automatically before MAX_HOURS and saves everything

Output files (in /kaggle/working/):
  rebar_run{RUN_ID}.npz          — rebar signals + depth labels
  delam_run{RUN_ID}/FILE____*.csv — delamination signals in SDNET2021 format
"""

# -- CONFIG -------------------------------------------------------------------
RUN_ID    = 2      # Increment each Kaggle session: 0, 1, 2, 3 ...
MAX_HOURS = 8    # Stop before this many hours (Kaggle limit is ~12h; 7.5 is safe)
# -----------------------------------------------------------------------------

import subprocess, sys, os, time

t_start = time.time()

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

print("Installing dependencies...")
pip("numpy", "scipy", "h5py")
pip("--no-build-isolation", "git+https://github.com/gprMax/gprMax.git")

# GPU detection
USE_GPU = False
try:
    pip("pycuda")
    import pycuda.driver as cuda
    cuda.init()
    if cuda.Device.count() > 0:
        USE_GPU = True
        print(f"GPU: {cuda.Device(0).name()} — GPU mode enabled")
    else:
        print("pycuda OK but no GPU device — falling back to CPU")
except Exception as e:
    print(f"GPU setup failed ({e}) — using CPU")

if not USE_GPU:
    print("\nWARNING: Running on CPU. Each simulation takes ~20s instead of ~1.5s.")
    print("At CPU speed, expect ~1,500 runs in 8 hours instead of ~9,000.")
    print("If this is unexpected, check GPU troubleshooting in KAGGLE_INSTRUCTIONS.md\n")

import random, shutil, tempfile
import numpy as np
from scipy.signal import resample
import h5py

# -- Shared constants ---------------------------------------------------------

N_OUT     = 512
DC_OFFSET = 32768
MAX_SECS  = MAX_HOURS * 3600

def time_left():
    return MAX_SECS - (time.time() - t_start)

def extract_ez(out_path):
    with h5py.File(out_path, "r") as f:
        return f["/rxs/rx1/Ez"][:].astype(np.float32)

def run_sim(in_path):
    cmd = [sys.executable, "-m", "gprMax", in_path]
    if USE_GPU:
        cmd += ["-gpu", "0"]
    return subprocess.run(cmd, capture_output=True, text=True)

def snap(v, cell=0.002):
    return round(round(v / cell) * cell, 6)

def print_progress(i, n_done, n_failed, phase):
    elapsed  = time.time() - t_start
    per_run  = elapsed / max(n_done + n_failed, 1)
    left_sec = time_left()
    est_more = int(left_sec / per_run) if per_run > 0 else 0
    bar_done = int(20 * n_done / max(i + 1, 1))
    bar      = "#" * bar_done + "-" * (20 - bar_done)
    print(f"  [{phase}] {i+1} runs | {n_done} ok, {n_failed} failed | "
          f"{per_run:.1f}s/run | time left {left_sec/60:.0f}min | "
          f"~{est_more} more possible  [{bar}]")

# =============================================================================
# PHASE 1 — REBAR DEPTH
# =============================================================================

REBAR_TEMPLATE = """\
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

def make_rebar(rng):
    epsr      = rng.uniform(4.0, 12.0)
    sigma     = rng.uniform(0.001, 0.05)
    depth_mm  = rng.uniform(40.0, 180.0)
    diam_mm   = rng.uniform(10.0, 32.0)
    freq_mhz  = rng.choice([900, 1500, 2000])
    offset_mm = rng.uniform(30.0, 60.0)
    return dict(
        epsr=epsr, sigma=sigma,
        depth_mm=depth_mm, diameter_mm=diam_mm,
        depth_m=snap(depth_mm / 1000), radius_m=snap(diam_mm / 2000),
        freq_hz=freq_mhz * 1e6,
        src_x=snap(0.300 - offset_mm / 2000),
        rx_x=snap(0.300 + offset_mm / 2000),
    )

def process_rebar(ez):
    ez = ez - ez.mean()
    ez = resample(ez, N_OUT).astype(np.float32)
    peak = np.abs(ez).max()
    if peak > 1e-20:
        ez /= peak
    return ez

def save_rebar_checkpoint(X_list, depth_list, epsr_list, diam_list, npz_path):
    if not X_list:
        return
    np.savez_compressed(
        npz_path,
        X=np.stack(X_list)[:, np.newaxis, :],
        depth_mm=np.array(depth_list,  dtype=np.float32),
        epsr=np.array(epsr_list,        dtype=np.float32),
        diameter_mm=np.array(diam_list, dtype=np.float32),
    )
    print(f"  [checkpoint] saved {len(X_list)} rebar signals -> {npz_path}")

def run_rebar_phase(rng, work_dir, budget_secs, npz_path):
    X_list, depth_list, epsr_list, diam_list = [], [], [], []
    n_failed = 0
    i = 0
    phase_start = time.time()
    CHECKPOINT_EVERY = 1000

    print(f"\n--- PHASE 1: REBAR (budget {budget_secs/3600:.1f}h) ---")

    while (time.time() - phase_start) < budget_secs and time_left() > 60:
        p        = make_rebar(rng)
        in_path  = os.path.join(work_dir, "r_cur.in")
        out_path = os.path.join(work_dir, "r_cur.out")

        with open(in_path, "w") as f:
            f.write(REBAR_TEMPLATE.format(**p))

        run_sim(in_path)

        if os.path.exists(out_path):
            try:
                sig = process_rebar(extract_ez(out_path))
                X_list.append(sig)
                depth_list.append(p["depth_mm"])
                epsr_list.append(p["epsr"])
                diam_list.append(p["diameter_mm"])
            except Exception as exc:
                print(f"  [rebar {i}] extract error: {exc}")
                n_failed += 1
            os.remove(out_path)
        else:
            n_failed += 1

        if os.path.exists(in_path):
            os.remove(in_path)

        i += 1
        if i % 200 == 0:
            print_progress(i, len(X_list), n_failed, "rebar")
        if len(X_list) % CHECKPOINT_EVERY == 0 and len(X_list) > 0:
            save_rebar_checkpoint(X_list, depth_list, epsr_list, diam_list, npz_path)

    print_progress(i, len(X_list), n_failed, "rebar FINAL")
    save_rebar_checkpoint(X_list, depth_list, epsr_list, diam_list, npz_path)
    return X_list, depth_list, epsr_list, diam_list

# =============================================================================
# PHASE 2 — DELAMINATION
# =============================================================================

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

FILL_PROPS = {"air": (1.0, 0.0), "water": (81.0, 0.01), "debris": (4.0, 0.005)}

def make_sound(rng):
    freq_mhz = rng.choice([900, 1500, 2000])
    return dict(epsr=rng.uniform(4.0, 10.0), sigma=rng.uniform(0.001, 0.05),
                freq_mhz=freq_mhz, freq_hz=freq_mhz * 1e6)

def make_delam(rng):
    freq_mhz     = rng.choice([900, 1500, 2000])
    gap_depth_mm = rng.uniform(30.0, 120.0)
    gap_thick_mm = rng.uniform(2.0, 20.0)
    fill         = rng.choice(["air", "water", "debris"])
    fe, fs       = FILL_PROPS[fill]
    return dict(
        epsr=rng.uniform(4.0, 10.0), sigma=rng.uniform(0.001, 0.05),
        freq_mhz=freq_mhz, freq_hz=freq_mhz * 1e6,
        gap_depth_mm=gap_depth_mm, gap_thick_mm=gap_thick_mm,
        fill=fill, fill_epsr=fe, fill_sigma=fs,
        gap_top_m=snap((0.50 - gap_depth_mm / 1000) - gap_thick_mm / 1000),
        gap_bot_m=snap(0.50 - gap_depth_mm / 1000),
    )

def process_delam(ez):
    ez = resample(ez, N_OUT).astype(np.float32)
    peak = np.abs(ez).max()
    if peak > 1e-20:
        ez = ez / peak * 30000.0
    return (ez + DC_OFFSET).astype(np.float32)

def write_sdnet_csv(path, signals, labels):
    n = len(labels)
    rows = []
    r0 = [""] * max(n + 1, 5); r0[4] = str(n); rows.append(r0)
    for _ in range(6): rows.append([""] * max(n + 1, 5))
    rows.append([""] + [str(int(l)) for l in labels])
    rows.append([""] * max(n + 1, 5))
    for s in range(N_OUT):
        rows.append([str(s)] + [f"{signals[j, s]:.2f}" for j in range(n)])
    with open(path, "w") as f:
        for row in rows: f.write(",".join(row) + "\n")

def run_delam_phase(rng, work_dir, out_dir, sigs_per_csv=1000):
    signals_acc, labels_acc = [], []
    csv_index = n_failed = 0
    i = 0
    # alternate sound/delam to keep labels balanced
    label_cycle = [0, 1]

    print(f"\n--- PHASE 2: DELAMINATION (remaining budget) ---")

    def flush():
        nonlocal signals_acc, labels_acc, csv_index
        if not signals_acc: return
        sigs = np.stack(signals_acc)
        path = os.path.join(out_dir, f"FILE____{RUN_ID:02d}_{csv_index:04d}.csv")
        write_sdnet_csv(path, sigs, labels_acc)
        print(f"  wrote {path} ({len(labels_acc)} signals)")
        signals_acc, labels_acc = [], []
        csv_index += 1

    while time_left() > 60:
        label    = label_cycle[i % 2]
        p        = make_delam(rng) if label == 0 else make_sound(rng)
        in_path  = os.path.join(work_dir, "d_cur.in")
        out_path = os.path.join(work_dir, "d_cur.out")
        tmpl     = DELAM_TEMPLATE if label == 0 else SOUND_TEMPLATE

        with open(in_path, "w") as f:
            f.write(tmpl.format(**p))

        run_sim(in_path)

        if os.path.exists(out_path):
            try:
                sig = process_delam(extract_ez(out_path))
                signals_acc.append(sig)
                labels_acc.append(label)
            except Exception as exc:
                print(f"  [delam {i}] extract error: {exc}")
                n_failed += 1
            os.remove(out_path)
        else:
            n_failed += 1

        if os.path.exists(in_path):
            os.remove(in_path)

        if len(signals_acc) >= sigs_per_csv:
            flush()

        i += 1
        if i % 200 == 0:
            print_progress(i, len(signals_acc) + csv_index * sigs_per_csv, n_failed, "delam")

    flush()
    print_progress(i, csv_index * sigs_per_csv, n_failed, "delam FINAL")
    return csv_index * sigs_per_csv

# =============================================================================
# MAIN
# =============================================================================

rng_rebar = random.Random(1000 + RUN_ID)
rng_delam = random.Random(2000 + RUN_ID)

work_dir  = tempfile.mkdtemp(prefix="gprmax_")
out_dir   = f"/kaggle/working/delam_run{RUN_ID}"
os.makedirs(out_dir, exist_ok=True)

install_elapsed = time.time() - t_start
usable_secs     = MAX_SECS - install_elapsed
rebar_budget    = usable_secs / 2
mode_str        = "GPU" if USE_GPU else "CPU"

print(f"\nRUN_ID={RUN_ID} | mode={mode_str} | usable time {usable_secs/3600:.2f}h")
print(f"Splitting evenly: {rebar_budget/3600:.2f}h rebar + {rebar_budget/3600:.2f}h delam\n")

# Phase 1 — rebar
rebar_npz = f"/kaggle/working/rebar_run{RUN_ID}.npz"
X_list, depth_list, epsr_list, diam_list = run_rebar_phase(
    rng_rebar, work_dir, rebar_budget, rebar_npz
)
print(f"\nRebar NPZ: {rebar_npz}  n={len(X_list)}")

# Phase 2 — delamination
n_delam = run_delam_phase(rng_delam, work_dir, out_dir)

shutil.rmtree(work_dir, ignore_errors=True)

total_min = (time.time() - t_start) / 60
print(f"\n=== DONE ===")
print(f"Total time: {total_min:.1f} min")
print(f"Rebar signals:      {len(X_list):,}  -> {rebar_npz}")
print(f"Delam signals:      {n_delam:,}  -> {out_dir}/")
print(f"Next session:       set RUN_ID = {RUN_ID + 1}")
