"""
generate_synthetic.py
Generates 50,000 labeled synthetic GPR A-scan signals using gprMax FDTD simulation.
Targets 30,000 Class-1 (sound) and 20,000 Class-2 (delaminated) signals.
Output format matches SDNET2021: 512 samples, 12 ns, DC offset = 32,768.

Installation (run once before using this script):
    pip install gprMax
  or on Colab:
    !pip install gprMax

Usage:
    python3 generate_synthetic.py

Output:
    ~/Desktop/verus/synthetic_data/synthetic_class1.csv
    ~/Desktop/verus/synthetic_data/synthetic_class2.csv
"""

import sys
import os
sys.path.insert(0, '/content/gprMax')

import subprocess
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import interp1d

# ── Check gprMax is importable ────────────────────────────────────────────────
try:
    import gprMax
    print(f"gprMax imported from: {gprMax.__file__}", flush=True)
except ImportError as e:
    print(f"ERROR: could not import gprMax: {e}", flush=True)
    print(f"  sys.path = {sys.path}", flush=True)
    print("  Make sure gprMax is installed or cloned to /content/gprMax", flush=True)
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("/content/drive/MyDrive/fluxspace/fluxspace_gpr_data/synthetic_data")
N_CLASS1    = 30_000
N_CLASS2    = 20_000
N_SAMPLES   = 512
TIME_WINDOW = 12e-9      # 12 ns
DC_OFFSET   = 32_768
FREQ        = 1.5e9      # 1.5 GHz

# FDTD domain parameters
DX          = 0.001      # 1 mm spatial resolution
DZ          = DX * 2     # thin z-dimension (2.5-D)
SLAB_Y      = 0.150      # 15 cm concrete slab (y = 0 at slab bottom, y = SLAB_Y at surface)
AIR_Y       = 0.040      # 4 cm air above surface
DOMAIN_X    = 0.100      # 10 cm horizontal width
DOMAIN_Y    = SLAB_Y + AIR_Y
ANT_X       = DOMAIN_X / 2          # centred horizontally
ANT_Y       = SLAB_Y + 0.005        # 5 mm above surface
RX_OFFSET   = 0.020                  # 2 cm Tx–Rx offset
PML_CELLS   = 10

random.seed(42)
np.random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Geometry helpers ───────────────────────────────────────────────────────────

def make_input_file(sim_id: int, label: int, params: dict, tmpdir: Path) -> Path:
    """
    Write a gprMax .in file for one A-scan simulation.

    Domain layout (y increasing upward):
        y = 0             : bottom of concrete slab
        y = SLAB_Y        : concrete surface (air interface)
        y = ANT_Y         : antenna/receiver height (5 mm above surface)
        y = DOMAIN_Y      : top of domain

    Rebar: PEC cylinder at y = SLAB_Y - rebar_depth (measured from surface downward)
    Delamination: horizontal box spanning full x-width, placed between surface and rebar
    """
    eps_r        = params["epsilon_r"]
    rebar_depth  = params["rebar_depth"]       # metres from surface
    rebar_radius = params["rebar_radius"]       # metres
    rebar_y      = SLAB_Y - rebar_depth         # y-coordinate of rebar centre

    waveform_id = f"ricker{sim_id}"

    lines = [
        f"#title: GPR bridge deck sim {sim_id} class{label}",
        f"#domain: {DOMAIN_X:.4f} {DOMAIN_Y:.4f} {DZ:.4f}",
        f"#dx_dy_dz: {DX:.4f} {DX:.4f} {DZ:.4f}",
        f"#time_window: {TIME_WINDOW:.3e}",
        "",
        f"#material: {eps_r:.2f} 0.010 1.0 0.0 concrete",
        f"#material: 81.0 0.001 1.0 0.0 water_gap",
        "",
        f"#box: 0.000 0.000 0.000  {DOMAIN_X:.4f} {SLAB_Y:.4f} {DZ:.4f}  concrete",
        "",
    ]

    # Delamination layer (Class 2 only)
    if params.get("has_delam", False):
        d_depth  = params["delam_depth"]      # metres from surface to TOP of gap
        d_thick  = params["delam_thickness"]  # metres
        d_mat    = params["delam_material"]   # "air" or "water"

        # top and bottom of delamination layer (in domain y-coords)
        d_top    = SLAB_Y - d_depth
        d_bottom = d_top - d_thick
        # ensure the gap sits above the rebar (with 3 mm clearance)
        d_bottom = max(d_bottom, rebar_y + rebar_radius + 0.003)

        if d_bottom < d_top:
            mat_name = "free_space" if d_mat == "air" else "water_gap"
            lines += [
                f"#box: 0.000 {d_bottom:.4f} 0.000  "
                f"{DOMAIN_X:.4f} {d_top:.4f} {DZ:.4f}  {mat_name}",
                "",
            ]

    # Rebar — PEC cylinder running in z
    lines += [
        f"#cylinder: {ANT_X:.4f} {rebar_y:.4f} 0.000  "
        f"{ANT_X:.4f} {rebar_y:.4f} {DZ:.4f}  "
        f"{rebar_radius:.4f}  pec",
        "",
        f"#waveform: ricker 1 {FREQ:.3e} {waveform_id}",
        f"#hertzian_dipole: z {ANT_X:.4f} {ANT_Y:.4f} {DZ/2:.4f}  {waveform_id}",
        "",
        f"#rx: {ANT_X + RX_OFFSET:.4f} {ANT_Y:.4f} {DZ/2:.4f}",
    ]

    fpath = tmpdir / f"sim_{sim_id:06d}.in"
    fpath.write_text("\n".join(lines) + "\n")
    return fpath


def run_gprmax(infile: Path) -> Path:
    """Run gprMax via Python API and return the path to the .out HDF5 file."""
    from gprMax.gprMax import api as gprmax_api
    gprmax_api(str(infile), n=1, gpu=[0])
    outfile = infile.with_suffix(".out")
    if not outfile.exists():
        raise FileNotFoundError(f"gprMax output not found: {outfile}")
    return outfile


def extract_ascan(outfile: Path) -> np.ndarray:
    """
    Read the Ez component from gprMax HDF5 output.
    Resample to N_SAMPLES over TIME_WINDOW.
    Apply DC offset to match SDNET2021 uint16 format.
    """
    with h5py.File(outfile, "r") as f:
        ez = f["/rxs/rx1/Ez"][:]    # (n_timesteps,)
        dt = float(f.attrs["dt"])   # seconds per sample

    n_steps  = len(ez)
    t_orig   = np.arange(n_steps) * dt
    t_target = np.linspace(0.0, TIME_WINDOW, N_SAMPLES)

    # Interpolate / resample to 512 samples
    ez_rs = interp1d(t_orig, ez, kind="linear",
                     bounds_error=False, fill_value=0.0)(t_target)

    # Normalise to SDNET amplitude range (~±3000 counts around DC_OFFSET)
    peak = np.abs(ez_rs).max()
    if peak > 0:
        ez_rs = ez_rs / peak * 3000.0
    ez_rs = np.clip(ez_rs + DC_OFFSET, 0, 65535).astype(np.float32)
    return ez_rs


# ── Random parameter generators ────────────────────────────────────────────────

def class1_params() -> dict:
    """Sound concrete: rebar, no delamination."""
    return {
        "epsilon_r":    random.uniform(6.0, 10.0),
        "rebar_depth":  random.uniform(0.04, 0.08),
        "rebar_radius": random.choice([0.008, 0.010]),  # 16 mm or 20 mm diameter
        "has_delam":    False,
    }


def class2_params() -> dict:
    """Delaminated: same but with air or water gap between surface and rebar."""
    rebar_depth = random.uniform(0.04, 0.08)
    # delamination top depth: at least 1 cm from surface, at least 1 cm above rebar
    max_delam_depth = rebar_depth - 0.015
    if max_delam_depth < 0.010:
        max_delam_depth = 0.010
    delam_depth = random.uniform(0.010, max(0.011, max_delam_depth))
    return {
        "epsilon_r":        random.uniform(6.0, 10.0),
        "rebar_depth":      rebar_depth,
        "rebar_radius":     random.choice([0.008, 0.010]),
        "has_delam":        True,
        "delam_depth":      delam_depth,
        "delam_thickness":  random.uniform(0.002, 0.010),
        "delam_material":   random.choice(["air", "water"]),
    }


# ── Main generation loop ───────────────────────────────────────────────────────

def generate(n_signals: int, label: int, param_fn, tmpdir: Path) -> list[np.ndarray]:
    signals   = []
    n_failed  = 0
    t_start   = time.time()

    for i in range(n_signals):
        sim_id = i
        params = param_fn()
        try:
            infile  = make_input_file(sim_id, label, params, tmpdir)
            outfile = run_gprmax(infile)
            ascan   = extract_ascan(outfile)
            signals.append(ascan)
        except Exception as e:
            n_failed += 1
            if n_failed <= 20:
                print(f"  WARNING sim {sim_id}: {e}", flush=True)
        finally:
            for f in [infile, outfile if 'outfile' in dir() else None]:
                if f and Path(f).exists():
                    Path(f).unlink(missing_ok=True)

        if (i + 1) % 1000 == 0:
            elapsed  = time.time() - t_start
            rate     = (i + 1) / elapsed
            eta_min  = (n_signals - i - 1) / rate / 60
            print(f"  [{i+1:>6,}/{n_signals:,}]  "
                  f"ok={len(signals):,}  fail={n_failed}  "
                  f"rate={rate:.1f}/s  ETA={eta_min:.0f} min", flush=True)

    elapsed = time.time() - t_start
    print(f"  Done: {len(signals):,} signals in {elapsed/60:.1f} min "
          f"({n_failed} failed)", flush=True)
    return signals


def save_csv(signals: list, label: int, outpath: Path) -> None:
    arr  = np.array(signals)                                     # (n, 512)
    cols = [f"time_{i}" for i in range(N_SAMPLES)] + ["class"]
    labs = np.full(len(signals), label, dtype=int)
    df   = pd.DataFrame(np.column_stack([arr, labs]), columns=cols)
    df.to_csv(outpath, index=False)
    print(f"  Saved {len(signals):,} signals → {outpath}", flush=True)


# ── Run ────────────────────────────────────────────────────────────────────────

print("=" * 60, flush=True)
print(f"GPR Synthetic Data Generation", flush=True)
print(f"  Class 1 target : {N_CLASS1:,}", flush=True)
print(f"  Class 2 target : {N_CLASS2:,}", flush=True)
print(f"  Samples / scan : {N_SAMPLES}  ({TIME_WINDOW*1e9:.0f} ns)", flush=True)
print(f"  Antenna freq   : {FREQ/1e9:.1f} GHz", flush=True)
print(f"  Domain         : {DOMAIN_X*100:.0f} cm × {DOMAIN_Y*100:.0f} cm  @  {DX*1000:.0f} mm", flush=True)
print("=" * 60, flush=True)

# Quick smoke-test with one simulation before committing to 50k
print("\nSmoke test (1 simulation)...", flush=True)
tmpdir = Path(tempfile.mkdtemp(prefix="gpr_synth_"))
try:
    test_params = class1_params()
    test_in     = make_input_file(0, 1, test_params, tmpdir)
    test_out    = run_gprmax(test_in)
    test_ascan  = extract_ascan(test_out)
    print(f"  OK — ascan shape={test_ascan.shape}  "
          f"min={test_ascan.min():.0f}  max={test_ascan.max():.0f}", flush=True)
    for f in [test_in, test_out]:
        f.unlink(missing_ok=True)
except Exception as e:
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"  FAILED: {e}", flush=True)
    print("  Fix the error above before running the full generation.", flush=True)
    sys.exit(1)

# ── Class 1 ───────────────────────────────────────────────────────────────────
print(f"\nGenerating {N_CLASS1:,} Class-1 (sound) signals...", flush=True)
class1_signals = generate(N_CLASS1, label=1, param_fn=class1_params, tmpdir=tmpdir)

# ── Class 2 ───────────────────────────────────────────────────────────────────
print(f"\nGenerating {N_CLASS2:,} Class-2 (delaminated) signals...", flush=True)
class2_signals = generate(N_CLASS2, label=2, param_fn=class2_params, tmpdir=tmpdir)

shutil.rmtree(tmpdir, ignore_errors=True)

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nSaving CSV files...", flush=True)
save_csv(class1_signals, 1, OUTPUT_DIR / "synthetic_class1.csv")
save_csv(class2_signals, 2, OUTPUT_DIR / "synthetic_class2.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("GENERATION COMPLETE", flush=True)
print(f"  Class 1 (sound):       {len(class1_signals):,}", flush=True)
print(f"  Class 2 (delaminated): {len(class2_signals):,}", flush=True)
print(f"  Total:                 {len(class1_signals)+len(class2_signals):,}", flush=True)
print(f"  Output dir:            {OUTPUT_DIR}", flush=True)

for label, signals, name in [(1, class1_signals, "sound"),
                               (2, class2_signals, "delaminated")]:
    print(f"\n  Sample signals — Class {label} ({name}):", flush=True)
    for j in range(min(3, len(signals))):
        s = signals[j]
        print(f"    [{j}]  min={s.min():.0f}  max={s.max():.0f}  "
              f"mean={s.mean():.0f}  p2p={s.max()-s.min():.0f}", flush=True)
