"""
generate_synthetic_bscan.py
Generates ~50,112 labeled synthetic GPR A-scan signals using gprMax FDTD
B-scan simulation.  Each simulation sweeps the antenna 64 steps × 1 mm,
producing 64 A-scans from a single FDTD run — ~783 total simulations
instead of ~50,000 one-off runs.

Targets
-------
  Class 1 (sound)      : 470 sims × 64 A-scans = 30,080 signals
  Class 2 (delaminated): 313 sims × 64 A-scans = 20,032 signals

Domain : 10 cm × 19 cm × 2 mm  @  1 mm resolution
Antenna: Tx starts at x = 0.010 m, steps +1 mm per iteration (64 steps)
         Rx offset = 20 mm (fixed Tx–Rx separation)

Output format matches SDNET2021: 512 samples over 12 ns, DC offset = 32,768.

Usage (Colab / Linux)
---------------------
    python generate_synthetic_bscan.py
"""

import subprocess
import sys

# Force colorama reinstall (required for gprMax on Colab)
subprocess.run([sys.executable, '-m', 'pip', 'install',
               '--force-reinstall', 'colorama'], check=True)

import os
sys.path.insert(0, '/content/gprMax')

import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import interp1d

# ── gprMax availability check ─────────────────────────────────────────────────
try:
    import gprMax
    print(f"gprMax imported from: {gprMax.__file__}", flush=True)
except ImportError as e:
    print(f"ERROR: could not import gprMax: {e}", flush=True)
    print(f"  sys.path = {sys.path}", flush=True)
    print("  Make sure gprMax is installed or cloned to /content/gprMax", flush=True)
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path("/kaggle/working/synthetic_data")
N_SIM_CLASS1  = 470          # 470 × 64 = 30,080  Class-1 A-scans
N_SIM_CLASS2  = 313          # 313 × 64 = 20,032  Class-2 A-scans
N_ASCAN       = 64           # A-scans per B-scan simulation
ASCAN_STEP    = 0.001        # 1 mm antenna step in x

N_SAMPLES     = 512
TIME_WINDOW   = 12e-9        # 12 ns
DC_OFFSET     = 32_768
FREQ          = 1.5e9        # 1.5 GHz centre frequency

# FDTD domain: 10 cm × 19 cm × 2 mm  @  1 mm resolution
DX            = 0.001        # 1 mm spatial resolution
DZ            = 0.002        # 2 mm z-thickness  (quasi-2-D)
SLAB_Y        = 0.150        # 15 cm concrete slab
AIR_Y         = 0.040        # 4 cm air above surface
DOMAIN_X      = 0.100        # 10 cm
DOMAIN_Y      = SLAB_Y + AIR_Y  # 19 cm

# Antenna geometry
ANT_X_START   = 0.010        # Tx x-position for first A-scan
ANT_X_END     = ANT_X_START + (N_ASCAN - 1) * ASCAN_STEP   # 0.073 m
ANT_Y         = SLAB_Y + 0.005   # 5 mm above concrete surface
RX_OFFSET     = 0.020            # 2 cm Tx–Rx separation

PROGRESS_EVERY = 50          # print ETA every N simulations

random.seed(42)
np.random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Input file builder ────────────────────────────────────────────────────────

def make_bscan_input(sim_id: int, label: int, params: dict, tmpdir: Path) -> Path:
    """
    Write a gprMax .in file for a single B-scan (64 A-scans).

    Domain layout (y-axis, increasing upward):
        y = 0          : bottom of concrete slab
        y = SLAB_Y     : concrete surface (air interface)
        y = ANT_Y      : antenna height (5 mm above surface)
        y = DOMAIN_Y   : top of simulation domain

    Rebar: PEC cylinder running the full z-extent at (rebar_x, rebar_y).
    Delamination (Class 2): horizontal box spanning full x-width, placed
        between the surface and the rebar.

    B-scan directives:
        #src_steps / #rx_steps advance the antenna 1 mm in x per iteration.
        The API is called with n=64 to execute all 64 A-scans.
    """
    eps_r        = params["epsilon_r"]
    rebar_depth  = params["rebar_depth"]      # metres from surface downward
    rebar_radius = params["rebar_radius"]      # metres
    rebar_x      = params["rebar_x"]           # x-position of rebar centre
    rebar_y      = SLAB_Y - rebar_depth        # y-coordinate of rebar centre

    waveform_id  = f"ricker{sim_id}"

    lines = [
        f"#title: GPR B-scan sim {sim_id} class{label}",
        f"#domain: {DOMAIN_X:.4f} {DOMAIN_Y:.4f} {DZ:.4f}",
        f"#dx_dy_dz: {DX:.4f} {DX:.4f} {DZ:.4f}",
        f"#time_window: {TIME_WINDOW:.3e}",
        "",
        f"#material: {eps_r:.2f} 0.010 1.0 0.0 concrete",
        f"#material: 81.0  0.001 1.0 0.0 water_gap",
        "",
        # Full concrete slab
        f"#box: 0.000 0.000 0.000  {DOMAIN_X:.4f} {SLAB_Y:.4f} {DZ:.4f}  concrete",
        "",
    ]

    # Delamination layer — Class 2 only
    if params.get("has_delam", False):
        d_depth = params["delam_depth"]        # metres from surface to top of gap
        d_thick = params["delam_thickness"]    # metres
        d_mat   = params["delam_material"]     # "air" or "water"

        d_top    = SLAB_Y - d_depth
        d_bottom = d_top - d_thick
        # Keep gap above rebar with 3 mm clearance
        d_bottom = max(d_bottom, rebar_y + rebar_radius + 0.003)

        if d_bottom < d_top:
            mat_name = "free_space" if d_mat == "air" else "water_gap"
            lines += [
                f"#box: 0.000 {d_bottom:.4f} 0.000  "
                f"{DOMAIN_X:.4f} {d_top:.4f} {DZ:.4f}  {mat_name}",
                "",
            ]

    # Rebar — PEC cylinder running full z-extent
    lines += [
        f"#cylinder: {rebar_x:.4f} {rebar_y:.4f} 0.000  "
        f"{rebar_x:.4f} {rebar_y:.4f} {DZ:.4f}  "
        f"{rebar_radius:.4f}  pec",
        "",
    ]

    # Source and receiver — Tx leads Rx by RX_OFFSET in x
    tx_x = ANT_X_START
    rx_x = ANT_X_START + RX_OFFSET
    lines += [
        f"#waveform: ricker 1 {FREQ:.3e} {waveform_id}",
        f"#hertzian_dipole: z {tx_x:.4f} {ANT_Y:.4f} {DZ/2:.4f}  {waveform_id}",
        f"#rx: {rx_x:.4f} {ANT_Y:.4f} {DZ/2:.4f}",
        "",
        # B-scan stepping directives: move Tx and Rx together by 1 mm each iteration
        f"#src_steps: {ASCAN_STEP:.4f} 0.000 0.000",
        f"#rx_steps:  {ASCAN_STEP:.4f} 0.000 0.000",
    ]

    fpath = tmpdir / f"bscan_{sim_id:06d}.in"
    fpath.write_text("\n".join(lines) + "\n")
    return fpath


# ── gprMax runner and output extractor ───────────────────────────────────────

def run_bscan(infile: Path) -> None:
    """
    Execute the B-scan via gprMax Python API (n=N_ASCAN iterations).

    gprMax writes one output file per iteration:
        {infile.stem}1.out, {infile.stem}2.out, …, {infile.stem}{N_ASCAN}.out
    No merging is performed — individual files are read directly by
    extract_bscan().
    """
    from gprMax.gprMax import api as gprmax_api
    gprmax_api(str(infile), n=N_ASCAN)


def extract_bscan(infile: Path) -> list[np.ndarray]:
    """
    Read Ez from each individual gprMax per-iteration .out file.

    gprMax names them {infile.stem}1.out … {infile.stem}{N_ASCAN}.out.
    Each file contains /rxs/rx1/Ez with shape (n_timesteps,).
    All traces are stacked into (n_timesteps, N_ASCAN), then each column
    is resampled to N_SAMPLES over TIME_WINDOW and normalised to the
    SDNET2021 amplitude range with DC_OFFSET.

    Returns a list of N_ASCAN float32 arrays each of length N_SAMPLES.
    """
    traces = []
    dt = None

    for i in range(1, N_ASCAN + 1):
        out_file = infile.parent / f"{infile.stem}{i}.out"
        with h5py.File(out_file, "r") as f:
            ez = f["/rxs/rx1/Ez"][:]        # shape (n_timesteps,)
            if dt is None:
                dt = float(f.attrs["dt"])
        traces.append(ez)

    # Stack into (n_timesteps, N_ASCAN)
    ez2d = np.column_stack(traces)

    n_timesteps = ez2d.shape[0]
    t_orig   = np.arange(n_timesteps) * dt
    t_target = np.linspace(0.0, TIME_WINDOW, N_SAMPLES)

    ascans = []
    for col in range(N_ASCAN):
        trace_rs = interp1d(
            t_orig, ez2d[:, col], kind="linear",
            bounds_error=False, fill_value=0.0
        )(t_target)

        # Normalise to ~±3 000 counts around DC_OFFSET
        peak = np.abs(trace_rs).max()
        if peak > 0:
            trace_rs = trace_rs / peak * 3000.0
        trace_rs = np.clip(trace_rs + DC_OFFSET, 0, 65535).astype(np.float32)
        ascans.append(trace_rs)

    return ascans


# ── Random parameter generators ───────────────────────────────────────────────

def _rebar_x() -> float:
    """Randomise rebar x so the hyperbola appears across the B-scan."""
    # Keep rebar within the scanned window with at least 5 mm margin each side
    margin = 0.005
    return random.uniform(ANT_X_START + margin, ANT_X_END - margin)


def class1_params() -> dict:
    """Sound concrete: rebar only, no delamination."""
    return {
        "epsilon_r":    random.uniform(6.0, 10.0),
        "rebar_depth":  random.uniform(0.04, 0.08),
        "rebar_radius": random.choice([0.008, 0.010]),  # 16 mm or 20 mm ⌀
        "rebar_x":      _rebar_x(),
        "has_delam":    False,
    }


def class2_params() -> dict:
    """Delaminated: rebar plus air or water gap between surface and rebar."""
    rebar_depth = random.uniform(0.04, 0.08)
    max_d_depth = rebar_depth - 0.015
    if max_d_depth < 0.010:
        max_d_depth = 0.010
    delam_depth = random.uniform(0.010, max(0.011, max_d_depth))
    return {
        "epsilon_r":        random.uniform(6.0, 10.0),
        "rebar_depth":      rebar_depth,
        "rebar_radius":     random.choice([0.008, 0.010]),
        "rebar_x":          _rebar_x(),
        "has_delam":        True,
        "delam_depth":      delam_depth,
        "delam_thickness":  random.uniform(0.002, 0.010),
        "delam_material":   random.choice(["air", "water"]),
    }


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_bscan(n_sims: int, label: int, param_fn, tmpdir: Path) -> list[np.ndarray]:
    """
    Run `n_sims` B-scan simulations and return all extracted A-scans.

    Each simulation yields up to N_ASCAN A-scans; failed simulations are
    skipped and reported.  Progress with ETA is printed every PROGRESS_EVERY
    simulations.
    """
    all_ascans = []
    n_failed   = 0
    t_start    = time.time()

    for i in range(n_sims):
        params = param_fn()
        infile = None
        try:
            infile = make_bscan_input(i, label, params, tmpdir)
            run_bscan(infile)
            ascans = extract_bscan(infile)
            all_ascans.extend(ascans)
        except Exception as e:
            n_failed += 1
            if n_failed <= 20:
                print(f"  WARNING sim {i}: {e}", flush=True)
        finally:
            # Remove .in file and all per-iteration .out files
            if infile and infile.exists():
                infile.unlink(missing_ok=True)
            if infile:
                for j in range(1, N_ASCAN + 1):
                    out_j = infile.parent / f"{infile.stem}{j}.out"
                    out_j.unlink(missing_ok=True)

        completed = i + 1
        if completed % PROGRESS_EVERY == 0 or completed == n_sims:
            elapsed  = time.time() - t_start
            rate     = completed / elapsed          # sims / s
            eta_min  = (n_sims - completed) / rate / 60 if rate > 0 else 0
            print(
                f"  [{completed:>4}/{n_sims}]  "
                f"A-scans collected={len(all_ascans):,}  "
                f"sims_failed={n_failed}  "
                f"rate={rate:.2f} sim/s  "
                f"ETA={eta_min:.0f} min",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"  Finished: {len(all_ascans):,} A-scans from {n_sims} sims "
        f"({n_failed} failed) in {elapsed/60:.1f} min",
        flush=True,
    )
    return all_ascans


def save_csv(signals: list, label: int, outpath: Path) -> None:
    arr  = np.array(signals)                            # (n, 512)
    cols = [f"time_{i}" for i in range(N_SAMPLES)] + ["class"]
    labs = np.full(len(signals), label, dtype=int)
    df   = pd.DataFrame(np.column_stack([arr, labs]), columns=cols)
    df.to_csv(outpath, index=False)
    print(f"  Saved {len(signals):,} signals → {outpath}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

print("=" * 64, flush=True)
print("GPR Synthetic B-scan Data Generation", flush=True)
print(f"  Class 1: {N_SIM_CLASS1} sims × {N_ASCAN} A-scans = {N_SIM_CLASS1*N_ASCAN:,} signals", flush=True)
print(f"  Class 2: {N_SIM_CLASS2} sims × {N_ASCAN} A-scans = {N_SIM_CLASS2*N_ASCAN:,} signals", flush=True)
print(f"  Samples / A-scan : {N_SAMPLES}  ({TIME_WINDOW*1e9:.0f} ns)", flush=True)
print(f"  Centre frequency : {FREQ/1e9:.1f} GHz", flush=True)
print(f"  Domain           : {DOMAIN_X*100:.0f} cm × {DOMAIN_Y*100:.0f} cm × {DZ*1000:.0f} mm", flush=True)
print(f"  Antenna sweep    : x = {ANT_X_START:.3f} → {ANT_X_END:.3f} m  ({N_ASCAN} steps × {ASCAN_STEP*1000:.0f} mm)", flush=True)
print(f"  Output dir       : {OUTPUT_DIR}", flush=True)
print("=" * 64, flush=True)

# ── Smoke test ────────────────────────────────────────────────────────────────
print("\nSmoke test (1 B-scan simulation)...", flush=True)
_tmpdir = Path(tempfile.mkdtemp(prefix="gpr_bscan_"))
try:
    _p   = class1_params()
    _inf = make_bscan_input(0, 1, _p, _tmpdir)
    run_bscan(_inf)
    _sc  = extract_bscan(_inf)
    print(
        f"  OK — {len(_sc)} A-scans extracted  "
        f"shape={_sc[0].shape}  "
        f"min={_sc[0].min():.0f}  max={_sc[0].max():.0f}",
        flush=True,
    )
    _inf.unlink(missing_ok=True)
    for _j in range(1, N_ASCAN + 1):
        (_inf.parent / f"{_inf.stem}{_j}.out").unlink(missing_ok=True)
except Exception as _e:
    shutil.rmtree(_tmpdir, ignore_errors=True)
    print(f"  FAILED: {_e}", flush=True)
    print("  Fix the error above before running the full generation.", flush=True)
    sys.exit(1)

# ── Class 1 ───────────────────────────────────────────────────────────────────
print(f"\nGenerating Class-1 (sound) — {N_SIM_CLASS1} B-scan sims...", flush=True)
class1_signals = generate_bscan(N_SIM_CLASS1, label=1, param_fn=class1_params, tmpdir=_tmpdir)

# ── Class 2 ───────────────────────────────────────────────────────────────────
print(f"\nGenerating Class-2 (delaminated) — {N_SIM_CLASS2} B-scan sims...", flush=True)
class2_signals = generate_bscan(N_SIM_CLASS2, label=2, param_fn=class2_params, tmpdir=_tmpdir)

shutil.rmtree(_tmpdir, ignore_errors=True)

# ── Save CSVs ─────────────────────────────────────────────────────────────────
print("\nSaving CSV files...", flush=True)
save_csv(class1_signals, 1, OUTPUT_DIR / "synthetic_bscan_class1.csv")
save_csv(class2_signals, 2, OUTPUT_DIR / "synthetic_bscan_class2.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(class1_signals) + len(class2_signals)
print("\n" + "=" * 64, flush=True)
print("GENERATION COMPLETE", flush=True)
print(f"  Class 1 (sound)      : {len(class1_signals):,}", flush=True)
print(f"  Class 2 (delaminated): {len(class2_signals):,}", flush=True)
print(f"  Total                : {total:,}", flush=True)
print(f"  Output dir           : {OUTPUT_DIR}", flush=True)

for lbl, sigs, name in [(1, class1_signals, "sound"), (2, class2_signals, "delaminated")]:
    print(f"\n  Sample A-scans — Class {lbl} ({name}):", flush=True)
    for j in range(min(3, len(sigs))):
        s = sigs[j]
        print(
            f"    [{j}]  min={s.min():.0f}  max={s.max():.0f}  "
            f"mean={s.mean():.0f}  p2p={s.max()-s.min():.0f}",
            flush=True,
        )
