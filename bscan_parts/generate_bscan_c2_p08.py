"""
generate_bscan_c2_p08.py
Part 8/10 — Class 2 (delaminated)
Sims 219–250  (31 simulations × 64 A-scans = up to 1,984 signals)
Run on Kaggle CPU.
Output: /kaggle/working/synthetic_bscan_c2_p08.csv
"""

SIM_START   = 219
SIM_END     = 250
PART_NUM    = 8
CLASS_LABEL = 2

import subprocess
import sys

# Force colorama reinstall (required for gprMax on Kaggle)
subprocess.run([sys.executable, '-m', 'pip', 'install',
               '--force-reinstall', 'colorama'], check=True)

import os
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages/gprMax')

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
    sys.exit(1)

# ── Fixed configuration ───────────────────────────────────────────────────────
OUTPUT_DIR    = Path("/kaggle/working")
N_ASCAN       = 64
ASCAN_STEP    = 0.001

N_SAMPLES     = 512
TIME_WINDOW   = 12e-9
DC_OFFSET     = 32_768
FREQ          = 1.5e9

DX            = 0.001
DZ            = 0.002
SLAB_Y        = 0.150
AIR_Y         = 0.040
DOMAIN_X      = 0.100
DOMAIN_Y      = SLAB_Y + AIR_Y

ANT_X_START   = 0.010
ANT_X_END     = ANT_X_START + (N_ASCAN - 1) * ASCAN_STEP
ANT_Y         = SLAB_Y + 0.005
RX_OFFSET     = 0.020

PROGRESS_EVERY = 10

random.seed(42)
np.random.seed(42)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Input file builder ────────────────────────────────────────────────────────

def make_bscan_input(sim_id: int, label: int, params: dict, tmpdir: Path) -> Path:
    eps_r        = params["epsilon_r"]
    rebar_depth  = params["rebar_depth"]
    rebar_radius = params["rebar_radius"]
    rebar_x      = params["rebar_x"]
    rebar_y      = SLAB_Y - rebar_depth

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
        f"#box: 0.000 0.000 0.000  {DOMAIN_X:.4f} {SLAB_Y:.4f} {DZ:.4f}  concrete",
        "",
    ]

    if params.get("has_delam", False):
        d_depth = params["delam_depth"]
        d_thick = params["delam_thickness"]
        d_mat   = params["delam_material"]

        d_top    = SLAB_Y - d_depth
        d_bottom = d_top - d_thick
        d_bottom = max(d_bottom, rebar_y + rebar_radius + 0.003)

        if d_bottom < d_top:
            mat_name = "free_space" if d_mat == "air" else "water_gap"
            lines += [
                f"#box: 0.000 {d_bottom:.4f} 0.000  "
                f"{DOMAIN_X:.4f} {d_top:.4f} {DZ:.4f}  {mat_name}",
                "",
            ]

    lines += [
        f"#cylinder: {rebar_x:.4f} {rebar_y:.4f} 0.000  "
        f"{rebar_x:.4f} {rebar_y:.4f} {DZ:.4f}  "
        f"{rebar_radius:.4f}  pec",
        "",
    ]

    tx_x = ANT_X_START
    rx_x = ANT_X_START + RX_OFFSET
    lines += [
        f"#waveform: ricker 1 {FREQ:.3e} {waveform_id}",
        f"#hertzian_dipole: z {tx_x:.4f} {ANT_Y:.4f} {DZ/2:.4f}  {waveform_id}",
        f"#rx: {rx_x:.4f} {ANT_Y:.4f} {DZ/2:.4f}",
        "",
        f"#src_steps: {ASCAN_STEP:.4f} 0.000 0.000",
        f"#rx_steps:  {ASCAN_STEP:.4f} 0.000 0.000",
    ]

    fpath = tmpdir / f"bscan_{sim_id:06d}.in"
    fpath.write_text("\n".join(lines) + "\n")
    return fpath


# ── gprMax runner ─────────────────────────────────────────────────────────────

def run_bscan(infile: Path) -> None:
    from gprMax.gprMax import api as gprmax_api
    gprmax_api(str(infile), n=N_ASCAN)


# ── Output extractor ──────────────────────────────────────────────────────────

def extract_bscan(infile: Path) -> list:
    traces = []
    dt = None

    for i in range(1, N_ASCAN + 1):
        out_file = infile.parent / f"{infile.stem}{i}.out"
        with h5py.File(out_file, "r") as f:
            ez = f["/rxs/rx1/Ez"][:]
            if dt is None:
                dt = float(f.attrs["dt"])
        traces.append(ez)

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

        peak = np.abs(trace_rs).max()
        if peak > 0:
            trace_rs = trace_rs / peak * 3000.0
        trace_rs = np.clip(trace_rs + DC_OFFSET, 0, 65535).astype(np.float32)
        ascans.append(trace_rs)

    return ascans


# ── Parameter generators ──────────────────────────────────────────────────────

def _rebar_x() -> float:
    margin = 0.005
    return random.uniform(ANT_X_START + margin, ANT_X_END - margin)


def class1_params() -> dict:
    return {
        "epsilon_r":    random.uniform(6.0, 10.0),
        "rebar_depth":  random.uniform(0.04, 0.08),
        "rebar_radius": random.choice([0.008, 0.010]),
        "rebar_x":      _rebar_x(),
        "has_delam":    False,
    }


def class2_params() -> dict:
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


# ── Generation loop ───────────────────────────────────────────────────────────

def generate_bscan(n_sims: int, label: int, param_fn, tmpdir: Path,
                   sim_id_offset: int = 0) -> list:
    all_ascans = []
    n_failed   = 0
    t_start    = time.time()

    for i in range(n_sims):
        sim_id = sim_id_offset + i
        params = param_fn()
        infile = None
        try:
            infile = make_bscan_input(sim_id, label, params, tmpdir)
            run_bscan(infile)
            ascans = extract_bscan(infile)
            all_ascans.extend(ascans)
        except Exception as e:
            n_failed += 1
            if n_failed <= 20:
                print(f"  WARNING sim {sim_id}: {e}", flush=True)
        finally:
            if infile and infile.exists():
                infile.unlink(missing_ok=True)
            if infile:
                for j in range(1, N_ASCAN + 1):
                    out_j = infile.parent / f"{infile.stem}{j}.out"
                    out_j.unlink(missing_ok=True)

        completed = i + 1
        if completed % PROGRESS_EVERY == 0 or completed == n_sims:
            elapsed  = time.time() - t_start
            rate     = completed / elapsed
            eta_min  = (n_sims - completed) / rate / 60 if rate > 0 else 0
            print(
                f"  [{completed:>4}/{n_sims}]  "
                f"A-scans={len(all_ascans):,}  "
                f"failed={n_failed}  "
                f"rate={rate:.2f} sim/s  "
                f"ETA={eta_min:.0f} min",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"  Done: {len(all_ascans):,} A-scans from {n_sims} sims "
        f"({n_failed} failed) in {elapsed/60:.1f} min",
        flush=True,
    )
    return all_ascans


# ── CSV saver ─────────────────────────────────────────────────────────────────

def save_csv(signals: list, label: int, outpath: Path) -> None:
    arr  = np.array(signals, dtype=np.int32)             # (n, 512)
    labs = np.full(len(signals), label, dtype=np.int32)
    data = np.column_stack([arr, labs])                  # (n, 513)
    np.savetxt(outpath, data, delimiter=",", fmt="%d")
    print(f"  Saved {len(signals):,} signals → {outpath}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

print("=" * 64, flush=True)
print(f"Part {PART_NUM} | Class {CLASS_LABEL} | Sims {SIM_START}-{SIM_END}", flush=True)
print(f"  Simulations : {SIM_END - SIM_START} × {N_ASCAN} A-scans = up to {(SIM_END - SIM_START) * N_ASCAN:,} signals", flush=True)
out_fname = f"synthetic_bscan_c{CLASS_LABEL}_p{PART_NUM:02d}.csv"
print(f"  Output      : {OUTPUT_DIR / out_fname}", flush=True)
print("=" * 64, flush=True)

_tmpdir = Path(tempfile.mkdtemp(prefix="gpr_bscan_"))
try:
    n_sims  = SIM_END - SIM_START
    signals = generate_bscan(n_sims, label=CLASS_LABEL, param_fn=class2_params,
                             tmpdir=_tmpdir, sim_id_offset=SIM_START)

    outpath = OUTPUT_DIR / f"synthetic_bscan_c{CLASS_LABEL}_p{PART_NUM:02d}.csv"
    save_csv(signals, CLASS_LABEL, outpath)

    print("\n" + "=" * 64, flush=True)
    print(f"  Part {PART_NUM} complete: {len(signals):,} signals saved → {outpath}", flush=True)
    print("=" * 64, flush=True)
finally:
    shutil.rmtree(_tmpdir, ignore_errors=True)
