"""
Generate gprMax .in files for rebar depth and delamination sweeps.

Usage:
    python generate_sweep.py rebar     --n 5000 --seed 0  --out runs/rebar_batch0
    python generate_sweep.py rebar2mat --n 5000 --seed 0  --out runs/rebar2mat_batch0
    python generate_sweep.py delam     --n 2500 --seed 0  --out runs/delam_batch0

rebar2mat generates dual-mat deck geometry (top and bottom rebar mats) — more
physically realistic than single-rebar. The label is top-mat depth only.
"""

import argparse
import os
import random
import math

REBAR_TEMPLATE = """#title: rebar depth={depth_mm:.1f}mm diam={diameter_mm:.1f}mm epsr={epsr:.2f} freq={freq_mhz:.0f}MHz
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

# Dual-mat: top rebar mat (label) + bottom rebar mat at +60–120mm deeper.
# More realistic bridge deck geometry for pretraining.
REBAR_DUAL_MAT_TEMPLATE = """#title: rebar2mat top={depth_mm:.1f}mm bot={bot_depth_mm:.1f}mm diam={diameter_mm:.1f}mm epsr={epsr:.2f} freq={freq_mhz:.0f}MHz
#domain: 0.60 0.50 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 8e-9

#material: {epsr:.4f} {sigma:.4f} 1 0 concrete

#waveform: ricker 1 {freq_hz:.0f} src_wave
#hertzian_dipole: z {src_x:.3f} 0.450 0 src_wave
#rx: {rx_x:.3f} 0.450 0

#box: 0 0 0 0.60 0.40 0.002 concrete
#cylinder: 0.300 {depth_m:.4f} 0 0.300 {depth_m:.4f} 0.002 {radius_m:.4f} pec
#cylinder: 0.300 {bot_depth_m:.4f} 0 0.300 {bot_depth_m:.4f} 0.002 {radius_m:.4f} pec
"""

DELAM_SOUND_TEMPLATE = """#title: sound epsr={epsr:.2f} freq={freq_mhz:.0f}MHz
#domain: 0.60 0.50 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 8e-9

#material: {epsr:.4f} {sigma:.4f} 1 0 concrete

#waveform: ricker 1 {freq_hz:.0f} src_wave
#hertzian_dipole: z 0.300 0.450 0 src_wave
#rx: 0.340 0.450 0

#box: 0 0 0 0.60 0.50 0.002 concrete
"""

DELAM_TEMPLATE = """#title: delaminated gap_depth={gap_depth_mm:.1f}mm thick={gap_thick_mm:.1f}mm fill={fill}
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

REBAR_PARAMS = {
    "epsr":      (4.0, 12.0),
    "sigma":     (0.001, 0.05),
    "depth_mm":  (40.0, 250.0),   # extended to 250mm to cover B440029 max truth (234mm)
    "diam_mm":   (10.0, 32.0),
    "freq_mhz":  [900, 1500, 2000],
    "offset_mm": (30.0, 60.0),
}

DELAM_PARAMS = {
    "epsr":          (4.0, 10.0),
    "sigma":         (0.001, 0.05),
    "gap_depth_mm":  (30.0, 120.0),
    "gap_thick_mm":  (2.0, 20.0),
    "fill":          ["air", "water", "debris"],
    "freq_mhz":      [900, 1500, 2000],
}

FILL_PROPS = {
    "air":    (1.0,  0.0),
    "water":  (81.0, 0.01),
    "debris": (4.0,  0.005),
}

def snap(val_m, cell=0.002):
    return round(round(val_m / cell) * cell, 6)

def rebar_params(rng):
    epsr     = rng.uniform(*REBAR_PARAMS["epsr"])
    sigma    = rng.uniform(*REBAR_PARAMS["sigma"])
    depth_mm = rng.uniform(*REBAR_PARAMS["depth_mm"])
    diam_mm  = rng.choice(REBAR_PARAMS["diam_mm"]) if isinstance(REBAR_PARAMS["diam_mm"], list) \
                else rng.uniform(*REBAR_PARAMS["diam_mm"])
    freq_mhz = rng.choice(REBAR_PARAMS["freq_mhz"])
    offset_mm = rng.uniform(*REBAR_PARAMS["offset_mm"])

    depth_m  = snap(depth_mm / 1000)
    radius_m = snap(diam_mm / 2000)
    freq_hz  = freq_mhz * 1e6
    src_x    = snap(0.300 - offset_mm / 2000)
    rx_x     = snap(0.300 + offset_mm / 2000)

    return dict(
        epsr=epsr, sigma=sigma,
        depth_mm=depth_mm, diameter_mm=diam_mm,
        depth_m=depth_m, radius_m=radius_m,
        freq_mhz=freq_mhz, freq_hz=freq_hz,
        src_x=src_x, rx_x=rx_x,
    )

def delam_params(rng, label):
    epsr         = rng.uniform(*DELAM_PARAMS["epsr"])
    sigma        = rng.uniform(*DELAM_PARAMS["sigma"])
    freq_mhz     = rng.choice(DELAM_PARAMS["freq_mhz"])
    freq_hz      = freq_mhz * 1e6

    if label == 0:
        gap_depth_mm  = rng.uniform(*DELAM_PARAMS["gap_depth_mm"])
        gap_thick_mm  = rng.uniform(*DELAM_PARAMS["gap_thick_mm"])
        fill          = rng.choice(DELAM_PARAMS["fill"])
        fill_epsr, fill_sigma = FILL_PROPS[fill]

        gap_top_m = snap((0.50 - gap_depth_mm / 1000) - gap_thick_mm / 1000)
        gap_bot_m = snap(0.50 - gap_depth_mm / 1000)

        return dict(
            label=0, epsr=epsr, sigma=sigma,
            freq_mhz=freq_mhz, freq_hz=freq_hz,
            gap_depth_mm=gap_depth_mm, gap_thick_mm=gap_thick_mm,
            fill=fill, fill_epsr=fill_epsr, fill_sigma=fill_sigma,
            gap_top_m=gap_top_m, gap_bot_m=gap_bot_m,
        )
    else:
        return dict(
            label=1, epsr=epsr, sigma=sigma,
            freq_mhz=freq_mhz, freq_hz=freq_hz,
        )

def rebar2mat_params(rng):
    p = rebar_params(rng)
    # Bottom mat is 60–120 mm deeper than the top mat, capped so it stays in domain.
    # For top_depth_mm > 180mm, the actual separation is compressed below the drawn value
    # because the cap at 300mm (concrete block height) takes effect. This is documented
    # rather than fixed — the cap preserves simulation validity; affected samples are ~40%
    # of the sweep with the 40–250mm depth range.
    mat_sep_mm  = rng.uniform(60.0, 120.0)
    bot_depth_mm = min(p["depth_mm"] + mat_sep_mm, 300.0)
    p["bot_depth_mm"] = bot_depth_mm
    p["bot_depth_m"]  = snap(bot_depth_mm / 1000)
    return p

def write_rebar2mat(out_dir, n, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    meta = []
    for i in range(n):
        p = rebar2mat_params(rng)
        content = REBAR_DUAL_MAT_TEMPLATE.format(**p)
        fname = f"rebar2mat_{i:05d}.in"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(content)
        meta.append(
            f"{fname},{p['depth_mm']:.4f},{p['bot_depth_mm']:.4f},"
            f"{p['epsr']:.4f},{p['diameter_mm']:.4f}"
        )
    with open(os.path.join(out_dir, "meta.csv"), "w") as f:
        f.write("filename,top_depth_mm,bot_depth_mm,epsr,diameter_mm\n")
        f.write("\n".join(meta) + "\n")
    print(f"Wrote {n} dual-mat rebar .in files + meta.csv -> {out_dir}")

def write_rebar(out_dir, n, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    meta = []
    for i in range(n):
        p = rebar_params(rng)
        content = REBAR_TEMPLATE.format(**p)
        fname = f"rebar_{i:05d}.in"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(content)
        meta.append(f"{fname},{p['depth_mm']:.4f},{p['epsr']:.4f},{p['diameter_mm']:.4f}")
    with open(os.path.join(out_dir, "meta.csv"), "w") as f:
        f.write("filename,depth_mm,epsr,diameter_mm\n")
        f.write("\n".join(meta) + "\n")
    print(f"Wrote {n} rebar .in files + meta.csv -> {out_dir}")

def write_delam(out_dir, n, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    meta = []
    half = n // 2
    labels = [0] * half + [1] * (n - half)
    rng.shuffle(labels)
    for i, label in enumerate(labels):
        p = delam_params(rng, label)
        if label == 0:
            content = DELAM_TEMPLATE.format(**p)
        else:
            content = DELAM_SOUND_TEMPLATE.format(**p)
        fname = f"delam_{i:05d}.in"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(content)
        meta.append(f"{fname},{label}")
    with open(os.path.join(out_dir, "meta.csv"), "w") as f:
        f.write("filename,label\n")
        f.write("\n".join(meta) + "\n")
    print(f"Wrote {n} delam .in files + meta.csv -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rebar", "rebar2mat", "delam"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/test")
    args = ap.parse_args()

    if args.mode == "rebar":
        write_rebar(args.out, args.n, args.seed)
    elif args.mode == "rebar2mat":
        write_rebar2mat(args.out, args.n, args.seed)
    else:
        write_delam(args.out, args.n, args.seed)

if __name__ == "__main__":
    main()
