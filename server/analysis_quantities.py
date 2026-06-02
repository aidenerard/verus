"""
analysis_quantities.py — dielectric/moisture map + ASTM D6087 deck quantities.
Re-exported by analysis.py. White-background style matches build_unified_depth_map
and render_corrosion_db_map so all Verus maps look consistent.
"""
from __future__ import annotations

import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Dielectric thresholds are heuristics (NOT codified by ASTM D6087): typical
# sound deck concrete sits at epsr 8-10; moisture/chloride pushes it higher.
_EPSR_MIN = 4.0
_EPSR_MAX = 16.0
_EPSR_MOISTURE_FLAG = 10.0


def build_dielectric_map(
    dielectric_values,
    output_path: str,
    analysis_name: str = "",
    x_coords=None,
    y_coords=None,
) -> str | None:
    """
    Plan-view per-trace dielectric (epsr) map — a moisture proxy. High epsr
    (>~10) = wet / chloride-laden = at-risk; low (4-6) = dry / sound.

    CALLER CONTRACT: only call this when REAL per-trace dielectric exists
    (plate calibration ran and epsr actually varies). A constant-epsr map is
    meaningless, so the pipelines gate on variation before calling.
    """
    epsr = np.clip(np.asarray(dielectric_values, dtype=np.float32), _EPSR_MIN, _EPSR_MAX)
    if epsr.size == 0 or np.all(np.isnan(epsr)):
        print("[DIELECTRIC MAP] empty input — skipping")
        return None

    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="white")
    have_gps = (
        x_coords is not None and y_coords is not None
        and len(x_coords) == len(epsr) == len(y_coords)
        and np.ptp(x_coords) > 0 and np.ptp(y_coords) > 0
    )
    if have_gps:
        from scipy.interpolate import griddata
        xc = np.asarray(x_coords, dtype=np.float64)
        yc = np.asarray(y_coords, dtype=np.float64)
        gx, gy = np.mgrid[xc.min():xc.max():400j, yc.min():yc.max():120j]
        Z = griddata((xc, yc), epsr, (gx, gy), method="linear")
        im = ax.pcolormesh(gx, gy, Z, cmap="RdYlBu_r", vmin=_EPSR_MIN, vmax=_EPSR_MAX, shading="auto")
    else:
        n = len(epsr)
        n_cols = max(1, int(np.sqrt(n * 4)))
        n_rows = max(1, int(np.ceil(n / n_cols)))
        padded = np.full(n_rows * n_cols, np.nan, dtype=np.float32)
        padded[:n] = epsr
        im = ax.imshow(padded.reshape(n_rows, n_cols), cmap="RdYlBu_r",
                       vmin=_EPSR_MIN, vmax=_EPSR_MAX, aspect="auto", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.6, pad=0.13, aspect=40)
    cbar.set_label("Dielectric constant εr  (low = dry/sound · high = wet/at-risk)", fontsize=10)
    cbar.set_ticks([4, 6, 8, 10, 12, 14, 16])
    high_moisture_pct = float((epsr > _EPSR_MOISTURE_FLAG).sum() / len(epsr) * 100)
    ax.set_title(f"{analysis_name or 'Dielectric Map'} — Moisture Proxy "
                 f"(εr>{_EPSR_MOISTURE_FLAG:.0f}: {high_moisture_pct:.1f}%)", fontsize=13, pad=10)
    ax.tick_params(colors="black", labelsize=10)
    for sp in ax.spines.values():
        sp.set_linewidth(0.9); sp.set_color("black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); plt.close("all"); gc.collect()
    print(f"[DIELECTRIC MAP] saved → {output_path} ({high_moisture_pct:.1f}% εr>{_EPSR_MOISTURE_FLAG:.0f})")
    return output_path


def calculate_deck_quantities(
    depths_in,
    deteriorated,
    dielectric_values=None,
    scan_spacing_ft: float = 2.0,
) -> dict:
    """
    ASTM D6087 deck condition quantities (Infrasense-style report table).

    `deteriorated` may be a bool array (per-pick flags) OR a scalar percentage
    already computed upstream. We never assert pass/fail compliance — ASTM sets
    its threshold from ground-truth cores, so we report method + threshold only.
    Moisture fields are included only when real per-trace dielectric is given.
    """
    depths = np.asarray(depths_in, dtype=np.float32)
    depths = depths[np.isfinite(depths)]
    n = int(depths.size)
    if n == 0:
        return {"n_picks": 0, "astm_method": "ASTM D6087-22", "astm_status": "Method applied"}

    if np.ndim(deteriorated) > 0:
        flags = np.asarray(deteriorated, dtype=bool)
        deteriorated_pct = round(float(flags.sum() / max(flags.size, 1) * 100), 1)
    else:
        deteriorated_pct = round(float(deteriorated), 1)

    q = {
        "n_picks":             n,
        "scan_spacing_ft":     scan_spacing_ft,
        "mean_cover_in":       round(float(np.mean(depths)), 2),
        "std_cover_in":        round(float(np.std(depths)), 2),
        "min_cover_in":        round(float(np.min(depths)), 2),
        "max_cover_in":        round(float(np.max(depths)), 2),
        "cover_below_2in_pct": round(float((depths < 2.0).sum() / n * 100), 1),
        "deteriorated_pct":    deteriorated_pct,
        "sound_pct":           round(100.0 - deteriorated_pct, 1),
        # No boolean compliance — ASTM D6087 thresholds require ground-truth cores.
        "astm_method":         "ASTM D6087-22",
        "astm_status":         "Method applied",
        "deterioration_method": "Depth-corrected amplitude (Pashoutani & Zhu 2023)",
        "threshold_db":        -8.0,
        "threshold_note":      "-8.0 dB default — verify with ground-truth cores",
    }

    if dielectric_values is not None:
        epsr = np.asarray(dielectric_values, dtype=np.float32)
        epsr = epsr[np.isfinite(epsr)]
        if epsr.size and float(np.std(epsr)) > 0.1:
            q["mean_dielectric"]  = round(float(np.mean(epsr)), 2)
            q["high_moisture_pct"] = round(float((epsr > _EPSR_MOISTURE_FLAG).sum() / epsr.size * 100), 1)
    return q
