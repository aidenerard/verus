"""
analysis.py — Generic GPR analysis functions for Verus backend.
Works on standardised output from ingest.py (read_proceq, read_dzt, etc.)
Proceq-specific pipeline lives in analysis_proceq.py (re-exported below).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
from scipy.signal import hilbert


def extract_amplitude_and_depth(
    traces: np.ndarray,
    search_start: int = 55,
    search_end: int = 150,
    epsr: float = 9.0,
    time_range_ns: float = 15.0,
) -> dict:
    n_samples = traces.shape[1]
    ns_per_sample = time_range_ns / n_samples
    velocity = 0.15 / np.sqrt(epsr)
    envelope = np.abs(hilbert(traces, axis=1))
    window = envelope[:, search_start:search_end]
    picks = np.argmax(window, axis=1) + search_start
    amps = envelope[np.arange(len(picks)), picks]
    depths_m = (picks * ns_per_sample * velocity) / 2.0
    depths_in = depths_m * 39.3701
    return {"picks": picks, "amplitudes": amps, "depths_m": depths_m, "depths_in": depths_in}


def _make_grid(easting, northing, values, grid_res_e=500, grid_res_n=80):
    e0, e1 = easting.min(), easting.max()
    n0, n1 = northing.min(), northing.max()
    if e1 == e0 or n1 == n0:
        return None, None, None
    ge = np.linspace(e0, e1, grid_res_e)
    gn = np.linspace(n0, n1, grid_res_n)
    GE, GN = np.meshgrid(ge, gn)
    GZ = griddata((easting, northing), values, (GE, GN), method="linear")
    GZ_near = griddata((easting, northing), values, (GE, GN), method="nearest")
    GZ[np.isnan(GZ)] = GZ_near[np.isnan(GZ)]
    GZ = median_filter(GZ, size=3)
    return GE, GN, GZ


def build_amplitude_map(
    all_easting: np.ndarray,
    all_northing: np.ndarray,
    all_amplitudes: np.ndarray,
    output_path: str,
    title: str = "Amplitude Map — Corrosion Risk",
) -> None:
    p2  = np.percentile(all_amplitudes, 2)
    p98 = np.percentile(all_amplitudes, 98)
    norm_amp = np.clip((all_amplitudes - p2) / (p98 - p2 or 1.0), 0.0, 1.0)
    print(f"[ANALYSIS] amplitude p2={p2:.1f}  p98={p98:.1f}")
    GE, GN, GZ = _make_grid(all_easting, all_northing, norm_amp)
    if GZ is None:
        print("[ANALYSIS] amplitude map: insufficient spatial variation — skipped")
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    cf = ax.contourf(GE, GN, GZ, levels=20, cmap="RdYlGn")
    plt.colorbar(cf, ax=ax, label="Amplitude (green=healthy, red=risk)")
    ax.set_xlabel("Along-track distance (m)")
    ax.set_ylabel("Cross-track distance (m)")
    ax.set_title(title)
    ax.set_aspect("auto")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ANALYSIS] amplitude map saved → {output_path}")


def build_depth_map(
    all_easting: np.ndarray,
    all_northing: np.ndarray,
    all_depths_in: np.ndarray,
    output_path: str,
    title: str = "Rebar Depth Map",
) -> None:
    GE, GN, GZ = _make_grid(all_easting, all_northing, all_depths_in)
    if GZ is None:
        print("[ANALYSIS] depth map: insufficient spatial variation — skipped")
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    cf = ax.contourf(GE, GN, GZ, levels=20, cmap="RdYlGn_r")
    cs = ax.contour(GE, GN, GZ, levels=8, colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, fmt="%.1f\"", fontsize=7)
    plt.colorbar(cf, ax=ax, label='Rebar depth (inches)')
    ax.set_xlabel("Along-track distance (m)")
    ax.set_ylabel("Cross-track distance (m)")
    ax.set_title(title)
    ax.set_aspect("auto")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ANALYSIS] depth map saved → {output_path}")


def build_corrosion_map(
    all_easting: np.ndarray,
    all_northing: np.ndarray,
    all_amplitudes: np.ndarray,
    output_path: str,
    title: str = "Corrosion Risk Map",
) -> None:
    if len(all_easting) == 0 or len(all_northing) == 0 or len(all_amplitudes) == 0:
        print("[ANALYSIS] corrosion map: empty input arrays — skipping")
        return
    p2  = np.percentile(all_amplitudes, 2)
    p98 = np.percentile(all_amplitudes, 98)
    norm_amp = np.clip((all_amplitudes - p2) / (p98 - p2 or 1.0), 0.0, 1.0)
    GE, GN, GZ = _make_grid(all_easting, all_northing, norm_amp)
    if GZ is None:
        print("[ANALYSIS] corrosion map: insufficient spatial variation — skipped")
        return
    threshold = float(np.median(GZ))
    print(f"[ANALYSIS] corrosion p2={p2:.1f}  p98={p98:.1f}  median_threshold={threshold:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(22, 5))
    cf0 = axes[0].contourf(GE, GN, GZ, levels=20, cmap="RdYlGn")
    plt.colorbar(cf0, ax=axes[0], label="Amplitude (green=healthy, red=risk)")
    axes[0].set_title("Continuous Amplitude")
    axes[0].set_aspect("auto")
    binary = (GZ >= threshold).astype(float)
    cf1 = axes[1].contourf(GE, GN, binary, levels=[0, 0.5, 1], cmap="RdYlGn")
    plt.colorbar(cf1, ax=axes[1], label="0=low amp (risk)  1=high amp (healthy)")
    axes[1].set_title(f"Binary (median threshold={threshold:.2f})")
    axes[1].set_aspect("auto")
    fig.suptitle(title)
    for ax in axes:
        ax.set_xlabel("Along-track distance (m)")
        ax.set_ylabel("Cross-track distance (m)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ANALYSIS] corrosion map saved → {output_path}")


def parse_pos_file(pos_path):
    """Parse Proceq .pos UTM file. Returns DataFrame[distance,easting,northing,elevation] or None."""
    try:
        rows: list[list[float]] = []
        in_data = False
        for raw in open(pos_path, "r"):
            line = raw.strip()
            if line.startswith("<VALID_MARKERS_SWEEP>"):
                in_data = True; continue
            if line.startswith("<INVALID_INTERVALS>"):
                in_data = False; continue
            if in_data and "," in line:
                parts = line.split(",")
                if len(parts) == 4:
                    try: rows.append([float(p) for p in parts])
                    except ValueError: pass
        if not rows:
            return None
        import pandas as pd
        return pd.DataFrame(rows, columns=["distance", "easting", "northing", "elevation"])
    except Exception as exc:
        print(f"[POS] Failed to parse {pos_path}: {exc}")
        return None


def get_trace_gps(pos_df, n_traces):
    """
    Interpolate GPS fixes to per-trace easting / northing.
    Returns (easting_arr, northing_arr) both shape (n_traces,),
    or (None, None) if pos_df is unusable.
    """
    if pos_df is None or len(pos_df) < 2:
        return None, None
    trace_indices = np.arange(n_traces)
    gps_indices   = np.linspace(0, n_traces - 1, len(pos_df))
    easting  = np.interp(trace_indices, gps_indices, pos_df["easting"].values)
    northing = np.interp(trace_indices, gps_indices, pos_df["northing"].values)
    return easting, northing


def build_model_depth_map(
    all_easting,
    all_northing,
    all_depths_in,
    output_dir,
    title: str = "Rebar Depth Map",
) -> str | None:
    """
    Infrasense-style contourf rebar depth heatmap from dense per-trace model
    predictions. Objective measurement — no analyst review needed.

    Writes <output_dir>/rebar_depth_map.png and returns the path, or None when
    inputs are empty (e.g. no GPS-valid traces).
    """
    import os
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter, median_filter

    if len(all_easting) == 0 or len(all_northing) == 0 or len(all_depths_in) == 0:
        print("[DEPTH MAP] no valid georeferenced traces — skipping depth map")
        return None

    # Snap per-trace depths to nearest 0.5" so downstream binning stays consistent.
    all_depths_in = np.round(np.asarray(all_depths_in) * 2) / 2

    e0, e1 = all_easting.min(),  all_easting.max()
    n0, n1 = all_northing.min(), all_northing.max()

    grid_e, grid_n = np.mgrid[e0:e1:500j, n0:n1:150j]

    grid_d = griddata((all_easting, all_northing), all_depths_in,
                      (grid_e, grid_n), method="cubic")
    grid_d_near = griddata((all_easting, all_northing), all_depths_in,
                           (grid_e, grid_n), method="nearest")
    grid_d[np.isnan(grid_d)] = grid_d_near[np.isnan(grid_d)]

    # Smoother spatial grouping — larger median + Gaussian, then re-snap to 0.5".
    grid_d = median_filter(grid_d, size=9)
    grid_d = gaussian_filter(grid_d, sigma=3)
    grid_d = np.round(grid_d * 2) / 2

    fig, ax = plt.subplots(figsize=(16, 6))

    # Relative colorscale per map, rounded outward to nearest 0.5".
    vmin = np.percentile(all_depths_in, 2)
    vmax = np.percentile(all_depths_in, 98)
    vmin = np.floor(vmin * 2) / 2
    vmax = np.ceil(vmax * 2) / 2
    levels_fill = np.arange(vmin, vmax + 0.5, 0.5)
    levels_line = np.arange(vmin, vmax + 0.5, 0.5)

    # Red-yellow only colormap (no green, no blue).
    red_yellow = LinearSegmentedColormap.from_list(
        "red_yellow",
        ["#FF0000", "#FF4400", "#FF8800", "#FFAA00", "#FFCC00", "#FFE800", "#FFFF00"],
        N=256,
    )

    cf = ax.contourf(grid_e, grid_n, grid_d,
                     levels=levels_fill, cmap=red_yellow, extend="both")
    cs = ax.contour(grid_e, grid_n, grid_d,
                    levels=levels_line, colors="black", linewidths=0.6)

    # Label each contour level exactly once — pick the longest path for that level
    # and put a single inline-text label at its midpoint.
    for level_idx, level_val in enumerate(cs.levels):
        if level_idx >= len(cs.collections):
            continue
        paths = cs.collections[level_idx].get_paths()
        if not paths:
            continue
        longest = max(paths, key=lambda p: len(p.vertices))
        mid_idx = len(longest.vertices) // 2
        x_mid, y_mid = longest.vertices[mid_idx]
        ax.text(x_mid, y_mid, f'{level_val:.1f}"',
                fontsize=8, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1",
                          facecolor="white", alpha=0.7, edgecolor="none"))

    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal",
                        pad=0.15, aspect=50,
                        ticks=np.arange(vmin, vmax + 0.5, 0.5))
    cbar.set_label("Rebar Depth (inches)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    ax.set_xlabel("Easting (m, UTM Zone 16N)", fontsize=11)
    ax.set_ylabel("Northing (m, UTM Zone 16N)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("auto")
    ax.tick_params(which="both", direction="in", top=True, right=True)

    plt.tight_layout()
    out = os.path.join(output_dir, "rebar_depth_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[DEPTH MAP] saved → {out}")
    return out


# Re-export Proceq pipeline so existing callers (jobs.py, diagnostics.py) keep working.
from analysis_proceq import (  # noqa: E402
    load_cscan_amplitudes,
    build_cscan_maps,
    process_proceq_dataset,
)

__all__ = [
    "extract_amplitude_and_depth",
    "_make_grid",
    "build_amplitude_map",
    "build_depth_map",
    "build_corrosion_map",
    "build_model_depth_map",
    "load_cscan_amplitudes",
    "build_cscan_maps",
    "process_proceq_dataset",
]
