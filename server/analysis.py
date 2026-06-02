"""
analysis.py — Generic GPR analysis functions for Verus backend.
Works on standardised output from ingest.py (read_proceq, read_dzt, etc.)
Proceq-specific pipeline lives in analysis_proceq.py (re-exported below).
"""
from __future__ import annotations

import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
from scipy.signal import hilbert

# Physics primitives live in physics.py (co-located sub-module per the
# 300-line cap); re-exported here so callers keep `from analysis import …`.
from physics import (  # noqa: F401
    find_time_zero,
    calculate_dielectric_fresnel,
    calculate_rebar_depth,
    calculate_astm_corrosion_index,
)


def extract_amplitude_and_depth(
    traces: np.ndarray,
    search_start: int = 55,
    search_end: int = 150,
    epsr: float = 9.0,
    time_range_ns: float = 15.0,
) -> dict:
    n_samples = traces.shape[1]
    ns_per_sample = time_range_ns / n_samples
    velocity = 0.3 / np.sqrt(epsr)
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
    plt.close('all')
    gc.collect()
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
    plt.close('all')
    gc.collect()
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
    plt.close('all')
    gc.collect()
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
    plt.close('all')
    gc.collect()
    print(f"[DEPTH MAP] saved → {out}")
    return out


def _epsr_for_frequency(frequency_mhz: int) -> float:
    """Rebar dielectric constant by GPR centre frequency. Mirrors inference.py."""
    return {400: 8.0, 900: 7.0, 1600: 6.0, 2000: 5.5, 2600: 5.0}.get(frequency_mhz, 6.0)


def detect_hyperbola_picks(
    traces,
    epsr:                float = 9.0,
    time_range_ns:       float = 16.0,
    min_depth_in:        float = 0.5,
    max_depth_in:        float = 10.0,
    min_distance_traces: int   = 15,
    model_mean_depth_in: float = None,
    amp_plate:           float = None,
) -> list[dict]:
    """
    Detect hyperbola apexes in a GPR B-scan (FHWA / Infrasense / GSSI RADAN).

    The apex is the point of minimum travel time, where the antenna is
    directly above the rebar — the ONLY valid depth-measurement point.
    Non-apex picks overestimate depth.

    Steps:
      1. Find time zero per trace (surface reflection), median for robustness
      2. Hilbert envelope; per-trace dominant peak in the search window
      3. Find LOCAL MINIMA in the depth profile = hyperbola apexes
      4. Symmetry check: depth increases on both sides of each apex
      5. Physically-correct depth (time-zero corrected + per-trace dielectric)

    Returns list[{trace_idx, sample_idx, depth_in, confidence, epsr, time_zero}].
    """
    from scipy.signal import hilbert, find_peaks
    from scipy.ndimage import uniform_filter1d

    n_traces, n_samples = traces.shape
    ns_per_sample = time_range_ns / n_samples

    # Time zero from the first few traces (median = robust to a noisy trace).
    t0_samples = [find_time_zero(traces[t]) for t in range(min(20, n_traces))]
    time_zero  = int(np.median(t0_samples))
    print(f"[PICKS] time_zero={time_zero} samples "
          f"({time_zero * ns_per_sample:.2f} ns)", flush=True)

    # Per-trace dielectric via Fresnel when a plate amplitude is available.
    def get_epsr(trace_idx: int) -> float:
        if amp_plate is not None:
            env = np.abs(hilbert(traces[trace_idx, :60].astype(np.float32)))
            amp_surf = float(env[time_zero]) if time_zero < 60 else float(env.max())
            return calculate_dielectric_fresnel(amp_surf, amp_plate)
        return epsr

    # Constrain the search window with the model estimate if we have one.
    if model_mean_depth_in is not None:
        search_min = max(min_depth_in, model_mean_depth_in - 2.5)
        search_max = min(max_depth_in, model_mean_depth_in + 3.5)
    else:
        search_min, search_max = min_depth_in, max_depth_in

    def depth_to_sample_offset(d_in: float, epsr_val: float) -> int:
        velocity = 0.3 / np.sqrt(epsr_val)
        twtt_ns  = (d_in / 39.3701) / velocity * 2
        return int(twtt_ns / ns_per_sample)

    epsr_median = float(np.median(
        [get_epsr(t) for t in range(0, n_traces, max(1, n_traces // 20))]
    ))
    s_min = max(time_zero + 1, time_zero + depth_to_sample_offset(search_min, epsr_median))
    s_max = min(n_samples - 1, time_zero + depth_to_sample_offset(search_max, epsr_median))
    if s_max - s_min < 2:
        return []
    print(f"[PICKS] search window: {search_min:.1f}\"-{search_max:.1f}\" "
          f"(samples {s_min}-{s_max}) epsr_median={epsr_median:.1f}", flush=True)

    envelope = np.abs(hilbert(traces, axis=1)).astype(np.float32)

    # Per-trace dominant peak sample inside the search window → depth profile.
    depth_profile = np.zeros(n_traces, dtype=np.float32)
    amp_profile   = np.zeros(n_traces, dtype=np.float32)
    for t in range(n_traces):
        seg = envelope[t, s_min:s_max]
        if seg.max() < 1e-6:
            depth_profile[t] = float(s_min + s_max) / 2
            continue
        pk = int(np.argmax(seg)) + s_min
        depth_profile[t] = float(pk)
        amp_profile[t]   = float(envelope[t, pk])

    smoothed = uniform_filter1d(depth_profile, size=max(5, min_distance_traces // 2))

    # Local minima in travel time = hyperbola apexes.
    candidates, _ = find_peaks(-smoothed, distance=min_distance_traces, prominence=1.5)
    print(f"[PICKS] {len(candidates)} candidates before symmetry check", flush=True)

    max_amp = float(amp_profile.max()) or 1.0
    result: list[dict] = []
    for idx in candidates:
        t      = int(idx)
        apex_s = int(depth_profile[t])
        # Reach out to ~2x the separation so the comparison lands on the
        # genuinely-risen hyperbola tail, not the near-flat plateau around
        # the apex. Compare the DEEPEST flank sample on each side so an
        # off-centre candidate (find_peaks on a flat plateau) still verifies.
        flank = min(2 * min_distance_traces, t, n_traces - t - 1)
        if flank < 3:
            continue
        left_seg  = depth_profile[max(0, t - flank):t]
        right_seg = depth_profile[t + 1:t + 1 + flank]
        left_deeper  = left_seg.size  > 0 and float(np.max(left_seg))  > apex_s + 0.5
        right_deeper = right_seg.size > 0 and float(np.max(right_seg)) > apex_s + 0.5
        if not (left_deeper and right_deeper):
            continue

        trace_epsr = get_epsr(t)
        depth_in = calculate_rebar_depth(
            apex_sample=apex_s, time_zero=time_zero, epsr=trace_epsr,
            time_range_ns=time_range_ns, n_samples=n_samples,
        )
        if not (search_min <= depth_in <= search_max):
            continue

        result.append({
            'trace_idx':  t,
            'sample_idx': apex_s,
            'depth_in':   depth_in,
            'confidence': round(float(amp_profile[t] / max_amp), 3),
            'epsr':       round(float(trace_epsr), 2),
            'time_zero':  time_zero,
        })

    print(f"[PICKS] {len(result)} verified apex picks", flush=True)
    return result


def safe_filename(name: str) -> str:
    """Convert an analysis name to a filesystem-safe filename token.
    'Bridge B440029 Oct 2024' → 'Bridge_B440029_Oct_2024'.
    Preserves alphanumerics + hyphens + underscores; spaces collapse to '_'."""
    import re
    safe = re.sub(r'[^\w\s\-]', '', name or 'Untitled Analysis')
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe or 'Untitled_Analysis'


def build_unified_depth_map(
    depths_in,
    output_path: str,
    analysis_name: str,
    x_coords=None,
    y_coords=None,
    vmin: float = 3.0,
    vmax: float = 8.5,
    bridge_length_ft: float | None = None,
    bridge_width_ft: float | None = None,
) -> str | None:
    """
    Unified rebar cover-depth map renderer used by both Proceq and DZT
    pipelines. This is the SINGLE canonical depth-map renderer — its output is
    the locked reference (see docs/depth_map_reference.png / DEPTH_MAP_SPEC.md).
    Do not change the palette, range, or banding without updating that fixture.
      - Title = analysis_name (verbatim)
      - Fixed 0.5" YlOrRd_r colour bands across the locked 3.0–8.5"
        range so every dataset uses the same palette mapping (color = depth
        is identical across instruments and never auto-rescales)
      - Depths snapped to nearest whole inch — cleaner integer transitions,
        less label clutter
      - Black isolines on integer inches only; **each** integer-level path
        gets its own label (one per disconnected region) placed in fixed
        horizontal orientation so "6" never reads as "9"
      - Bare integer ticks, no axis labels
      - Rectangular horizontal colorbar at the bottom (no extend arrows)

    Input modes:
      • 1D `depths_in` + (`x_coords`, `y_coords`)  → griddata onto 500×150
        regular UTM grid (Proceq with GPS)
      • 2D `depths_in`                            → used as (swath, trace) grid

    Raises ValueError on 1D depths with no GPS coordinates — that input
    shape has no spatial coherence and the previous tile-to-40-rows
    fallback only produced a vertical-stripe artifact.
    """
    import os
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy.interpolate import griddata as _griddata
    from scipy.ndimage import gaussian_filter, median_filter, zoom as nd_zoom

    arr = np.asarray(depths_in, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        print("[DEPTH MAP] empty input — skipping"); return None

    # Snap to nearest whole inch and clip to the fixed reference range so
    # everything maps inside the rectangular colorbar.
    arr_clip = np.clip(np.round(arr), vmin, vmax)

    # ── Assemble Z grid + X/Y meshgrid ───────────────────────────────────
    have_gps = (
        x_coords is not None and y_coords is not None and arr.ndim == 1
        and len(x_coords) == len(arr) == len(y_coords)
        and np.ptp(x_coords) > 0 and np.ptp(y_coords) > 0
    )
    if have_gps:
        xc = np.asarray(x_coords, dtype=np.float64)
        yc = np.asarray(y_coords, dtype=np.float64)
        gx, gy = np.mgrid[xc.min():xc.max():500j, yc.min():yc.max():150j]
        Z = _griddata((xc, yc), arr_clip, (gx, gy), method="cubic")
        Z_nn = _griddata((xc, yc), arr_clip, (gx, gy), method="nearest")
        Z[np.isnan(Z)] = Z_nn[np.isnan(Z)]
        Z = median_filter(Z, size=9)
        Z = gaussian_filter(Z, sigma=3)
        Z = np.clip(np.round(Z), vmin, vmax)
        X, Y = gx, gy
        x_lim, y_lim = (xc.min(), xc.max()), (yc.min(), yc.max())
    else:
        if arr_clip.ndim != 2:
            raise ValueError(
                "build_unified_depth_map requires either a 2D depth grid OR "
                "1D depths_in paired with x_coords/y_coords. Got 1D-only input "
                f"of shape {arr.shape} — no spatial layout to render."
            )
        grid_src = arr_clip
        n_rows, n_cols = grid_src.shape
        nan_mask = np.isnan(grid_src)
        filled = np.where(nan_mask, float(np.nanmean(grid_src)), grid_src)
        target_rows = max(n_rows, 40)
        upsampled = (nd_zoom(filled, (target_rows / n_rows, 1.0), order=1)
                     if target_rows != n_rows else filled)
        sigma_y = max(0.8, target_rows / 50.0)
        sigma_x = max(1.2, n_cols / 200.0)
        Z = gaussian_filter(upsampled, sigma=(sigma_y, sigma_x))
        Z = np.clip(np.round(Z), vmin, vmax)
        X, Y = np.meshgrid(np.arange(n_cols), np.linspace(0, n_rows - 1, target_rows))
        x_lim, y_lim = (0, max(1, n_cols - 1)), (n_rows - 1, 0)

    # ── Real bridge dimensions (user-input feet) ─────────────────────────
    # When the inspector enters deck length × width, linearly rescale the
    # meshgrid to those feet so the axes read true distance-along-bridge ×
    # width instead of UTM metres / grid indices. Plan-view shape is
    # preserved (linear remap); the locked palette/banding is untouched.
    feet_axes = bool(bridge_length_ft and bridge_width_ft
                     and bridge_length_ft > 0 and bridge_width_ft > 0)
    if feet_axes:
        X = (X - X.min()) / (np.ptp(X) or 1.0) * float(bridge_length_ft)
        Y = (Y - Y.min()) / (np.ptp(Y) or 1.0) * float(bridge_width_ft)
        x_lim, y_lim = (0.0, float(bridge_length_ft)), (0.0, float(bridge_width_ft))

    # ── Fixed reference palette: 0.5" bands across the full vmin→vmax range ──
    levels = np.arange(vmin, vmax + 0.5, 0.5)
    integer_levels = [float(l) for l in levels if abs(l - round(l)) < 1e-6]
    n_bands = max(1, len(levels) - 1)
    cmap = ListedColormap(plt.cm.YlOrRd_r(np.linspace(0, 1, n_bands)))  # type: ignore[attr-defined]
    norm = BoundaryNorm(levels, cmap.N)

    # ── Render ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor='white')
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    # Integer-inch isolines only — no half-inch lines clutter the map.
    cs_int = ax.contour(X, Y, Z, levels=integer_levels, colors='black', linewidths=0.8)
    # Multiple labels per integer level — one per connected path. Skip very
    # short paths (artifacts) and drop labels that fall within `min_sep` of
    # an existing label so the result is dense but not piled up.
    x_span = abs(x_lim[1] - x_lim[0]) or 1.0
    y_span = abs(y_lim[1] - y_lim[0]) or 1.0
    min_sep_x = x_span / 12.0
    min_sep_y = y_span / 6.0
    placed: list[tuple[float, float]] = []
    for li, lv in enumerate(cs_int.levels):
        if li >= len(cs_int.collections): continue
        for path in cs_int.collections[li].get_paths():
            v = path.vertices
            if len(v) < 30: continue
            mid = v[len(v) // 2]
            if any(abs(mid[0] - px) < min_sep_x and abs(mid[1] - py) < min_sep_y
                   for px, py in placed):
                continue
            ax.text(mid[0], mid[1], f'{int(round(lv))}',
                    fontsize=10, ha='center', va='center', color='black',
                    fontweight='normal')
            placed.append((float(mid[0]), float(mid[1])))

    ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
    if feet_axes:
        import matplotlib.ticker as _mticker
        def _nice_step(span: float) -> float:
            for s in (5, 10, 20, 25, 50, 100, 200, 500):
                if span / s <= 14: return s
            return 1000
        ax.set_xlabel('Distance Along Bridge (ft)', fontsize=10, color='black')
        ax.set_ylabel('Bridge Width (ft)', fontsize=10, color='black')
        ax.xaxis.set_major_locator(_mticker.MultipleLocator(_nice_step(float(bridge_length_ft))))
        ax.yaxis.set_major_locator(_mticker.MultipleLocator(_nice_step(float(bridge_width_ft))))
        ax.set_title(f'{analysis_name}  —  {bridge_length_ft:.0f} ft × {bridge_width_ft:.0f} ft',
                     fontsize=13, pad=10)
    else:
        ax.set_title(analysis_name, fontsize=13, pad=10)
    ax.tick_params(colors='black', length=4, width=0.7, labelsize=10)
    for sp in ax.spines.values():
        sp.set_linewidth(0.9); sp.set_color('black')

    # Rectangular colorbar — explicit extend='neither' overrides the
    # arrows contourf would otherwise propagate.
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', ticks=levels,
                        spacing='proportional', shrink=0.6, pad=0.13, aspect=40,
                        extend='neither')
    cbar.ax.tick_params(labelsize=9, length=3, colors='black')
    cbar.outline.set_linewidth(0.5); cbar.outline.set_edgecolor('black')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig); plt.close('all'); gc.collect()
    print(f"[DEPTH MAP] saved → {output_path}")
    return output_path


# Re-export Proceq pipeline so existing callers (jobs.py, diagnostics.py) keep working.
from analysis_proceq import (  # noqa: E402
    load_cscan_amplitudes,
    build_cscan_maps,
    render_corrosion_db_map,
    process_proceq_dataset,
)
from analysis_bscan import build_bscan_image  # noqa: E402
from analysis_quantities import (  # noqa: E402
    build_dielectric_map,
    calculate_deck_quantities,
)

__all__ = [
    "extract_amplitude_and_depth",
    "_make_grid",
    "build_amplitude_map",
    "build_depth_map",
    "build_corrosion_map",
    "build_model_depth_map",
    "build_unified_depth_map",
    "safe_filename",
    "detect_hyperbola_picks",
    "find_time_zero",
    "calculate_dielectric_fresnel",
    "calculate_rebar_depth",
    "calculate_astm_corrosion_index",
    "build_bscan_image",
    "load_cscan_amplitudes",
    "build_cscan_maps",
    "render_corrosion_db_map",
    "process_proceq_dataset",
    "build_dielectric_map",
    "calculate_deck_quantities",
]
