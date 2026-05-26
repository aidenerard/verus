"""
analysis_proceq.py — Proceq RIS-specific GPR analysis pipeline.
Imported and re-exported by analysis.py.
"""
from __future__ import annotations

import gc
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_RIS_EPSR          = 9.0
_RIS_VELOCITY      = 0.15 / np.sqrt(_RIS_EPSR)  # ≈ 0.05 m/ns in concrete
_RIS_NS_PER_SAMPLE = 15.0 / 510                  # ≈ 0.02941 ns per sample


def load_cscan_amplitudes(cscan_paths: list[str]) -> list[dict | None]:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ingest import read_cscan

    result = []
    for path in cscan_paths:
        try:
            cs = read_cscan(path)
            result.append({"amp": cs["peak_amplitude"], "dx": cs["dx_m"], "dy": cs["dy_m"]})
        except Exception as exc:
            print(f"[ANALYSIS] CScan load failed {os.path.basename(path)}: {exc}")
            result.append(None)
    return result


def build_horizon_picks_plot(odd_scans, ts_data, output_dir):
    """
    Plot B-scan for each swath with TS ground truth horizon overlaid.
    Shows rebar pick (red) and deck bottom pick (blue) per trace.
    """
    from ingest import read_proceq

    n_swaths = min(len(odd_scans) // 4, ts_data['n_swaths'])
    fig, axes = plt.subplots(n_swaths, 1, figsize=(24, n_swaths * 2))
    if n_swaths == 1:
        axes = [axes]

    for swath_idx in range(n_swaths):
        ax = axes[swath_idx]
        scan_path = odd_scans[swath_idx * 4]
        try:
            result = read_proceq(scan_path)
            traces = result['traces']  # (n_traces, 510)

            ax.imshow(traces.T, aspect='auto', cmap='gray',
                      vmin=-0.3, vmax=0.3, origin='upper')

            rebar_cm  = ts_data['rebar_cm'][swath_idx]
            bottom_cm = ts_data['bottom_cm'][swath_idx]

            velocity      = _RIS_VELOCITY
            ns_per_sample = _RIS_NS_PER_SAMPLE

            def cm_to_sample(cm_arr):
                depth_m = cm_arr / 100.0
                t_ns    = depth_m / velocity * 2.0
                return t_ns / ns_per_sample

            x_picks  = np.linspace(0, len(traces) - 1, len(rebar_cm))
            x_traces = np.arange(len(traces))
            rebar_samples  = np.interp(x_traces, x_picks, cm_to_sample(rebar_cm))
            bottom_samples = np.interp(x_traces, x_picks, cm_to_sample(bottom_cm))

            ax.plot(x_traces, rebar_samples,  'r-', linewidth=0.8,
                    label='Rebar' if swath_idx == 0 else '')
            ax.plot(x_traces, bottom_samples, 'b-', linewidth=0.8,
                    label='Deck bottom' if swath_idx == 0 else '')
            ax.set_ylabel(f'Swath {swath_idx + 1}', fontsize=8)
            ax.set_yticks([])
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes,
                    ha='center', va='center')

    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Trace index')
    plt.suptitle('Horizon Picks — All Swaths\nRed=Rebar, Blue=Deck Bottom',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(output_dir, 'horizon_picks.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[PICKS] saved → {out}")
    return out


def build_ts_depth_map(ts_data, output_dir):
    from analysis import build_depth_map
    E = np.concatenate(ts_data['easting'])
    N = np.concatenate(ts_data['northing'])
    D = np.concatenate(ts_data['depths_in'])
    out = os.path.join(output_dir, 'rebar_depth_map.png')
    build_depth_map(E, N, D, out, title='Rebar Depth Map (B-scan picks)')
    return out


def build_cscan_maps(cscan_slices, output_dir, title_prefix=''):
    """
    Build corrosion risk map directly from CScan raster data.
    cscan_slices: list of dicts from load_cscan_amplitudes()
    Returns a stats dict; returns a zero-filled shape-matching dict when there
    are no usable slices so callers reading ['high_risk_pct'] keep working.
    """
    empty_stats = {'corrosion_map_path': None, 'high_risk_pct': 0.0,
                   'threshold': 0.0, 'shape': (0, 0)}
    if not cscan_slices:
        print("[ANALYSIS] no CScan slices — skipping corrosion map")
        return empty_stats
    all_amps = [s['amp'] for s in cscan_slices]
    full_map = np.concatenate(all_amps, axis=1)  # (n_cross, total_cols)
    if full_map.size == 0:
        print("[ANALYSIS] CScan slices contained no data — skipping corrosion map")
        return empty_stats
    dx = cscan_slices[0]['dx']
    dy = cscan_slices[0]['dy']
    total_length_m = full_map.shape[1] * dx
    total_width_m  = full_map.shape[0] * dy

    print(f"[ANALYSIS] CScan raster: {full_map.shape} "
          f"({total_length_m:.1f}m × {total_width_m:.2f}m)")
    print(f"[ANALYSIS] Amplitude range: {full_map.min():.1f} – {full_map.max():.1f}")

    p2, p98 = np.percentile(full_map, 2), np.percentile(full_map, 98)
    norm_map = np.clip((full_map - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
    extent = [0, total_length_m, total_width_m, 0]

    threshold     = float(np.median(norm_map))
    risk_map      = (norm_map < threshold).astype(np.float32)
    high_risk_pct = float(risk_map.mean() * 100)
    print(f"[ANALYSIS] corrosion threshold: {threshold:.3f}, "
          f"high risk: {high_risk_pct:.1f}%")

    fig, ax = plt.subplots(1, 1, figsize=(24, 3))
    im = ax.imshow(risk_map, aspect='auto', cmap='RdYlGn_r',
                   origin='upper', extent=extent, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='0=healthy  1=at-risk')
    ax.set_title(f'{title_prefix}Corrosion Risk Map  (threshold={threshold:.2f}, '
                 f'{high_risk_pct:.1f}% flagged at risk)')
    ax.set_xlabel('Along-track distance (m)')
    ax.set_ylabel('Cross-track (m)')
    plt.tight_layout()
    cor_path = os.path.join(output_dir, 'corrosion_map.png')
    plt.savefig(cor_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ANALYSIS] corrosion map saved → {cor_path}")

    return {
        'corrosion_map_path': cor_path,
        'high_risk_pct':      high_risk_pct,
        'threshold':          threshold,
        'shape':              full_map.shape,
    }


from analysis_proceq_utils import median_swath_length as _median_swath_length  # noqa: E402


def process_proceq_dataset(
    data_dir: str,
    output_dir: str,
    epsr: float = _RIS_EPSR,
    search_start: int = 55,
    search_end: int = 150,
    inference_sample_rate: int = 16,
) -> dict | None:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import read_proceq
    from analysis import extract_amplitude_and_depth, parse_pos_file, get_trace_gps

    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    scan_files     = sorted(glob.glob(os.path.join(data_dir, "**", "PRC_*.scan"), recursive=True))
    pos_files      = sorted(glob.glob(os.path.join(data_dir, "**", "Swath_*.pos"), recursive=True))
    cscan_01_files = sorted(glob.glob(os.path.join(data_dir, "**", "CScan_01.CScan"), recursive=True))

    odd_scans = [
        f for f in scan_files
        if int(os.path.basename(f).replace("PRC_", "").replace(".scan", "")) % 2 == 1
    ]
    print(f"[ANALYSIS] {len(scan_files)} PRC  {len(odd_scans)} odd  "
          f"{len(pos_files)} pos  {len(cscan_01_files)} CScan_01")

    CHANNELS_PER_SWATH = 4
    DY = 0.05
    swath_groups = [
        odd_scans[i:i + CHANNELS_PER_SWATH]
        for i in range(0, len(odd_scans), CHANNELS_PER_SWATH)
    ]

    cscan_raw     = load_cscan_amplitudes(cscan_01_files)
    median_length = _median_swath_length(pos_files)
    print(f"[ANALYSIS] median GPS swath length: {median_length:.1f}m")

    ts_data: dict = {
        'n_swaths':  len(swath_groups),
        'rebar_cm':  [],
        'bottom_cm': [],
        'rebar_in':  [],
        'easting':   [],
        'northing':  [],
        'depths_in': [],
    }
    total_traces = 0

    for swath_idx, swath_scans in enumerate(swath_groups):
        pos_path = pos_files[swath_idx] if swath_idx < len(pos_files) else None
        pos_df   = parse_pos_file(pos_path) if pos_path else None

        for ch_idx, scan_path in enumerate(swath_scans):
            base = os.path.basename(scan_path)
            try:
                result = read_proceq(scan_path, pos_path)
            except Exception as exc:
                print(f"[ANALYSIS] SKIP {base}: {exc}")
                continue

            traces   = result["traces"]
            n_traces = len(traces)
            total_traces += n_traces

            depth_result = extract_amplitude_and_depth(
                traces, search_start=search_start, search_end=search_end, epsr=epsr,
            )

            east, north = get_trace_gps(pos_df, n_traces)
            if east is not None:
                ts_data['easting'].append(east)
                ts_data['northing'].append(north)
                print(f"  [GPS] swath {swath_idx+1} ch {ch_idx+1}: "
                      f"real GPS {east.min():.1f}–{east.max():.1f}E")
            else:
                print(f"  [GPS] swath {swath_idx+1} ch {ch_idx+1}: "
                      f"no GPS, skipping from depth map")
                ts_data['easting'].append(None)
                ts_data['northing'].append(None)
            ts_data['depths_in'].append(depth_result['depths_in'])

            if ch_idx == 0:
                bottom = extract_amplitude_and_depth(
                    traces, search_start=200, search_end=400, epsr=epsr,
                )
                ts_data['rebar_cm'].append(depth_result['depths_m'] * 100)
                ts_data['bottom_cm'].append(bottom['depths_m'] * 100)
                ts_data['rebar_in'].append(depth_result['depths_in'])

            print(f"[ANALYSIS]   swath {swath_idx+1:02d} ch {ch_idx+1}: "
                  f"{n_traces} traces  depth {depth_result['depths_in'].mean():.2f}\"")

            # Release the (n_traces, 510) raw trace array now that depth picking
            # is done — the rest of the loop only needs the small per-trace
            # depth/easting/northing scalars accumulated in ts_data.
            del result, traces, depth_result
            gc.collect()

    if not ts_data['easting']:
        print("[ANALYSIS] No data — aborting")
        return None

    print(f"[ANALYSIS] total_traces={total_traces:,}")

    # Output 1 — horizon picks
    build_horizon_picks_plot(odd_scans, ts_data, output_dir)

    # Output 2 — rebar depth map (real GPS coords, signal-proc depths; no-GPS swaths dropped)
    valid_mask  = [e is not None for e in ts_data['easting']]
    east_valid_chunks  = [e for e in ts_data['easting']  if e is not None]
    north_valid_chunks = [n for n in ts_data['northing'] if n is not None]
    depth_valid_chunks = [d for d, v in zip(ts_data['depths_in'], valid_mask) if v]
    east_valid  = np.concatenate(east_valid_chunks)  if east_valid_chunks  else np.array([])
    north_valid = np.concatenate(north_valid_chunks) if north_valid_chunks else np.array([])
    depth_valid = np.concatenate(depth_valid_chunks) if depth_valid_chunks else np.array([])
    if len(east_valid) > 100:
        from analysis import build_model_depth_map
        build_model_depth_map(east_valid, north_valid, depth_valid,
                              output_dir, title='Rebar Depth Map')
        print(f"[DEPTH MAP] built from {len(east_valid):,} GPS-valid traces")
    else:
        print("[DEPTH MAP] insufficient GPS traces, skipping depth map")

    # Output 3 — corrosion map
    cscan_slices  = [s for s in cscan_raw if s is not None]
    high_risk_pct = 0.0
    if cscan_slices:
        cscan_result  = build_cscan_maps(cscan_slices, output_dir)
        high_risk_pct = cscan_result['high_risk_pct']

    print("[ANALYSIS] all outputs written to", output_dir)
    all_rebar_in = np.concatenate(ts_data['rebar_in']) if ts_data['rebar_in'] else np.array([])
    return {
        'n_traces':      total_traces,
        'mean_depth_in': float(all_rebar_in.mean()) if len(all_rebar_in) > 0 else 0.0,
        'high_risk_pct': high_risk_pct,
    }
