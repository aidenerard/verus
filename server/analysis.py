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
    "load_cscan_amplitudes",
    "build_cscan_maps",
    "process_proceq_dataset",
]
