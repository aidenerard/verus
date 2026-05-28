"""
server/analysis_bscan.py
B-scan rendering for the Proceq pipeline. Split out of analysis.py to keep
both modules under the 300-line cap. Output style targets Proceq OneVision /
GSSI RADAN / Screening Eagle conventions: grayscale amplitude (white=+, black=-),
travel-time axis on left, depth-in-inches on right, distance on bottom.
"""
from __future__ import annotations

import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def build_bscan_image(
    traces,              # np.ndarray (n_traces, n_samples) normalized float32
    output_path,
    title: str = "B-Scan",
    epsr: float = 9.0,
    time_range_ns: float = 16.0,
    picks_in=None,       # optional np.ndarray (n_traces,) — rebar picks in inches
    swath_idx: int = 0,
    dx: float = 0.00517, # meters per trace
) -> str:
    """
    Render a Proceq/GSSI-style B-scan PNG and return the output path.
    Grayscale (vmin=-1, vmax=1) with optional red rebar-horizon overlay.
    """
    n_traces, n_samples = traces.shape
    velocity     = 0.15 / np.sqrt(epsr)            # m/ns in concrete
    max_depth_m  = (time_range_ns * velocity) / 2.0
    max_depth_in = max_depth_m * 39.3701
    total_dist_m = n_traces * dx

    # Contrast stretch via 2nd–98th percentile clip, then normalize to [-1, 1]
    p2, p98 = np.percentile(traces, 2), np.percentile(traces, 98)
    display = np.clip(traces, p2, p98)
    max_abs = max(abs(p2), abs(p98))
    if max_abs > 0:
        display = display / max_abs

    fig, ax = plt.subplots(figsize=(20, 6))
    fig.patch.set_facecolor("white")

    # Transpose so traces are columns (distance on X, time on Y)
    ax.imshow(
        display.T,
        aspect="auto", cmap="gray", vmin=-1, vmax=1, origin="upper",
        extent=[0, total_dist_m, time_range_ns, 0],
        interpolation="bilinear",
    )

    if picks_in is not None and len(picks_in) > 0:
        picks_m  = np.asarray(picks_in) / 39.3701
        picks_ns = picks_m / velocity * 2.0
        x_axis   = np.linspace(0, total_dist_m, len(picks_ns))
        ax.plot(x_axis, picks_ns, color="#FF4400", linewidth=1.2,
                alpha=0.85, label="Rebar horizon")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7, facecolor="white")

    ax.set_ylabel("Two-way travel time (ns)", fontsize=10)
    ax.set_xlabel("Distance (m)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(time_range_ns, 0)  # time increases downward

    # Right-side depth axis (inches)
    ax2 = ax.twinx()
    ax2.set_ylim(max_depth_in, 0)
    ax2.set_ylabel("Depth (inches)", fontsize=10)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f"'))

    # Tick the time axis at 1-inch depth multiples
    depth_ticks_in = np.arange(0, max_depth_in + 1, 1.0)
    depth_ticks_ns = depth_ticks_in / 39.3701 / velocity * 2.0
    ax.set_yticks(depth_ticks_ns)
    ax.set_yticklabels([f"{ns:.1f}" for ns in depth_ticks_ns], fontsize=8)
    ax.grid(True, axis="y", alpha=0.2, color="white", linewidth=0.5)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.tick_params(axis="both", which="both", direction="in", top=True, labelsize=8)

    plt.tight_layout()
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close("all")
    gc.collect()
    print(f"[BSCAN] saved → {output_path}", flush=True)
    return output_path


def render_swath_bscan(traces, depth_result, output_dir: str,
                      swath_idx: int, epsr: float) -> str | None:
    """One-line helper for the swath loop. Best-effort — returns None on failure."""
    try:
        out = os.path.join(output_dir, f"bscan_swath{swath_idx + 1:02d}.png")
        return build_bscan_image(
            traces=traces, output_path=out,
            title=f"B-Scan — Swath {swath_idx + 1}, Channel 1",
            epsr=epsr, time_range_ns=16.0,
            picks_in=depth_result.get("depths_in") if depth_result else None,
            swath_idx=swath_idx,
        )
    except Exception as exc:
        print(f"[BSCAN] swath {swath_idx + 1} failed: {exc}", flush=True)
        return None
