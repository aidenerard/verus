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

    # Per-trace DC removal (subtract each A-scan's own mean) so reflector
    # polarity is centered around zero regardless of any DC offset on that
    # individual trace.
    dc_removed = traces - traces.mean(axis=1, keepdims=True)

    # Per-trace 2nd/98th percentile contrast stretch — each A-scan is
    # independently mapped so its p2 → -1 and p98 → +1. Every column in the
    # rendered B-scan then fills the full -1..1 dynamic range, matching the
    # per-trace AGC look of pro GPR software (Proceq OneVision, GSSI RADAN,
    # ReflexW). Trade-off: this hides systematic amplitude attenuation along
    # the scan — fine for delamination/rebar interpretation, less appropriate
    # for absolute reflector strength comparisons across distance.
    p2  = np.percentile(dc_removed, 2,  axis=1, keepdims=True)
    p98 = np.percentile(dc_removed, 98, axis=1, keepdims=True)
    span = p98 - p2
    span_safe = np.where(span > 0, span, 1.0)
    display = np.clip((dc_removed - p2) / span_safe * 2 - 1, -1, 1)
    # Flat traces (constant signal) would saturate to black under the above;
    # collapse them to neutral mid-gray instead.
    display[(span <= 0)[:, 0], :] = 0.0

    fig, ax = plt.subplots(figsize=(20, 6))
    fig.patch.set_facecolor("white")

    # Transpose so traces are columns (distance on X, time on Y). nearest
    # interpolation gives a crisp, pixel-accurate B-scan instead of the soft
    # bilinear blend matplotlib uses by default for non-uniform extents.
    ax.imshow(
        display.T,
        aspect="auto", cmap="gray", vmin=-1, vmax=1, origin="upper",
        extent=[0, total_dist_m, time_range_ns, 0],
        interpolation="nearest",
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


def encode_traces_for_frontend(traces, max_traces: int = 500, max_samples: int = 256) -> dict:
    """
    Downsample + int8-quantize + zlib-compress + base64-encode a B-scan trace
    array so the frontend canvas can render it cheaply.

    Encoding: zlib+base64+int8. Decode order (browser side):
      base64 → zlib decompress → Int8Array shape (n_traces, n_samples).
    """
    import base64
    import zlib

    n_traces, n_samples = traces.shape
    if n_traces > max_traces:
        idx = np.linspace(0, n_traces - 1, max_traces, dtype=int)
        traces = traces[idx]
    if n_samples > max_samples:
        idx = np.linspace(0, n_samples - 1, max_samples, dtype=int)
        traces = traces[:, idx]

    t_min, t_max = float(traces.min()), float(traces.max())
    if t_max > t_min:
        normalized = ((traces - t_min) / (t_max - t_min) * 254 - 127).astype(np.int8)
    else:
        normalized = np.zeros_like(traces, dtype=np.int8)

    compressed = zlib.compress(normalized.tobytes(), level=6)
    return {
        "data":      base64.b64encode(compressed).decode("ascii"),
        "n_traces":  int(normalized.shape[0]),
        "n_samples": int(normalized.shape[1]),
        "encoding":  "zlib+base64+int8",
    }


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
