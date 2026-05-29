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


def preprocess_for_display(traces, frequency_mhz: float = 2000):
    """
    Apply standard GPR signal processing for B-scan display, matching the
    output quality of Geolitix / GSSI RADAN / Proceq OneVision.

    Steps:
      1. DC removal (dewow) via running-mean subtraction per trace
      2. Time-zero correction via Hilbert-envelope surface pick + shift
      3. Linear gain with depth (1× → 4×)
      4. Bandpass filter (4th-order Butterworth, 0.02–0.8 of Nyquist)
      5. AGC — per-trace RMS normalization
      6. Global contrast stretch — 1st/99th percentile clip → [-1, 1]
    """
    from scipy import signal as sp_signal
    from scipy.signal import hilbert

    raw = traces.astype(np.float32)
    n_in, n_samples = raw.shape
    print(f"[PREPROCESS] input shape={raw.shape} "
          f"range=[{float(raw.min()):.4g}, {float(raw.max()):.4g}] "
          f"mean={float(raw.mean()):.4g} std={float(raw.std()):.4g}", flush=True)

    # Dead-trace filter. Calibration / dead traces (constant or near-constant)
    # poison the global percentile stretch in Step 6 by yanking p1/p99 toward
    # their plateau value. Spec calls for std<1.0, which is correct for
    # unscaled DZT readgssi output (millions-range) but would drop every
    # trace for Proceq data already normalized to [-1, 1] on ingest. Gate on
    # the input scale and fall back to a relative floor.
    trace_stds = raw.std(axis=1)
    amp_max = float(np.abs(raw).max())
    if amp_max > 100.0:
        std_threshold = 1.0
    else:
        std_threshold = max(float(raw.std()) * 0.01, 1e-6)
    valid_mask = trace_stds >= std_threshold
    n_valid = int(valid_mask.sum())
    print(f"[PREPROCESS] trace stdev range=[{float(trace_stds.min()):.4g}, "
          f"{float(trace_stds.max()):.4g}], threshold={std_threshold:.4g}, "
          f"dropping {n_in - n_valid} of {n_in} traces", flush=True)
    if n_valid < 10:
        raise ValueError(
            f"only {n_valid} valid traces remain after std<{std_threshold:.4g} "
            f"filter (need ≥10) — input appears to be all-zero, calibration-only, "
            f"or corrupted"
        )
    processed = raw[valid_mask].copy()
    n_traces = processed.shape[0]

    # Explicit per-trace DC mean subtraction (axis=1, keepdims=True). Belt-and-
    # suspenders before the sliding-window dewow that follows — guarantees the
    # whole trace is centred on zero regardless of dewow window edge effects.
    processed = processed - processed.mean(axis=1, keepdims=True)

    # Step 1 — DC removal (dewow): subtract a sliding mean per trace to kill
    # low-frequency drift / DC bias without flattening reflectors.
    window = max(n_samples // 8, 10)
    kernel = np.ones(window) / window
    for i in range(n_traces):
        dc = np.convolve(processed[i], kernel, mode="same")
        processed[i] -= dc

    # Step 2 — Time-zero correction: find the first strong arrival per trace
    # via Hilbert envelope inside the first quarter, then shift each trace so
    # surface reflections line up at the median pick.
    search_end = n_samples // 4
    surface_picks = []
    for i in range(n_traces):
        env = np.abs(hilbert(processed[i, :search_end]))
        surface_picks.append(int(np.argmax(env)))
    t0_ref = int(np.median(surface_picks))
    corrected = np.zeros_like(processed)
    for i in range(n_traces):
        shift = t0_ref - surface_picks[i]
        if shift > 0:
            corrected[i, shift:] = processed[i, :n_samples - shift]
        elif shift < 0:
            corrected[i, :n_samples + shift] = processed[i, -shift:]
        else:
            corrected[i] = processed[i]
    processed = corrected

    # Step 3 — Linear gain with depth: compensate signal attenuation, boosting
    # deeper samples up to 4× to keep late reflectors visible.
    gain = np.linspace(1.0, 4.0, n_samples).astype(np.float32)
    processed *= gain[np.newaxis, :]

    # Step 4 — Bandpass: drop DC residual and high-frequency noise outside the
    # GPR signal band. Best-effort — short / pathological traces may fail
    # filtfilt's padding requirement; pass-through on error.
    try:
        b, a = sp_signal.butter(4, [0.02, 0.8], btype="band")
        for i in range(n_traces):
            processed[i] = sp_signal.filtfilt(b, a, processed[i])
    except Exception:
        pass

    # Step 5 — AGC: divide each trace by its RMS so weak reflectors come up to
    # parity with strong ones (independent per A-scan).
    for i in range(n_traces):
        rms = float(np.sqrt(np.mean(processed[i] ** 2)))
        if rms > 1e-10:
            processed[i] /= rms

    # Step 6 — Global contrast stretch: 1st/99th percentile clip mapped to
    # [-1, 1] for matplotlib's vmin/vmax range.
    p1, p99 = np.percentile(processed, 1), np.percentile(processed, 99)
    processed = np.clip(processed, p1, p99)
    if p99 > p1:
        processed = (processed - p1) / (p99 - p1) * 2 - 1
    else:
        processed = np.zeros_like(processed)

    print(f"[PREPROCESS] output shape={processed.shape} "
          f"range=[{float(processed.min()):.4g}, {float(processed.max()):.4g}] "
          f"mean={float(processed.mean()):.4g} std={float(processed.std()):.4g}", flush=True)
    return processed


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

    # Full 6-step GPR preprocessing — dewow, time-zero alignment, depth gain,
    # bandpass, per-trace AGC, then a global percentile contrast stretch into
    # [-1, 1]. See preprocess_for_display() for details.
    display = preprocess_for_display(traces)

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
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close("all")
    gc.collect()
    print(f"[BSCAN] saved → {output_path}", flush=True)
    return output_path


def encode_traces_for_frontend(traces, max_traces: int = 500, max_samples: int = 256) -> dict:
    """
    Run the full preprocess_for_display() pipeline, then downsample,
    int8-quantize, zlib-compress, and base64-encode so the frontend canvas
    can render it cheaply at parity with the PNG B-scan.

    Encoding: zlib+base64+int8. Decode order (browser side):
      base64 → zlib decompress → Int8Array shape (n_traces, n_samples).
    """
    import base64
    import zlib

    # Apply the same 6-step GPR signal processing the PNG uses so the canvas
    # viewer matches what analysts see in the static B-scan image.
    processed = preprocess_for_display(traces)

    n_traces, n_samples = processed.shape
    if n_traces > max_traces:
        idx = np.linspace(0, n_traces - 1, max_traces, dtype=int)
        processed = processed[idx]
    if n_samples > max_samples:
        idx = np.linspace(0, n_samples - 1, max_samples, dtype=int)
        processed = processed[:, idx]

    t_min, t_max = float(processed.min()), float(processed.max())
    if t_max > t_min:
        normalized = ((processed - t_min) / (t_max - t_min) * 254 - 127).astype(np.int8)
    else:
        normalized = np.zeros_like(processed, dtype=np.int8)

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
