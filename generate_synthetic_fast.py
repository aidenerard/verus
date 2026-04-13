"""
generate_synthetic_fast.py
──────────────────────────
Generates synthetic GPR bridge-deck A-scan signals using a physics-based
Ricker wavelet model.  Runs on CPU with numpy only — no GPU, no gprMax.
Target: <60 seconds for 50,000 signals.

Output format (exact SDNET2021 layout — loadable by cnn.py without changes):
  Each file is a (521, n_signals+1) integer CSV:
    Row 0, col 4     : n_signals
    Row 7, cols 1..n : labels  (1 = sound,  2 = delaminated)
    Rows 9..520      : amplitude data — col 0 = 0 (placeholder),
                       cols 1..n = A-scan amplitudes as uint16 counts
                       (DC offset 32,768, peak ±3,000 counts)
  Files are split into chunks of MAX_SIGS_PER_FILE (≤1,000) and saved as:
    synthetic_data/synthetic_sound/FILE____001.csv  …  (sound)
    synthetic_data/synthetic_delam/FILE____001.csv  …  (delaminated)

  Labels as read by cnn.py load_csv:
    raw_labels = row7[1:n+1]
    labels = (raw_labels == 1).astype(int)   ← 1→sound(1), 2→delaminated(0)
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = Path("~/Desktop/verus/synthetic_data").expanduser()

# ── Constants ──────────────────────────────────────────────────────────────────
N_SAMPLES          = 512
T_MAX_NS           = 12.0
T                  = np.linspace(0, T_MAX_NS, N_SAMPLES)    # (512,) ns
DC_OFFSET          = 32_768
MAX_COUNTS         = 3_000
MAX_SIGS_PER_FILE  = 1_000    # match SDNET2021 file-size convention

N_CLASS1   = 30_000    # sound signals
N_CLASS2   = 20_000    # delaminated signals
BATCH_SIZE = 1_000     # generation batch size

RNG = np.random.default_rng(42)


# ── Physics helpers ────────────────────────────────────────────────────────────

def ricker_batch(t_centers: np.ndarray,
                 amplitudes: np.ndarray,
                 f0: np.ndarray) -> np.ndarray:
    """
    Vectorised Ricker wavelet for a batch of signals.

    Args:
        t_centers  : (B,) arrival times in ns
        amplitudes : (B,) peak amplitudes (may be negative for phase reversal)
        f0         : (B,) antenna frequency in GHz

    Returns:
        (B, N_SAMPLES) float64 wavelet array

    Note: GHz × ns = dimensionless — no unit conversion needed.
    """
    tau = T[np.newaxis, :] - t_centers[:, np.newaxis]      # (B, 512) ns
    u   = (np.pi * f0[:, np.newaxis] * tau) ** 2           # dimensionless
    return (1.0 - 2.0 * u) * np.exp(-u) * amplitudes[:, np.newaxis]


def generate_batch(n_sound: int, n_delam: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one batch of signals.

    Args:
        n_sound : number of sound signals (label 0)
        n_delam : number of delaminated signals (label 1)

    Returns:
        signals : (n_sound + n_delam, N_SAMPLES) int32 with DC offset
        labels  : (n_sound + n_delam,) int32, 1=sound / 2=delaminated (SDNET convention)
    """
    B = n_sound + n_delam
    if B == 0:
        return np.empty((0, N_SAMPLES), dtype=np.int32), np.empty(0, dtype=np.int32)

    # ── Shared randomised physical parameters ─────────────────────────────────
    eps_r   = RNG.uniform(6.0, 10.0, B)
    v_mps   = 3e8 / np.sqrt(eps_r)                         # wave velocity (m/s)
    f0      = RNG.uniform(1.0, 2.5, B)                     # antenna frequency (GHz)
    d_rebar = RNG.uniform(0.05, 0.12, B)                   # rebar depth (m)
    alpha   = RNG.uniform(0.1, 0.4, B)                     # attenuation coefficient

    t_rebar_ns = 2.0 * d_rebar / v_mps * 1e9               # two-way travel time (ns)

    # ── Build signal waveforms (float, zero-centred) ──────────────────────────
    sig = np.zeros((B, N_SAMPLES), dtype=np.float64)

    # 1. Direct wave at 0.5 ns, amplitude = 1.0 (fixed, reference)
    sig += ricker_batch(np.full(B, 0.5), np.ones(B), f0)

    # 2. Rebar reflection at t_rebar
    rebar_amp = np.exp(-alpha * t_rebar_ns) * RNG.uniform(0.3, 0.8, B)
    sig      += ricker_batch(t_rebar_ns, rebar_amp, f0)

    # 3. Bottom-of-deck reflection
    d_bottom    = RNG.uniform(0.15, 0.30, B)               # deck thickness (m)
    t_bottom_ns = 2.0 * d_bottom / v_mps * 1e9
    bottom_amp  = np.exp(-alpha * t_bottom_ns) * RNG.uniform(0.1, 0.3, B)
    sig        += ricker_batch(t_bottom_ns, bottom_amp, f0)

    # 4. Gaussian noise (proportional to peak amplitude)
    peak_amp  = np.abs(sig).max(axis=1)
    noise_std = RNG.uniform(0.01, 0.04, B) * peak_amp
    sig      += RNG.standard_normal((B, N_SAMPLES)) * noise_std[:, np.newaxis]

    # 5. Delamination reflection (delaminated signals only) ────────────────────
    if n_delam > 0:
        ds = n_sound           # start index of delaminated block

        # Delamination sits between direct wave and rebar, shallower than rebar
        d_delam    = RNG.uniform(0.02, d_rebar[ds:] * 0.7)
        t_delam_ns = 2.0 * d_delam / v_mps[ds:] * 1e9

        # Delamination is stronger than rebar (higher coefficient, shorter path)
        delam_amp  = (np.exp(-alpha[ds:] * t_delam_ns)
                      * RNG.uniform(0.4, 0.9, n_delam))

        # Phase reversal: -1 = air-gap delamination, +1 = water/debris gap
        phase      = np.where(RNG.random(n_delam) < 0.5, -1.0, 1.0)
        delam_amp *= phase

        sig[ds:] += ricker_batch(t_delam_ns, delam_amp, f0[ds:])

    # ── Post-processing: scale → ±3000 counts, add DC offset 32768 ────────────
    peak_abs = np.abs(sig).max(axis=1, keepdims=True)
    peak_abs = np.where(peak_abs < 1e-12, 1.0, peak_abs)   # guard zero-energy
    scaled   = sig / peak_abs * MAX_COUNTS
    output   = np.round(scaled).astype(np.int32) + DC_OFFSET
    output   = np.clip(output, 0, 65535)

    # SDNET label convention: 1=sound, 2=delaminated
    labels = np.concatenate([
        np.ones(n_sound, dtype=np.int32),       # sound      → 1
        np.full(n_delam, 2, dtype=np.int32),    # delaminated → 2
    ])
    return output, labels


# ── Main generation loop ───────────────────────────────────────────────────────

def generate_class(n_total: int, is_delaminated: bool,
                   label_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Generate n_total signals of one class in BATCH_SIZE chunks."""
    X_chunks, y_chunks = [], []
    done = 0
    while done < n_total:
        b = min(BATCH_SIZE, n_total - done)
        xb, yb = generate_batch(
            n_sound=0 if is_delaminated else b,
            n_delam=b if is_delaminated else 0,
        )
        X_chunks.append(xb)
        y_chunks.append(yb)
        done += b
        print(f"  {label_name:12}  [{done:>7,} / {n_total:>7,}]", flush=True)
    return np.concatenate(X_chunks), np.concatenate(y_chunks)


# ── SDNET2021 file writer ──────────────────────────────────────────────────────

def save_as_sdnet(signals: np.ndarray, labels: np.ndarray,
                  out_subdir: Path) -> int:
    """
    Save signals in exact SDNET2021 CSV format loadable by cnn.py.

    File layout  (521 rows × (n+1) cols):
      Row 0   : [0, 0, 0, 0, n, 0, ...]   ← n = n_signals in this file
      Rows 1-6: all zeros
      Row 7   : [0, lbl_1, lbl_2, ..., lbl_n]
      Row 8   : all zeros
      Rows 9-520: amplitude columns — col 0 = 0, cols 1..n = A-scan amps
                  shape (512 time samples) × n signals (transposed from signals)

    Args:
        signals   : (N, 512) int32 with DC offset
        labels    : (N,) int32, 1=sound or 2=delaminated
        out_subdir: directory to write FILE____NNN.csv files into

    Returns:
        Number of files written.
    """
    out_subdir.mkdir(parents=True, exist_ok=True)
    N = len(signals)
    n_files = 0

    for start in range(0, N, MAX_SIGS_PER_FILE):
        end      = min(start + MAX_SIGS_PER_FILE, N)
        sigs_blk = signals[start:end]   # (n, 512)
        lbl_blk  = labels[start:end]    # (n,)
        n        = len(sigs_blk)

        # Build (521, n+1) integer grid
        grid = np.zeros((521, n + 1), dtype=np.int32)

        # Row 0, col 4: number of signals
        grid[0, 4] = n

        # Row 7, cols 1..n: labels
        grid[7, 1 : n + 1] = lbl_blk

        # Rows 9..520: amplitude data
        # sigs_blk is (n, 512); transposed → (512, n); write to cols 1..n
        grid[9 : 9 + N_SAMPLES, 1 : n + 1] = sigs_blk.T.astype(np.int32)

        n_files += 1
        fname = out_subdir / f"FILE____{n_files:03d}.csv"
        np.savetxt(fname, grid, delimiter=",", fmt="%d")

    return n_files


# ── Validation helpers ─────────────────────────────────────────────────────────

def early_window_energy(X: np.ndarray, t_cutoff_ns: float = 4.0) -> np.ndarray:
    """Mean squared amplitude in the t < t_cutoff_ns window."""
    cutoff_idx = int(t_cutoff_ns / T_MAX_NS * N_SAMPLES)
    sig_centered = X[:, :cutoff_idx].astype(np.float64) - DC_OFFSET
    return (sig_centered ** 2).mean(axis=1)


def print_signal_stats(X: np.ndarray, label: str, n_samples: int = 10) -> None:
    """Print mean and std for n_samples random signals from X."""
    idx = RNG.choice(len(X), n_samples, replace=False)
    sigs = X[idx].astype(np.float64) - DC_OFFSET   # centre at zero
    print(f"\n  {label}  ({n_samples} random signals, centred at DC=0):")
    print(f"    Mean across samples: {sigs.mean(axis=1).mean():>8.2f}  "
          f"std of means: {sigs.mean(axis=1).std():>8.2f}")
    print(f"    Std  across samples: {sigs.std(axis=1).mean():>8.2f}  "
          f"std of stds : {sigs.std(axis=1).std():>8.2f}")


def plot_samples(X1: np.ndarray, X2: np.ndarray, out_path: Path) -> None:
    """2×3 grid: top row = 3 random Class 1 (sound), bottom = 3 Class 2 (delaminated)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), facecolor="white")
    fig.suptitle("Synthetic GPR A-scan Waveforms", fontsize=12, fontweight="bold")

    idx1 = RNG.choice(len(X1), 3, replace=False)
    idx2 = RNG.choice(len(X2), 3, replace=False)

    for col, (i1, i2) in enumerate(zip(idx1, idx2)):
        ax = axes[0, col]
        ax.plot(T, X1[i1].astype(np.float64) - DC_OFFSET,
                color="#1a2f5a", linewidth=0.8)
        ax.set_title(f"Sound #{i1}", fontsize=9)
        ax.set_xlim(0, T_MAX_NS)
        ax.set_ylabel("Amplitude (counts)" if col == 0 else "")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.tick_params(labelsize=7)

        ax = axes[1, col]
        ax.plot(T, X2[i2].astype(np.float64) - DC_OFFSET,
                color="#c0392b", linewidth=0.8)
        ax.set_title(f"Delaminated #{i2}", fontsize=9)
        ax.set_xlim(0, T_MAX_NS)
        ax.set_xlabel("Time (ns)" if col == 1 else "")
        ax.set_ylabel("Amplitude (counts)" if col == 0 else "")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Waveform plot saved → {out_path}", flush=True)


# ── Smoke test: verify one saved file round-trips through cnn.py load_csv ─────

def smoke_test_roundtrip(out_subdir: Path, expected_sdnet_label: int) -> None:
    """
    Load the first FILE____001.csv back with the same logic as cnn.py load_csv
    and assert the shape and label values are correct.
    """
    import pandas as pd
    fpath = out_subdir / "FILE____001.csv"
    if not fpath.exists():
        print("  [smoke test] FILE____001.csv not found — skipping", flush=True)
        return

    raw       = pd.read_csv(fpath, header=None).values
    n_sigs    = int(raw[0, 4])
    raw_lbls  = raw[7, 1 : n_sigs + 1].astype(int)
    data_row  = None
    for row in range(9, 14):
        try:
            val = float(raw[row, 0])
            if not np.isnan(val):
                data_row = row
                break
        except (ValueError, TypeError):
            continue
    if data_row is None:
        print("  [smoke test] FAIL — cannot locate amplitude rows", flush=True)
        return

    amp_block = raw[data_row : data_row + N_SAMPLES, 0 : n_sigs + 1].astype(np.float32)
    amps      = (amp_block[:, 1:] - DC_OFFSET).T          # (n, 512)

    ok_shape  = amp_block.shape == (N_SAMPLES, n_sigs + 1)
    ok_labels = np.all(raw_lbls == expected_sdnet_label)
    ok_amp    = np.abs(amps).max() <= MAX_COUNTS + 5      # tiny rounding margin

    print(f"\n  [smoke test] {fpath.name}", flush=True)
    print(f"    n_signals = {n_sigs}  {'✓' if n_sigs <= MAX_SIGS_PER_FILE else '✗'}", flush=True)
    print(f"    amp shape  = {amp_block.shape}  {'✓' if ok_shape else '✗'}", flush=True)
    print(f"    labels all = {expected_sdnet_label}  {'✓' if ok_labels else '✗'} "
          f"(sample: {raw_lbls[:5]})", flush=True)
    print(f"    amp range  = [{amps.min():.0f}, {amps.max():.0f}]  "
          f"{'✓' if ok_amp else '✗'}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("Verus Synthetic GPR Data Generator (V17 / CPU-only)", flush=True)
    print(f"  Sound signals     : {N_CLASS1:,}", flush=True)
    print(f"  Delaminated       : {N_CLASS2:,}", flush=True)
    print(f"  Total             : {N_CLASS1 + N_CLASS2:,}", flush=True)
    print(f"  Batch size        : {BATCH_SIZE:,}", flush=True)
    print(f"  Max sigs/file     : {MAX_SIGS_PER_FILE:,}", flush=True)
    print(f"  Output directory  : {OUT_DIR}", flush=True)
    print(f"  Format            : SDNET2021 (521×(n+1) per file)", flush=True)
    print("=" * 60, flush=True)

    t_start = time.perf_counter()

    # ── Generate ───────────────────────────────────────────────────────────────
    print(f"\n[1/2] Generating sound signals …", flush=True)
    X1, y1 = generate_class(N_CLASS1, is_delaminated=False, label_name="Sound")

    print(f"\n[2/2] Generating delaminated signals …", flush=True)
    X2, y2 = generate_class(N_CLASS2, is_delaminated=True,  label_name="Delaminated")

    t_gen = time.perf_counter() - t_start
    print(f"\n  Generation complete in {t_gen:.1f}s", flush=True)

    # ── Save in SDNET2021 format ───────────────────────────────────────────────
    sound_dir = OUT_DIR / "synthetic_sound"
    delam_dir = OUT_DIR / "synthetic_delam"

    print(f"\nSaving SDNET2021 files …", flush=True)
    n1 = save_as_sdnet(X1, y1, sound_dir)
    print(f"  Saved {n1} sound file(s)        → {sound_dir}/FILE____001.csv … "
          f"FILE____{n1:03d}.csv  [{X1.shape[0]:,} signals]", flush=True)

    n2 = save_as_sdnet(X2, y2, delam_dir)
    print(f"  Saved {n2} delaminated file(s)  → {delam_dir}/FILE____001.csv … "
          f"FILE____{n2:03d}.csv  [{X2.shape[0]:,} signals]", flush=True)

    # ── Validation ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("VALIDATION", flush=True)
    print(f"{'─'*60}", flush=True)

    print_signal_stats(X1, "Class 1 — Sound")
    print_signal_stats(X2, "Class 2 — Delaminated")

    # Early-window energy check (t < 4 ns)
    e1 = early_window_energy(X1, t_cutoff_ns=4.0)
    e2 = early_window_energy(X2, t_cutoff_ns=4.0)
    print(f"\n  Early-window energy  (t < 4 ns, mean²):", flush=True)
    print(f"    Class 1 (sound)       : mean = {e1.mean():>10.1f}  "
          f"std = {e1.std():>10.1f}", flush=True)
    print(f"    Class 2 (delaminated) : mean = {e2.mean():>10.1f}  "
          f"std = {e2.std():>10.1f}", flush=True)
    ratio = e2.mean() / e1.mean() if e1.mean() > 0 else float("nan")
    if ratio > 1.0:
        print(f"    ✓ Delaminated early-window energy is {ratio:.2f}× higher "
              f"(delamination reflection confirmed)", flush=True)
    else:
        print(f"    ✗ WARNING: delaminated energy not higher than sound "
              f"(ratio={ratio:.2f}). Check physics parameters.", flush=True)

    # Round-trip smoke tests
    smoke_test_roundtrip(sound_dir, expected_sdnet_label=1)
    smoke_test_roundtrip(delam_dir, expected_sdnet_label=2)

    # Waveform plot
    plot_samples(X1, X2, OUT_DIR / "sample_waveforms.png")

    # ── Summary ────────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"  Total time  : {t_total:.1f}s", flush=True)
    print(f"  Throughput  : {(N_CLASS1+N_CLASS2)/t_total:,.0f} signals/second", flush=True)
    print(f"  {'✓ UNDER 60s' if t_total < 60 else '✗ EXCEEDED 60s'}  "
          f"({t_total:.1f}s)", flush=True)
    print(f"  Sound files     : {n1}  ({X1.shape[0]:,} signals in {sound_dir.name}/)", flush=True)
    print(f"  Delam files     : {n2}  ({X2.shape[0]:,} signals in {delam_dir.name}/)", flush=True)
    print("=" * 60, flush=True)
