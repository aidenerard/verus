"""
GPR Bridge Deck Inspection Pipeline
SDNET2021 dataset — ASTM D6087-inspired A/A0 classification

Data notes for this dataset:
  - DC offset: amplitudes stored as 16-bit unsigned centered at 32768
  - Surface reflection: ~2.5 ns
  - Class-1 (sound) rebar reflection: ~7.5–8.5 ns
  - Class-2 delamination reflection:  ~5.9–7.0 ns  [earlier than Class-1 rebar]
  → Rebar window widened to 3-11 ns; the FIRST significant peak (> 20% of
    surface amplitude) is used rather than the largest, so delamination
    reflectors at 5-7 ns are found before the deeper rebar in sound signals.
  → Class-2 A/A0 is LOWER than Class-1 (shallower reflector → smaller
    geometric-spreading correction). D6087 flags LOW A/A0 as delaminated.

Usage:
    python pipeline.py            # single test file
    python pipeline.py --all      # all files across all bridges
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import find_peaks, hilbert

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path("~/Desktop/verus/gpr_data").expanduser()

# ── Time windows (ns) ──────────────────────────────────────────────────────
SURF_WIN  = (0.0, 3.0)   # first surface reflection
REBAR_WIN = (3.0, 11.0)  # rebar / delamination reflection window

# ── D6087 thresholds relative to Class-1 median A/A0 ──────────────────────
# Low A/A0 → delaminated (delamination reflector is shallower → smaller
# geometric-spreading correction → smaller corrected ratio).
THRESH_SUSPECT      = 0.75   # A/A0 < 0.75 × median_class1 → suspect
THRESH_DETERIORATED = 0.50   # A/A0 < 0.50 × median_class1 → deteriorated

DC_OFFSET = 32768   # 16-bit GPR data centred at 2^15


# ── Peak detection ──────────────────────────────────────────────────────────

def _find_peak_in_window(time_ns: np.ndarray, signal: np.ndarray,
                         t_lo: float, t_hi: float) -> tuple[float, float]:
    """
    Return (peak_time, |peak_amplitude|) for the largest absolute peak in
    [t_lo, t_hi].  Falls back to argmax when find_peaks finds nothing.
    Used for the surface window.
    """
    mask = (time_ns >= t_lo) & (time_ns <= t_hi)
    if not mask.any():
        return np.nan, np.nan

    region   = signal[mask]
    region_t = time_ns[mask]
    abs_r    = np.abs(region)

    if np.all(np.isnan(abs_r)):
        return np.nan, np.nan

    prom_threshold = 0.05 * np.nanmax(abs_r)
    peaks, _ = find_peaks(np.nan_to_num(abs_r), prominence=prom_threshold)

    best = peaks[np.argmax(abs_r[peaks])] if len(peaks) > 0 else np.nanargmax(abs_r)
    return float(region_t[best]), float(abs_r[best])


def _find_first_significant_rebar_peak(time_ns: np.ndarray, signal: np.ndarray,
                                       t_lo: float, t_hi: float,
                                       min_amp: float) -> tuple[float, float]:
    """
    Return (peak_time, |peak_amplitude|) for the FIRST local maximum in
    [t_lo, t_hi] whose absolute amplitude exceeds min_amp (20% of the surface
    reflection amplitude).  Falls back to the largest peak if none qualify.
    """
    mask = (time_ns >= t_lo) & (time_ns <= t_hi)
    if not mask.any():
        return np.nan, np.nan

    region   = signal[mask]
    region_t = time_ns[mask]
    abs_r    = np.abs(region)

    if np.all(np.isnan(abs_r)):
        return np.nan, np.nan

    prom_threshold = 0.05 * np.nanmax(abs_r)
    peaks, _ = find_peaks(np.nan_to_num(abs_r), prominence=prom_threshold)

    if len(peaks) == 0:
        peaks = np.array([np.nanargmax(abs_r)])

    # First peak (lowest index = earliest time) that exceeds the amplitude threshold
    significant = peaks[abs_r[peaks] >= min_amp]
    best = significant[0] if len(significant) > 0 else peaks[np.argmax(abs_r[peaks])]
    return float(region_t[best]), float(abs_r[best])


# ── A/A0 formula ────────────────────────────────────────────────────────────

def _compute_ratio(t_surf: float, a_surf: float,
                   t_reb:  float, a_reb:  float) -> float:
    """A/A0 = |A_rebar / A_surface| × (t_rebar / t_surface)²"""
    if np.any(np.isnan([t_surf, a_surf, t_reb, a_reb])):
        return np.nan
    if a_surf == 0 or t_surf == 0:
        return np.nan
    return (a_reb / a_surf) * (t_reb / t_surf) ** 2


# ── Hilbert envelope FWHM ───────────────────────────────────────────────────

def _compute_fwhm_ns(env_window: np.ndarray, dt: float) -> float:
    """
    Given a 1-D envelope array (absolute Hilbert values) covering a time
    window, find the peak and return its FWHM in nanoseconds.

    FWHM is measured by walking left and right from the peak index until the
    envelope drops below 50% of the peak value, then converting the sample
    count to ns via dt (sample interval in ns).
    """
    if len(env_window) == 0 or np.all(np.isnan(env_window)):
        return np.nan

    peak_idx  = int(np.nanargmax(env_window))
    half_max  = 0.5 * env_window[peak_idx]

    left = peak_idx
    while left > 0 and env_window[left - 1] >= half_max:
        left -= 1

    right = peak_idx
    while right < len(env_window) - 1 and env_window[right + 1] >= half_max:
        right += 1

    return (right - left) * dt


# ── Core file processor ─────────────────────────────────────────────────────

def process_file(filepath: Path) -> dict | None:
    filepath = Path(filepath)

    # 1. Load raw Excel (no header — all rows as-is)
    try:
        raw = pd.read_excel(filepath, header=None, engine="openpyxl")
    except Exception as e:
        print(f"ERROR reading {filepath.name}: {e}")
        return None

    # 2. Metadata (0-indexed rows)
    #    Row 0 col E → total signal count
    #    Row 5  → X positions, Row 6  → Y positions, Row 7  → class labels
    n_signals = int(raw.iloc[0, 4])
    labels    = raw.iloc[7, 1:n_signals + 1].values.astype(int)

    # 3. Amplitude block: rows 10–521, col 0 = time_ns, cols 1+ = signals
    amp_block = raw.iloc[10:522, 0:n_signals + 1].values.astype(float)
    time_ns   = amp_block[:, 0]
    amps      = amp_block[:, 1:] - DC_OFFSET   # shape (512, n_signals)

    # 4. Compute per-signal A/A0
    ratios = np.empty(n_signals, dtype=float)
    for i in range(n_signals):
        sig = amps[:, i]
        t_s, a_s = _find_peak_in_window(time_ns, sig, *SURF_WIN)
        t_r, a_r = _find_first_significant_rebar_peak(
            time_ns, sig, *REBAR_WIN, min_amp=0.20 * a_s
        )
        ratios[i] = _compute_ratio(t_s, a_s, t_r, a_r)

    # 5. D6087 classification
    #    Reference = median A/A0 of in-file Class-1 (sound) signals.
    #    Signals with LOW A/A0 are flagged as delaminated per D6087.
    class1_mask = labels == 1
    median_ref  = float(np.nanmedian(ratios[class1_mask]))

    thresh_susp = THRESH_SUSPECT      * median_ref   # below → suspect
    thresh_det  = THRESH_DETERIORATED * median_ref   # below → deteriorated

    predicted = np.where(
        ratios <= thresh_det,  "deteriorated",
        np.where(ratios <= thresh_susp, "suspect", "sound")
    )

    # 6. Band energy ratio (diagnostic feature)
    shallow_mask = (time_ns >= 5.0) & (time_ns <= 7.0)
    deep_mask    = (time_ns >= 7.0) & (time_ns <= 9.0)
    shallow_energy = np.nansum(amps[shallow_mask, :] ** 2, axis=0)  # shape (n_signals,)
    deep_energy    = np.nansum(amps[deep_mask,    :] ** 2, axis=0)
    band_ratios    = shallow_energy / (deep_energy + 1e-9)

    # 7. Hilbert envelope FWHM (diagnostic feature)
    dt           = float(time_ns[1] - time_ns[0])          # ~0.0234 ns/sample
    rebar_mask   = (time_ns >= 3.0) & (time_ns <= 11.0)
    envelope     = np.abs(hilbert(amps, axis=0))            # shape (512, n_signals)
    env_rebar    = envelope[rebar_mask, :]                  # shape (n_rebar_samples, n_signals)
    fwhm_ns      = np.array([_compute_fwhm_ns(env_rebar[:, i], dt) for i in range(n_signals)])

    # 8. Evaluation
    gt_delaminated = labels >= 2          # class 2 or 3
    pred_flagged   = predicted != "sound"

    n_gt_del            = int(gt_delaminated.sum())
    n_correctly_flagged = int((gt_delaminated & pred_flagged).sum())
    n_missed            = int((gt_delaminated & ~pred_flagged).sum())

    tn       = int((~gt_delaminated & ~pred_flagged).sum())
    accuracy = (n_correctly_flagged + tn) / n_signals * 100
    fnr      = (n_missed / n_gt_del * 100) if n_gt_del > 0 else 0.0

    return dict(
        filename=filepath.name,
        bridge=filepath.parent.name,
        n_signals=n_signals,
        n_gt_del=n_gt_del,
        n_correctly_flagged=n_correctly_flagged,
        n_missed=n_missed,
        accuracy=accuracy,
        fnr=fnr,
        median_ref=median_ref,
        thresh_susp=thresh_susp,
        thresh_det=thresh_det,
        labels=labels,
        ratios=ratios,
        band_ratios=band_ratios,
        fwhm_ns=fwhm_ns,
    )


def print_result(r: dict) -> None:
    print(
        f"{r['filename']} | "
        f"Signals: {r['n_signals']} | "
        f"GT Delaminated: {r['n_gt_del']} | "
        f"Correctly Flagged: {r['n_correctly_flagged']} | "
        f"Missed: {r['n_missed']} | "
        f"Accuracy: {r['accuracy']:.1f}% | "
        f"FNR: {r['fnr']:.1f}%"
    )


# ── Entry points ────────────────────────────────────────────────────────────

def run_single():
    test_file = DATA_PATH / "forest_river_north_bound" / "FILE____050.xlsx"
    print(f"Processing single file: {test_file.name}\n")
    result = process_file(test_file)
    if result:
        print_result(result)
        print(
            f"\n  [debug] Class-1 median A/A0 : {result['median_ref']:.4f}\n"
            f"          Suspect threshold    : <{THRESH_SUSPECT:.2f}× = {result['thresh_susp']:.4f}\n"
            f"          Deteriorated thresh  : <{THRESH_DETERIORATED:.2f}× = {result['thresh_det']:.4f}"
        )

        # ── Band energy ratio diagnostics ──────────────────────────────────
        labels      = result["labels"]
        band_ratios = result["band_ratios"]
        ratios      = result["ratios"]

        c1 = band_ratios[labels == 1]
        c2 = band_ratios[labels >= 2]
        c1_median = float(np.median(c1))

        valid = ~(np.isnan(band_ratios) | np.isnan(ratios))
        corr, _ = pearsonr(band_ratios[valid], ratios[valid])

        print("\n  [band energy ratio]")
        print(f"  {'':20s} {'Median':>10} {'P5':>10} {'P95':>10}")
        print(f"  {'Class-1 (sound)':20s} {np.median(c1):>10.4f} {np.percentile(c1, 5):>10.4f} {np.percentile(c1, 95):>10.4f}")
        print(f"  {'Class-2 (delaminated)':20s} {np.median(c2):>10.4f} {np.percentile(c2, 5):>10.4f} {np.percentile(c2, 95):>10.4f}")
        print(f"\n  Class-2 signals with band_ratio > Class-1 median ({c1_median:.4f}): "
              f"{(c2 > c1_median).sum()} / {len(c2)} ({(c2 > c1_median).mean()*100:.1f}%)")
        print(f"  Pearson r(band_ratio, A/A0) : {corr:.4f}")

        # ── Hilbert envelope FWHM diagnostics ─────────────────────────────
        fwhm_ns   = result["fwhm_ns"]

        fw1 = fwhm_ns[labels == 1]
        fw2 = fwhm_ns[labels >= 2]
        fw1_median = float(np.median(fw1))

        valid_fw = ~(np.isnan(fwhm_ns) | np.isnan(ratios))
        corr_fw, _ = pearsonr(fwhm_ns[valid_fw], ratios[valid_fw])

        print("\n  [Hilbert envelope FWHM]")
        print(f"  {'':20s} {'Median (ns)':>12} {'P5 (ns)':>10} {'P95 (ns)':>10}")
        print(f"  {'Class-1 (sound)':20s} {np.median(fw1):>12.4f} {np.percentile(fw1, 5):>10.4f} {np.percentile(fw1, 95):>10.4f}")
        print(f"  {'Class-2 (delaminated)':20s} {np.median(fw2):>12.4f} {np.percentile(fw2, 5):>10.4f} {np.percentile(fw2, 95):>10.4f}")
        print(f"\n  Class-2 signals with FWHM > Class-1 median ({fw1_median:.4f} ns): "
              f"{(fw2 > fw1_median).sum()} / {len(fw2)} ({(fw2 > fw1_median).mean()*100:.1f}%)")
        print(f"  Pearson r(FWHM, A/A0)       : {corr_fw:.4f}")

        # ── FWHM threshold sweep ───────────────────────────────────────────
        gt_del = labels >= 2
        n_pos  = gt_del.sum()           # total Class-2
        n_neg  = (~gt_del).sum()        # total Class-1

        thresholds = np.percentile(fwhm_ns, np.arange(10, 91, 5))  # P10..P90

        print("\n  [FWHM threshold sweep]  flag if FWHM > threshold")
        print(f"  {'Threshold (ns)':>15} {'Percentile':>10} {'TPR':>8} {'FPR':>8} {'FNR':>8}")
        print(f"  {'-'*52}")

        best_thresh = None
        best_tpr    = 0.0
        best_row    = None

        for pct, thresh in zip(np.arange(10, 91, 5), thresholds):
            flagged = fwhm_ns > thresh
            tpr = flagged[ gt_del].sum() / n_pos
            fpr = flagged[~gt_del].sum() / n_neg
            fnr = 1.0 - tpr
            marker = " <--" if fpr < 0.15 and tpr > best_tpr else ""
            if fpr < 0.15 and tpr > best_tpr:
                best_tpr    = tpr
                best_thresh = thresh
                best_row    = (pct, thresh, tpr, fpr, fnr)
            print(f"  {thresh:>15.4f} {pct:>9.0f}% {tpr*100:>7.1f}% {fpr*100:>7.1f}% {fnr*100:>7.1f}%{marker}")

        print()
        if best_row is not None:
            pct, thresh, tpr, fpr, fnr = best_row
            print(f"  Best threshold with FPR < 15%: {thresh:.4f} ns (P{pct:.0f})")
            print(f"    TPR: {tpr*100:.1f}%  FPR: {fpr*100:.1f}%  FNR: {fnr*100:.1f}%")
        else:
            # Find overall best tradeoff (closest to top-left of ROC)
            rows = []
            for pct, thresh in zip(np.arange(10, 91, 5), thresholds):
                flagged = fwhm_ns > thresh
                tpr = flagged[ gt_del].sum() / n_pos
                fpr = flagged[~gt_del].sum() / n_neg
                rows.append((pct, thresh, tpr, fpr))
            best = min(rows, key=lambda x: (x[3] - 0.0)**2 + (x[2] - 1.0)**2)  # nearest (0,1)
            pct, thresh, tpr, fpr = best
            print(f"  No threshold achieved FPR < 15% with meaningful TPR.")
            print(f"  Best tradeoff (nearest ROC top-left): {thresh:.4f} ns (P{pct:.0f})")
            print(f"    TPR: {tpr*100:.1f}%  FPR: {fpr*100:.1f}%  FNR: {(1-tpr)*100:.1f}%")


def run_all():
    all_results = []
    for bridge_dir in sorted(DATA_PATH.iterdir()):
        if not bridge_dir.is_dir():
            continue
        files = sorted(bridge_dir.glob("*.xlsx"))
        print(f"\n--- {bridge_dir.name} ({len(files)} files) ---")
        for f in files:
            r = process_file(f)
            if r:
                print_result(r)
                all_results.append(r)

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n=== Aggregate Summary ===")
        print(f"Total files    : {len(df)}")
        print(f"Total signals  : {df['n_signals'].sum():,}")
        print(f"Total GT delam : {df['n_gt_del'].sum():,}")
        print(f"Total flagged  : {df['n_correctly_flagged'].sum():,}")
        print(f"Total missed   : {df['n_missed'].sum():,}")
        print(f"Mean accuracy  : {df['accuracy'].mean():.1f}%")
        print(f"Mean FNR       : {df['fnr'].mean():.1f}%")


if __name__ == "__main__":
    if "--all" in sys.argv:
        run_all()
    else:
        run_single()
