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
from scipy.signal import find_peaks, hilbert, windows
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path("~/Desktop/verus/gpr_data").expanduser()

# ── Time windows (ns) ──────────────────────────────────────────────────────
SURF_WIN  = (0.0, 3.0)   # first surface reflection
REBAR_WIN = (6.0, 9.0)   # rebar / delamination reflection window — confirmed 6.5–8.0 ns from diagnostic

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


# ── Attenuation model ───────────────────────────────────────────────────────

def _exp_decay(t: np.ndarray, A: float, alpha: float) -> np.ndarray:
    """Exponential decay model: A × e^(−α×t)"""
    return A * np.exp(-alpha * t)


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

    # 4. Tapered Hilbert envelope — computed once, used for both rebar peak
    #    detection (step 4) and FWHM (step 7).
    #    Hann taper on samples 410–512 suppresses the FFT periodicity artifact
    #    that inflated the envelope at ~11.97 ns in un-tapered signals.
    dt           = float(time_ns[1] - time_ns[0])          # ~0.0234 ns/sample
    _taper       = np.ones(512)
    _taper[410:] = windows.hann(204)[102:]                  # falling half: 1 → 0 over last 102 samples
    envelope     = np.abs(hilbert(amps * _taper[:, np.newaxis], axis=0))   # shape (512, n_signals)

    rebar_mask   = (time_ns >= REBAR_WIN[0]) & (time_ns <= REBAR_WIN[1])
    env_rebar    = envelope[rebar_mask, :]     # shape (n_rebar_samples, n_signals)
    t_rebar_win  = time_ns[rebar_mask]

    # 5. Compute per-signal A/A0
    #    Surface peak: largest absolute peak in raw signal (0–3 ns, no artifact).
    #    Rebar peak:   largest peak on the tapered Hilbert envelope in 6–9 ns.
    #                  Argmax is safe now that the 11.97 ns artifact is gone.
    ratios      = np.empty(n_signals, dtype=float)
    rebar_times = np.empty(n_signals, dtype=float)
    for i in range(n_signals):
        sig = amps[:, i]
        t_s, a_s = _find_peak_in_window(time_ns, sig, *SURF_WIN)
        best      = int(np.argmax(env_rebar[:, i]))
        t_r       = float(t_rebar_win[best])
        a_r       = float(env_rebar[best, i])
        rebar_times[i] = t_r
        ratios[i]      = _compute_ratio(t_s, a_s, t_r, a_r)

    # 6. D6087 classification
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

    # 7. Band energy ratio (diagnostic feature)
    shallow_mask = (time_ns >= 5.0) & (time_ns <= 7.0)
    deep_mask    = (time_ns >= 7.0) & (time_ns <= 9.0)
    shallow_energy = np.nansum(amps[shallow_mask, :] ** 2, axis=0)  # shape (n_signals,)
    deep_energy    = np.nansum(amps[deep_mask,    :] ** 2, axis=0)
    band_ratios    = shallow_energy / (deep_energy + 1e-9)

    # 8. Hilbert envelope FWHM — reuses env_rebar sliced in step 4.
    fwhm_ns      = np.array([_compute_fwhm_ns(env_rebar[:, i], dt) for i in range(n_signals)])

    # 9. Signal attenuation coefficient α (diagnostic feature)
    #    Fit A×e^(−α×t) to the Hilbert envelope in the 2–10 ns window.
    #    Uses the same envelope matrix already computed for FWHM (step 7).
    attn_mask = (time_ns >= 2.0) & (time_ns <= 10.0)
    t_attn    = time_ns[attn_mask]                   # 1-D time axis for the window
    env_attn  = envelope[attn_mask, :]               # shape (n_attn, n_signals)

    alpha_arr = np.full(n_signals, np.nan)
    for i in range(n_signals):
        env_i = env_attn[:, i]
        if np.all(np.isnan(env_i)) or env_i.max() == 0:
            continue
        try:
            popt, _ = curve_fit(
                _exp_decay, t_attn, env_i,
                p0=[float(env_i.max()), 0.3],
                bounds=([0.0, 0.0], [np.inf, np.inf]),
                maxfev=800,
            )
            alpha_arr[i] = popt[1]   # negative α already excluded by lower bound=0
        except (RuntimeError, ValueError):
            pass  # remains NaN

    # 10. Evaluation
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
        rebar_times=rebar_times,
        band_ratios=band_ratios,
        fwhm_ns=fwhm_ns,
        alpha_arr=alpha_arr,
        pred_flagged=pred_flagged,
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

        # ── Rebar window validation ────────────────────────────────────────
        labels       = result["labels"]
        ratios       = result["ratios"]
        rebar_times  = result["rebar_times"]
        pred_flagged = result["pred_flagged"]

        c1_mask = labels == 1
        c2_mask = labels >= 2
        gt_del  = c2_mask
        n_pos   = gt_del.sum();  n_neg = (~gt_del).sum()

        # t_rebar and derived depth
        t_ref_ns = float(np.nanmedian(rebar_times[c1_mask]))
        d_ref_cm = (3e8 * t_ref_ns * 1e-9) / (2.0 * np.sqrt(7.0)) * 100

        # A/A0 distributions by class
        r1 = ratios[c1_mask];  r2 = ratios[c2_mask]
        c1_med_ao = float(np.nanmedian(r1))

        # NaN counts
        nan_r1 = int(np.isnan(r1).sum());  nan_r2 = int(np.isnan(r2).sum())
        nan_t1 = int(np.isnan(rebar_times[c1_mask]).sum())
        nan_t2 = int(np.isnan(rebar_times[c2_mask]).sum())

        # FNR and FPR from D6087 result
        fnr_d6 = result["fnr"]
        fpr_d6 = pred_flagged[~gt_del].sum() / n_neg * 100

        print("\n  [rebar window validation — 6.0–9.0 ns, largest envelope peak]")
        print(f"  Class-1 median t_rebar : {t_ref_ns:.4f} ns  →  depth {d_ref_cm:.2f} cm at ε=7.0")
        print(f"  Class-1 median A/A0    : {c1_med_ao:.4f}")
        print(f"\n  {'':22} {'Median A/A0':>12} {'P5':>8} {'P95':>8} {'NaN':>6}")
        print(f"  {'Class-1 (sound)':22} {np.nanmedian(r1):>12.4f} "
              f"{np.nanpercentile(r1,5):>8.4f} {np.nanpercentile(r1,95):>8.4f} {nan_r1:>6}")
        print(f"  {'Class-2 (delaminated)':22} {np.nanmedian(r2):>12.4f} "
              f"{np.nanpercentile(r2,5):>8.4f} {np.nanpercentile(r2,95):>8.4f} {nan_r2:>6}")
        print(f"\n  Class-2 signals with A/A0 > Class-1 median ({c1_med_ao:.4f}): "
              f"{(r2 > c1_med_ao).sum()} / {len(r2)} ({(~np.isnan(r2) & (r2 > c1_med_ao)).sum() / (~np.isnan(r2)).sum() * 100:.1f}%)")
        print(f"  NaN t_rebar — Class-1: {nan_t1}  Class-2: {nan_t2}")
        print(f"\n  D6087 at <0.75× Class-1 median:  FNR {fnr_d6:.1f}%  FPR {fpr_d6:.1f}%")
        print(f"  V3 original (3–11 ns, first-sig-peak): FNR 60.2%  FPR 10.3%")
        print(f"  4.5–9.0 ns run:                        FNR 52.9%  FPR 25.8%")
        print(f"  5.5–9.0 ns run:                        FNR 66.4%  FPR 27.5%")

        # ── Band energy ratio diagnostics ──────────────────────────────────
        band_ratios = result["band_ratios"]
        c1 = band_ratios[c1_mask]
        c2 = band_ratios[c2_mask]
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

        fw1 = fwhm_ns[c1_mask]
        fw2 = fwhm_ns[c2_mask]
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

        # ── Combined A/A0 + FWHM classifier sweep ─────────────────────────
        gt_del       = labels >= 2
        n_pos        = gt_del.sum()
        n_neg        = (~gt_del).sum()
        pred_flagged = result["pred_flagged"]

        # V3 baseline stats (recomputed from stored flags)
        v3_tpr = pred_flagged[ gt_del].sum() / n_pos
        v3_fpr = pred_flagged[~gt_del].sum() / n_neg
        v3_fnr = 1.0 - v3_tpr
        v3_acc = ((pred_flagged & gt_del).sum() + (~pred_flagged & ~gt_del).sum()) / len(labels) * 100

        t1_pcts  = [25, 35, 50]
        t2_pcts  = [65, 75, 85]
        t1_vals  = np.percentile(ratios[labels == 1], t1_pcts)   # Class-1 A/A0 percentiles
        t2_vals  = np.percentile(fwhm_ns,             t2_pcts)   # full FWHM percentiles

        print("\n  [combined A/A0 + FWHM sweep]  flag if (A/A0 < t1) OR (FWHM > t2)")
        print(f"  {'t1 (A/A0)':>10} {'t1%':>5} {'t2 (FWHM ns)':>13} {'t2%':>5} "
              f"{'TPR':>7} {'FPR':>7} {'FNR':>7} {'Acc':>7} {'New':>6}")
        print(f"  {'-'*73}")

        best_combined = None

        for t1, p1 in zip(t1_vals, t1_pcts):
            for t2, p2 in zip(t2_vals, t2_pcts):
                ao_flag   = ratios   < t1
                fw_flag   = fwhm_ns  > t2
                combined  = ao_flag | fw_flag

                tpr = combined[ gt_del].sum() / n_pos
                fpr = combined[~gt_del].sum() / n_neg
                fnr = 1.0 - tpr
                acc = (( combined &  gt_del).sum() + (~combined & ~gt_del).sum()) / len(labels) * 100
                new_catches = int((fw_flag & ~ao_flag & gt_del).sum())  # caught by FWHM, not A/A0

                marker = " <" if fpr < 0.15 else ""
                print(f"  {t1:>10.4f} {p1:>4}% {t2:>13.4f} {p2:>4}% "
                      f"{tpr*100:>6.1f}% {fpr*100:>6.1f}% {fnr*100:>6.1f}% {acc:>6.1f}% {new_catches:>5}{marker}")

                if fpr < 0.15 and (best_combined is None or fnr < best_combined["fnr"]):
                    best_combined = dict(t1=t1, p1=p1, t2=t2, p2=p2,
                                        tpr=tpr, fpr=fpr, fnr=fnr, acc=acc,
                                        new_catches=new_catches)

        print(f"\n  V3 baseline:  TPR {v3_tpr*100:.1f}%  FPR {v3_fpr*100:.1f}%  "
              f"FNR {v3_fnr*100:.1f}%  Acc {v3_acc:.1f}%")

        if best_combined:
            b = best_combined
            fnr_delta = (v3_fnr - b["fnr"]) * 100
            print(f"\n  Best combined (FPR < 15%): A/A0 < {b['t1']:.4f} (P{b['p1']}) "
                  f"OR FWHM > {b['t2']:.4f} ns (P{b['p2']})")
            print(f"    TPR {b['tpr']*100:.1f}%  FPR {b['fpr']*100:.1f}%  "
                  f"FNR {b['fnr']*100:.1f}%  Acc {b['acc']:.1f}%")
            print(f"    FNR improvement over V3: {fnr_delta:+.1f} pp")
            print(f"    New catches vs A/A0 alone: {b['new_catches']} signals")
        else:
            print("\n  No combined threshold achieved FPR < 15%.")

        # ── Logistic regression: A/A0 + FWHM ──────────────────────────────
        gt_del       = labels >= 2
        pred_flagged = result["pred_flagged"]

        # Feature matrix — drop any signal where either feature is NaN
        X_raw  = np.column_stack([ratios, fwhm_ns])
        y      = gt_del.astype(int)
        valid  = ~np.isnan(X_raw).any(axis=1)
        X_raw, y, orig_idx = X_raw[valid], y[valid], np.where(valid)[0]

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        n_pos = y.sum()
        n_neg = (y == 0).sum()

        # 5-fold stratified CV — collect OOF probabilities and per-fold metrics
        skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y))
        cv_rows   = []

        for train_idx, val_idx in skf.split(X_scaled, y):
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(X_scaled[train_idx], y[train_idx])
            probs = clf.predict_proba(X_scaled[val_idx])[:, 1]
            oof_probs[val_idx] = probs

            pred  = (probs >= 0.5).astype(int)
            yv    = y[val_idx]
            np_   = yv.sum();  nn_ = (yv == 0).sum()
            tp = ((pred == 1) & (yv == 1)).sum()
            tn = ((pred == 0) & (yv == 0)).sum()
            fp = ((pred == 1) & (yv == 0)).sum()
            fn = ((pred == 0) & (yv == 1)).sum()
            cv_rows.append([tp/np_, fp/nn_, fn/np_, (tp+tn)/len(yv)])

        cv = np.array(cv_rows)
        print("\n  [logistic regression — 5-fold CV at threshold 0.5]")
        print(f"  {'Metric':<10} {'Mean':>8} {'Std':>8}")
        for i, name in enumerate(["TPR", "FPR", "FNR", "Accuracy"]):
            print(f"  {name:<10} {cv[:, i].mean()*100:>7.1f}% {cv[:, i].std()*100:>7.1f}%")

        # Threshold sweep on OOF probabilities
        print(f"\n  [threshold sweep on OOF probabilities]")
        print(f"  {'Threshold':>10} {'TPR':>8} {'FPR':>8} {'FNR':>8} {'Acc':>8}")
        print(f"  {'-'*44}")

        best_lr = None
        for thresh in np.round(np.arange(0.1, 0.91, 0.05), 2):
            pred = (oof_probs >= thresh).astype(int)
            tp = ((pred == 1) & (y == 1)).sum()
            tn = ((pred == 0) & (y == 0)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            tpr = tp / n_pos;  fpr = fp / n_neg
            fnr = fn / n_pos;  acc = (tp + tn) / len(y) * 100
            marker = " <--" if fpr < 0.15 and (best_lr is None or fnr < best_lr["fnr"]) else \
                     " <"   if fpr < 0.15 else ""
            if fpr < 0.15 and (best_lr is None or fnr < best_lr["fnr"]):
                best_lr = dict(thresh=thresh, tp=tp, tn=tn, fp=fp, fn=fn,
                               tpr=tpr, fpr=fpr, fnr=fnr, acc=acc, pred=pred)
            print(f"  {thresh:>10.2f} {tpr*100:>7.1f}% {fpr*100:>7.1f}% {fnr*100:>7.1f}% {acc:>7.1f}%{marker}")

        # Full model for coefficients
        clf_full = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf_full.fit(X_scaled, y)
        coef_ao, coef_fw = clf_full.coef_[0]

        print(f"\n  Learned coefficients (standardised features):")
        print(f"  A/A0  coef : {coef_ao:>+.4f}  ({'↑ higher A/A0 → delaminated' if coef_ao > 0 else '↓ lower A/A0 → delaminated'})")
        print(f"  FWHM  coef : {coef_fw:>+.4f}  ({'↑ higher FWHM → delaminated' if coef_fw > 0 else '↓ lower FWHM → delaminated'})")
        print(f"  Intercept  : {clf_full.intercept_[0]:>+.4f}")

        if best_lr:
            b = best_lr
            # New catches vs V3: LR catches & GT delaminated & V3 missed
            v3_missed_mask = (~pred_flagged[orig_idx].astype(bool)) & (y == 1)
            lr_new_catches = int(((b["pred"] == 1) & v3_missed_mask).sum())
            v3_total_missed = int(v3_missed_mask.sum())

            print(f"\n  [best LR threshold: {b['thresh']:.2f}]")
            print(f"  Confusion matrix:")
            print(f"  {'':22} Pred Sound   Pred Delam")
            print(f"  {'GT Sound':22} {b['tn']:>10}   {b['fp']:>10}")
            print(f"  {'GT Delaminated':22} {b['fn']:>10}   {b['tp']:>10}")
            print(f"\n  TPR {b['tpr']*100:.1f}%  FPR {b['fpr']*100:.1f}%  "
                  f"FNR {b['fnr']*100:.1f}%  Acc {b['acc']:.1f}%")
            print(f"\n  V3 baseline:  TPR {v3_tpr*100:.1f}%  FPR {v3_fpr*100:.1f}%  "
                  f"FNR {v3_fnr*100:.1f}%  Acc {v3_acc:.1f}%")
            print(f"  FNR improvement over V3: {(v3_fnr - b['fnr'])*100:+.1f} pp")
            print(f"  New catches (LR finds, V3 missed): {lr_new_catches} / {v3_total_missed}")
        else:
            print("\n  No LR threshold achieved FPR < 15%.")

        # ── Dielectric constant ε ──────────────────────────────────────────
        rebar_times = result["rebar_times"]

        C          = 3e8          # speed of light, m/s
        EPS_START  = 7.0          # assumed ε for sound concrete

        # Anchor depth from Class-1 median t_rebar
        t_ref_ns   = float(np.nanmedian(rebar_times[labels == 1]))
        t_ref_s    = t_ref_ns * 1e-9
        d_ref      = (C * t_ref_s) / (2.0 * np.sqrt(EPS_START))  # metres

        # Per-signal ε: ε_i = (c · t_i / (2·d_ref))²  ≡  EPS_START·(t_i/t_ref)²
        t_s_arr = rebar_times * 1e-9
        eps     = (C * t_s_arr / (2.0 * d_ref)) ** 2

        eps_c1     = eps[labels == 1]
        eps_c2     = eps[labels >= 2]
        c1_med_eps = float(np.nanmedian(eps_c1))

        valid_ao = ~(np.isnan(eps) | np.isnan(ratios))
        valid_fw = ~(np.isnan(eps) | np.isnan(fwhm_ns))
        corr_eps_ao, _ = pearsonr(eps[valid_ao], ratios[valid_ao])
        corr_eps_fw, _ = pearsonr(eps[valid_fw], fwhm_ns[valid_fw])

        n_nan_rebar = int(np.isnan(rebar_times).sum())

        print("\n  [dielectric constant ε  (anchored to Class-1 median rebar depth)]")
        print(f"  t_ref (Class-1 median t_rebar) : {t_ref_ns:.4f} ns")
        print(f"  d_ref (anchor depth, ε=7.0)    : {d_ref*100:.2f} cm")
        print(f"\n  {'':22} {'Median ε':>10} {'P5 ε':>10} {'P95 ε':>10}")
        print(f"  {'Class-1 (sound)':22} {np.nanmedian(eps_c1):>10.4f} "
              f"{np.nanpercentile(eps_c1, 5):>10.4f} {np.nanpercentile(eps_c1, 95):>10.4f}")
        print(f"  {'Class-2 (delaminated)':22} {np.nanmedian(eps_c2):>10.4f} "
              f"{np.nanpercentile(eps_c2, 5):>10.4f} {np.nanpercentile(eps_c2, 95):>10.4f}")
        print(f"\n  Class-2 signals with ε > Class-1 median ({c1_med_eps:.4f}): "
              f"{(eps_c2 > c1_med_eps).sum()} / {len(eps_c2)} "
              f"({(eps_c2 > c1_med_eps).mean()*100:.1f}%)")
        print(f"  Pearson r(ε, A/A0)  : {corr_eps_ao:.4f}")
        print(f"  Pearson r(ε, FWHM)  : {corr_eps_fw:.4f}")
        print(f"  Signals with NaN t_rebar: {n_nan_rebar}")

        # ── Signal attenuation coefficient α ───────────────────────────────
        alpha_arr = result["alpha_arr"]

        a1 = alpha_arr[c1_mask]
        a2 = alpha_arr[c2_mask]
        c1_med_alpha = float(np.nanmedian(a1))

        nan_a1 = int(np.isnan(a1).sum())
        nan_a2 = int(np.isnan(a2).sum())

        valid_ao = ~(np.isnan(alpha_arr) | np.isnan(ratios))
        valid_fw = ~(np.isnan(alpha_arr) | np.isnan(fwhm_ns))
        corr_a_ao, _ = pearsonr(alpha_arr[valid_ao], ratios[valid_ao])
        corr_a_fw, _ = pearsonr(alpha_arr[valid_fw], fwhm_ns[valid_fw])

        c2_above = int((~np.isnan(a2) & (a2 > c1_med_alpha)).sum())
        c2_valid = int((~np.isnan(a2)).sum())  # non-NaN Class-2 count

        print("\n  [signal attenuation coefficient α  (ns⁻¹)]")
        print(f"  {'':22} {'Median α':>10} {'P5 α':>10} {'P95 α':>10} {'NaN':>6}")
        print(f"  {'Class-1 (sound)':22} {np.nanmedian(a1):>10.4f} "
              f"{np.nanpercentile(a1,  5):>10.4f} {np.nanpercentile(a1, 95):>10.4f} {nan_a1:>6}")
        print(f"  {'Class-2 (delaminated)':22} {np.nanmedian(a2):>10.4f} "
              f"{np.nanpercentile(a2,  5):>10.4f} {np.nanpercentile(a2, 95):>10.4f} {nan_a2:>6}")
        print(f"\n  Class-2 signals with α > Class-1 median ({c1_med_alpha:.4f}): "
              f"{c2_above} / {c2_valid} ({c2_above / max(c2_valid, 1) * 100:.1f}%)")
        print(f"  Pearson r(α, A/A0) : {corr_a_ao:.4f}")
        print(f"  Pearson r(α, FWHM) : {corr_a_fw:.4f}")


def run_large_files():
    """Run the current pipeline on files 050–055 and print per-file + aggregate stats."""
    BRIDGE     = DATA_PATH / "forest_river_north_bound"
    LARGE_NUMS = range(50, 56)
    THRESH     = 0.75

    print("=" * 72)
    print("CURRENT PIPELINE — large files 050–055  (6.0–9.0 ns, largest env peak)")
    print("=" * 72)

    hdr = (f"  {'File':12} {'N':>6} {'C1':>6} {'C2':>6} {'C2%':>5}  "
           f"{'t_reb':>6}  {'C1 med':>8} {'C2 med':>8}  "
           f"{'C2<C1med':>9}  {'FNR':>6} {'FPR':>6}")
    print(f"\n{hdr}")
    print(f"  {'-'*70}")

    agg_tp = agg_fp = agg_fn = agg_tn = 0

    for num in LARGE_NUMS:
        fname = BRIDGE / f"FILE____{num:03d}.xlsx"
        r = process_file(fname)
        if r is None:
            print(f"  FILE____{num:03d}.xlsx  ERROR")
            continue

        labels       = r["labels"]
        ratios       = r["ratios"]
        rebar_times  = r["rebar_times"]
        pred_flagged = r["pred_flagged"]
        n            = r["n_signals"]
        c1_mask      = labels == 1
        c2_mask      = labels >= 2
        n_c1         = int(c1_mask.sum())
        n_c2         = int(c2_mask.sum())
        c2_pct       = n_c2 / n * 100

        t_reb   = float(np.nanmedian(rebar_times[c1_mask]))
        r1      = ratios[c1_mask]
        r2      = ratios[c2_mask]
        c1_med  = float(np.nanmedian(r1))
        c2_med  = float(np.nanmedian(r2))
        c2_below = int((~np.isnan(r2) & (r2 < c1_med)).sum())
        c2_valid = int((~np.isnan(r2)).sum())
        c2_below_pct = c2_below / max(c2_valid, 1) * 100

        gt_del = c2_mask
        n_pos  = int(gt_del.sum())
        n_neg  = int((~gt_del).sum())
        tp = int(( pred_flagged &  gt_del).sum())
        fp = int(( pred_flagged & ~gt_del).sum())
        fn = int((~pred_flagged &  gt_del).sum())
        tn = int((~pred_flagged & ~gt_del).sum())
        fnr = fn / n_pos * 100 if n_pos > 0 else float("nan")
        fpr = fp / n_neg * 100 if n_neg > 0 else float("nan")

        agg_tp += tp; agg_fp += fp; agg_fn += fn; agg_tn += tn

        print(f"  FILE____{num:03d}   {n:>6} {n_c1:>6} {n_c2:>6} {c2_pct:>4.1f}%  "
              f"{t_reb:>6.3f}  {c1_med:>8.3f} {c2_med:>8.3f}  "
              f"{c2_below_pct:>8.1f}%  {fnr:>5.1f}% {fpr:>5.1f}%")

        # Detailed A/A0 distribution block
        print(f"\n    {'':22} {'Median':>8} {'P5':>8} {'P95':>8}")
        print(f"    {'Class-1 (sound)':22} {np.nanmedian(r1):>8.3f} "
              f"{np.nanpercentile(r1, 5):>8.3f} {np.nanpercentile(r1, 95):>8.3f}")
        if n_c2 > 0:
            print(f"    {'Class-2 (delaminated)':22} {np.nanmedian(r2):>8.3f} "
                  f"{np.nanpercentile(r2, 5):>8.3f} {np.nanpercentile(r2, 95):>8.3f}")
        print()

    # Aggregate
    total_pos = agg_tp + agg_fn
    total_neg = agg_fp + agg_tn
    agg_fnr   = agg_fn / total_pos * 100 if total_pos > 0 else float("nan")
    agg_fpr   = agg_fp / total_neg * 100 if total_neg > 0 else float("nan")
    agg_acc   = (agg_tp + agg_tn) / (total_pos + total_neg) * 100

    print("=" * 72)
    print("AGGREGATE — files 050–055")
    print("=" * 72)
    print(f"\n  TP {agg_tp:,}  FP {agg_fp:,}  FN {agg_fn:,}  TN {agg_tn:,}")
    print(f"\n  FNR {agg_fnr:.1f}%  FPR {agg_fpr:.1f}%  Acc {agg_acc:.1f}%")
    print(f"\n  V3 baseline: FNR 60.2%  FPR 10.3%")
    print(f"  Delta:       FNR {60.2 - agg_fnr:+.1f} pp  FPR {10.3 - agg_fpr:+.1f} pp")
    print()


def run_tpeak():
    """Test peak time (t_rebar) as the primary classification feature on files 050–055."""
    BRIDGE     = DATA_PATH / "forest_river_north_bound"
    LARGE_NUMS = range(50, 56)
    FPR_CAP    = 0.15

    print("=" * 72)
    print("PEAK TIME (t_rebar) AS CLASSIFIER — files 050–055")
    print("=" * 72)

    # Accumulators for aggregate sweep
    agg_t_all    = []   # (t_peak, is_delaminated) tuples across all files
    agg_tp = agg_fp = agg_fn = agg_tn = 0
    best_agg_fnr = None

    for num in LARGE_NUMS:
        fname = BRIDGE / f"FILE____{num:03d}.xlsx"
        r = process_file(fname)
        if r is None:
            print(f"\nFILE____{num:03d}.xlsx  ERROR")
            continue

        labels      = r["labels"]
        t_peak      = r["rebar_times"]   # t_peak in 6–9 ns for every signal
        c1_mask     = labels == 1
        c2_mask     = labels >= 2
        n_pos       = int(c2_mask.sum())
        n_neg       = int(c1_mask.sum())

        t1 = t_peak[c1_mask]
        t2 = t_peak[c2_mask]
        c1_med = float(np.nanmedian(t1))
        c2_below = int((~np.isnan(t2) & (t2 < c1_med)).sum())
        c2_valid  = int((~np.isnan(t2)).sum())

        print(f"\n{'─'*72}")
        print(f"FILE____{num:03d}  |  N={r['n_signals']:,}  C1={n_neg:,}  C2={n_pos:,} "
              f"({n_pos/r['n_signals']*100:.1f}%)")
        print(f"{'─'*72}")
        print(f"\n  {'':22} {'Median':>8} {'P5':>8} {'P95':>8}")
        print(f"  {'Class-1 (sound)':22} {np.nanmedian(t1):>8.3f} "
              f"{np.nanpercentile(t1, 5):>8.3f} {np.nanpercentile(t1, 95):>8.3f}")
        if n_pos > 0:
            print(f"  {'Class-2 (delaminated)':22} {np.nanmedian(t2):>8.3f} "
                  f"{np.nanpercentile(t2, 5):>8.3f} {np.nanpercentile(t2, 95):>8.3f}")
        print(f"\n  Class-2 with t_peak < Class-1 median ({c1_med:.3f} ns): "
              f"{c2_below}/{c2_valid} ({c2_below/max(c2_valid,1)*100:.1f}%)")

        # Threshold sweep: flag as delaminated if t_peak < threshold
        # (earlier peak = shallower reflector = delamination)
        gt_del   = c2_mask
        valid    = ~np.isnan(t_peak)
        thresholds = np.percentile(t_peak[valid], np.arange(25, 76, 5))
        pcts       = np.arange(25, 76, 5)

        print(f"\n  [threshold sweep]  flag if t_peak < threshold")
        print(f"  {'Threshold (ns)':>15} {'Pct':>5} {'TPR':>7} {'FPR':>7} {'FNR':>7}")
        print(f"  {'-'*44}")

        best_fnr = float("inf")
        best_row = None
        for pct, thresh in zip(pcts, thresholds):
            flagged = valid & (t_peak < thresh)
            tp = int(( flagged &  gt_del).sum())
            fp = int(( flagged & ~gt_del).sum())
            fn = int((~flagged &  gt_del).sum())
            tn = int((~flagged & ~gt_del).sum())
            tpr = tp / n_pos if n_pos > 0 else 0.0
            fpr = fp / n_neg if n_neg > 0 else 0.0
            fnr = fn / n_pos if n_pos > 0 else 0.0
            marker = " <--" if fpr <= FPR_CAP and fnr < best_fnr else ""
            if fpr <= FPR_CAP and fnr < best_fnr:
                best_fnr = fnr
                best_row = (thresh, pct, tpr, fpr, fnr, tp, fp, fn, tn)
            print(f"  {thresh:>15.3f} {pct:>4.0f}% {tpr*100:>6.1f}% {fpr*100:>6.1f}% {fnr*100:>6.1f}%{marker}")

        if best_row:
            thresh, pct, tpr, fpr, fnr, tp, fp, fn, tn = best_row
            print(f"\n  Best (FPR ≤ 15%): t_peak < {thresh:.3f} ns (P{pct:.0f})")
            print(f"    TPR {tpr*100:.1f}%  FPR {fpr*100:.1f}%  FNR {fnr*100:.1f}%")
            agg_tp += tp; agg_fp += fp; agg_fn += fn; agg_tn += tn
        else:
            print(f"\n  No threshold achieved FPR ≤ 15%.")
            # use P25 as fallback for aggregate
            thresh = thresholds[0]
            flagged = valid & (t_peak < thresh)
            tp = int(( flagged &  gt_del).sum())
            fp = int(( flagged & ~gt_del).sum())
            fn = int((~flagged &  gt_del).sum())
            tn = int((~flagged & ~gt_del).sum())
            agg_tp += tp; agg_fp += fp; agg_fn += fn; agg_tn += tn

        # Collect for aggregate analysis
        for i in range(len(labels)):
            if not np.isnan(t_peak[i]):
                agg_t_all.append((t_peak[i], int(labels[i] >= 2)))

    # Aggregate
    print(f"\n{'='*72}")
    print("AGGREGATE — files 050–055  (best per-file threshold, FPR ≤ 15%)")
    print(f"{'='*72}")
    total_pos = agg_tp + agg_fn
    total_neg = agg_fp + agg_tn
    agg_fnr   = agg_fn / total_pos * 100 if total_pos > 0 else float("nan")
    agg_fpr   = agg_fp / total_neg * 100 if total_neg > 0 else float("nan")
    agg_acc   = (agg_tp + agg_tn) / (total_pos + total_neg) * 100

    print(f"\n  TP {agg_tp:,}  FP {agg_fp:,}  FN {agg_fn:,}  TN {agg_tn:,}")
    print(f"\n  FNR {agg_fnr:.1f}%  FPR {agg_fpr:.1f}%  Acc {agg_acc:.1f}%")
    print(f"\n  V3 baseline:    FNR 60.2%  FPR 10.3%")
    print(f"  Delta vs V3:    FNR {60.2-agg_fnr:+.1f} pp  FPR {10.3-agg_fpr:+.1f} pp")
    print()


def run_bscan():
    """
    Per-file B-scan lateral consistency analysis — large files (050–055) only.

    For each file independently:
      1. Find rebar depth (dominant Hilbert-envelope peak in 2–12 ns).
      2. Build Class-1 P90 baseline: A_corrected = A_peak × t_peak².
      3. Score every signal: score = A_corrected / baseline.
      4. Report per-file and aggregate FNR/FPR at score < 0.75.
    """
    BRIDGE     = DATA_PATH / "forest_river_north_bound"
    LARGE_NUMS = range(50, 56)   # 050 .. 055
    PEAK_LO    = 2.0
    PEAK_HI    = 12.0
    THRESH     = 0.75

    print("=" * 70)
    print("PER-FILE B-SCAN ANALYSIS — forest_river_north_bound (files 050–055)")
    print("=" * 70)

    # Accumulators for aggregate Step 5
    agg_tp = agg_fp = agg_fn = agg_tn = 0

    for num in LARGE_NUMS:
        fname = BRIDGE / f"FILE____{num:03d}.xlsx"
        print(f"\n{'─'*70}")
        print(f"FILE____{num:03d}.xlsx")
        print(f"{'─'*70}")

        # ── Load ─────────────────────────────────────────────────────────────
        try:
            raw = pd.read_excel(fname, header=None, engine="openpyxl")
            n_signals = int(raw.iloc[0, 4])
            labels    = raw.iloc[7, 1:n_signals + 1].values.astype(int)
            amp_block = raw.iloc[10:522, 0:n_signals + 1].values.astype(float)
            time_ns   = amp_block[:, 0]
            amps      = amp_block[:, 1:] - DC_OFFSET   # (512, n_signals)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        c1_mask = labels == 1
        c2_mask = labels >= 2
        n_c1    = int(c1_mask.sum())
        n_c2    = int(c2_mask.sum())
        print(f"  Signals: {n_signals}  |  Class-1: {n_c1}  |  Class-2: {n_c2}")

        # ── Step 1: Rebar depth from dominant peak ────────────────────────────
        peak_mask = (time_ns >= PEAK_LO) & (time_ns <= PEAK_HI)
        t_win     = time_ns[peak_mask]

        envelope  = np.abs(hilbert(amps, axis=0))   # (512, n_signals)
        env_win   = envelope[peak_mask, :]           # (n_win, n_signals)

        peak_idx  = np.argmax(env_win, axis=0)       # (n_signals,)
        t_peak    = t_win[peak_idx]                  # (n_signals,)

        p25, med, p75 = np.percentile(t_peak, [25, 50, 75])
        t_rebar = float(med)

        print(f"\n  [Step 1] t_peak distribution (ns):")
        print(f"  P25={p25:.3f}  Median={med:.3f}  P75={p75:.3f}  →  t_rebar={t_rebar:.3f} ns")

        # ── Step 2: Class-1 baseline (single P90 value) ───────────────────────
        # For each signal: amplitude of largest peak within ±1.0 ns of t_rebar
        SEARCH_NS = 1.0
        lo = t_rebar - SEARCH_NS
        hi = t_rebar + SEARCH_NS
        search_mask = (time_ns >= lo) & (time_ns <= hi)

        env_search = envelope[search_mask, :]              # (n_search, n_signals)
        t_search   = time_ns[search_mask]
        best_local = np.argmax(env_search, axis=0)         # (n_signals,)
        a_peak     = env_search[best_local, np.arange(n_signals)]   # (n_signals,)
        t_peak_local = t_search[best_local]                # (n_signals,)

        # Geometric spreading correction
        a_corrected = a_peak * (t_peak_local ** 2)

        # Class-1 P90 — single scalar baseline for this file
        a_c1        = a_corrected[c1_mask]
        baseline    = float(np.percentile(a_c1, 90))

        print(f"\n  [Step 2] Class-1 A_corrected P90 baseline: {baseline:.2f}")
        print(f"           (median Class-1 A_corrected: {np.median(a_c1):.2f})")

        # ── Step 3: Score ─────────────────────────────────────────────────────
        score = a_corrected / baseline   # scalar division — no NaNs expected

        s1 = score[c1_mask]
        s2 = score[c2_mask]
        c1_med = float(np.median(s1))

        c2_below = int((s2 < c1_med).sum())

        print(f"\n  [Step 4] Score distributions:")
        print(f"  {'':22} {'Median':>8} {'P5':>8} {'P95':>8}")
        print(f"  {'Class-1 (sound)':22} {np.median(s1):>8.4f} "
              f"{np.percentile(s1,  5):>8.4f} {np.percentile(s1, 95):>8.4f}")
        if n_c2 > 0:
            print(f"  {'Class-2 (delaminated)':22} {np.median(s2):>8.4f} "
                  f"{np.percentile(s2,  5):>8.4f} {np.percentile(s2, 95):>8.4f}")
            print(f"\n  Class-2 below Class-1 median ({c1_med:.4f}): "
                  f"{c2_below}/{n_c2} ({c2_below/n_c2*100:.1f}%)")
        else:
            print(f"  Class-2 (delaminated)     [no Class-2 signals in this file]")

        # ── FNR / FPR ─────────────────────────────────────────────────────────
        gt_del  = c2_mask
        n_pos   = int(gt_del.sum())
        n_neg   = int((~gt_del).sum())
        flagged = score < THRESH

        tp = int(( flagged &  gt_del).sum())
        fp = int(( flagged & ~gt_del).sum())
        fn = int((~flagged &  gt_del).sum())
        tn = int((~flagged & ~gt_del).sum())

        fnr = fn / n_pos * 100 if n_pos > 0 else float("nan")
        fpr = fp / n_neg * 100 if n_neg > 0 else float("nan")

        print(f"\n  FNR {fnr:.1f}%  FPR {fpr:.1f}%  (threshold < {THRESH})")

        agg_tp += tp;  agg_fp += fp
        agg_fn += fn;  agg_tn += tn

    # ── Step 5: Aggregate ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE — files 050–055 combined")
    print(f"{'='*70}")

    total_pos = agg_tp + agg_fn
    total_neg = agg_fp + agg_tn
    agg_fnr   = agg_fn / total_pos * 100 if total_pos > 0 else float("nan")
    agg_fpr   = agg_fp / total_neg * 100 if total_neg > 0 else float("nan")
    agg_acc   = (agg_tp + agg_tn) / (total_pos + total_neg) * 100

    print(f"\n  TP: {agg_tp:,}  FP: {agg_fp:,}  FN: {agg_fn:,}  TN: {agg_tn:,}")
    print(f"  FNR {agg_fnr:.1f}%  FPR {agg_fpr:.1f}%  Acc {agg_acc:.1f}%")
    print(f"\n  V3 baseline:  FNR 60.2%  FPR 10.3%  Acc ~88.4%")
    fnr_delta = 60.2 - agg_fnr
    fpr_delta = 10.3 - agg_fpr
    print(f"  Delta vs V3:  FNR {fnr_delta:+.1f} pp  FPR {fpr_delta:+.1f} pp")
    print()


def run_diagnostic():
    """
    Pure diagnostic: find where rebar actually sits in FILE____050.
    Prints Hilbert envelope at 0.5 ns intervals for 20 Class-1 and 10 Class-2
    signals, then the 3 largest peaks per signal and the dominant peak time.
    """
    fname = DATA_PATH / "forest_river_north_bound" / "FILE____050.xlsx"
    print(f"Loading {fname.name} …")

    raw       = pd.read_excel(fname, header=None, engine="openpyxl")
    n_signals = int(raw.iloc[0, 4])
    labels    = raw.iloc[7, 1:n_signals + 1].values.astype(int)
    amp_block = raw.iloc[10:522, 0:n_signals + 1].values.astype(float)
    time_ns   = amp_block[:, 0]
    amps      = amp_block[:, 1:] - DC_OFFSET   # (512, n_signals)

    dt       = float(time_ns[1] - time_ns[0])
    envelope = np.abs(hilbert(amps, axis=0))   # (512, n_signals)

    rng     = np.random.default_rng(42)
    c1_idx  = np.where(labels == 1)[0]
    c2_idx  = np.where(labels >= 2)[0]
    sample_c1 = rng.choice(c1_idx, size=20, replace=False)
    sample_c2 = rng.choice(c2_idx, size=10, replace=False)

    # ── Interpolate envelope at 0.5 ns grid points ──────────────────────────
    grid = np.arange(2.0, 12.01, 0.5)   # 2.0, 2.5, …, 12.0

    def interp_env(sig_idx: int) -> np.ndarray:
        return np.interp(grid, time_ns, envelope[:, sig_idx])

    # ── Print envelope table for one signal ──────────────────────────────────
    def print_signal(sig_idx: int, label: str) -> None:
        env_grid = interp_env(sig_idx)
        peak_val = env_grid.max()
        print(f"\nSignal {sig_idx} ({label}):")
        for t, v in zip(grid, env_grid):
            bar   = "█" * min(40, int(v / peak_val * 40))
            note  = "  ← peak" if v == peak_val else ""
            print(f"  {t:4.1f} ns: {v:>8,.0f}  {bar}{note}")

    # ── Find top-3 peaks in 3–12 ns via find_peaks on the envelope ───────────
    def top3_peaks(sig_idx: int) -> list[tuple[float, float]]:
        mask  = (time_ns >= 3.0) & (time_ns <= 12.0)
        t_win = time_ns[mask]
        e_win = envelope[mask, sig_idx]
        peaks, props = find_peaks(e_win, prominence=0.02 * e_win.max())
        if len(peaks) == 0:
            # fall back: just sort by amplitude
            order = np.argsort(e_win)[::-1][:3]
            return [(float(t_win[i]), float(e_win[i])) for i in order]
        order = np.argsort(e_win[peaks])[::-1][:3]
        return [(float(t_win[peaks[i]]), float(e_win[peaks[i]])) for i in order]

    # ── Class-1 signals ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLASS-1 SIGNALS (20 random samples)")
    print("=" * 70)

    for idx in sample_c1:
        print_signal(idx, "Class-1")

    # ── Class-2 signals ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLASS-2 SIGNALS (10 random samples)")
    print("=" * 70)

    for idx in sample_c2:
        print_signal(idx, "Class-2")

    # ── Top-3 peaks per signal ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP-3 ENVELOPE PEAKS in 3–12 ns (largest first)")
    print("=" * 70)

    c1_first_peaks = []   # collect #1 peak times for mode analysis

    print(f"\n{'Signal':>8}  {'Class':>7}  {'#1 peak':>10}  {'#2 peak':>10}  {'#3 peak':>10}")
    print(f"  {'-'*54}")

    for idx in list(sample_c1) + list(sample_c2):
        cls  = "Class-1" if labels[idx] == 1 else "Class-2"
        tops = top3_peaks(idx)
        t1   = f"{tops[0][0]:.2f} ns" if len(tops) > 0 else "—"
        t2   = f"{tops[1][0]:.2f} ns" if len(tops) > 1 else "—"
        t3   = f"{tops[2][0]:.2f} ns" if len(tops) > 2 else "—"
        print(f"  {idx:>6}  {cls:>7}  {t1:>10}  {t2:>10}  {t3:>10}")
        if labels[idx] == 1:
            c1_first_peaks.append(tops[0][0])

    # ── Mode of Class-1 primary peaks (rounded to 0.5 ns) ────────────────────
    rounded = [round(t * 2) / 2 for t in c1_first_peaks]
    from collections import Counter
    counts = Counter(rounded)
    mode_t, mode_n = counts.most_common(1)[0]

    print(f"\n{'='*70}")
    print("DOMINANT REBAR PEAK TIME — Class-1 signals")
    print(f"{'='*70}")
    print(f"\n  Peak times (rounded to 0.5 ns):")
    for t, n in sorted(counts.items()):
        bar = "█" * n
        print(f"  {t:5.1f} ns : {n:2d}  {bar}")
    print(f"\n  Most common: {mode_t:.1f} ns  (found in {mode_n}/20 Class-1 signals)")
    print()


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
    elif "--bscan" in sys.argv:
        run_bscan()
    elif "--diag" in sys.argv:
        run_diagnostic()
    elif "--large" in sys.argv:
        run_large_files()
    elif "--tpeak" in sys.argv:
        run_tpeak()
    else:
        run_single()
