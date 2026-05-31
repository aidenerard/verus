"""
server/physics.py
GPR physics primitives: time-zero detection, Fresnel dielectric estimation,
physically-correct rebar depth, and the ASTM D6087 corrosion index.

Re-exported through analysis.py so callers do `from analysis import ...`.
These are the ground-truth formulas used by Infrasense winDECARR / GSSI
RADAN and mandated by ASTM D6087.
"""
from __future__ import annotations

import numpy as np


def find_time_zero(trace: np.ndarray, search_samples: int = 60) -> int:
    """
    Find the sample index where the GPR pulse enters the surface.

    Time zero is the first strong reflection — the air/surface interface.
    All depth calculations must measure travel time FROM this sample, not
    from sample 0.

    Returns: sample index of surface entry (time zero).
    """
    from scipy.signal import hilbert
    env = np.abs(hilbert(trace[:search_samples].astype(np.float32)))
    return int(np.argmax(env))


def calculate_dielectric_fresnel(amp_surface: float, amp_plate: float) -> float:
    """
    Per-trace concrete dielectric via the Fresnel reflection-coefficient
    method (ASTM D6087, Infrasense winDECARR).

    epsr = ((Ap + As) / (Ap - As))^2
      Ap = reflection amplitude from a metal calibration plate (perfect reflector)
      As = surface reflection amplitude (air/concrete interface) at this trace

    Concrete dielectric ranges 4 (dry) → 20 (saturated / chloride-laden).
    A fixed value causes 20-40% depth errors.

    Returns: relative permittivity, clamped to [4.0, 20.0].
    """
    if amp_plate <= 0 or abs(amp_surface) >= abs(amp_plate):
        return 9.0  # typical concrete fallback
    r = abs(amp_surface / amp_plate)
    r = min(r, 0.99)  # guard the (1 - r) denominator
    epsr = ((1 + r) / (1 - r)) ** 2
    return float(np.clip(epsr, 4.0, 20.0))


def calculate_rebar_depth(
    apex_sample: int,
    time_zero: int,
    epsr: float,
    time_range_ns: float = 16.0,
    n_samples: int = 512,
) -> float:
    """
    Rebar depth from a hyperbola apex sample index — the only physically
    correct formula:
      1. travel time measured FROM time_zero (not sample 0)
      2. per-trace dielectric → wave velocity
      3. depth = (TWTT * velocity) / 2

    Returns: depth in inches.
    """
    ns_per_sample = time_range_ns / n_samples
    velocity_m_ns = 0.3 / np.sqrt(epsr)            # one-way velocity, m/ns
    twtt_samples  = max(0, apex_sample - time_zero)
    twtt_ns       = twtt_samples * ns_per_sample
    depth_m       = (twtt_ns * velocity_m_ns) / 2
    depth_in      = depth_m * 39.3701
    return round(float(depth_in), 3)


def calculate_astm_corrosion_index(
    amp_rebar: np.ndarray,
    depth_in: np.ndarray,
    threshold_db: float = -8.0,
) -> dict:
    """
    ASTM D6087 deterioration index (Infrasense winDECARR / GSSI RADAN):
      1. depth-correct rebar amplitude for geometric spreading (×depth^2)
      2. convert to dB relative to the maximum
      3. ASTM D6087 threshold (-6 to -8 dB): below = deteriorated

    Low depth-corrected amplitude = signal attenuation = corrosive
    environment / chloride contamination / delamination.

    Returns: {corrected_db, deteriorated, high_risk_pct, threshold_db, n_picks}.
    """
    depth_m   = np.array(depth_in, dtype=np.float32) / 39.3701
    corrected = np.array(amp_rebar, dtype=np.float32) * (depth_m ** 2)
    corrected = np.maximum(corrected, 1e-10)

    a_max        = float(corrected.max())
    corrected_db = 20 * np.log10(corrected / a_max)

    deteriorated  = corrected_db < threshold_db
    high_risk_pct = float(deteriorated.sum() / len(deteriorated) * 100)

    return {
        'corrected_db':  corrected_db.tolist(),
        'deteriorated':  deteriorated.tolist(),
        'high_risk_pct': round(high_risk_pct, 1),
        'threshold_db':  threshold_db,
        'n_picks':       len(amp_rebar),
    }
