"""
server/ingest_utils.py
Shared utilities for GPR format converters: resampling, CSV writing, companion lookup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def resample_to_512(signal: np.ndarray, original_samples: int) -> np.ndarray:
    """Resample a 1-D A-scan to exactly 512 samples using Fourier-based resampling."""
    if original_samples == 512:
        return signal.astype(np.float32)
    from scipy.signal import resample
    return resample(signal.astype(np.float64), 512).astype(np.float32)


def zscore_normalize(amps: np.ndarray) -> np.ndarray:
    """Z-score normalize each signal row: (x - mean) / (std + 1e-8)."""
    means = amps.mean(axis=1, keepdims=True)
    stds  = amps.std(axis=1, keepdims=True)
    return ((amps - means) / (stds + 1e-8)).astype(np.float32)


def write_csv(csv_path: Path, amps: np.ndarray) -> None:
    """Write (n_signals, 512) float32 array to CSV for load_csv()."""
    np.savetxt(str(csv_path), amps, delimiter=",", fmt="%.4f")


def find_companion(stem: str, ext: str, *search_dirs: Path) -> Optional[Path]:
    """
    Case-insensitive search for a companion file with matching stem and extension.
    Returns the first match found across all search_dirs, or None.
    """
    ext_lower = ext.lower()
    for d in search_dirs:
        if not d.is_dir():
            continue
        for candidate in d.iterdir():
            if candidate.suffix.lower() == ext_lower and candidate.stem == stem:
                return candidate
    return None
