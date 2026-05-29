"""
Extract Ez time-series from gprMax .out (HDF5) files.

Used by both local testing and the Kaggle batch scripts.
"""

import h5py
import numpy as np
from scipy.signal import resample

N_OUT     = 512
DC_OFFSET = 32768


def extract_ez(out_path: str) -> np.ndarray:
    """Return raw Ez array (n_iterations,) float32 from a gprMax .out file."""
    with h5py.File(out_path, "r") as f:
        ez = f["/rxs/rx1/Ez"][:]
    return ez.astype(np.float32)


def process_rebar(ez: np.ndarray) -> np.ndarray:
    """DC-remove → resample to 512 → max-abs normalize. Returns (512,) float32."""
    ez = ez - ez.mean()
    ez = resample(ez, N_OUT).astype(np.float32)
    peak = np.abs(ez).max()
    if peak > 1e-20:
        ez /= peak
    return ez


def process_delam(ez: np.ndarray) -> np.ndarray:
    """
    Resample to 512 → scale to ±30000 digitizer range → add DC_OFFSET.
    Produces values that match what load_csv() in cnn.py expects:
      amps = (amp_block - DC_OFFSET) * taper
    """
    ez = resample(ez, N_OUT).astype(np.float32)
    peak = np.abs(ez).max()
    if peak > 1e-20:
        ez = ez / peak * 30000.0
    ez = ez + DC_OFFSET
    return ez.astype(np.float32)
