"""
server/data.py
CSV loading: format sniffing + normalisation → (n_signals, 512) float32.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from model import DC_OFFSET, N_SAMPLES


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _normalise_key(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")


_TIME_AXIS_KEYS = {
    _normalise_key(k) for k in (
        "time_ns", "time", "time(ns)", "t(ns)", "depth_m", "depth",
        "sample", "sample_no", "twt", "twt(ns)",
    )
}


def _sniff_csv(fpath: Path) -> tuple[str, int]:
    """
    Return (delimiter, skiprows) by scanning up to 300 lines.

    Strategy A — keyword: first field matches a known time-axis label →
      data starts on the next line.
    Strategy B — numeric: first row with ≥10 fields, no alpha, ≥80% parseable.
    """
    delimiter  = ","
    best_count = 0

    with open(fpath, "r", errors="replace") as f:
        lines = []
        for _ in range(300):
            line = f.readline()
            if not line:
                break
            lines.append(line)

    for line in lines:
        for d in (",", "\t", ";"):
            c = line.count(d)
            if c > best_count:
                best_count = c
                delimiter = d

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(delimiter)
        first = _normalise_key(parts[0].strip())

        if first in _TIME_AXIS_KEYS and len(parts) > 1:
            print(f"[sniff_csv] time-axis header '{parts[0].strip()}' "
                  f"at row {i} → data starts at row {i + 1}", flush=True)
            return delimiter, i + 1

        if len(parts) < 10:
            continue
        has_alpha = any(
            any(c.isalpha() for c in p.strip())
            for p in parts if p.strip()
        )
        if has_alpha:
            continue
        numeric = sum(1 for p in parts if _is_float(p.strip()))
        if numeric >= 0.8 * len(parts):
            print(f"[sniff_csv] numeric data at row {i} "
                  f"({len(parts)} fields, {numeric} numeric)", flush=True)
            return delimiter, i

    print("[sniff_csv] WARNING: no data row in first 300 lines → skiprows=0", flush=True)
    return delimiter, 0


# ── Public API ────────────────────────────────────────────────────────────────

def load_csv(fpath: Path) -> np.ndarray:
    """
    Load one CSV file → normalised signals (n_signals, 512) float32.

    Accepts SDNET2021, simple row-per-A-scan, and transposed layouts.
    Applies per-signal z-score normalisation.
    """
    delimiter, skiprows = _sniff_csv(fpath)
    print(f"[load_csv] delimiter={repr(delimiter)} skiprows={skiprows}", flush=True)

    try:
        df = pd.read_csv(
            fpath, header=None, sep=delimiter,
            skiprows=skiprows, dtype=np.float32,
            on_bad_lines="skip",
        )
    except Exception as exc:
        raise ValueError(f"pd.read_csv failed: {exc}")

    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    if df.empty:
        raise ValueError("No numeric data found in CSV")

    data_array = df.to_numpy(dtype=np.float32, na_value=0.0)
    del df
    gc.collect()

    print(f"[load_csv] raw shape: {data_array.shape}", flush=True)

    rows, cols = data_array.shape

    if 400 <= rows <= 600:
        col0 = data_array[:, 0]
        if np.abs(col0).max() < 500:
            print("[load_csv] Dropping time/index column 0", flush=True)
            data_array = np.ascontiguousarray(data_array[:, 1:])
        amps = np.ascontiguousarray(data_array.T)
        del data_array
    elif 400 <= cols <= 600:
        col0 = data_array[:, 0]
        if np.abs(col0).max() < rows + 2:
            print("[load_csv] Dropping row-index column 0", flush=True)
            amps = np.ascontiguousarray(data_array[:, 1:])
            del data_array
        else:
            amps = data_array
    elif rows > cols:
        amps = np.ascontiguousarray(data_array.T)
        del data_array
    else:
        amps = data_array
    gc.collect()

    n_signals, n_samples = amps.shape
    if n_signals == 0:
        raise ValueError("No A-scan signals found in CSV")

    if n_samples > N_SAMPLES:
        amps = np.ascontiguousarray(amps[:, :N_SAMPLES])
    elif n_samples < N_SAMPLES:
        amps = np.pad(amps, ((0, 0), (0, N_SAMPLES - n_samples)), mode="constant")

    if np.abs(amps.mean()) > 1000:
        amps -= DC_OFFSET

    mean = amps.mean(axis=1, keepdims=True)
    std  = amps.std(axis=1,  keepdims=True) + 1e-8
    amps -= mean
    amps /= std

    print(f"[load_csv] done: {n_signals} signals normalised", flush=True)
    return amps
