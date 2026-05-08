"""
server/grids.py
Grid builders: prob_grid (C-scan), rebar depth, amplitude, TWT.
"""

import numpy as np
from scipy.ndimage import zoom

MAX_GRID_ROWS = 200
MAX_GRID_COLS = 500

_REBAR_MAX_ROWS = 100
_REBAR_MAX_COLS = 512

_FREQ_TIME_WINDOW_NS: dict[int, float] = {
    400:  40.0,
    900:  20.0,
    1600: 15.0,
    2000: 10.0,
    2600:  8.0,
}

_FREQ_ER: dict[int, float] = {
    400:  8.0,
    900:  7.0,
    1600: 6.0,
    2000: 6.0,
    2600: 5.0,
}


def _otsu_threshold(probs: np.ndarray, bins: int = 256) -> float:
    """
    Otsu's method on P(sound) histogram.
    Clipped to [0.30, 0.85] to remain physically reasonable.
    """
    counts, edges = np.histogram(probs, bins=bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(counts.sum())
    if total == 0:
        return 0.5

    w0  = np.cumsum(counts) / total
    mu0 = np.cumsum(centers * counts) / np.maximum(np.cumsum(counts), 1)
    mu_all = float(np.sum(centers * counts) / total)
    w1  = 1.0 - w0
    mu1 = np.where(w1 > 0, (mu_all - w0 * mu0) / w1, mu_all)

    between_var = w0 * w1 * (mu0 - mu1) ** 2
    t = float(centers[np.argmax(between_var)])
    return float(np.clip(t, 0.30, 0.85))


def _fill_trailing_nan(grid: np.ndarray) -> None:
    """Fill trailing NaN cells in each row with the last valid value (in-place)."""
    for i in range(grid.shape[0]):
        valid = np.where(~np.isnan(grid[i, :]))[0]
        if len(valid) and valid[-1] < grid.shape[1] - 1:
            grid[i, valid[-1] + 1:] = grid[i, valid[-1]]


def _downsample(grid: np.ndarray) -> np.ndarray:
    """Downsample grid to ≤ MAX_GRID_ROWS × MAX_GRID_COLS."""
    r, c = grid.shape
    if r > MAX_GRID_ROWS:
        idx  = np.linspace(0, r - 1, MAX_GRID_ROWS, dtype=int)
        grid = grid[idx, :]
    if c > MAX_GRID_COLS:
        idx  = np.linspace(0, c - 1, MAX_GRID_COLS, dtype=int)
        grid = grid[:, idx]
    return grid


def extract_peak_info(
    signals: np.ndarray,
    skip_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per A-scan: find sample index and amplitude of peak absolute value,
    skipping the first *skip_samples* (air-wave zone).

    Returns
    -------
    peak_idx : int32 (n,)
    peak_amp : float32 (n,)
    """
    window    = np.abs(signals[:, skip_samples:])
    local_idx = np.argmax(window, axis=1)
    peak_idx  = (local_idx + skip_samples).astype(np.int32)
    peak_amp  = window[np.arange(len(signals)), local_idx].astype(np.float32)
    return peak_idx, peak_amp


def build_prob_grid(
    file_preds: list[np.ndarray],
    file_confs: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """
    Build downsampled P(sound) grid and Otsu threshold.

    Returns
    -------
    prob_grid : float32 (rows, cols) — NaN for padding
    otsu_T    : Otsu threshold float
    """
    n_files  = len(file_preds)
    max_sigs = max(len(p) for p in file_preds)

    grid = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    for row, (preds, confs) in enumerate(zip(file_preds, file_confs)):
        for col, (pred, conf) in enumerate(zip(preds, confs)):
            grid[row, col] = conf if pred == 1 else 1.0 - conf

    grid = _downsample(grid)
    _fill_trailing_nan(grid)

    if grid.shape[1] < 50:
        scale = max(1, 50 // grid.shape[1])
        grid  = zoom(grid, (1, scale), order=1)

    all_preds  = np.concatenate(file_preds)
    all_confs  = np.concatenate(file_confs)
    all_psound = np.where(all_preds == 1, all_confs, 1.0 - all_confs)
    T = _otsu_threshold(all_psound)
    del all_psound, all_confs, all_preds

    return grid.astype(np.float32), T


def build_extra_grids(
    file_peak_idxs: list[np.ndarray],
    file_peak_amps: list[np.ndarray],
    frequency_mhz: int = 1600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build depth, amplitude, and TWT grids from per-file peak info.

    Returns (depth_grid_in, amplitude_grid, twt_grid_ns) — all float32,
    downsampled to ≤ MAX_GRID_ROWS × MAX_GRID_COLS, NaN for padding.
    """
    time_window = _FREQ_TIME_WINDOW_NS.get(frequency_mhz, 15.0)
    er          = _FREQ_ER.get(frequency_mhz, 6.0)
    velocity    = 0.3 / np.sqrt(er)

    n_files  = len(file_peak_idxs)
    max_sigs = max(len(a) for a in file_peak_idxs)

    depth_grid = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    amp_grid   = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    twt_grid   = np.full((n_files, max_sigs), np.nan, dtype=np.float32)

    for row, (peak_idx, peak_amp) in enumerate(zip(file_peak_idxs, file_peak_amps)):
        n        = len(peak_idx)
        twt_ns   = (peak_idx.astype(np.float32) / 512.0) * time_window
        depth_m  = velocity * twt_ns / 2.0
        depth_in = depth_m * 39.3701
        max_amp  = peak_amp.max() if peak_amp.max() > 0 else 1.0
        norm_amp = peak_amp / max_amp

        depth_grid[row, :n] = depth_in
        amp_grid[row, :n]   = norm_amp
        twt_grid[row, :n]   = twt_ns

    depth_grid = _downsample(depth_grid)
    amp_grid   = _downsample(amp_grid)
    twt_grid   = _downsample(twt_grid)

    for g in (depth_grid, amp_grid, twt_grid):
        _fill_trailing_nan(g)

    return depth_grid, amp_grid, twt_grid


def build_rebar_grids(
    file_depth_arrs: list[np.ndarray],
    file_twt_arrs:   list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack per-file rebar depth and TWT arrays into 2D grids,
    then downsample to ≤ _REBAR_MAX_ROWS × _REBAR_MAX_COLS.

    Returns (depth_grid_in, twt_grid_ns) as float32, NaN for padding.
    """
    n_files  = len(file_depth_arrs)
    max_sigs = max(len(a) for a in file_depth_arrs)

    depth_g = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    twt_g   = np.full((n_files, max_sigs), np.nan, dtype=np.float32)

    for row, (d, t) in enumerate(zip(file_depth_arrs, file_twt_arrs)):
        n = len(d)
        depth_g[row, :n] = d
        twt_g[row,   :n] = t

    r, c = depth_g.shape
    if r > _REBAR_MAX_ROWS:
        idx    = np.linspace(0, r - 1, _REBAR_MAX_ROWS, dtype=int)
        depth_g = depth_g[idx, :]
        twt_g   = twt_g[idx, :]
    if c > _REBAR_MAX_COLS:
        idx    = np.linspace(0, c - 1, _REBAR_MAX_COLS, dtype=int)
        depth_g = depth_g[:, idx]
        twt_g   = twt_g[:, idx]

    for g in (depth_g, twt_g):
        _fill_trailing_nan(g)

    return depth_g, twt_g


def build_peak_grid(
    file_peak_arrs: list[np.ndarray],
    max_rows: int = 100,
    max_cols: int = 300,
) -> np.ndarray:
    """
    Stack per-file rebar peak sample indices into a 2D grid (n_files, max_signals).
    Downsampled to ≤ max_rows × max_cols, NaN for padding.
    """
    n_files  = len(file_peak_arrs)
    max_sigs = max(len(a) for a in file_peak_arrs)

    grid = np.full((n_files, max_sigs), np.nan, dtype=np.float32)
    for row, arr in enumerate(file_peak_arrs):
        n = len(arr)
        grid[row, :n] = arr.astype(np.float32)

    r, c = grid.shape
    if r > max_rows:
        idx  = np.linspace(0, r - 1, max_rows, dtype=int)
        grid = grid[idx, :]
    if c > max_cols:
        idx  = np.linspace(0, c - 1, max_cols, dtype=int)
        grid = grid[:, idx]

    _fill_trailing_nan(grid)
    return grid


def grid_to_list(arr: np.ndarray) -> list:
    """Convert float32 grid (possibly with NaN) to nested Python list; NaN → None."""
    return [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in arr]
