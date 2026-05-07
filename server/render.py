"""
server/render.py
Matplotlib heatmap renderers → base64 PNG strings.
"""

import base64
import gc
import io

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap

from grids import build_prob_grid

_DEPTH_ACC_IN = {400: 1.0, 900: 0.5, 1600: 0.25, 2000: 0.25, 2600: 0.125}


def _fig_to_b64(fig: plt.Figure, dpi: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Replace NaNs with nanmean, then apply Gaussian filter."""
    fill = float(np.nanmean(grid)) if not np.all(np.isnan(grid)) else 0.0
    return gaussian_filter(np.where(np.isnan(grid), fill, grid), sigma=sigma)


def _styled_axes(ax: plt.Axes) -> None:
    ax.tick_params(which='major', top=True, right=True, direction='in',
                   length=4, width=0.7, labelsize=9)
    ax.tick_params(which='minor', top=True, right=True, direction='in',
                   length=2, width=0.5)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.set_facecolor('white')


def render_rebar_cscan_b64(
    depth_grid: np.ndarray,
    swath_spacing_ft: float = 1.0,
    along_track_ft_per_col: float = 1.0,
    structure_name: str = "Bridge Deck",
    smoothing_sigma: float = 1.0,
    dpi: int = 300,
    rebar_picks: list[tuple[float, float]] | None = None,
) -> str:
    """
    ASTM-style rebar cover depth map.
    Discrete 0.5" colour bands, black isolines, inline depth labels.
    Optional rebar_picks: (x_ft, y_ft) dots showing raw inference positions.
    """
    n_rows, n_cols = depth_grid.shape
    x_extent_ft = n_cols * along_track_ft_per_col
    y_extent_ft = n_rows * swath_spacing_ft

    d_min = float(np.floor(np.nanmin(depth_grid) * 2) / 2)
    d_max = float(np.ceil(np.nanmax(depth_grid) * 2) / 2)
    if d_max - d_min < 1.0:
        d_max = d_min + 1.0
    levels = np.arange(d_min, d_max + 0.5, 0.5)

    n_bands = max(1, len(levels) - 1)
    cmap = ListedColormap(plt.cm.YlOrRd_r(np.linspace(0, 1, n_bands)))  # type: ignore[attr-defined]
    norm = BoundaryNorm(levels, cmap.N)

    X, Y = np.meshgrid(
        np.linspace(0, x_extent_ft, n_cols),
        np.linspace(0, y_extent_ft, n_rows),
    )
    smoothed = _smooth(depth_grid, smoothing_sigma)

    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi, facecolor='white')

    cf = ax.contourf(X, Y, smoothed, levels=levels, cmap=cmap, norm=norm)
    cs = ax.contour(X, Y, smoothed, levels=levels, colors='black', linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=10, fmt='%g', colors='black')

    if rebar_picks:
        ax.scatter(
            [p[0] for p in rebar_picks], [p[1] for p in rebar_picks],
            s=2, c='black', alpha=0.4, marker='.', linewidths=0, zorder=10,
        )

    ax.set_xlim(0, x_extent_ft)
    ax.set_ylim(0, y_extent_ft)
    ax.set_xlabel('Along-track (ft)', fontsize=11)
    ax.set_ylabel('Lateral (ft)', fontsize=11)
    _styled_axes(ax)

    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', ticks=levels,
                        spacing='proportional', shrink=0.7, pad=0.15, aspect=40)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Rebar Cover Depth (in)', fontsize=11)

    fig.suptitle(f'{structure_name} — Rebar Cover Depth', fontsize=12, fontweight='bold')
    return _fig_to_b64(fig, dpi)


def render_cscan_b64(
    file_preds: list[np.ndarray],
    file_confs: list[np.ndarray],
    file_names: list[str],
    swath_spacing_ft: float = 1.0,
    along_track_ft_per_col: float = 1.0,
    structure_name: str = "Bridge Deck",
    smoothing_sigma: float = 1.5,
    dpi: int = 300,
) -> str:
    """
    ASTM-style delamination probability map.
    Fixed 10-band Spectral_r scale, black isolines, inline % labels.
    """
    prob_grid, T = build_prob_grid(file_preds, file_confs)
    print(f"[render] Otsu threshold: {T:.4f}", flush=True)

    n_rows, n_cols = prob_grid.shape
    x_extent_ft = n_cols * along_track_ft_per_col
    y_extent_ft = n_rows * swath_spacing_ft

    levels = [0, 5, 10, 20, 30, 40, 50, 60, 70, 85, 100]
    n_bands = len(levels) - 1
    cmap = ListedColormap(plt.cm.Spectral_r(np.linspace(0, 1, n_bands)))  # type: ignore[attr-defined]
    norm = BoundaryNorm(levels, cmap.N)

    Z = _smooth((1.0 - prob_grid) * 100.0, smoothing_sigma)
    del prob_grid
    gc.collect()

    X, Y = np.meshgrid(
        np.linspace(0, x_extent_ft, n_cols),
        np.linspace(0, y_extent_ft, n_rows),
    )

    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi, facecolor='white')

    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    cs = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=10, fmt='%g%%', colors='black')

    ax.set_xlim(0, x_extent_ft)
    ax.set_ylim(0, y_extent_ft)
    ax.set_xlabel('Along-track (ft)', fontsize=11)
    ax.set_ylabel('Lateral (ft)', fontsize=11)
    _styled_axes(ax)

    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', ticks=levels,
                        spacing='proportional', shrink=0.7, pad=0.15, aspect=40)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Delamination Probability (%)', fontsize=11)

    fig.suptitle(f'{structure_name} — Condition Map', fontsize=12, fontweight='bold')
    del Z
    gc.collect()
    return _fig_to_b64(fig, dpi)


def render_rebar_depth_b64(depth_grid: np.ndarray, dpi: int = 100) -> str:
    """Render rebar depth heatmap (imshow fallback). Blue=shallow, red=deep."""
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "depth", ['#2563EB', '#10B981', '#FBBF24', '#EF4444'], N=256)
    cmap_obj.set_bad(color='#F0EFEC')
    nan_mask = np.isnan(depth_grid)
    valid = depth_grid[~nan_mask]
    vmin, vmax = (max(0.0, np.percentile(valid, 5)), max(np.percentile(valid, 5) + 0.5, np.percentile(valid, 95))) if len(valid) else (1.0, 4.0)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    smooth = gaussian_filter(np.where(nan_mask, (vmin + vmax) / 2, depth_grid), sigma=(1.5, 3.0))
    masked = np.ma.array(smooth, mask=nan_mask)
    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22)
    ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect='auto', origin='upper', interpolation='bilinear')
    for sp in ax.spines.values(): sp.set_linewidth(0.8); sp.set_color('#CCCCCC')
    ax.set_xlabel('Scan line (longitudinal →)', fontsize=8, color='#555555', labelpad=6)
    ax.set_ylabel('Swath', fontsize=8, color='#555555', labelpad=6)
    ax.tick_params(colors='#777777', length=3, width=0.6, labelsize=7)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm),
                        ax=ax, orientation='horizontal', pad=0.18, fraction=0.04, aspect=50)
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_ticklabels([f'Shallow ({vmin:.1f}")', f'{(vmin+vmax)/2:.1f}"', f'Deep ({vmax:.1f}")'], fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_linewidth(0.5); cbar.outline.set_edgecolor('#CCCCCC')
    return _fig_to_b64(fig, dpi)


def render_amplitude_b64(amplitude_grid: np.ndarray, dpi: int = 100) -> str:
    """Render amplitude heatmap. Red=low (deteriorated), blue=high (sound)."""
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "amp", ['#C0392B', '#E67E22', '#F1C40F', '#27AE60', '#2980B9'], N=256)
    cmap_obj.set_bad(color='#F0EFEC')
    nan_mask = np.isnan(amplitude_grid)
    norm   = mcolors.Normalize(vmin=0.0, vmax=1.0)
    smooth = gaussian_filter(np.where(nan_mask, 0.5, amplitude_grid), sigma=(1.5, 3.0))
    masked = np.ma.array(smooth, mask=nan_mask)
    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22)
    ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect='auto', origin='upper', interpolation='bilinear')
    for sp in ax.spines.values(): sp.set_linewidth(0.8); sp.set_color('#CCCCCC')
    ax.set_xlabel('Scan line (longitudinal →)', fontsize=8, color='#555555', labelpad=6)
    ax.set_ylabel('Swath', fontsize=8, color='#555555', labelpad=6)
    ax.tick_params(colors='#777777', length=3, width=0.6, labelsize=7)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm),
                        ax=ax, orientation='horizontal', pad=0.18, fraction=0.04, aspect=50)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['Low Amplitude (Deteriorated)', 'Boundary', 'High Amplitude (Sound)'], fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_linewidth(0.5); cbar.outline.set_edgecolor('#CCCCCC')
    return _fig_to_b64(fig, dpi)


def compute_confidence_metrics(
    all_confs: np.ndarray,
    amplitude_grid: np.ndarray,
    frequency_mhz: int = 1600,
) -> tuple[float, float, str]:
    """Returns (model_confidence_pct, depth_accuracy_in, signal_quality)."""
    conf_pct  = float(np.mean(all_confs) * 100) if len(all_confs) else 50.0
    depth_acc = _DEPTH_ACC_IN.get(frequency_mhz, 0.5)
    valid_amp = amplitude_grid[~np.isnan(amplitude_grid)]
    mean_amp  = float(np.mean(valid_amp)) if len(valid_amp) else 0.5
    quality   = 'Good' if mean_amp >= 0.65 else ('Fair' if mean_amp >= 0.40 else 'Poor')
    return round(conf_pct, 1), depth_acc, quality


if __name__ == "__main__":
    import pathlib
    rng = np.random.default_rng(42)

    rows, cols = 10, 80
    base = np.full((rows, cols), 3.5, dtype=np.float32)
    cx_grid, _ = np.meshgrid(np.arange(cols), np.arange(rows))
    for cx in [25, 60]:
        base -= (2.0 * np.exp(-((cx_grid - cx) ** 2) / 30.0)).astype(np.float32)
    base = np.clip(base, 1.5, 6.0) + rng.normal(0, 0.2, base.shape).astype(np.float32)

    picks = [(float(si) / max(cols - 1, 1) * 80.0, float(ri)) for ri in range(rows) for si in range(0, cols, 3)]
    rebar_b64 = render_rebar_cscan_b64(
        base, swath_spacing_ft=1.0, along_track_ft_per_col=1.0,
        structure_name="Bridge 440029 (demo)", rebar_picks=picks,
    )
    pathlib.Path("rebar_demo.png").write_bytes(base64.b64decode(rebar_b64))
    print("rebar_demo.png written")

    preds_list, confs_list = [], []
    for row in range(rows):
        p = np.ones(cols, dtype=np.int32)
        c = rng.uniform(0.60, 0.90, cols).astype(np.float32)
        for cx in [25, 60]:
            mask = np.abs(np.arange(cols) - cx) < 8
            p[mask] = 0; c[mask] = rng.uniform(0.55, 0.78, mask.sum())
        preds_list.append(p); confs_list.append(c)

    cond_b64 = render_cscan_b64(
        preds_list, confs_list, [f"swath_{i:02d}.csv" for i in range(rows)],
        swath_spacing_ft=1.0, along_track_ft_per_col=1.0,
        structure_name="Bridge 440029 (demo)",
    )
    pathlib.Path("condition_demo.png").write_bytes(base64.b64decode(cond_b64))
    print("condition_demo.png written")
