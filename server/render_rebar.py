"""
server/render_rebar.py
Rebar cover depth, amplitude/attenuation, and confidence metrics renderers.
"""

import gc

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap

from render_cscan import _fig_to_b64, _smooth, _styled_axes

_DEPTH_ACC_IN = {400: 1.0, 900: 0.5, 1600: 0.25, 2000: 0.25, 2600: 0.125}


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
    """Render depth-normalised attenuation heatmap. Red=high attenuation (deteriorated), blue=low (sound)."""
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "amp", ['#C0392B', '#E67E22', '#F1C40F', '#27AE60', '#2980B9'], N=256)
    cmap_obj.set_bad(color='#F0EFEC')
    nan_mask = np.isnan(amplitude_grid)
    valid    = amplitude_grid[~nan_mask]
    vmin     = float(np.percentile(valid, 5))  if len(valid) else 0.0
    vmax     = float(np.percentile(valid, 95)) if len(valid) else 1.0
    if vmax - vmin < 0.05:
        vmax = vmin + 0.05
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    smooth = gaussian_filter(np.where(nan_mask, (vmin + vmax) / 2, amplitude_grid), sigma=(1.5, 3.0))
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
    cbar.set_ticklabels(['High Attenuation', '', 'Low Attenuation'], fontsize=8)
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
