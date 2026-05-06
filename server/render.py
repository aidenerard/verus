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

from grids import build_prob_grid

_DEPTH_ACC_IN = {400: 1.0, 900: 0.5, 1600: 0.25, 2000: 0.25, 2600: 0.125}


def _fig_to_b64(fig: plt.Figure, dpi: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _styled_hbar(fig: plt.Figure, ax: plt.Axes, cmap_obj, norm, ticks, labels) -> None:
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm),
        ax=ax, orientation='horizontal', pad=0.18, fraction=0.04, aspect=50,
    )
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels, fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#CCCCCC')


def _styled_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#CCCCCC')
    ax.set_xlabel('Scan line (longitudinal →)', fontsize=8, color='#555555', labelpad=6)
    ax.set_ylabel('Swath', fontsize=8, color='#555555', labelpad=6)
    ax.tick_params(colors='#777777', length=3, width=0.6, labelsize=7)


def render_cscan_b64(
    file_preds:  list[np.ndarray],
    file_confs:  list[np.ndarray],
    file_names:  list[str],
    bridge_name: str = "Bridge Deck",
    dpi:         int = 100,
) -> str:
    """Render GPR condition heatmap. Red=deteriorated, blue=sound."""
    prob_grid, T = build_prob_grid(file_preds, file_confs)
    print(f"[render] Otsu threshold: {T:.4f}", flush=True)

    nan_mask = np.isnan(prob_grid)
    p = gaussian_filter(np.where(nan_mask, T, prob_grid), sigma=(1.5, 3.0))
    display = np.where(
        p <= T,
        0.5 * p / T,
        0.5 + 0.5 * (p - T) / (1.0 - T),
    )
    masked = np.ma.array(display, mask=nan_mask)
    del prob_grid, p, display

    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "gpr", ['#C0392B', '#E67E22', '#F1C40F', '#27AE60', '#2980B9'], N=256,
    )
    cmap_obj.set_bad(color='#F0EFEC')
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22)
    ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect='auto', origin='upper',
              interpolation='bilinear')
    _styled_axes(ax)
    _styled_hbar(fig, ax, cmap_obj, norm,
                 [0.0, 0.5, 1.0], ['Deteriorated', 'Boundary', 'Sound'])

    return _fig_to_b64(fig, dpi)


def render_rebar_depth_b64(depth_grid: np.ndarray, dpi: int = 100) -> str:
    """Render rebar depth heatmap. Blue=shallow, red=deep."""
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "depth", ['#2563EB', '#10B981', '#FBBF24', '#EF4444'], N=256,
    )
    cmap_obj.set_bad(color='#F0EFEC')

    nan_mask = np.isnan(depth_grid)
    valid    = depth_grid[~nan_mask]
    vmin, vmax = 1.0, 4.0
    if len(valid):
        vmin = max(0.0, np.percentile(valid, 5))
        vmax = max(vmin + 0.5, np.percentile(valid, 95))

    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    smooth = gaussian_filter(
        np.where(nan_mask, (vmin + vmax) / 2, depth_grid), sigma=(1.5, 3.0)
    )
    masked = np.ma.array(smooth, mask=nan_mask)

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22)
    ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect='auto', origin='upper',
              interpolation='bilinear')
    _styled_axes(ax)
    _styled_hbar(fig, ax, cmap_obj, norm,
                 [vmin, (vmin + vmax) / 2, vmax],
                 [f'Shallow ({vmin:.1f}")', f'{(vmin+vmax)/2:.1f}"', f'Deep ({vmax:.1f}")'])

    return _fig_to_b64(fig, dpi)


def render_amplitude_b64(amplitude_grid: np.ndarray, dpi: int = 100) -> str:
    """Render amplitude heatmap. Red=low (deteriorated), blue=high (sound)."""
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "amp", ['#C0392B', '#E67E22', '#F1C40F', '#27AE60', '#2980B9'], N=256,
    )
    cmap_obj.set_bad(color='#F0EFEC')

    nan_mask = np.isnan(amplitude_grid)
    norm     = mcolors.Normalize(vmin=0.0, vmax=1.0)
    smooth   = gaussian_filter(
        np.where(nan_mask, 0.5, amplitude_grid), sigma=(1.5, 3.0)
    )
    masked   = np.ma.array(smooth, mask=nan_mask)

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor='white')
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22)
    ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect='auto', origin='upper',
              interpolation='bilinear')
    _styled_axes(ax)
    _styled_hbar(fig, ax, cmap_obj, norm,
                 [0.0, 0.5, 1.0],
                 ['Low Amplitude (Deteriorated)', 'Boundary', 'High Amplitude (Sound)'])

    return _fig_to_b64(fig, dpi)


def render_rebar_cscan_b64(
    depth_grid:    np.ndarray,
    frequency_mhz: int  = 1600,
    model_used:    bool = False,
    dpi:           int  = 200,
) -> str:
    """
    Landscape (20×4 in, 200 DPI) rebar depth map.
    Blue = shallow (0.5"), red = deep (4"+).
    """
    cmap_obj = mcolors.LinearSegmentedColormap.from_list(
        "rebar_cs", ['#2563EB', '#10B981', '#FBBF24', '#EF4444'], N=256,
    )
    cmap_obj.set_bad(color='#F0EFEC')

    nan_mask = np.isnan(depth_grid)
    norm     = mcolors.Normalize(vmin=0.5, vmax=4.0)
    smooth   = gaussian_filter(
        np.where(nan_mask, 2.25, depth_grid), sigma=(1.0, 2.0)
    )
    masked = np.ma.array(smooth, mask=nan_mask)

    fig, ax = plt.subplots(figsize=(20, 4), facecolor='white')
    fig.subplots_adjust(left=0.04, right=0.98, top=0.82, bottom=0.22)
    ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect='auto',
              origin='upper', interpolation='bilinear')
    ax.set_xlabel('Longitudinal Distance (ft)', fontsize=9, color='#555555', labelpad=6)
    ax.set_ylabel('Lateral Distance (ft)',      fontsize=9, color='#555555', labelpad=6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#CCCCCC')
    ax.tick_params(colors='#777777', length=3, width=0.6, labelsize=8)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm),
        ax=ax, orientation='horizontal', pad=0.22, fraction=0.04, aspect=60,
    )
    cbar.set_ticks([0.5, 4.0])
    cbar.set_ticklabels(['Shallow Cover (0.5")', 'Deep Cover (4")'], fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#CCCCCC')

    acc   = _DEPTH_ACC_IN.get(frequency_mhz, 0.5)
    src   = 'AI Model' if model_used else 'Physics Estimate'
    fig.text(0.02, 0.93,
             f'Rebar Depth Map — estimated cover depth in inches [{src}]',
             fontsize=9, fontweight='bold', va='top', color='#333333')
    fig.text(0.98, 0.93,
             f'±{acc}" accuracy at {frequency_mhz} MHz',
             fontsize=8, va='top', ha='right', color='#777777')

    return _fig_to_b64(fig, dpi)


def compute_confidence_metrics(
    all_confs: np.ndarray,
    amplitude_grid: np.ndarray,
    frequency_mhz: int = 1600,
) -> tuple[float, float, str]:
    """
    Returns (model_confidence_pct, depth_accuracy_in, signal_quality).
    """
    conf_pct  = float(np.mean(all_confs) * 100) if len(all_confs) else 50.0
    depth_acc = {400: 1.0, 900: 0.5, 1600: 0.25, 2000: 0.25, 2600: 0.125}.get(
        frequency_mhz, 0.5
    )
    valid_amp = amplitude_grid[~np.isnan(amplitude_grid)]
    mean_amp  = float(np.mean(valid_amp)) if len(valid_amp) else 0.5
    quality   = 'Good' if mean_amp >= 0.65 else ('Fair' if mean_amp >= 0.40 else 'Poor')
    return round(conf_pct, 1), depth_acc, quality
