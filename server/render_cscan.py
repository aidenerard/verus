"""
server/render_cscan.py
4-class GPR condition map renderer + shared matplotlib helpers.
"""

import base64
import gc
import io

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap

# 4-class condition map: upper bound of P(deterioration) % for each class.
# Sound < 35% anchors to production binary threshold (P(sound) < 0.65).
_BOUNDS  = [0, 35, 55, 75, 100]
_COLORS  = ['#27AE60', '#F1C40F', '#E67E22', '#C0392B']
_LABELS  = ['Sound\n(<35%)', 'Monitor\n(35–55%)', 'Anomalous\n(55–75%)', 'Significant\n(>75%)']
_KEYS    = ['sound', 'monitor', 'anomalous_response', 'significant_anomaly']
_CONF_T  = 0.55

_DISCLAIMER = (
    "Signal anomalies detected by GPR; classes do not constitute direct delamination determination. "
    "Field verification and professional engineering judgment required. Ref: ASTM D6432."
)


def _fig_to_b64(fig: plt.Figure, dpi: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
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


def render_cscan_b64(
    prob_grid: np.ndarray,
    conf_grid: np.ndarray,
    swath_spacing_ft: float = 1.0,
    along_track_ft_per_col: float = 1.0,
    structure_name: str = "Bridge Deck",
    smoothing_sigma: float = 1.5,
    dpi: int = 300,
) -> tuple[str, dict[str, float]]:
    """
    4-class GPR condition map. Accepts pre-computed prob_grid (P(sound)) and conf_grid.
    Returns (base64_png, class_area_pcts). Caller owns build_prob_grid and passes results in.
    Low-confidence zones (conf < 0.55) rendered with 40% white overlay.
    """
    valid = ~np.isnan(prob_grid)
    p_del = (1.0 - prob_grid[valid]) * 100.0
    n     = max(p_del.size, 1)
    area_pcts: dict[str, float] = {
        _KEYS[0]: round(100.0 * float(np.mean(p_del < 35)),                    1),
        _KEYS[1]: round(100.0 * float(np.mean((p_del >= 35) & (p_del < 55))),  1),
        _KEYS[2]: round(100.0 * float(np.mean((p_del >= 55) & (p_del < 75))),  1),
        _KEYS[3]: round(100.0 * float(np.mean(p_del >= 75)),                   1),
    }

    n_rows, n_cols = prob_grid.shape
    x_ft = n_cols * along_track_ft_per_col
    y_ft = n_rows * swath_spacing_ft

    Z     = np.clip(_smooth((1.0 - prob_grid) * 100.0, smoothing_sigma), 0.001, 99.999)
    Z_conf = _smooth(conf_grid, smoothing_sigma)
    del conf_grid
    gc.collect()

    X, Y = np.meshgrid(np.linspace(0, x_ft, n_cols), np.linspace(0, y_ft, n_rows))

    cmap = ListedColormap(_COLORS)
    norm = BoundaryNorm(_BOUNDS, cmap.N)

    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi, facecolor='white')
    ax.contourf(X, Y, Z, levels=_BOUNDS, cmap=cmap, norm=norm)

    low_conf = np.where(Z_conf < _CONF_T, 1.0, np.nan)
    if not np.all(np.isnan(low_conf)):
        ax.contourf(X, Y, low_conf, levels=[0.5, 1.5], colors=['white'], alpha=0.4)
        ax.contour(X, Y, low_conf, levels=[0.5], colors=['#777777'],
                   linewidths=0.4, linestyles='--')

    ax.set_xlim(0, x_ft)
    ax.set_ylim(0, y_ft)
    ax.set_xlabel('Along-track (ft)', fontsize=11)
    ax.set_ylabel('Lateral (ft)', fontsize=11)
    _styled_axes(ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.65, pad=0.18, aspect=40)
    cbar.set_ticks([17.5, 45, 65, 87.5])
    cbar.set_ticklabels(_LABELS)
    cbar.ax.tick_params(labelsize=8, length=0)
    cbar.set_label('Condition Class  (P(deterioration), %)', fontsize=10)

    stats = (f"Sound {area_pcts['sound']}%  |  Monitor {area_pcts['monitor']}%  |  "
             f"Anomalous {area_pcts['anomalous_response']}%  |  "
             f"Significant {area_pcts['significant_anomaly']}%")
    fig.suptitle(f'{structure_name} — Condition Map\n{stats}',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.text(0.5, -0.01, _DISCLAIMER, ha='center', fontsize=6,
             color='#555555', style='italic')

    del Z, Z_conf
    gc.collect()
    return _fig_to_b64(fig, dpi), area_pcts
