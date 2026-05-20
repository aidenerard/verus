"""
server/gridding.py
Gridding algorithms for rebar depth maps. Each algorithm accepts scattered
(xs, ys, zs) picks and produces a regular grid suitable for rendering.

Algorithms:
  nearest            — scipy.interpolate.griddata method='nearest'
  idw                — KDTree inverse-distance weighting
  natural_neighbor   — scipy.interpolate.griddata method='cubic' (approximation —
                       true Sibson natural neighbor needs CGAL/Voronoi tesselation)
  minimum_curvature  — scipy.interpolate.RBFInterpolator thin-plate-spline kernel
  kriging            — 501 stub; requires pykrige which is not in requirements.txt
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as _MplPath
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import ConvexHull, KDTree


GRID_ALGORITHMS: set[str] = {
    "nearest", "idw", "natural_neighbor", "minimum_curvature", "kriging",
}


# ── Individual algorithms ─────────────────────────────────────────────────────

def _grid_nearest(xs, ys, zs, grid_x, grid_y, **_):
    pts = np.column_stack([xs, ys])
    return griddata(pts, zs, (grid_x, grid_y), method="nearest")


def _grid_idw(xs, ys, zs, grid_x, grid_y,
              search_radius_ft: float = 10.0, power: float = 2.0, **_):
    tree   = KDTree(np.column_stack([xs, ys]))
    flat   = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    out    = np.full(flat.shape[0], np.nan, dtype=np.float32)
    k_neigh = min(8, len(xs))

    dists, idxs = tree.query(flat, k=k_neigh, distance_upper_bound=search_radius_ft)
    # When k=1, scipy returns 1-D arrays; promote so the loop below is uniform.
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    zs_arr = np.asarray(zs)
    for i in range(flat.shape[0]):
        d, ix = dists[i], idxs[i]
        in_range = d < np.inf
        if not in_range.any():
            continue
        # Zero distance → exact match
        if (d == 0).any():
            out[i] = zs_arr[ix[d == 0][0]]
            continue
        w = 1.0 / (d[in_range] ** power)
        out[i] = (w * zs_arr[ix[in_range]]).sum() / w.sum()
    return out.reshape(grid_x.shape)


def _grid_natural_neighbor(xs, ys, zs, grid_x, grid_y, **_):
    pts = np.column_stack([xs, ys])
    return griddata(pts, zs, (grid_x, grid_y), method="cubic")


def _grid_minimum_curvature(xs, ys, zs, grid_x, grid_y,
                            smoothing: float = 0.0, **_):
    pts  = np.column_stack([xs, ys])
    rbf  = RBFInterpolator(pts, zs, kernel="thin_plate_spline", smoothing=smoothing)
    flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return rbf(flat).reshape(grid_x.shape)


def _grid_kriging(*args, **kwargs):
    raise NotImplementedError(
        "Kriging requires the pykrige package (not currently in requirements.txt). "
        "Add 'pykrige>=1.7.0' and wire pykrige.OrdinaryKriging here."
    )


_ALGORITHMS = {
    "nearest":           _grid_nearest,
    "idw":               _grid_idw,
    "natural_neighbor":  _grid_natural_neighbor,
    "minimum_curvature": _grid_minimum_curvature,
    "kriging":           _grid_kriging,
}


# ── Edge clipping ─────────────────────────────────────────────────────────────

def clip_to_hull(grid: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                 xs, ys, buffer_ft: float = 1.0) -> np.ndarray:
    """Mask grid cells outside the convex hull of (xs, ys), expanded by buffer_ft."""
    pts = np.column_stack([np.asarray(xs), np.asarray(ys)])
    if len(pts) < 3:
        return grid
    try:
        hull = ConvexHull(pts)
    except Exception:
        return grid

    centroid = pts.mean(axis=0)
    hull_pts = pts[hull.vertices]
    dirs     = hull_pts - centroid
    norms    = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    expanded = hull_pts + (dirs / norms) * buffer_ft

    poly      = _MplPath(expanded)
    flat_pts  = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    inside    = poly.contains_points(flat_pts).reshape(grid.shape)
    out       = grid.copy().astype(np.float32)
    out[~inside] = np.nan
    return out


def clip_to_polygon(grid: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                    polygon: list[tuple[float, float]]) -> np.ndarray:
    """Mask grid cells outside a user-supplied polygon (list of (x,y) tuples)."""
    if not polygon or len(polygon) < 3:
        return grid
    poly     = _MplPath(np.asarray(polygon))
    flat_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    inside   = poly.contains_points(flat_pts).reshape(grid.shape)
    out      = grid.copy().astype(np.float32)
    out[~inside] = np.nan
    return out


# ── Anisotropy ────────────────────────────────────────────────────────────────

def _apply_anisotropy(xs, ys, angle_deg: float, ratio: float):
    """Rotate to angle, scale cross-axis by 1/ratio, rotate back.
    ratio > 1 stretches along the rotated x-axis (sees variations along that axis as 'closer')."""
    if ratio == 1.0:
        return xs, ys
    ang = np.radians(angle_deg)
    c, s = np.cos(ang), np.sin(ang)
    xr =  c * xs + s * ys
    yr = -s * xs + c * ys
    yr = yr * ratio
    xs_out =  c * xr - s * yr
    ys_out =  s * xr + c * yr
    return xs_out, ys_out


# ── Main entry ────────────────────────────────────────────────────────────────

def run_gridding(
    algorithm: str,
    xs, ys, zs,
    *,
    cell_size_ft:     float = 0.5,
    search_radius_ft: float = 10.0,
    edge_clip:        bool  = True,
    anisotropy_angle: float = 0.0,
    anisotropy_ratio: float = 1.0,
    edge_polygon:     list[tuple[float, float]] | None = None,
) -> dict:
    """
    Run a gridding algorithm over scattered picks.

    Returns dict with keys: grid, grid_x, grid_y, extent (x_min,x_max,y_min,y_max),
    cell_size_ft, algorithm, n_points.
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. "
                         f"Supported: {sorted(GRID_ALGORITHMS)}")
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    if len(xs) < 3:
        raise ValueError(f"Gridding needs at least 3 picks; got {len(xs)}")

    xs_a, ys_a = _apply_anisotropy(xs, ys, anisotropy_angle, anisotropy_ratio)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    nx = max(2, int((x_max - x_min) / cell_size_ft) + 1)
    ny = max(2, int((y_max - y_min) / cell_size_ft) + 1)
    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(gx, gy)

    grid = _ALGORITHMS[algorithm](
        xs_a, ys_a, zs, grid_x, grid_y,
        search_radius_ft=search_radius_ft,
    )

    if edge_polygon:
        grid = clip_to_polygon(grid, grid_x, grid_y, edge_polygon)
    elif edge_clip:
        grid = clip_to_hull(grid, grid_x, grid_y, xs, ys,
                            buffer_ft=search_radius_ft / 2.0)

    return {
        "grid":         grid,
        "grid_x":       grid_x,
        "grid_y":       grid_y,
        "extent":       (x_min, x_max, y_min, y_max),
        "cell_size_ft": cell_size_ft,
        "algorithm":    algorithm,
        "n_points":     int(len(xs)),
    }
