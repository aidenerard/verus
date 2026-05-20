"""
server/picks_writer.py
Persists per-trace rebar picks to the Supabase picks table after inference.

Subsamples to ~500 picks per scan line (the regrid endpoint reconstructs grids
from this set; full per-trace persistence would be ~100k+ rows for typical jobs).
"""
from __future__ import annotations

from math import floor
from typing import Optional

import numpy as np


PICKS_PER_SCAN_LINE = 500
INSERT_BATCH_SIZE   = 1000


def persist_picks_for_job(
    sb,
    job_id: str,
    file_names: list[str],
    rebar_depth_arrs: list[np.ndarray],
    file_peak_amps: list[np.ndarray],
    file_confs: list[np.ndarray],
    per_file_summary: list[dict],
    swath_spacing_ft: float,
    along_track_ft_per_col: float,
) -> int:
    """
    Write picks to the picks table. Returns number of rows inserted.
    Silently no-ops if sb is None (Supabase not configured).
    """
    if not sb:
        return 0

    rows: list[dict] = []
    for fi, (name, depths, amps, confs, summary) in enumerate(zip(
        file_names, rebar_depth_arrs, file_peak_amps, file_confs, per_file_summary,
    )):
        n = len(depths)
        if n == 0:
            continue
        step = max(1, n // PICKS_PER_SCAN_LINE)

        gps        = summary.get("gps") or {}
        gps_coords = gps.get("coordinates") or []

        y_ft     = fi * swath_spacing_ft
        x_extent = max(n - 1, 1) * along_track_ft_per_col

        for ti in range(0, n, step):
            lat = lon = None
            if gps_coords:
                ci = min(int(ti / max(n, 1) * len(gps_coords)),
                         len(gps_coords) - 1)
                pt = gps_coords[ci]
                # GPS coords stored as [(lat, lon), ...]
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    lat, lon = float(pt[0]), float(pt[1])

            rows.append({
                "job_id":       job_id,
                "scan_line_id": name,
                "trace_index":  int(ti),
                "depth_in":     float(depths[ti]),
                "amplitude":    float(amps[ti])  if ti < len(amps)  else None,
                "confidence":   float(confs[ti]) if ti < len(confs) else None,
                "lat":          lat,
                "lon":          lon,
                "x_ft":         float((ti / max(n - 1, 1)) * x_extent),
                "y_ft":         float(y_ft),
            })

    inserted = 0
    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        chunk = rows[start:start + INSERT_BATCH_SIZE]
        try:
            sb.table("picks").insert(chunk).execute()
            inserted += len(chunk)
        except Exception as exc:
            print(f"[picks] insert batch failed ({start}..{start+len(chunk)}): {exc}",
                  flush=True)
    return inserted


def estimate_along_track_ft_per_col(
    per_file_summary: list[dict],
    grid_cols: int,
) -> float:
    """Match pipeline.py's haversine-based along-track scale, defaulting to 1.0 ft/col."""
    from math import radians, cos, sin, sqrt, atan2
    try:
        total_ft = 0.0
        for f in per_file_summary:
            gps = f.get("gps")
            if not gps:
                continue
            coords = gps.get("coordinates") or []
            if len(coords) < 2:
                continue
            lat1, lon1 = radians(coords[0][0]),  radians(coords[0][1])
            lat2, lon2 = radians(coords[-1][0]), radians(coords[-1][1])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
            total_ft += 2 * 6_371_000 * atan2(sqrt(a), sqrt(1 - a)) * 3.28084
        if total_ft > 0 and grid_cols > 0:
            return total_ft / grid_cols
    except Exception:
        pass
    return 1.0
