"""
server/ingest_gps.py
NMEA GPS parsing for GSSI .dzg companion files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _nmea_lat(raw: str, hemi: str) -> float:
    raw = raw.strip()
    if not raw:
        return 0.0
    dd = int(float(raw) / 100)
    mm = float(raw) - dd * 100
    lat = dd + mm / 60.0
    return -lat if hemi.strip().upper() == "S" else lat


def _nmea_lon(raw: str, hemi: str) -> float:
    raw = raw.strip()
    if not raw:
        return 0.0
    dd = int(float(raw) / 100)
    mm = float(raw) - dd * 100
    lon = dd + mm / 60.0
    return -lon if hemi.strip().upper() == "W" else lon


def parse_dzg(dzg_path: Path) -> list[tuple[float, float]]:
    """Parse a GSSI .dzg GPS log → list of (lat, lon) per trace."""
    coords: list[tuple[float, float]] = []
    if not dzg_path.exists():
        return coords
    try:
        with open(dzg_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("$GPGGA"):
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                try:
                    lat = _nmea_lat(parts[2], parts[3])
                    lon = _nmea_lon(parts[4], parts[5])
                    if lat != 0.0 or lon != 0.0:
                        coords.append((lat, lon))
                except (ValueError, IndexError):
                    continue
    except OSError:
        pass
    return coords


def gps_summary(coords: list[tuple[float, float]]) -> Optional[dict]:
    """Build GPS summary dict. Thins to ≤100 points. Returns None if no coords."""
    if not coords:
        return None
    n = len(coords)
    if n > 100:
        indices = np.linspace(0, n - 1, 100, dtype=int)
        thinned = [coords[int(i)] for i in indices]
    else:
        thinned = coords
    return {
        "lat_start":   coords[0][0],
        "lon_start":   coords[0][1],
        "lat_end":     coords[-1][0],
        "lon_end":     coords[-1][1],
        "coordinates": [[lat, lon] for lat, lon in thinned],
    }
