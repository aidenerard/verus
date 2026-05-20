"""
server/ingest_converters_ids.py
IDS GeoRadar .scan binary → CSV converter + GPS helpers.
"""

from __future__ import annotations

from math import cos, radians
from pathlib import Path
from typing import Optional

import numpy as np

from ingest_gps import gps_summary
from ingest_utils import resample_to_512, zscore_normalize, write_csv


# ── Public converter ──────────────────────────────────────────────────────────

def convert_ids(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """
    Convert IDS GeoRadar .scan binary → CSV.

    Binary layout: flat int16 little-endian, row-major, shape (n_traces, 512).
    Standard IDS file: 16,576,896 bytes / 2 = 8,288,448 samples → 16,188 traces.
    """
    print(f"[INGEST] convert_ids: reading {file_path.name} …", flush=True)

    raw = file_path.read_bytes()
    n_bytes = len(raw)

    n_samples = 512
    n_traces = n_bytes // (n_samples * 2)

    if n_traces == 0:
        raise ValueError(
            f"IDS .scan file {file_path.name} is too small "
            f"({n_bytes} bytes) for {n_samples} samples/trace."
        )

    data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    data = data[: n_traces * n_samples].reshape(n_traces, n_samples)

    if n_samples != 512:
        amps = np.stack([resample_to_512(data[i], n_samples) for i in range(n_traces)])
    else:
        amps = data

    amps = zscore_normalize(amps)

    csv_path = upload_dir / (file_path.stem + ".csv")
    write_csv(csv_path, amps)

    gps = _parse_ids_pos(file_path, upload_dir)

    print(
        f"[INGEST] IDS .scan → {csv_path.name}: {n_traces} traces, "
        f"{n_samples} samp/trace → 512, normalized, GPS={'yes' if gps else 'no'}",
        flush=True,
    )
    return csv_path, gps


# ── GPS helpers ───────────────────────────────────────────────────────────────

def _parse_ids_pos(scan_path: Path, upload_dir: Path) -> Optional[dict]:
    """
    Locate a companion Swath_XXXX.pos file and extract GPS coordinates.

    Mapping: PRC_000001–000008 → Swath_0001, PRC_000009–000016 → Swath_0002, …
    RAW files use the file number directly as the swath index.
    """
    stem = scan_path.stem          # e.g. "PRC_000001" or "RAW_000008"
    parts = stem.split("_", 1)
    swath_idx = None

    if len(parts) == 2:
        try:
            n = int(parts[1])
            prefix = parts[0].upper()
            if prefix == "PRC":
                # 112 PRC files, 8 channels per swath → swath = ceil(n / 8)
                swath_idx = max(1, (n - 1) // 8 + 1)
            else:
                # RAW files: one channel per swath index
                swath_idx = n
        except ValueError:
            pass

    candidates: list[str] = []
    if swath_idx is not None:
        candidates.append(f"Swath_{swath_idx:04d}.pos")
        candidates.append(f"Swath_{swath_idx}.pos")

    for search_dir in (upload_dir, scan_path.parent):
        for name in candidates:
            pos_path = search_dir / name
            if pos_path.exists():
                return _read_pos_gps(pos_path)
        # Fallback: first .pos in directory
        pos_files = sorted(search_dir.glob("*.pos"))
        if pos_files:
            return _read_pos_gps(pos_files[0])

    return None


def _read_pos_gps(pos_path: Path) -> Optional[dict]:
    """
    Parse Swath_XXXX.pos format into a gps_summary dict.

    Format:
      Header lines start with '<'
      Data lines: encoder_pos, UTM_E, UTM_N, Elev  (comma-separated)
      <INVALID_INTERVALS> section marks encoder ranges with bad GPS — skip them.
    CRS: UTM Zone 16N (EPSG:32616).  Converted to WGS84 lat/lon for gps_summary.
    """
    try:
        import pyproj
        _transformer = pyproj.Transformer.from_crs(
            "EPSG:32616", "EPSG:4326", always_xy=True
        )
        def _to_latlon(utm_e: float, utm_n: float) -> tuple[float, float]:
            lon, lat = _transformer.transform(utm_e, utm_n)
            return lat, lon
    except ImportError:
        def _to_latlon(utm_e: float, utm_n: float) -> tuple[float, float]:
            # Rough approximation for UTM16N when pyproj is unavailable (~10 m error)
            lat = utm_n / 111_320.0
            lon = (utm_e - 500_000.0) / (111_320.0 * abs(cos(radians(lat)))) - 87.0
            return lat, lon

    # Parse invalid encoder intervals to skip bad-GPS rows
    invalid_ranges: list[tuple[float, float]] = []
    in_invalid = False
    coords: list[tuple[float, float]] = []

    with open(pos_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("<"):
                in_invalid = "INVALID_INTERVALS" in line.upper()
                continue
            parts = line.split(",")
            if in_invalid:
                if len(parts) >= 2:
                    try:
                        invalid_ranges.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
                continue
            if len(parts) < 3:
                continue
            try:
                enc   = float(parts[0])
                utm_e = float(parts[1])
                utm_n = float(parts[2])
            except ValueError:
                continue
            # Skip rows whose encoder position falls in an invalid interval
            if any(lo <= enc <= hi for lo, hi in invalid_ranges):
                continue
            try:
                coords.append(_to_latlon(utm_e, utm_n))
            except Exception:
                continue

    return gps_summary(coords) if coords else None
