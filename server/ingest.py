"""
server/ingest.py
Format metadata + main dispatcher for GPR binary → CSV conversion.

Import from sub-modules:
  ingest_gps        — NMEA GPS parsing
  ingest_utils      — resample, write_csv, find_companion
  ingest_converters — per-format converters
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ingest_converters import (
    convert_dzt, convert_dt1, convert_mala, convert_segy,
    convert_impulseradar, passthrough_csv,
)
from ingest_converters_ids import convert_ids

# ── Format metadata ───────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: set[str] = {
    ".csv", ".dzt", ".dt1", ".rd3", ".rd7", ".segy", ".sgy",
    ".dt", ".gec",           # IDS GeoRadar legacy
    ".scan",                 # IDS GeoRadar OneVision binary
    ".iprb", ".iprh",        # ImpulseRadar
}
COMPANION_EXTENSIONS: set[str] = {
    ".dzg", ".hd", ".rad",
    ".pos", ".prcs", ".svy", ".nmea",  # IDS GeoRadar companions
}

FORMAT_INFO: list[dict] = [
    {"ext": ".csv",  "label": "CSV",       "description": "Pass-through CSV"},
    {"ext": ".dzt",  "label": "GSSI DZT",  "description": "GSSI SIR series raw data"},
    {"ext": ".dt1",  "label": "S&S DT1",   "description": "Sensors & Software pulseEKKO"},
    {"ext": ".rd3",  "label": "MALA RD3",  "description": "MALA GeoScience 16-bit"},
    {"ext": ".rd7",  "label": "MALA RD7",  "description": "MALA GeoScience 32-bit"},
    {"ext": ".segy", "label": "SEG-Y",     "description": "SEG-Y revision 1 / 2"},
    {"ext": ".sgy",  "label": "SEG-Y",     "description": "SEG-Y revision 1 / 2"},
]

# ── Dispatch maps ─────────────────────────────────────────────────────────────

MANUFACTURER_FORMAT_MAP: dict[str, any] = {
    "gssi":             convert_dzt,
    "sensors_software": convert_dt1,
    "mala":             convert_mala,
    "ids":              convert_ids,
    "impulseradar":     convert_impulseradar,
    "segy":             convert_segy,
    "csv":              passthrough_csv,
}

_EXT_MAP: dict[str, any] = {
    ".csv":  passthrough_csv,
    ".dzt":  convert_dzt,
    ".dt1":  convert_dt1,
    ".rd3":  convert_mala,
    ".rd7":  convert_mala,
    ".segy": convert_segy,
    ".sgy":  convert_segy,
    ".dt":   convert_ids,
    ".gec":  convert_ids,
    ".scan": convert_ids,
    ".iprb": convert_impulseradar,
    ".iprh": convert_impulseradar,
}


# ── Main dispatcher ───────────────────────────────────────────────────────────

def detect_and_convert(
    file_path: Path,
    upload_dir: Path,
    manufacturer: Optional[str] = None,
) -> tuple[Path, Optional[dict]]:
    """
    Detect format and convert to CSV.  When *manufacturer* is provided,
    dispatch directly without extension sniffing.

    Returns (csv_path, gps_data_or_None).
    """
    ext = file_path.suffix.lower()
    print(
        f"[INGEST] detect_and_convert: {file_path.name}, ext={ext}, manufacturer={manufacturer!r}",
        flush=True,
    )

    if manufacturer and manufacturer in MANUFACTURER_FORMAT_MAP:
        print(f"[INGEST] Using manufacturer converter: {manufacturer}", flush=True)
        return MANUFACTURER_FORMAT_MAP[manufacturer](file_path, upload_dir)

    converter = _EXT_MAP.get(ext)
    if converter is None:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return converter(file_path, upload_dir)
