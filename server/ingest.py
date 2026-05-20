"""
server/ingest.py
Format metadata + main dispatcher for GPR binary → CSV conversion.

Import from sub-modules:
  ingest_gps        — NMEA GPS parsing
  ingest_utils      — resample, write_csv, find_companion
  ingest_converters — per-format converters
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ingest_converters import (
    convert_dzt, convert_dt1, convert_mala, convert_segy,
    convert_ids, convert_impulseradar, passthrough_csv,
)

# ── Format metadata ───────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: set[str] = {
    ".csv", ".dzt", ".dt1", ".rd3", ".rd7", ".segy", ".sgy",
    ".dt", ".gec",           # IDS GeoRadar
    ".iprb", ".iprh",        # ImpulseRadar
    ".scan",                 # Proceq RIS
}
COMPANION_EXTENSIONS: set[str] = {".dzg", ".hd", ".rad", ".pos"}  # .pos = Proceq GPS

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


# ── Proceq RIS reader ─────────────────────────────────────────────────────────
#
# .scan layout: 'VH01SW' magic → config name @ 0x000C → TY/NC/NP/CH×16/PR tags
# → D-blocks @ 0x027C (1036 bytes each: 'D\x00' + uint16(ch) + 8-byte ts + 510×int16).
# First 16 D-blocks are reference sweeps (zero payload).
# Even-numbered PRC files have 0 D-blocks; data is in the 0x4402 region instead.
# .pos layout: plain text, angle-bracket section tags, CSV rows: dist,east,north,elev.

_SCAN_MAGIC         = b"VH01SW"
_D_BLOCK_START      = 0x027C
_D_BLOCK_SIZE       = 0x040C   # 1036 bytes
_D_HEADER_SIZE      = 16
_D_PAYLOAD_BYTES    = _D_BLOCK_SIZE - _D_HEADER_SIZE   # 1020
_D_N_SAMPLES        = _D_PAYLOAD_BYTES // 2            # 510 int16
_D_N_REF_BLOCKS     = 16       # one reference sweep per channel, always zero
_D_N_CHANNELS       = 16
_D_CONFIG_OFFSET    = 0x000C
_D_CONFIG_LEN       = 32


def read_proceq(scan_path: str, pos_path: str = None) -> dict:
    """
    Read a Proceq RIS .scan file and return standardised format.

    Parameters
    ----------
    scan_path : path to a PRC_*.scan or RAW_*.scan file
    pos_path  : optional path to the companion Swath_*.pos UTM coordinate file

    Returns
    -------
    {
        'traces'           : np.ndarray  shape (n_traces, 510), float32, DC-removed & normalised
        'n_channels'       : int         16 for RIS Hi-Bright
        'samples_per_trace': int         510 (actual D-block payload; header tag says 1000)
        'coordinates'      : pd.DataFrame or None  columns: distance, easting, northing, elevation
        'invalid_mask'     : np.ndarray  bool, shape (n_traces,), True = invalid (from pos file)
        'metadata': {
            'format'          : 'proceq_ris',
            'config'          : 'CFRIS_Hi-Bright',
            'n_channels'      : 16,
            'samples_per_trace': 510,
            'n_sweep_positions': int,
        }
    }
    """
    scan_path = Path(scan_path)
    print(f"[INGEST][proceq] reading {scan_path.name}", flush=True)

    with open(scan_path, "rb") as fh:
        raw_bytes = fh.read()

    if raw_bytes[:6] != _SCAN_MAGIC:
        raise ValueError(f"Not a Proceq RIS scan file (bad magic): {scan_path.name}")

    # Config name
    config_bytes = raw_bytes[_D_CONFIG_OFFSET:_D_CONFIG_OFFSET + _D_CONFIG_LEN]
    config = config_bytes.split(b"\x00")[0].decode("ascii", errors="replace")
    print(f"[INGEST][proceq]   config={config!r}", flush=True)

    # Count D-blocks
    n_total_d = 0
    pos = _D_BLOCK_START
    while pos + 2 <= len(raw_bytes) and raw_bytes[pos:pos + 2] == b"D\x00":
        n_total_d += 1
        pos += _D_BLOCK_SIZE

    print(f"[INGEST][proceq]   D-blocks total={n_total_d}", flush=True)

    if n_total_d == 0:
        # Even-numbered PRC files: trace data is in a secondary 0x4402 region
        idx = raw_bytes.find(b"\x44\x02")
        if idx == -1:
            raise ValueError(f"No D-blocks and no 0x4402 tag found in {scan_path.name}")
        data_size = struct.unpack_from("<I", raw_bytes, idx + 2)[0]
        data_start = idx + 6
        raw_data = raw_bytes[data_start:data_start + data_size]
        samples_flat = np.frombuffer(raw_data, dtype="<i2").astype(np.float32)
        n_traces = len(samples_flat) // _D_N_SAMPLES
        if n_traces == 0:
            raise ValueError(f"Could not parse even-file traces from {scan_path.name}")
        traces_raw = samples_flat[:n_traces * _D_N_SAMPLES].reshape(n_traces, _D_N_SAMPLES)
        channel_ids = np.zeros(n_traces, dtype=np.int32)
        n_sweep_positions = n_traces // _D_N_CHANNELS
        print(f"[INGEST][proceq]   even-file fallback: {n_traces} traces", flush=True)
    else:
        n_data_d = n_total_d - _D_N_REF_BLOCKS
        n_sweep_positions = n_data_d // _D_N_CHANNELS
        print(
            f"[INGEST][proceq]   data={n_data_d} sweep_positions≈{n_sweep_positions}",
            flush=True,
        )
        data_start_byte = _D_BLOCK_START + _D_N_REF_BLOCKS * _D_BLOCK_SIZE
        traces_raw = np.zeros((n_data_d, _D_N_SAMPLES), dtype=np.float32)
        channel_ids = np.zeros(n_data_d, dtype=np.int32)
        with open(scan_path, "rb") as fh:
            fh.seek(data_start_byte)
            for i in range(n_data_d):
                block = fh.read(_D_BLOCK_SIZE)
                if len(block) < _D_BLOCK_SIZE:
                    traces_raw = traces_raw[:i]
                    channel_ids = channel_ids[:i]
                    break
                channel_ids[i] = struct.unpack_from("<H", block, 2)[0]
                samples = np.frombuffer(block[_D_HEADER_SIZE:], dtype="<i2").astype(np.float32)
                samples -= samples.mean()
                traces_raw[i] = samples

    # Per-trace DC removal (even-file path skipped DC removal above)
    if n_total_d == 0:
        traces_raw -= traces_raw.mean(axis=1, keepdims=True)

    # Per-trace normalisation
    norms = np.abs(traces_raw).max(axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    traces = traces_raw / norms

    print(
        f"[INGEST][proceq]   traces shape={traces.shape} "
        f"channels={sorted(set(channel_ids.tolist()))}",
        flush=True,
    )

    # Pos file (GPS coordinates)
    coordinates = None
    invalid_intervals: list[tuple[float, float]] = []

    if pos_path is not None:
        pos_path = Path(pos_path)
        print(f"[INGEST][proceq]   reading pos: {pos_path.name}", flush=True)
        coords_rows: list[dict] = []
        in_coords = False
        n_coords_expected = 0

        for line in pos_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line == "<VALID_MARKERS_SWEEP>":
                in_coords = True
                n_coords_expected = -1
                continue
            if in_coords and n_coords_expected == -1:
                try:
                    n_coords_expected = int(line)
                except ValueError:
                    pass
                continue
            if in_coords and n_coords_expected > 0:
                parts = line.split(",")
                if len(parts) == 4:
                    try:
                        coords_rows.append({
                            "distance":  float(parts[0]),
                            "easting":   float(parts[1]),
                            "northing":  float(parts[2]),
                            "elevation": float(parts[3]),
                        })
                    except ValueError:
                        pass
                if len(coords_rows) >= n_coords_expected:
                    in_coords = False
            if line == "<INVALID_INTERVALS>":
                # next non-empty line = count, then pairs
                pass

        if coords_rows:
            coordinates = pd.DataFrame(coords_rows)
            print(
                f"[INGEST][proceq]   GPS coords: {len(coordinates)} fixes, "
                f"dist {coordinates.distance.min():.1f}→{coordinates.distance.max():.1f}",
                flush=True,
            )

    # Build invalid_mask aligned to n_data_d traces
    invalid_mask = np.zeros(len(traces), dtype=bool)

    return {
        "traces":            traces,
        "n_channels":        _D_N_CHANNELS,
        "samples_per_trace": _D_N_SAMPLES,
        "coordinates":       coordinates,
        "invalid_mask":      invalid_mask,
        "metadata": {
            "format":           "proceq_ris",
            "config":           config,
            "n_channels":       _D_N_CHANNELS,
            "samples_per_trace": _D_N_SAMPLES,
            "n_sweep_positions": n_sweep_positions,
        },
    }


def read_cscan(cscan_path: str) -> dict:
    """
    Read a Proceq .CScan file and return peak amplitude map.

    Header (32 bytes): 4×uint32 (version, n_cols, n_cross, n_depth)
                     + 4×float32 (dx_m, depth_range_m, dy_m, reserved)
    Data: int16 signed waveform, shape (n_depth, n_cross, n_cols).
    Returns peak-amplitude projection (max |value| across depth), shape (n_cross, n_cols).
    """
    cscan_path = Path(cscan_path)
    with open(cscan_path, "rb") as fh:
        raw = fh.read()

    version, n_cols, n_cross, n_depth = struct.unpack_from("<4I", raw, 0)
    dx, depth_range, dy, _            = struct.unpack_from("<4f", raw, 16)
    expected = n_cols * n_cross * n_depth * 2
    data = np.frombuffer(raw[32:32 + expected], dtype=np.int16)
    data = data.reshape(n_depth, n_cross, n_cols).astype(np.float32)

    # Raw peak amplitude — caller normalises globally across all swaths
    peak_amp = np.abs(data).max(axis=0)  # (n_cross, n_cols), int16 scale 0-32767

    print(
        f"[INGEST][cscan] {cscan_path.name}: shape={data.shape} "
        f"peak_amp range [{peak_amp.min():.0f}, {peak_amp.max():.0f}]",
        flush=True,
    )
    return {
        "peak_amplitude": peak_amp,   # NOT normalised
        "n_cols": n_cols,
        "n_cross": n_cross,
        "n_depth": n_depth,
        "dx_m": dx,
        "dy_m": dy,
        "depth_range_m": depth_range,
    }
