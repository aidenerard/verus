"""
server/ingest.py
Convert raw GPR binary formats → CSV for inference by run.py load_csv().

Supported primary formats
--------------------------
  .dzt         GSSI SIR series (via readgssi)
  .dt1         Sensors & Software pulseEKKO
  .rd3 / .rd7  MALA GeoScience (16-bit / 32-bit)
  .segy / .sgy SEG-Y rev 1/2 (via segyio)
  .csv         Pass-through (no conversion needed)

Companion / metadata files (never sent to inference):
  .dzg         GSSI GPS log (NMEA sentences)
  .hd          Sensors & Software pulseEKKO header
  .rad         MALA GeoScience header

GPS is extracted where available and returned as:
  {
    "lat_start":   float,
    "lon_start":   float,
    "lat_end":     float,
    "lon_end":     float,
    "coordinates": [[lat, lon], ...]   # thinned to ≤100 points
  }
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Optional heavy dependencies — checked at call time, not import time ───────
# readgssi and segyio are only required when the corresponding format is used.
# We deliberately do NOT import them at module level so that the server starts
# successfully even if they are not installed; a clear RuntimeError is raised
# when the format is actually requested.

# ── Format metadata ───────────────────────────────────────────────────────────
#
# All extensions are stored lowercase.  Every membership check must normalise
# the candidate extension with .lower() first — which detect_and_convert() and
# server.py both do before touching these sets.

SUPPORTED_EXTENSIONS: set[str] = {
    ".csv", ".dzt", ".dt1", ".rd3", ".rd7", ".segy", ".sgy",
    ".dt", ".gec",          # IDS GeoRadar
    ".iprb", ".iprh",       # ImpulseRadar
}
COMPANION_EXTENSIONS: set[str] = {".dzg", ".hd", ".rad"}  # paired metadata / GPS files

FORMAT_INFO: list[dict] = [
    {"ext": ".csv",  "label": "CSV",       "description": "Pass-through CSV (SDNET or amplitude matrix)"},
    {"ext": ".dzt",  "label": "GSSI DZT",  "description": "GSSI SIR series raw data"},
    {"ext": ".dt1",  "label": "S&S DT1",   "description": "Sensors & Software pulseEKKO"},
    {"ext": ".rd3",  "label": "MALA RD3",  "description": "MALA GeoScience 16-bit"},
    {"ext": ".rd7",  "label": "MALA RD7",  "description": "MALA GeoScience 32-bit"},
    {"ext": ".segy", "label": "SEG-Y",     "description": "SEG-Y revision 1 / 2"},
    {"ext": ".sgy",  "label": "SEG-Y",     "description": "SEG-Y revision 1 / 2"},
]


# ── GPS helpers ───────────────────────────────────────────────────────────────

def _nmea_lat(raw: str, hemi: str) -> float:
    """Convert NMEA DDMM.MMMM + hemisphere to decimal degrees."""
    raw = raw.strip()
    if not raw:
        return 0.0
    dd = int(float(raw) / 100)
    mm = float(raw) - dd * 100
    lat = dd + mm / 60.0
    return -lat if hemi.strip().upper() == "S" else lat


def _nmea_lon(raw: str, hemi: str) -> float:
    """Convert NMEA DDDMM.MMMM + hemisphere to decimal degrees."""
    raw = raw.strip()
    if not raw:
        return 0.0
    dd = int(float(raw) / 100)
    mm = float(raw) - dd * 100
    lon = dd + mm / 60.0
    return -lon if hemi.strip().upper() == "W" else lon


def _parse_dzg(dzg_path: Path) -> list[tuple[float, float]]:
    """
    Parse a GSSI .dzg GPS file → list of (lat, lon) per trace.
    Each GPGGA line: $GPGGA,time,lat,N/S,lon,E/W,fix,sats,hdop,alt,M,...
    """
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


def _gps_summary(coords: list[tuple[float, float]]) -> Optional[dict]:
    """
    Build GPS summary dict. Thins to ≤100 points for payload efficiency.
    Returns None if no valid coords.
    """
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


# ── Companion file lookup ─────────────────────────────────────────────────────

def _find_companion(stem: str, ext: str, *search_dirs: Path) -> Optional[Path]:
    """
    Case-insensitive search for a companion file.

    Looks in each directory for any file whose stem matches *stem* (exact, case-
    sensitive — the stem comes from the already-saved upload path) and whose
    extension matches *ext* when both are lowercased.  This handles .DZG, .Dzg,
    .dzg, .RAD, .HDS, etc. from Windows equipment without needing an explicit
    list of uppercase variants.

    Returns the first match found, or None.
    """
    ext_lower = ext.lower()
    for d in search_dirs:
        if not d.is_dir():
            continue
        for candidate in d.iterdir():
            if candidate.suffix.lower() == ext_lower and candidate.stem == stem:
                return candidate
    return None


# ── Signal utilities ──────────────────────────────────────────────────────────

def resample_to_512(signal: np.ndarray, original_samples: int) -> np.ndarray:
    """Resample a 1-D A-scan to 512 samples via linear interpolation."""
    if original_samples == 512:
        return signal.astype(np.float32)
    from scipy.interpolate import interp1d
    t_orig   = np.linspace(0, 1, original_samples)
    t_target = np.linspace(0, 1, 512)
    return interp1d(t_orig, signal.astype(np.float64), kind="linear")(t_target).astype(np.float32)


def _write_csv(csv_path: Path, amps: np.ndarray) -> None:
    """
    Write an (n_signals, 512) float32 array to CSV.
    run.py load_csv() auto-detects this layout: 512 columns triggers the
    '400 <= cols <= 600' branch, treating each row as one A-scan.
    """
    np.savetxt(str(csv_path), amps, delimiter=",", fmt="%.4f")


# ── DZT (GSSI SIR) ───────────────────────────────────────────────────────────

def convert_dzt(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """
    Convert a GSSI .dzt file → CSV using readgssi.
    Only channel 0 is extracted (most bridge decks are single-channel).
    Companion .dzg file is parsed for GPS if present in upload_dir.
    """
    print(f"[INGEST] convert_dzt: reading {file_path.name} …", flush=True)

    # ── Import readgssi — must be listed in requirements.txt ─────────────────
    try:
        from readgssi import readgssi as rgssi
    except ImportError as exc:
        raise RuntimeError(
            "readgssi is not installed on this server. "
            "Add 'readgssi>=0.0.18' to requirements.txt and redeploy."
        ) from exc

    # ── Parse the DZT binary ──────────────────────────────────────────────────
    try:
        header, arrs, _ = rgssi.readdzt(str(file_path))
    except Exception as exc:
        raise ValueError(
            f"readgssi could not parse {file_path.name}: {exc}"
        ) from exc

    print(
        f"[INGEST] convert_dzt: readdzt OK — "
        f"type(arrs)={type(arrs).__name__}, header keys={list(header.keys())[:8]}",
        flush=True,
    )

    # readgssi ≥0.0.16 returns arrs as a dict {channel_idx: (n_samples, n_traces)}.
    # Older versions returned a single interleaved numpy array (n_samples, n_traces*nchan).
    # Handle both cases gracefully.
    nchan = int(header.get("nchan", 1))

    if isinstance(arrs, dict):
        # Modern API: channels already separated — grab channel 0
        keys = sorted(arrs.keys())
        ch0  = arrs[keys[0]]            # (n_samples, n_traces)
        print(f"[INGEST] convert_dzt: dict API, {len(keys)} channel(s), ch0.shape={ch0.shape}", flush=True)
    elif isinstance(arrs, (list, tuple)):
        # List API
        ch0  = arrs[0]
        print(f"[INGEST] convert_dzt: list API, {len(arrs)} channel(s), ch0.shape={ch0.shape}", flush=True)
    else:
        # Legacy interleaved array — de-interleave manually
        if "_r" in file_path.stem.lower():
            arrs = np.fliplr(arrs)
        ch0 = arrs[:, 0::nchan]         # (n_samples, n_traces)
        print(f"[INGEST] convert_dzt: legacy array API, ch0.shape={ch0.shape}", flush=True)

    ch0      = np.asarray(ch0, dtype=np.float32)
    n_samples, n_traces = ch0.shape[0], ch0.shape[1]

    if n_traces == 0:
        raise ValueError(f"DZT file {file_path.name} contains 0 traces.")

    # Flip reverse-pass files (stem ends in _R or _r)
    if "_r" in file_path.stem.lower() and not isinstance(arrs, (list, tuple, dict)):
        pass  # already flipped above for legacy path
    elif "_r" in file_path.stem.lower():
        ch0 = np.fliplr(ch0)

    # Transpose to (n_traces, n_samples) then resample to 512
    amps_raw = ch0.T                    # (n_traces, n_samples)
    if n_samples != 512:
        amps = np.stack([resample_to_512(amps_raw[i], n_samples) for i in range(n_traces)])
    else:
        amps = amps_raw.astype(np.float32)

    csv_path = upload_dir / (file_path.stem + "_ch0.csv")
    print(f"[INGEST] convert_dzt: writing CSV → {csv_path.name} ({n_traces} traces)", flush=True)
    _write_csv(csv_path, amps)

    if not csv_path.exists():
        raise RuntimeError(f"CSV write failed — {csv_path} does not exist after _write_csv.")

    # GPS: case-insensitive search for companion .dzg in upload_dir then source dir
    dzg_path = _find_companion(file_path.stem, ".dzg", upload_dir, file_path.parent)
    coords   = _parse_dzg(dzg_path) if dzg_path else []
    gps      = _gps_summary(coords)

    print(
        f"[INGEST] DZT → {csv_path.name}: {n_traces} traces, "
        f"ch0/{nchan} ch, {n_samples} samp/scan → 512, GPS={'yes' if gps else 'no'}",
        flush=True,
    )
    return csv_path, gps


# ── DT1 (Sensors & Software pulseEKKO) ───────────────────────────────────────

def convert_dt1(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """
    Convert a Sensors & Software .dt1 file → CSV.
    Header (.hd): plain-text key: value pairs.
    Binary layout per trace: 50-byte ASCII trace header + samples × int16 big-endian.
    """
    # Locate companion .hd header — case-insensitive search
    hd_path = _find_companion(file_path.stem, ".hd", upload_dir, file_path.parent)
    params: dict[str, str] = {}
    if hd_path:
        with open(hd_path, "r", errors="replace") as f:
            for line in f:
                if ":" in line:
                    k, _, v = line.partition(":")
                    params[k.strip().upper()] = v.strip()

    # SAMPLES/SCAN is the canonical key; fall back to common alternatives
    samples = int(
        params.get("SAMPLES/SCAN",
        params.get("SAMPLES PER SCAN",
        params.get("SAMPLES", "512")))
    )

    # Each trace = 50-byte header + samples × 2 bytes
    trace_size = 50 + samples * 2
    raw = file_path.read_bytes()
    n_traces = len(raw) // trace_size
    if n_traces == 0:
        raise ValueError(
            f"DT1 file too small: {len(raw)} bytes with {samples} samp/trace "
            f"(need ≥{trace_size} bytes per trace)"
        )

    amps_list = []
    for i in range(n_traces):
        offset = i * trace_size + 50                    # skip 50-byte trace header
        raw_bytes = raw[offset : offset + samples * 2]
        trace = np.frombuffer(raw_bytes, dtype=">i2").astype(np.float32)
        amps_list.append(resample_to_512(trace, samples))

    amps = np.stack(amps_list)                          # (n_traces, 512)
    csv_path = upload_dir / (file_path.stem + ".csv")
    _write_csv(csv_path, amps)

    print(
        f"[ingest] DT1 → {csv_path.name}: {n_traces} traces, {samples} samp/trace",
        flush=True,
    )
    return csv_path, None


# ── MALA (RD3 / RD7) ─────────────────────────────────────────────────────────

def convert_mala(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """
    Convert a MALA .rd3 (16-bit) or .rd7 (32-bit) file → CSV.
    Header (.rad): plain-text key: value pairs.
    Binary layout: packed samples, little-endian, no per-trace header.
    """
    # Locate companion .rad header — case-insensitive search
    rad_path = _find_companion(file_path.stem, ".rad", upload_dir, file_path.parent)
    params: dict[str, str] = {}
    if rad_path:
        with open(rad_path, "r", errors="replace") as f:
            for line in f:
                if ":" in line:
                    k, _, v = line.partition(":")
                    params[k.strip().upper()] = v.strip()

    samples = int(params.get("SAMPLES", "512"))
    ext     = file_path.suffix.lower()

    raw = file_path.read_bytes()
    if ext == ".rd3":
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    else:                                               # .rd7
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32)

    n_traces = len(data) // samples
    if n_traces == 0:
        raise ValueError(
            f"MALA {ext} file too small: {len(raw)} bytes, {samples} samp/trace expected"
        )

    data = data[: n_traces * samples].reshape(n_traces, samples)

    if samples != 512:
        amps = np.stack([resample_to_512(data[i], samples) for i in range(n_traces)])
    else:
        amps = data

    csv_path = upload_dir / (file_path.stem + ".csv")
    _write_csv(csv_path, amps)

    print(
        f"[ingest] MALA {ext.upper()} → {csv_path.name}: {n_traces} traces, {samples} samp/trace",
        flush=True,
    )
    return csv_path, None


# ── SEG-Y ─────────────────────────────────────────────────────────────────────

def convert_segy(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """
    Convert a SEG-Y file → CSV using segyio.
    GPS is read from SourceX / SourceY trace header fields and the coordinate
    scalar; only used if values fall within geographic lon/lat ranges.
    """
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError(
            "segyio is not installed on this server. "
            "Add 'segyio>=1.9.12' to requirements.txt and redeploy."
        ) from exc

    amps_list: list[np.ndarray] = []
    src_x_list: list[float]    = []
    src_y_list: list[float]    = []

    with segyio.open(str(file_path), ignore_geometry=True) as f:
        n_traces  = f.tracecount
        n_samples = len(f.trace[0])

        for i in range(n_traces):
            trace = f.trace[i].astype(np.float32)
            amps_list.append(resample_to_512(trace, n_samples))

            try:
                hdr    = f.header[i]
                scalar = hdr.get(segyio.TraceField.ElevationScalar, 1)
                if scalar == 0:
                    scalar = 1
                elif scalar < 0:
                    scalar = 1.0 / abs(scalar)
                sx = hdr.get(segyio.TraceField.SourceX, 0) * scalar
                sy = hdr.get(segyio.TraceField.SourceY, 0) * scalar
                if sx != 0.0 or sy != 0.0:
                    src_x_list.append(sx)
                    src_y_list.append(sy)
            except Exception:
                pass

    amps     = np.stack(amps_list)
    csv_path = upload_dir / (file_path.stem + ".csv")
    _write_csv(csv_path, amps)

    # Use GPS only if coordinates look geographic (lon/lat range)
    gps = None
    if src_x_list and src_y_list:
        xs = np.array(src_x_list)
        ys = np.array(src_y_list)
        if np.abs(xs).max() <= 180.0 and np.abs(ys).max() <= 90.0:
            coords = list(zip(ys.tolist(), xs.tolist()))  # (lat, lon)
            gps    = _gps_summary(coords)

    print(
        f"[ingest] SEG-Y → {csv_path.name}: {n_traces} traces, {n_samples} samp/trace, "
        f"GPS={'yes' if gps else 'no'}",
        flush=True,
    )
    return csv_path, gps


# ── IDS GeoRadar stub ─────────────────────────────────────────────────────────

def convert_ids(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    raise RuntimeError(
        f"IDS GeoRadar format ({file_path.suffix}) is not yet supported. "
        "Export to SEG-Y or CSV from ReflexW and re-upload."
    )


# ── ImpulseRadar stub ─────────────────────────────────────────────────────────

def convert_impulseradar(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    raise RuntimeError(
        f"ImpulseRadar format ({file_path.suffix}) is not yet supported. "
        "Export to SEG-Y or CSV from ImpulseRadar Studio and re-upload."
    )


# ── CSV pass-through (explicit function for manufacturer dispatch) ────────────

def passthrough_csv(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    print(f"[INGEST] CSV pass-through: {file_path.name}", flush=True)
    return file_path, None


# ── Manufacturer → converter map ──────────────────────────────────────────────

MANUFACTURER_FORMAT_MAP: dict[str, any] = {
    "gssi":            convert_dzt,
    "sensors_software": convert_dt1,
    "mala":            convert_mala,
    "ids":             convert_ids,
    "impulseradar":    convert_impulseradar,
    "segy":            convert_segy,
    "csv":             passthrough_csv,
}

# Extension → converter fallback (used when manufacturer not specified)
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
    Detect file format and convert to CSV.  When *manufacturer* is provided,
    dispatch directly to the correct converter without extension sniffing.

    Parameters
    ----------
    file_path    : Path to the uploaded file (already saved to disk in upload_dir)
    upload_dir   : Directory where converted CSV is written
    manufacturer : Manufacturer key from MANUFACTURER_FORMAT_MAP, or None

    Returns
    -------
    (csv_path, gps_data_or_None)

    Raises
    ------
    ValueError   – unsupported extension or corrupt/unreadable file
    RuntimeError – required library not installed (readgssi, segyio)
    """
    ext = file_path.suffix.lower()
    print(
        f"[INGEST] detect_and_convert: {file_path.name}, ext={ext}, manufacturer={manufacturer!r}",
        flush=True,
    )

    # Manufacturer-based dispatch — skips extension sniffing entirely
    if manufacturer and manufacturer in MANUFACTURER_FORMAT_MAP:
        converter = MANUFACTURER_FORMAT_MAP[manufacturer]
        print(f"[INGEST] Using manufacturer converter: {manufacturer}", flush=True)
        return converter(file_path, upload_dir)

    # Extension-based fallback
    converter = _EXT_MAP.get(ext)
    if converter is None:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return converter(file_path, upload_dir)
