"""
server/ingest_converters.py
Binary GPR format → CSV converters (GSSI, Sensors & Software, MALA, SEG-Y, stubs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ingest_gps import parse_dzg, gps_summary
from ingest_utils import resample_to_512, zscore_normalize, write_csv, find_companion


# ── GSSI DZT ─────────────────────────────────────────────────────────────────

def convert_dzt(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """Convert GSSI .dzt → CSV (channel 0 only). Parses .dzg for GPS."""
    print(f"[INGEST] convert_dzt: reading {file_path.name} …", flush=True)

    try:
        from readgssi import readgssi as rgssi
    except ImportError as exc:
        raise RuntimeError(
            "readgssi is not installed on this server. "
            "Add 'readgssi>=0.0.18' to requirements.txt and redeploy."
        ) from exc

    try:
        header, arrs, _ = rgssi.readdzt(str(file_path))
    except Exception as exc:
        raise ValueError(f"readgssi could not parse {file_path.name}: {exc}") from exc

    print(
        f"[INGEST] convert_dzt: readdzt OK — "
        f"type(arrs)={type(arrs).__name__}, header keys={list(header.keys())[:8]}",
        flush=True,
    )

    nchan = int(header.get("nchan", 1))

    if isinstance(arrs, dict):
        keys = sorted(arrs.keys())
        ch0  = arrs[keys[0]]
        print(f"[INGEST] convert_dzt: dict API, {len(keys)} channel(s), ch0.shape={ch0.shape}", flush=True)
    elif isinstance(arrs, (list, tuple)):
        ch0  = arrs[0]
        print(f"[INGEST] convert_dzt: list API, {len(arrs)} channel(s), ch0.shape={ch0.shape}", flush=True)
    else:
        if "_r" in file_path.stem.lower():
            arrs = np.fliplr(arrs)
        ch0 = arrs[:, 0::nchan]
        print(f"[INGEST] convert_dzt: legacy array API, ch0.shape={ch0.shape}", flush=True)

    ch0 = np.asarray(ch0, dtype=np.float32)
    n_samples, n_traces = ch0.shape[0], ch0.shape[1]

    if n_traces == 0:
        raise ValueError(f"DZT file {file_path.name} contains 0 traces.")

    if "_r" in file_path.stem.lower() and not isinstance(arrs, (list, tuple, dict)):
        pass  # already flipped above
    elif "_r" in file_path.stem.lower():
        ch0 = np.fliplr(ch0)

    amps_raw = ch0.T
    if n_samples != 512:
        amps = np.stack([resample_to_512(amps_raw[i], n_samples) for i in range(n_traces)])
    else:
        amps = amps_raw.astype(np.float32)
    amps = zscore_normalize(amps)

    csv_path = upload_dir / (file_path.stem + "_ch0.csv")
    print(f"[INGEST] convert_dzt: writing CSV → {csv_path.name} ({n_traces} traces)", flush=True)
    write_csv(csv_path, amps)

    if not csv_path.exists():
        raise RuntimeError(f"CSV write failed — {csv_path} does not exist after write_csv.")

    dzg_path = find_companion(file_path.stem, ".dzg", upload_dir, file_path.parent)
    coords   = parse_dzg(dzg_path) if dzg_path else []
    gps      = gps_summary(coords)

    print(
        f"[INGEST] DZT → {csv_path.name}: {n_traces} traces, "
        f"ch0/{nchan} ch, {n_samples} samp/scan → 512, GPS={'yes' if gps else 'no'}",
        flush=True,
    )
    return csv_path, gps


# ── Sensors & Software DT1 ────────────────────────────────────────────────────

def convert_dt1(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """Convert Sensors & Software .dt1 → CSV."""
    hd_path = find_companion(file_path.stem, ".hd", upload_dir, file_path.parent)
    params: dict[str, str] = {}
    if hd_path:
        with open(hd_path, "r", errors="replace") as f:
            for line in f:
                if ":" in line:
                    k, _, v = line.partition(":")
                    params[k.strip().upper()] = v.strip()

    samples = int(
        params.get("SAMPLES/SCAN",
        params.get("SAMPLES PER SCAN",
        params.get("SAMPLES", "512")))
    )

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
        offset    = i * trace_size + 50
        raw_bytes = raw[offset : offset + samples * 2]
        trace     = np.frombuffer(raw_bytes, dtype=">i2").astype(np.float32)
        amps_list.append(resample_to_512(trace, samples))

    amps     = zscore_normalize(np.stack(amps_list))
    csv_path = upload_dir / (file_path.stem + ".csv")
    write_csv(csv_path, amps)

    print(f"[ingest] DT1 → {csv_path.name}: {n_traces} traces, {samples} samp/trace → 512, normalized", flush=True)
    return csv_path, None


# ── MALA RD3 / RD7 ───────────────────────────────────────────────────────────

def convert_mala(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """Convert MALA .rd3 (16-bit) or .rd7 (32-bit) → CSV."""
    rad_path = find_companion(file_path.stem, ".rad", upload_dir, file_path.parent)
    params: dict[str, str] = {}
    if rad_path:
        with open(rad_path, "r", errors="replace") as f:
            for line in f:
                if ":" in line:
                    k, _, v = line.partition(":")
                    params[k.strip().upper()] = v.strip()

    samples = int(params.get("SAMPLES", "512"))
    ext     = file_path.suffix.lower()
    raw     = file_path.read_bytes()

    if ext == ".rd3":
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    else:
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
    amps = zscore_normalize(amps)

    csv_path = upload_dir / (file_path.stem + ".csv")
    write_csv(csv_path, amps)

    print(f"[ingest] MALA {ext.upper()} → {csv_path.name}: {n_traces} traces, {samples} samp/trace → 512, normalized", flush=True)
    return csv_path, None


# ── SEG-Y ─────────────────────────────────────────────────────────────────────

def convert_segy(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    """Convert SEG-Y → CSV using segyio. GPS from SourceX/SourceY headers."""
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

    if not amps_list:
        raise ValueError(f"SEG-Y file {file_path.name} contains 0 traces.")

    amps     = zscore_normalize(np.stack(amps_list))
    csv_path = upload_dir / (file_path.stem + ".csv")
    write_csv(csv_path, amps)

    gps = None
    if src_x_list and src_y_list:
        xs = np.array(src_x_list)
        ys = np.array(src_y_list)
        if np.abs(xs).max() <= 180.0 and np.abs(ys).max() <= 90.0:
            coords = list(zip(ys.tolist(), xs.tolist()))
            gps    = gps_summary(coords)

    print(
        f"[ingest] SEG-Y → {csv_path.name}: {n_traces} traces, {n_samples} samp/trace → 512, "
        f"normalized, GPS={'yes' if gps else 'no'}",
        flush=True,
    )
    return csv_path, gps


# ── Stub converters ───────────────────────────────────────────────────────────

def convert_ids(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    raise RuntimeError(
        f"IDS GeoRadar format ({file_path.suffix}) is not yet supported. "
        "Export to SEG-Y or CSV from ReflexW and re-upload."
    )


def convert_impulseradar(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    raise RuntimeError(
        f"ImpulseRadar format ({file_path.suffix}) is not yet supported. "
        "Export to SEG-Y or CSV from ImpulseRadar Studio and re-upload."
    )


def passthrough_csv(file_path: Path, upload_dir: Path) -> tuple[Path, Optional[dict]]:
    print(f"[INGEST] CSV pass-through: {file_path.name}", flush=True)
    try:
        raw = np.loadtxt(str(file_path), delimiter=",", dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"CSV {file_path.name} could not be read: {exc}") from exc
    if raw.ndim != 2:
        raise ValueError(f"CSV {file_path.name}: expected 2-D array, got shape {raw.shape}")
    n_traces, n_samples = raw.shape
    if n_traces == 0:
        raise ValueError(f"CSV {file_path.name} contains 0 traces.")
    if n_samples != 512:
        raw = np.stack([resample_to_512(raw[i], n_samples) for i in range(n_traces)])
    amps     = zscore_normalize(raw)
    csv_path = upload_dir / (file_path.stem + "_norm.csv")
    write_csv(csv_path, amps)
    print(f"[INGEST] CSV: {n_traces} traces, {n_samples} samp/trace → 512, normalized", flush=True)
    return csv_path, None
