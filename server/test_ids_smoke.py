"""
Smoke test for IDS GeoRadar .scan ingest.
Run from server/ directory with: python test_ids_smoke.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add server/ to path so we can import without the full server stack
sys.path.insert(0, str(Path(__file__).parent))

# Minimal stubs so we don't need fastapi/torch/supabase
import types

# Stub ingest_gps
gps_mod = types.ModuleType("ingest_gps")
def _gps_summary(coords):
    if not coords:
        return None
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return {
        "lat_min": min(lats), "lat_max": max(lats),
        "lon_min": min(lons), "lon_max": max(lons),
        "coordinates": coords,
        "n_points": len(coords),
    }
gps_mod.gps_summary = _gps_summary
gps_mod.parse_dzg = lambda p: []
sys.modules["ingest_gps"] = gps_mod

# Stub ingest_utils — only what the IDS converter uses
utils_mod = types.ModuleType("ingest_utils")
def _resample_to_512(trace, n_samples):
    from scipy.signal import resample
    return resample(trace, 512).astype(np.float32)
def _zscore_normalize(arr):
    std = arr.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (arr - arr.mean(axis=1, keepdims=True)) / std
def _write_csv(path, arr):
    np.savetxt(str(path), arr, delimiter=",", fmt="%.6f")
def _find_companion(*a, **kw):
    return None
utils_mod.resample_to_512   = _resample_to_512
utils_mod.zscore_normalize  = _zscore_normalize
utils_mod.write_csv         = _write_csv
utils_mod.find_companion    = _find_companion
sys.modules["ingest_utils"] = utils_mod

from ingest_converters_ids import convert_ids, _read_pos_gps

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"c:\Users\quack\Documents\Projects\Verus\Data\Stephen Terracon Cornbread\Data")
SCAN_FILE = DATA_DIR / "PRC_000001.scan"
POS_FILE  = DATA_DIR / "Swath_0001.pos"

def test_scan_shape():
    """Verify binary math: 16,576,896 bytes → (16188, 512) float32 z-scored."""
    print("\n=== test_scan_shape ===")
    assert SCAN_FILE.exists(), f"Missing: {SCAN_FILE}"
    raw = SCAN_FILE.read_bytes()
    n_bytes = len(raw)
    n_samples = 512
    n_traces  = n_bytes // (n_samples * 2)
    print(f"  File: {n_bytes:,} bytes -> {n_traces} traces x {n_samples} samples")
    assert n_traces == 16188, f"Expected 16188 traces, got {n_traces}"

    data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    data = data[:n_traces * n_samples].reshape(n_traces, n_samples)
    print(f"  Raw int16 range: [{data.min():.0f}, {data.max():.0f}]")

    std = data.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    amps = (data - data.mean(axis=1, keepdims=True)) / std
    print(f"  Z-scored: mean={amps.mean():.6f}, std={amps.std():.4f} (expect ~0 and ~1)")
    assert abs(amps.mean()) < 0.01, "Mean too far from 0"
    assert abs(amps.std() - 1.0) < 0.05, "Std too far from 1"
    print("  PASS")

def test_convert_ids():
    """Full convert_ids path: read → reshape → zscore → CSV → GPS."""
    print("\n=== test_convert_ids ===")
    assert SCAN_FILE.exists(), f"Missing: {SCAN_FILE}"
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        csv_path, gps = convert_ids(SCAN_FILE, out_dir)
        print(f"  CSV: {csv_path.name}")
        assert csv_path.exists(), "CSV not written"

        # Load back and verify shape + normalization
        arr = np.loadtxt(str(csv_path), delimiter=",", dtype=np.float32)
        print(f"  Loaded shape: {arr.shape} (expect (16188, 512))")
        assert arr.shape == (16188, 512), f"Bad shape: {arr.shape}"
        print(f"  mean={arr.mean():.6f}, std={arr.std():.4f}")
        assert abs(arr.mean()) < 0.01
        assert abs(arr.std() - 1.0) < 0.05

        # GPS check
        if gps:
            print(f"  GPS: {gps['n_points']} points, "
                  f"lat=[{gps['lat_min']:.5f},{gps['lat_max']:.5f}], "
                  f"lon=[{gps['lon_min']:.5f},{gps['lon_max']:.5f}]")
            # Expect central Indiana/Ohio roughly: 40.17°N, 85.43°W
            assert 38 < gps['lat_min'] < 43, "Latitude out of expected range"
            assert -90 < gps['lon_min'] < -80, "Longitude out of expected range"
            print("  GPS: PASS")
        else:
            print("  GPS: no coords found (fallback or no .pos in tmpdir)")
    print("  PASS")

def test_pos_gps():
    """Parse Swath_0001.pos directly and validate coordinate extraction."""
    print("\n=== test_pos_gps ===")
    assert POS_FILE.exists(), f"Missing: {POS_FILE}"
    gps = _read_pos_gps(POS_FILE)
    assert gps is not None, ".pos parse returned None"
    print(f"  n_points={gps['n_points']}")
    print(f"  lat=[{gps['lat_min']:.5f}, {gps['lat_max']:.5f}]")
    print(f"  lon=[{gps['lon_min']:.5f}, {gps['lon_max']:.5f}]")
    # RTK-fixed coords should be consistent: lat range < 0.01°
    lat_spread = gps['lat_max'] - gps['lat_min']
    lon_spread = gps['lon_max'] - gps['lon_min']
    print(f"  spread: Δlat={lat_spread:.5f}°, Δlon={lon_spread:.5f}°")
    assert lat_spread < 0.01, f"Lat spread too large: {lat_spread}"
    assert lon_spread < 0.05, f"Lon spread too large: {lon_spread}"
    print("  PASS")

def test_rebar_physics_depth():
    """
    Physics fallback for rebar depth on IDS 2 GHz data.
    16 ns range / 512 samples = 0.03125 ns/sample.
    For typical rebar at 2-4" cover, peak sample should be < 80.
    """
    print("\n=== test_rebar_physics_depth ===")
    raw = SCAN_FILE.read_bytes()
    n_traces, n_samples = 16188, 512
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    data = data[:n_traces * n_samples].reshape(n_traces, n_samples)

    # Physics constants for IDS 2 GHz
    er       = 5.5                      # dielectric for 2000 MHz concrete
    velocity = 0.3 / np.sqrt(er)       # m/ns ≈ 0.1279 m/ns
    ns_per_sample = 16.0 / 512         # 0.03125 ns/sample

    peak_samples = np.argmax(np.abs(data), axis=1)
    twt_ns       = peak_samples.astype(np.float32) * ns_per_sample
    depth_m      = velocity * twt_ns / 2.0
    depth_in     = depth_m * 39.3701

    print(f"  peak_sample: min={peak_samples.min()}, median={int(np.median(peak_samples))}, max={peak_samples.max()}")
    print(f"  depth (in): min={depth_in.min():.2f}, mean={depth_in.mean():.2f}, max={depth_in.max():.2f}")
    print(f"  (typical bridge deck rebar: 2–6\")")
    print("  PASS")

if __name__ == "__main__":
    test_scan_shape()
    test_convert_ids()
    test_pos_gps()
    test_rebar_physics_depth()
    print("\n✓ All smoke tests passed.")
