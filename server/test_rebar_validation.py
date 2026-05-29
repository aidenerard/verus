"""
End-to-end rebar depth validation against Infrasense ground-truth CSV.

Run from server/ directory:
    python test_rebar_validation.py

What it tests:
  1. HorizonCNN loads from models/horizon_model.pth
  2. DZT traces resampled to 512 samples, then run through run_rebar_inference()
     (which applies per-trace DC-remove + max-abs normalize internally)
  3. MAE vs L2Depth_inches from the Infrasense depth-report CSVs
  4. Per-file and per-lane breakdown
"""

import sys, csv, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# ── Minimal stubs so we can import without fastapi/supabase ──────────────────
import types

for mod_name in ["fastapi", "fastapi.middleware.cors", "fastapi.responses",
                 "supabase", "psutil", "auth"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# ── Real imports ─────────────────────────────────────────────────────────────
import torch
from model import HorizonCNN, DEVICE, INFER_BATCH
from inference import run_rebar_inference
from ingest_utils import resample_to_512

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = Path(r"C:\Users\quack\Documents\Projects\Verus\Data\Ken Infrasense Data #1")
MODEL_PATH = ROOT / "models" / "horizon_model.pth"

BRIDGES = {
    "B170020": {
        "data_dir": DATA_DIR / "B170020" / "raw data",
        "csv":      DATA_DIR / "B170020 Rebar Depth Report.csv",
        "ns_total": 15.0,
        "freq_mhz": 1600,
        "layout": [
            # (file_num, ch, start_scan, end_scan, reversed)
            (95, 1,  610, 1802, False),
            (95, 2,  617, 1806, False),
            (96, 2,  741, 1930, True),
            (97, 1,  437, 1623, False),
            (97, 2,  443, 1645, False),
            (98, 1,  505, 1703, True),
            (98, 2,  498, 1690, True),
        ],
    },
    "B440029": {
        "data_dir": DATA_DIR / "B440029" / "raw data",
        "csv":      DATA_DIR / "B440029 Rebar Depth Report.csv",
        "ns_total": 15.0,
        "freq_mhz": 1600,
        "layout": [
            (799, 1, 3535, 4163, False),
            (799, 2, 3535, 4163, False),
            (801, 1, 4737, 5373, False),
            (801, 2, 4737, 5373, False),
            (803, 1, 4835, 5462, False),
            (803, 2, 4835, 5462, False),
            (805, 1, 4984, 5614, False),
            (805, 2, 4984, 5614, False),
            (807, 2, 4016, 4643, False),
        ],
    },
}


def load_rebar_model():
    assert MODEL_PATH.exists(), f"horizon_model.pth not found at {MODEL_PATH}"
    rm = HorizonCNN().to(DEVICE)
    rm.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False))
    rm.eval()
    n = sum(p.numel() for p in rm.parameters() if p.requires_grad)
    print(f"  Loaded HorizonCNN: {n:,} params, device={DEVICE}")
    model_cfg = {"max_depth_mm": 120.0}
    return rm, model_cfg


def load_csv_gt(csv_path):
    """
    Returns dict: {(file_stem_upper, scan_number): depth_inches}
    Keys use the preprocessed filename stem (uppercased) and int scan number.
    Skips rows with empty L2Depth_inches.
    """
    gt = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_d = row.get("L2Depth_inches", "").strip()
            if not raw_d:
                continue
            try:
                depth = float(raw_d)
                # preProcessedFileName like "WISDOT24_095 P_1_PREP_ch01.DZT"
                fname = row["preProcessedFileName"].strip().upper()
                scan  = int(row["scanNumber"].strip())
                gt[(fname, scan)] = depth
            except (ValueError, KeyError):
                continue
    return gt


def dzt_key(file_num, ch_num):
    """Build the uppercased preprocessed filename key matching the CSV."""
    return f"WISDOT24_{file_num:03d} P_1_PREP_CH0{ch_num}.DZT"


def run_bridge(bridge_name, cfg, rebar_model, model_cfg):
    try:
        from readgssi import readgssi as rgssi
    except ImportError:
        print("  readgssi not installed — skipping DZT loading")
        return None

    gt      = load_csv_gt(cfg["csv"])
    ns_total = cfg["ns_total"]

    all_pred, all_true = [], []
    print(f"\n  {'File':>6} ch  traces  matched   MAE(in)  RMSE(in)")
    print(f"  {'-'*55}")

    for file_num, ch_num, start_scan, end_scan, reversed_flag in cfg["layout"]:
        fname = f"WISDOT24_{file_num:03d} P_1.DZT"
        fpath = cfg["data_dir"] / fname
        if not fpath.exists():
            print(f"  {file_num:>6}  {ch_num}  MISSING: {fpath}")
            continue

        try:
            header, data, _ = rgssi.readdzt(str(fpath))
        except Exception as e:
            print(f"  {file_num:>6}  {ch_num}  ERROR: {e}")
            continue

        if isinstance(data, dict):
            arr = list(data.values())[ch_num - 1]
        elif isinstance(data, (list, tuple)):
            arr = data[ch_num - 1]
        else:
            nchan = int(header.get("nchan", 1))
            arr = data[:, ch_num - 1 :: nchan]

        arr = np.asarray(arr, dtype=np.float32)
        n_samples_raw = arr.shape[0]     # e.g. 256

        # Trim to layout range
        arr = arr[:, start_scan:end_scan]
        scan_indices = np.arange(start_scan, end_scan)

        if reversed_flag:
            arr = arr[:, ::-1].copy()
            scan_indices = scan_indices[::-1].copy()

        n_traces = arr.shape[1]

        # Transpose to (n_traces, n_samples), resample to 512.
        # run_rebar_inference applies per-trace DC-remove + max-abs normalize internally,
        # matching colab_train_horizon.py preprocessing exactly.
        amps = arr.T.copy()
        if n_samples_raw != 512:
            amps = np.stack([resample_to_512(amps[i], n_samples_raw)
                             for i in range(n_traces)])
        depth_pred, _, _ = run_rebar_inference(
            rebar_model, amps, frequency_mhz=cfg["freq_mhz"],
            model_config=model_cfg,
        )

        # ── Match to ground truth ────────────────────────────────────────────
        key_prefix = dzt_key(file_num, ch_num)
        matched_pred, matched_true = [], []
        for i, scan_idx in enumerate(scan_indices):
            k = (key_prefix, int(scan_idx))
            if k in gt:
                matched_pred.append(float(depth_pred[i]))
                matched_true.append(gt[k])

        if not matched_true:
            print(f"  {file_num:>6}  {ch_num}  {n_traces:>6}  no GT matches")
            continue

        mp   = np.array(matched_pred)
        mt   = np.array(matched_true)
        mae  = np.abs(mp - mt).mean()
        rmse = np.sqrt(((mp - mt) ** 2).mean())
        pred_range = f"[{mp.min():.2f}\"-{mp.max():.2f}\"]"
        print(f"  {file_num:>6}  {ch_num}  {n_traces:>6}  {len(mt):>7}  "
              f"{mae:>8.3f}\"  {rmse:>8.3f}\"  pred={pred_range}")

        all_pred.extend(matched_pred)
        all_true.extend(matched_true)

    if not all_true:
        print("  No ground truth matches found for this bridge.")
        return None

    ap = np.array(all_pred)
    at = np.array(all_true)
    mae  = np.abs(ap - at).mean()
    rmse = np.sqrt(((ap - at) ** 2).mean())
    bias = (ap - at).mean()
    print(f"\n  {bridge_name} TOTAL  n={len(at):,}  "
          f"MAE={mae:.3f}\"  RMSE={rmse:.3f}\"  bias={bias:+.3f}\"")
    print(f"  pred range: [{ap.min():.2f}\", {ap.max():.2f}\"]  "
          f"true range: [{at.min():.2f}\", {at.max():.2f}\"]")
    return {"bridge": bridge_name, "n": len(at), "mae": mae, "rmse": rmse, "bias": bias}


if __name__ == "__main__":
    print("=== HorizonCNN Rebar Depth Validation vs Infrasense Ground Truth ===\n")

    rebar_model, model_cfg = load_rebar_model()

    results = []
    for bname, bcfg in BRIDGES.items():
        print(f"\n--- {bname} ---")
        r = run_bridge(bname, bcfg, rebar_model, model_cfg)
        if r:
            results.append(r)

    if results:
        print("\n=== Summary ===")
        print(f"  {'Bridge':>10}  {'N':>7}  {'MAE':>8}  {'RMSE':>9}  {'Bias':>8}")
        for r in results:
            print(f"  {r['bridge']:>10}  {r['n']:>7,}  "
                  f"{r['mae']:>7.3f}\"  {r['rmse']:>8.3f}\"  {r['bias']:>+7.3f}\"")
        target = 0.3
        overall_mae = np.mean([r["mae"] for r in results])
        print(f"\n  Overall MAE: {overall_mae:.3f}\"  "
              f"(target < {target}\")  "
              f"{'PASS' if overall_mae < target else 'NEEDS WORK'}")
    else:
        print("\nNo results — check readgssi is installed and DZT files exist.")
