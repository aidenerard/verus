"""
run.py — GPR bridge deck delamination inference: re-export shim + CLI.

Importable by server.py:
    from run import (CNN1D, load_csv, run_inference, render_cscan_b64,
                     extract_bscan_b64, extract_peak_info, build_prob_grid,
                     build_extra_grids, render_rebar_depth_b64,
                     render_amplitude_b64, compute_confidence_metrics,
                     INFER_BATCH, MAX_GRID_ROWS, MAX_GRID_COLS, DEVICE)

CLI usage:
    python run.py --input /path/to/csvs --model model.pth [--output results.json]
"""

# ── Re-exports (keep server.py imports working unchanged) ─────────────────────

from model import (                                         # noqa: F401
    CNN1D, TemporalAttention, THRESHOLD, DC_OFFSET,
    N_SAMPLES, INFER_BATCH, DEVICE,
)
from data import load_csv                                   # noqa: F401
from inference import (                                     # noqa: F401
    run_inference, make_predictions_list, extract_bscan_b64,
)
from grids import (                                         # noqa: F401
    MAX_GRID_ROWS, MAX_GRID_COLS,
    extract_peak_info, build_prob_grid, build_extra_grids,
)
from render import (                                        # noqa: F401
    render_cscan_b64, render_rebar_depth_b64, render_amplitude_b64,
    compute_confidence_metrics,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

import argparse
import gc
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import torch

_DEFAULT_MODEL = Path("models/model_v13.pth")
_DEFAULT_INPUT = Path(".")


def _resolve_inputs(argv: list[str]) -> list[Path]:
    if not argv:
        found = sorted(_DEFAULT_INPUT.rglob("FILE____*.csv"))
        if not found:
            sys.exit(f"No FILE____*.csv files found in {_DEFAULT_INPUT}")
        return found
    paths = [Path(p) for p in argv]
    if len(paths) == 1 and paths[0].is_dir():
        found = sorted(paths[0].rglob("FILE____*.csv"))
        if not found:
            sys.exit(f"No FILE____*.csv files found in {paths[0]}")
        return found
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"File(s) not found: {missing}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Verus GPR Inference")
    parser.add_argument("--input",     help="Input folder or CSV file(s)")
    parser.add_argument("--model",     help="Path to model .pth file")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output",    help="Write JSON results to this path")
    parser.add_argument("--dpi",       type=int, default=72)
    parser.add_argument("inputs",      nargs="*",
                        help="Positional CSV files / folder")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else _DEFAULT_MODEL
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    threshold = args.threshold if args.threshold is not None else THRESHOLD
    csv_files = _resolve_inputs([args.input] if args.input else args.inputs)

    print("=" * 60, flush=True)
    print("Verus GPR Bridge Deck Inference", flush=True)
    print(f"  Model      : {model_path}", flush=True)
    print(f"  Device     : {DEVICE}", flush=True)
    print(f"  Threshold  : {threshold}", flush=True)
    print(f"  Input files: {len(csv_files)}", flush=True)
    print("=" * 60, flush=True)

    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Loaded model — {n_params:,} trainable parameters", flush=True)

    t0          = time.perf_counter()
    file_preds: list[np.ndarray] = []
    file_confs: list[np.ndarray] = []
    file_names: list[str]        = []
    per_file: list[dict]         = []
    total_sigs = 0

    print(f"\n  {'File':36} {'Signals':>8} {'Sound%':>8} {'Delam%':>8}", flush=True)
    print(f"  {'-'*63}", flush=True)

    for fpath in csv_files:
        try:
            signals = load_csv(fpath)
        except Exception as e:
            print(f"  WARNING  {fpath.name}: {e}", flush=True)
            continue

        preds, confs = run_inference(model, signals, threshold=threshold)
        del signals
        gc.collect()

        n       = len(preds)
        n_snd   = int(preds.sum())
        pct_del = (n - n_snd) / n * 100
        pct_snd = n_snd / n * 100

        tag = f"{fpath.parent.name}/{fpath.name}"
        print(f"  {tag:36} {n:>8,} {pct_snd:>7.1f}% {pct_del:>7.1f}%", flush=True)

        file_preds.append(preds)
        file_confs.append(confs)
        file_names.append(str(fpath))
        per_file.append({"filename": fpath.name, "signals": n, "delam_pct": round(pct_del, 2)})
        total_sigs += n

    elapsed   = time.perf_counter() - t0
    all_preds = np.concatenate(file_preds)
    delam_pct = round(int((all_preds == 0).sum()) / total_sigs * 100, 2)
    sound_pct = round(100.0 - delam_pct, 2)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Files: {len(file_preds)}  |  Signals: {total_sigs:,}  |  "
          f"Sound: {sound_pct}%  |  Delam: {delam_pct}%  |  "
          f"Time: {elapsed:.2f}s", flush=True)

    print("\n  Rendering C-scan …", flush=True)
    cscan_b64   = render_cscan_b64(file_preds, file_confs, file_names, dpi=args.dpi)
    predictions = make_predictions_list(file_names, file_preds, file_confs)

    output = {
        "signals_analyzed":  total_sigs,
        "delamination_pct":  delam_pct,
        "sound_pct":         sound_pct,
        "analysis_time_sec": round(elapsed, 2),
        "cscan_image":       cscan_b64,
        "per_file_summary":  per_file,
        "predictions_count": len(predictions),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"  Results JSON → {out_path.resolve()}", flush=True)
    else:
        print("\n" + json.dumps({k: v for k, v in output.items()
                                 if k != "cscan_image"}, indent=2))
        print(f'  "cscan_image": "<{len(cscan_b64)} chars base64>"')


if __name__ == "__main__":
    main()
