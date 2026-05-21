"""
server/ensemble_inference.py
Runs the GPR model ensemble across a Proceq dataset as a parallel stats
layer on top of the existing signal-processing pipeline. Kept separate
from analysis_proceq.py so the deterministic pipeline stays independent
of the (optional) model weights.

Strategy: sample ~per_file traces uniformly from every odd-numbered PRC
file, run them through GPRModelEnsemble.predict (which internally batches
to bound memory), aggregate into a flat model_stats dict.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Optional

import numpy as np


def _load_trace_sample(data_dir: str, per_file: int = 200) -> Optional[np.ndarray]:
    """Uniformly subsample traces from every odd PRC file under data_dir.
    Returns (N, 510) float32, or None if no readable scans."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ingest import read_proceq

    scans = sorted(glob.glob(os.path.join(data_dir, "**", "PRC_*.scan"), recursive=True))
    odd_scans = [
        f for f in scans
        if int(os.path.basename(f).replace("PRC_", "").replace(".scan", "")) % 2 == 1
    ]

    chunks: list[np.ndarray] = []
    for path in odd_scans:
        try:
            result = read_proceq(path)
        except Exception as exc:
            print(f"[ensemble] skip {os.path.basename(path)}: {exc}", flush=True)
            continue
        traces = result.get("traces")
        if traces is None or len(traces) == 0:
            continue
        n = len(traces)
        if n <= per_file:
            chunks.append(np.asarray(traces, dtype=np.float32))
        else:
            idx = np.linspace(0, n - 1, per_file, dtype=int)
            chunks.append(np.asarray(traces[idx], dtype=np.float32))

    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def run_ensemble_stats(
    data_dir: str,
    ensemble,
    per_file: int = 200,
) -> Optional[dict]:
    """
    Sample traces from the dataset, run ensemble.predict, build a flat
    model_stats dict suitable for the job result. Returns None on any
    failure so callers can safely default to signal-processing-only.
    """
    try:
        sample = _load_trace_sample(data_dir, per_file=per_file)
        if sample is None or len(sample) == 0:
            return None
        print(f"[ensemble] predict on {len(sample)} sampled traces", flush=True)
        predictions = ensemble.predict(sample)
    except Exception as exc:
        print(f"[ensemble] inference failed: {exc}", flush=True)
        return None

    depth_in = predictions.get("rebar_depth_in")
    thick_in = predictions.get("deck_thickness_in")
    corr     = predictions.get("corrosion_risk")

    return {
        "mean_depth_inches":     float(depth_in.mean()) if depth_in is not None else None,
        "depth_range":           [float(depth_in.min()), float(depth_in.max())]
                                  if depth_in is not None else None,
        "mean_thickness_inches": float(thick_in.mean()) if thick_in is not None else None,
        "high_risk_pct":         float((corr > 0.5).mean() * 100)
                                  if corr is not None else None,
        "model_version":         "horizon_v1",
        "n_traces_sampled":      int(len(sample)),
    }
