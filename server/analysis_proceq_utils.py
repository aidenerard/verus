"""
server/analysis_proceq_utils.py
Small self-contained helpers for the Proceq pipeline. Split out of
analysis_proceq.py to keep that module under the 300-line cap.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def group_files_by_swath(data_dir: str) -> dict:
    """
    Auto-group a flat upload of Proceq files into swaths.

    Naming convention:
      PRC_NNNNNN.scan — odd = trace data, even = processed
      Swath_NNNN.pos  — GPS for swath NNNN
      CScan_NN.CScan  — corrosion amplitude for swath NN

    Files-per-swath is autodetected from len(PRC) / len(pos). The hardcoded
    16 from the prior pass broke the Terracon dataset (which has 8 PRC per
    swath); /pos count handles both layouts.

    Uses rglob so flat uploads AND nested local layouts both work.
    Returns: {swath_idx: {odd_scans, even_scans, pos_file, cscan_file}}
    """
    data_path = Path(data_dir)
    swaths: dict[int, dict] = {}

    prc_paths = sorted(data_path.rglob("PRC_*.scan"))
    pos_paths = sorted(data_path.rglob("Swath_*.pos"))

    if len(pos_paths) == 0:
        print("[GROUP] WARNING: no pos files found — falling back to 16 PRC per swath",
              flush=True)
        files_per_swath = 16
    else:
        files_per_swath = max(1, len(prc_paths) // len(pos_paths))
        print(f"[GROUP] auto-detected {files_per_swath} PRC files per swath "
              f"({len(prc_paths)} PRC / {len(pos_paths)} pos)", flush=True)

    for f in prc_paths:
        try:
            num = int(f.stem.replace("PRC_", ""))
        except ValueError:
            continue
        swath_idx = math.ceil(num / files_per_swath)  # 1-indexed
        bucket = swaths.setdefault(swath_idx, {
            "odd_scans": [], "even_scans": [], "pos_file": None, "cscan_file": None,
        })
        if num % 2 == 1:
            bucket["odd_scans"].append(str(f))
        else:
            bucket["even_scans"].append(str(f))

    for f in sorted(data_path.rglob("Swath_*.pos")):
        try:
            num = int(f.stem.replace("Swath_", ""))
        except ValueError:
            continue
        if num in swaths:
            swaths[num]["pos_file"] = str(f)

    for f in sorted(data_path.rglob("CScan_*.CScan")):
        try:
            num = int(f.stem.replace("CScan_", ""))
        except ValueError:
            continue
        if num in swaths:
            swaths[num]["cscan_file"] = str(f)

    print(f"[GROUP] found {len(swaths)} swaths from uploaded files", flush=True)
    for idx, s in sorted(swaths.items()):
        print(f"  swath {idx}: {len(s['odd_scans'])} odd + "
              f"{len(s['even_scans'])} even PRC, "
              f"pos={'yes' if s['pos_file'] else 'no'}, "
              f"cscan={'yes' if s['cscan_file'] else 'no'}",
              flush=True)
    return swaths


def median_swath_length(pos_files: list[str]) -> float:
    """Median of max along-track distance reported in <VALID_MARKERS_SWEEP> blocks.
    Returns 35.0 m fallback if no .pos files are parseable."""
    lengths: list[float] = []
    for path in pos_files:
        try:
            dists, in_sec, n_exp = [], False, -1
            for s in (l.strip() for l in open(path, errors="replace")):
                if s == "<VALID_MARKERS_SWEEP>":
                    in_sec = True; n_exp = -1; continue
                if in_sec and n_exp < 0:
                    try: n_exp = int(s)
                    except: pass
                    continue
                if in_sec and n_exp > 0 and len(s.split(",")) == 4:
                    try: dists.append(float(s.split(",")[0]))
                    except: pass
                if in_sec and len(dists) >= n_exp > 0: break
            if dists: lengths.append(max(dists))
        except Exception:
            pass
    return float(np.median(lengths)) if lengths else 35.0
