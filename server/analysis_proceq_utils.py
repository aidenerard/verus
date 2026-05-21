"""
server/analysis_proceq_utils.py
Small self-contained helpers for the Proceq pipeline. Split out of
analysis_proceq.py to keep that module under the 300-line cap.
"""
from __future__ import annotations

import numpy as np


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
