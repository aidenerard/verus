"""
server/processing_state.py
Defaults and helpers for the analysis_jobs.processing_state JSONB column.
"""
from __future__ import annotations

import hashlib
import json


DEFAULT_PROCESSING_STATE: dict = {
    "time_zero_shifts": {},
    "filters": [],
    "gridding": {
        "algorithm":         "nearest",
        "search_radius_ft":  10.0,
        "edge_clip":         True,
        "anisotropy_angle":  0.0,
        "anisotropy_ratio":  1.0,
        "cell_size_ft":      0.5,
    },
    "gps_latency_ms": 0,
    "needs_regrid":   False,
}


def settings_hash(d: dict) -> str:
    """Stable short hash of a settings dict — keys regrid result cache."""
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def merge_state(current: dict | None, patch: dict) -> dict:
    """Shallow-merge a patch into the current state, preserving defaults."""
    out = dict(DEFAULT_PROCESSING_STATE)
    if current:
        out.update(current)
    out.update(patch)
    return out
