import os

from .gpr_ensemble import (
    GPRModelEnsemble,
    preprocess_trace,
    HORIZON_PATH,
    THICKNESS_PATH,
    CORROSION_PATH,
)


def get_ensemble():
    """
    Load GPRModelEnsemble if at least horizon weights exist; otherwise None.
    Server must work without models loaded — callers should treat a None
    return as 'fall back to signal-processing outputs only'.
    """
    if not os.path.exists(HORIZON_PATH):
        return None
    return GPRModelEnsemble(
        horizon_path=HORIZON_PATH,
        thickness_path=THICKNESS_PATH if os.path.exists(THICKNESS_PATH) else None,
        corrosion_path=CORROSION_PATH if os.path.exists(CORROSION_PATH) else None,
    )


__all__ = ["GPRModelEnsemble", "preprocess_trace", "get_ensemble"]
