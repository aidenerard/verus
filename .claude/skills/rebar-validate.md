# rebar-validate

Use after training or tuning the rebar depth model (HorizonCNN / RebarDepthCNN) to get MAE and RMSE against Infrasense ground truth.

## What it runs

`server/test_rebar_validation.py` — loads the rebar model, runs direct preprocessing (DC-remove + max-abs normalise + Hilbert envelope, matching training exactly), matches predictions to ground truth by scan index, reports per-file and per-bridge MAE/RMSE/bias.

Target: overall MAE < 0.3 inches.

## Data

Ground truth is from Ken Infrasense Data #1 — two WisDOT bridges:

| Bridge | DZT files | GT CSV | Scan range |
|---|---|---|---|
| B170020 | `WISDOT24_095–098 P_1.DZT` | `B170020 Rebar Depth Report.csv` | 4 files, channels 1–2 |
| B440029 | `WISDOT24_799–807 P_1.DZT` | `B440029 Rebar Depth Report.csv` | 5 files (odd numbers), channels 1–2 |

Raw data root: `C:\Users\quack\Documents\Projects\Verus\Data\Ken Infrasense Data #1\`

Ground truth CSV columns: `preProcessedFileName`, `scanNumber`, `L2Depth_inches`

## Usage

```bash
# From the repo root — must run from server/ because of relative imports
cd server
python test_rebar_validation.py
```

Requires `readgssi` installed (`pip install readgssi`). Without it the DZT loading step is skipped and results will be empty.

## Hardcoded paths

`DATA_DIR` and `MODEL_PATH` in `test_rebar_validation.py` are hardcoded to local paths. If the data or model location changes, update those constants at the top of the file:

```python
DATA_DIR   = Path(r"C:\Users\quack\Documents\Projects\Verus\Data\Ken Infrasense Data #1")
MODEL_PATH = ROOT / "models" / "rebar_model.pth"
```

## Output

Per-file table: `file_num  ch  n_traces  matched  MAE(in)  RMSE(in)  pred=[min"-max"]`

Per-bridge summary: `n  MAE  RMSE  bias` with PASS/NEEDS WORK vs 0.3" target.
