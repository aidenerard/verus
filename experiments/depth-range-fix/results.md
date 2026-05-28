# Experiment D — Depth-Range Fix (MAX_DEPTH_MM=120)

## Configuration

| Parameter       | Value  |
|-----------------|--------|
| `MAX_DEPTH_MM`  | 120.0 (was 300.0) |
| `Dropout rate`  | 0.3 (unchanged) |
| Architecture    | HorizonCNN baseline (depth_head only) |
| `SKIP_SWATHS`   | {0, 4} — top-mat only, actual data range 0-120mm |
| Output          | `horizon_model_depth_fix.pth` |

## Motivation

Baseline set `MAX_DEPTH_MM=300` but training data (with `SKIP_SWATHS={0,4}`) only
contains depths 0-120mm (top-mat rebar). Labels `y = depth_mm / 300` place all
ground-truth values in [0, 0.4], leaving 60% of the model's sigmoid output range
permanently unused. Correcting to `MAX_DEPTH_MM=120` makes labels span [0, 1] fully.

Full TS dataset distribution (all 14 files):
- Min: 0mm, Max: 485mm, Mean: 79mm, p95: 432mm
- 0-120mm range is solely due to SKIP_SWATHS={0,4} excluding bottom-mat swath 0

## Results

| Run          | Best val MAE (mm) | Best val MAE (in) | Delta vs baseline |
|--------------|-------------------|-------------------|-------------------|
| Baseline     | 14.78             | 0.582             | —                 |
| Experiment D | TBD               | TBD               | TBD               |

## Training log

```
(paste Kaggle output here)
```

## Model weights

Google Drive file ID: TBD
URL: TBD

## Verdict

[ ] Promote to main
[ ] Discard
[ ] Iterate

**Reason:** TBD
