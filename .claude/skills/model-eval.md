# model-eval

Use after any model training run, hyperparameter change (focal loss alpha, threshold, architecture), or before a demo to get a current accuracy baseline.

## What it does

Runs `scripts/eval_model.py` against a checkpoint and data directory. Outputs to `eval_results/<timestamp>/`:
- `metrics.json` — precision, recall, F1, FNR, confusion matrix, PR-AUC at both the requested threshold and the best-F1 threshold
- `per_bridge.csv` — per-bridge breakdown (each data subdirectory treated as one bridge)
- `confusion_matrix.png`, `pr_curve.png`, `threshold_sweep.png`

## Usage

```bash
# From the repo root (verus/)
python scripts/eval_model.py \
  --model server/model.pth \
  --data  data/csv/sdnet2021/

# With a specific threshold (e.g., the one in model_config.json)
python scripts/eval_model.py \
  --model server/model.pth \
  --data  data/csv/ \
  --threshold 0.65

# All data sources combined
python scripts/eval_model.py \
  --model server/model.pth \
  --data  data/csv/
```

## Notes

- Reads `model_config.json` from the same directory as `model.pth` automatically. If absent, defaults to `CNN1D(in_channels=2)`.
- Expects SDNET2021 `FILE____.csv` format. Also accepts plain CSVs; if no `FILE____` files are found it falls back to `*.csv`.
- Plots require `matplotlib`. Metrics JSON and CSV are always saved regardless.
- Label convention: `1=sound`, `0=delaminated`. Positive class for all metrics = delaminated.
