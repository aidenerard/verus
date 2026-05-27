# gen-synthetic

Use when expanding the delamination training dataset — either to rebalance classes, add noise augmentation coverage, or increase total signal count before a Kaggle training run.

## Two synthetic data strategies

### Fast (numpy Ricker wavelet) — local, minutes
Generates physically-plausible A-scans using Ricker wavelets with randomised reflector depths, amplitudes, and noise. No GPU needed.

```bash
# From the repo root
python scripts/generate_synthetic_fast.py \
  --out data/csv/synthetic_numpy/ \
  --n-sound 25000 \
  --n-delam 25000
```

Output: `FILE____*.csv` files in SDNET2021 format, ready to drop into Kaggle dataset `aidenerard/synthetic-data`.

### Physics-based (gprMax B-scan) — Kaggle GPU notebooks, hours
Full electromagnetic simulation. Runs as 20 parallel Kaggle notebooks (10 sound + 10 delaminated), each covering a slice of the simulation budget. Scripts are in `bscan_parts/`.

```bash
# After downloading all Kaggle output CSVs to a local folder:
python scripts/combine_bscan_parts.py \
  --parts-dir /path/to/downloaded/parts \
  --out data/csv/synthetic_gprmax/
```

## Datasets in use for delamination training

The Kaggle training script (`kaggle_push/cnn.py`) pulls from three Kaggle datasets:

| Kaggle dataset | Local source | Contents |
|---|---|---|
| `aidenerard/all-bridges-csv` | `data/csv/sdnet2021/` | 5 SDNET2021 bridges, ~658K signals with delamination labels |
| `aidenerard/synthetic-data` (numpy) | `data/csv/synthetic_numpy/` | 50K fast Ricker-wavelet synthetic signals |
| `aidenerard/synthetic-data` (gprMax) | `data/csv/synthetic_gprmax/` | ~50K physics-simulation signals |

## When to regenerate

- Adding a new bridge's data: convert it to SDNET2021 CSV format, add to the Kaggle dataset, retrain.
- Class imbalance shifted (new real data is heavily one-sided): regenerate synthetic to rebalance.
- Augmentation coverage gap: targeted synthetic generation (e.g., more highly attenuated signals).

## Notes on real datasets

| Dataset | Format | Use |
|---|---|---|
| SDNET 2021 (raw) | `.xlsx` in `Data/SDNET 2021 GPR Data/` | Source of truth for delamination labels — 206 files, 5 bridges |
| Terracon Proceq | `.scan` in `Data/Stephen Terracon Cornbread/Data/` | HorizonCNN rebar training only — NOT used for delamination |
| Infrasense (Ken) | `.DZT` in `Data/Ken Infrasense Data #1/` | Rebar depth validation only (`rebar-validate` skill) |

Synthetic data supplements SDNET2021 only. Terracon and Infrasense data use separate training pipelines (`colab_train_horizon.py`, `notebooks/train_rebar_horizon.ipynb`).
