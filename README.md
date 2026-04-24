# Verus Project Structure

## data/raw/
Raw unprocessed data files. Never modify these directly.
- `sdnet2021/` — original SDNET2021 CSV files (source of truth)
- `gatech/` — raw DZT files from GT analyst (Infrasense)
- `synthetic_gprmax/` — raw gprMax B-scan output CSVs from Kaggle notebooks

## data/csv/
Processed data in SDNET2021-compatible CSV format, ready for training.
- `sdnet2021/` — 5 bridges, ~657,938 signals
- `synthetic_numpy/` — 50,000 fast numpy synthetic signals (Ricker wavelet)
- `synthetic_gprmax/` — ~50,000 gprMax physics simulation signals
- `gatech/` — GT bridge data after running `ingest_gpr_data.py`

## scripts/
All Python scripts for training, inference, and data processing.
Run all scripts from the `scripts/` directory.

| Script | Purpose |
|---|---|
| `cnn.py` | Train CNN1D delamination classifier |
| `run.py` | Standalone inference / evaluation |
| `generate_synthetic_fast.py` | Fast CPU synthetic data (numpy Ricker) |
| `generate_synthetic_bscan.py` | Physics-based gprMax B-scan data |
| `combine_bscan_parts.py` | Combine Kaggle CSV parts into SDNET2021 format |
| `ingest_gpr_data.py` | Convert raw GPR data from external sources |

## server/
FastAPI inference server for Render deployment.
Self-contained — do not change relative paths inside `server/`.

## models/
Trained model weights.
- `model.pth` — current production model (CNN1D V17, in_channels=2)

## bscan_parts/
20 parallel gprMax generation scripts for Kaggle CPU notebooks.
Each script covers a slice of the total simulation budget:
- `generate_bscan_c1_p01.py` … `p10.py` — 470 sound sims (47/part)
- `generate_bscan_c2_p01.py` … `p10.py` — 313 delam sims (~31/part)
- `combine_bscan_parts.py` — combine downloaded CSVs into SDNET2021 format
