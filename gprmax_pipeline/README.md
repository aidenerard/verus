# gprMax Synthetic Data Pipeline

Generates labelled GPR training data via physics simulation for two tasks:
- **Rebar depth regression** â€” label = depth in mm
- **Delamination classification** â€” label = 0 (delaminated) / 1 (sound)

---

## Local setup (already done)

```powershell
cd verus
python -m venv .venv-gprmax
.venv-gprmax\Scripts\Activate.ps1
pip install numpy setuptools wheel Cython scipy h5py
pip install --no-build-isolation git+https://github.com/gprMax/gprMax.git
```

## Generate test .in files locally

```powershell
# 5 rebar runs
python gprmax_pipeline/generate_sweep.py rebar --n 5 --seed 42 --out gprmax_pipeline/runs/rebar_test

# 6 delam runs (3 delaminated + 3 sound)
python gprmax_pipeline/generate_sweep.py delam --n 6 --seed 42 --out gprmax_pipeline/runs/delam_test

# Run one through gprMax to verify (~10s on CPU)
.venv-gprmax\Scripts\python.exe -m gprMax gprmax_pipeline\runs\rebar_test\rebar_00000.in
```

---

## Kaggle â€” full scale (50,000 rebar + 50,000 delam)

See **`KAGGLE_INSTRUCTIONS.md`** for the complete step-by-step guide (written for Kaggle beginners).

### Rebar depth (10 parallel notebooks)

1. Go to kaggle.com â†’ New Notebook â†’ switch to **Script** view
2. Paste the entire contents of `gprmax_pipeline/kaggle_rebar.py` as one cell
3. Change `BATCH_ID = 0` at the top (use 0â€“9 for the 10 parallel notebooks)
4. Enable **GPU T4 Ã— 1** in notebook settings
5. Run â€” each notebook takes ~3â€“5 hours and saves `rebar_batch{N}.npz`
6. Download all 10 `.npz` files

**Merge the 10 batches locally:**
```python
import numpy as np, glob

files = sorted(glob.glob("rebar_batch*.npz"))
X    = np.concatenate([np.load(f)["X"]           for f in files])
dep  = np.concatenate([np.load(f)["depth_mm"]    for f in files])
eps  = np.concatenate([np.load(f)["epsr"]        for f in files])
diam = np.concatenate([np.load(f)["diameter_mm"] for f in files])

np.savez_compressed("data/synthetic_rebar_gprmax.npz",
                    X=X, depth_mm=dep, epsr=eps, diameter_mm=diam)
print(X.shape)  # (50000, 1, 512)
```

### Delamination classification (10 parallel notebooks)

1. Same setup, paste `gprmax_pipeline/kaggle_delam.py`
2. Change `BATCH_ID = 0` (use 0â€“9)
3. Each notebook saves a folder `delam_batch{N}/` with CSV files in SDNET2021 format
4. Download all 10 folders and move them to `data/csv/synthetic_gprmax/`
5. The existing `cnn.py` `load_csv()` reads them directly â€” no format conversion needed

---

## File map

| File | Purpose |
|---|---|
| `generate_sweep.py` | Local: generate .in files for any batch size |
| `extract_signals.py` | Local: extract Ez from .out HDF5 files |
| `kaggle_rebar.py` | Kaggle cell: 5,000 rebar runs â†’ NPZ |
| `kaggle_delam.py` | Kaggle cell: 5,000 delam runs â†’ SDNET2021 CSVs |
| `templates/rebar_template.in` | Reference domain template (rebar) |
| `templates/delam_template.in` | Reference domain template (delam) |

## Output formats

**Rebar:** `data/synthetic_rebar_gprmax.npz`
- `X`: `(50000, 1, 512)` float32 â€” DC-removed, max-abs normalised Ez
- `depth_mm`: `(50000,)` float32
- `epsr`: `(50000,)` float32
- `diameter_mm`: `(50000,)` float32

**Delamination:** `data/csv/synthetic_gprmax/FILE____NN_MMMM.csv`
- SDNET2021 format â€” directly loadable by `kaggle_push/cnn.py`
- 1,000 signals per file, 50 files total
- Labels: 1 = sound, 0 = delaminated

## Parameter ranges

| Parameter | Rebar | Delam |
|---|---|---|
| Concrete Îµáµ£ | 4â€“12 | 4â€“10 |
| Ïƒ (S/m) | 0.001â€“0.05 | 0.001â€“0.05 |
| Rebar depth | 40â€“180 mm | â€” |
| Rebar diameter | 10â€“32 mm | â€” |
| Gap depth | â€” | 30â€“120 mm |
| Gap thickness | â€” | 2â€“20 mm |
| Gap fill | â€” | air / water / debris |
| Antenna freq | 900 / 1500 / 2000 MHz | 900 / 1500 / 2000 MHz |
