# Experiment: gprMax Pretraining for HorizonCNN

**Branch:** research/accuracy-roadmap
**Status:** pipeline built, training pending Exp E results

---

## Setup

| Parameter | Value |
|---|---|
| Template | `rebar2mat` (dual-mat: top + bottom rebar, 60–120mm separation) |
| Depth range | 40–250mm (extended from 180mm to cover B440029 max 234mm) |
| εr range | 4–12 |
| Antenna frequencies | 900 / 1500 / 2000 MHz |
| Normalization ceiling (pretrain) | 120mm (depths >120mm clipped to 1.0 — head is discarded anyway) |
| Pretrain epochs | 50 (early stop patience=15) |
| Fine-tune script | `train_rebar_exp_e.py --reinit-head` |
| Loss | SmoothL1Loss |

## Advisory findings (before training)

- ML Advisor: run Exp E first; pretraining is low priority if B440029 MAE < 2.0" after Exp E.
- Geophysics Advisor: dual-mat template is physically meaningful. Asphalt overlay is the largest unaddressed gap (not in template). 2D spreading affects amplitude but not arrival time; antenna coupling absence is the key morphological mismatch — mitigated by ±30 sample time-shift augmentation.

## Synthetic training data

| Metric | Value |
|---|---|
| Number of synthetic traces | _TBD (run after Exp E)_ |
| Source | gprMax dual-mat rebar2mat template |
| Kaggle GPU sessions needed | ~6 (T4, 9K traces/session) |

## Pretrain results (on synthetic val)

| Metric | Value |
|---|---|
| Pretrain val MAE | _TBD mm_ |
| Pretrain val MAE (inches) | _TBD in_ |

## Fine-tune results vs baseline

Val bridge: **B170020** (Infrasense GSSI, held-out gold standard)

| Metric | Exp E only (no pretrain) | Exp E + gprMax pretrain | Delta |
|---|---|---|---|
| B170020 MAE mm | _TBD_ | _TBD_ | _TBD_ |
| B170020 MAE in | _TBD_ | _TBD_ | _TBD_ |
| B440029 MAE mm | _TBD_ | _TBD_ | _TBD_ |
| B440029 MAE in | _TBD_ | _TBD_ | _TBD_ |

## Verdict

_Pending Exp E results. Promote if B170020 MAE improves by >10% vs Exp E alone and B440029 MAE < 30mm (1.18"). Discard if no improvement._

---

## When to run pretraining

**Condition:** Exp E B440029 MAE > 2.5" (63mm) after running `train_rebar_exp_e.py`.

**If yes:** Run 50K dual-mat synthetic sweep on Kaggle/RunPod, then:
```bash
python train_rebar_pretrain.py data/synthetic_rebar_gprmax_dualmat.npz
python train_rebar_exp_e.py --reinit-head   # default; loads horizon_model_pretrained.pth
```

**If no (B440029 MAE < 2.0"):** Skip pretraining. Move to Exp F (ordinal ranking loss).
