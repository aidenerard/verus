"""One-time script to adapt Kaggle notebooks for local execution."""
import json

LOCAL_DATA   = r'C:/Users/quack/Documents/Projects/Verus/Data/Stephen Terracon Cornbread/Data'
LOCAL_MODELS = r'C:/Users/quack/Documents/Projects/Verus/verus/server/models'


def set_cell(nb, cell_id, lines):
    for c in nb['cells']:
        if c.get('id') == cell_id:
            c['source'] = lines
            return
    raise KeyError(cell_id)


# ── train_rebar_horizon.ipynb ─────────────────────────────────────────────────
with open('train_rebar_horizon.ipynb') as f:
    nb = json.load(f)

set_cell(nb, 'cell_config', [
    f"DATA_DIR       = r'{LOCAL_DATA}'\n",
    f"WORKING_DIR    = r'{LOCAL_MODELS}'\n",
    "TARGET_SAMPLES = 512\n",
    "MAX_DEPTH_MM   = 300.0  # TS picks are in mm\n",
    "BATCH_SIZE     = 512\n",
    "EPOCHS         = 100\n",
    "PATIENCE       = 15\n",
    "LR             = 1e-3\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Device: {DEVICE}')",
])

set_cell(nb, 'cell_preprocess', [
    "def preprocess_trace(raw_trace, target_samples=512):\n",
    "    if len(raw_trace) != target_samples:\n",
    "        raw_trace = scipy_resample(raw_trace, target_samples)\n",
    "    raw_trace = raw_trace - raw_trace.mean()\n",
    "    max_abs = np.abs(raw_trace).max()\n",
    "    if max_abs > 0:\n",
    "        raw_trace = raw_trace / max_abs\n",
    "    return raw_trace.astype(np.float32)\n",
    "\n",
    "\n",
    "def preprocess_batch(raw_traces, target_samples=512):\n",
    "    if raw_traces.shape[1] != target_samples:\n",
    "        raw_traces = scipy_resample(raw_traces, target_samples, axis=1)\n",
    "    raw_traces = raw_traces - raw_traces.mean(axis=1, keepdims=True)\n",
    "    norms = np.abs(raw_traces).max(axis=1, keepdims=True)\n",
    "    norms[norms == 0] = 1.0\n",
    "    return (raw_traces / norms).astype(np.float32)\n",
    "\n",
    "\n",
    "def mm_to_normalized_depth(depth_mm, max_depth_mm=300.0):\n",
    "    '''Normalize depth labels to 0-1. max_depth_mm=300 covers 0-30 cm.'''\n",
    "    return np.clip(depth_mm / max_depth_mm, 0.0, 1.0)",
])

set_cell(nb, 'cell_data', [
    "def load_rebar_dataset():\n",
    "    scan_files = sorted(glob.glob(os.path.join(DATA_DIR, 'PRC_*.scan')))\n",
    "    odd_scans  = [\n",
    "        f for f in scan_files\n",
    "        if int(os.path.basename(f).replace('PRC_', '').replace('.scan', '')) % 2 == 1\n",
    "    ]\n",
    "    ts_files = sorted(glob.glob(os.path.join(DATA_DIR, 'TS_*_1.txt')))\n",
    "    n_swaths = min(len(odd_scans) // 4, len(ts_files))\n",
    "    print(f'Odd scans: {len(odd_scans)}, TS rebar files: {len(ts_files)}, swaths: {n_swaths}')\n",
    "\n",
    "    all_traces, all_labels, all_swath_ids = [], [], []\n",
    "\n",
    "    for sw_idx in range(n_swaths):\n",
    "        swath_scans = odd_scans[sw_idx * 4 : sw_idx * 4 + 4]\n",
    "        ts_picks    = np.loadtxt(ts_files[sw_idx])  # 3393 values in mm\n",
    "\n",
    "        for scan_path in swath_scans:\n",
    "            raw_traces, n_data = read_proceq_traces(scan_path)\n",
    "            if raw_traces is None or n_data < 10:\n",
    "                continue\n",
    "            N         = len(raw_traces)\n",
    "            x_ts      = np.linspace(0, N - 1, len(ts_picks))\n",
    "            labels_mm = np.interp(np.arange(N), x_ts, ts_picks)\n",
    "            processed = preprocess_batch(raw_traces, TARGET_SAMPLES)\n",
    "            all_traces.append(processed)\n",
    "            all_labels.append(mm_to_normalized_depth(labels_mm, MAX_DEPTH_MM).astype(np.float32))\n",
    "            all_swath_ids.extend([sw_idx] * N)\n",
    "            print(f'  sw{sw_idx+1:02d}  {os.path.basename(scan_path):25s}  '\n",
    "                  f'{N:6d} traces  depth {labels_mm.mean():.1f} mm ({labels_mm.mean()/25.4:.2f} in)')\n",
    "\n",
    "    return all_traces, all_labels, np.array(all_swath_ids), n_swaths\n",
    "\n",
    "\n",
    "swath_traces, swath_labels, swath_ids, n_swaths = load_rebar_dataset()\n",
    "\n",
    "all_traces = np.concatenate(swath_traces)\n",
    "all_labels = np.concatenate(swath_labels)\n",
    "\n",
    "train_mask = swath_ids < 11\n",
    "val_mask   = swath_ids >= 11\n",
    "\n",
    "X_train, y_train = all_traces[train_mask], all_labels[train_mask]\n",
    "X_val,   y_val   = all_traces[val_mask],   all_labels[val_mask]\n",
    "\n",
    "depths_all_mm = all_labels * MAX_DEPTH_MM\n",
    "print(f'\\nDataset summary:')\n",
    "print(f'  n_train_traces: {len(X_train):,}')\n",
    "print(f'  n_val_traces:   {len(X_val):,}')\n",
    "print(f'  depth range:    {depths_all_mm.min():.1f} - {depths_all_mm.max():.1f} mm')\n",
    "print(f'  mean depth:     {(y_train * MAX_DEPTH_MM).mean():.1f} mm  '\n",
    "      f'({(y_train * MAX_DEPTH_MM / 25.4).mean():.2f} in)')",
])

# cell_train: rename mae/rmse cm -> mm
src = ''.join(nb['cells'][8]['source'])
src = (src
    .replace('mae_cm',  'mae_mm')
    .replace('rmse_cm', 'rmse_mm')
    .replace('* MAX_DEPTH_CM', '* MAX_DEPTH_MM')
    .replace("best_mae:.3f} cm  ({best_mae / 2.54:.3f} in)", "best_mae:.3f} mm  ({best_mae / 25.4:.3f} in)")
)
nb['cells'][8]['source'] = [src]

# cell_eval: unit conversions and epsr fix
src = ''.join(nb['cells'][9]['source'])
src = (src
    .replace('* MAX_DEPTH_CM', '* MAX_DEPTH_MM')
    .replace('preds_cm', 'preds_mm')
    .replace('tgts_cm',  'tgts_mm')
    .replace('mae_cm',   'mae_mm')
    .replace('rmse_cm',  'rmse_mm')
    .replace('mae_in  = mae_cm  / 2.54',  'mae_in  = mae_mm  / 25.4')
    .replace('rmse_in = rmse_cm / 2.54',  'rmse_in = rmse_mm / 25.4')
    .replace("velocity_m_ns = 0.15 / (9.0 ** 0.5)  # epsr=9.0",
             "velocity_m_ns = 0.15 / (6.0 ** 0.5)  # epsr=6.0 for IDS 2GHz on concrete")
    .replace("def cm_to_sample(cm_arr):\n    return (cm_arr / 100.0) / velocity_m_ns * 2.0 / ns_per_sample",
             "def mm_to_sample(mm_arr):\n    return (mm_arr / 1000.0) / velocity_m_ns * 2.0 / ns_per_sample")
    .replace("cm_to_sample(true_show)", "mm_to_sample(true_show)")
    .replace("cm_to_sample(pred_show)", "mm_to_sample(pred_show)")
    .replace("mae_cm:.3f} cm",   "mae_mm:.3f} mm")
    .replace("rmse_cm:.3f} cm",  "rmse_mm:.3f} mm")
    .replace("mae_in:.3f} in)",  "mae_in:.3f} in)")
    .replace("rmse_in:.3f} in)", "rmse_in:.3f} in)")
    .replace("'Actual depth (cm)'",    "'Actual depth (mm)'")
    .replace("'Predicted depth (cm)'", "'Predicted depth (mm)'")
    .replace("'Residual (cm)'",        "'Residual (mm)'")
    .replace("std:.3f} cm'",           "std:.3f} mm'")
)
nb['cells'][9]['source'] = [src]

src = ''.join(nb['cells'][10]['source'])
src = src.replace('mae_cm', 'mae_mm').replace('rmse_cm', 'rmse_mm')
nb['cells'][10]['source'] = [src]

with open('train_rebar_horizon.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("train_rebar_horizon.ipynb updated")


# ── train_corrosion.ipynb: paths only ─────────────────────────────────────────
with open('train_corrosion.ipynb') as f:
    nb2 = json.load(f)

set_cell(nb2, 'cell_config', [
    f"DATA_DIR    = r'{LOCAL_DATA}'\n",
    f"WORKING_DIR = r'{LOCAL_MODELS}'\n",
    "TARGET_SAMPLES = 512\n",
    "BATCH_SIZE  = 512\n",
    "EPOCHS      = 100\n",
    "PATIENCE    = 15\n",
    "LR          = 1e-3\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Device: {DEVICE}')",
])

with open('train_corrosion.ipynb', 'w') as f:
    json.dump(nb2, f, indent=1)
print("train_corrosion.ipynb updated")
