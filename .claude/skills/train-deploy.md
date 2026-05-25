# train-deploy

Use immediately after any training run completes. Walks the full post-training checklist to get new weights live on Render without silently loading the wrong architecture.

## Checklist

### 1. Verify outputs exist (Kaggle / Colab)
- `model.pth` — trained weights
- `model_config.json` — architecture record written by the training script

If `model_config.json` is missing, the server falls back to probing known architectures in order. It may load the wrong layer sizes silently — no crash, wrong predictions.

### 2. Upload both files to Google Drive
1. Download `model.pth` and `model_config.json` from Kaggle → Output Files (or Colab → Drive if Colab run).
2. Upload both to Google Drive.
3. For each file: right-click → Share → "Anyone with the link (Viewer)" → copy link.
4. Extract the file ID from the share URL: `https://drive.google.com/file/d/<FILE_ID>/view`

### 3. Update both Render env vars
In Render dashboard → your service → Environment:

| Variable | Value |
|---|---|
| `MODEL_GDRIVE_URL` | `https://drive.google.com/uc?export=download&id=<model_pth_file_id>` |
| `MODEL_CONFIG_GDRIVE_URL` | `https://drive.google.com/uc?export=download&id=<model_config_file_id>` |

For the rebar model:

| Variable | Value |
|---|---|
| `REBAR_MODEL_GDRIVE_URL` | `https://drive.google.com/uc?export=download&id=<rebar_model_pth_file_id>` |

### 4. Trigger redeploy
```bash
git push origin main   # auto-deploys both Vercel (frontend) and Render (backend)
```
Or manually: Render dashboard → Manual Deploy → Deploy latest commit.

### 5. Verify on startup
Watch Render logs for:
```
[model] Config loaded: {"in_channels": 2, "conv_channels": [...], ...}
[model] CNN1D loaded — X params
```
If you see `fallback probe` in the logs, `MODEL_CONFIG_GDRIVE_URL` is wrong or the file isn't publicly shared.

## Notes
- The threshold in `model_config.json` overrides the hardcoded `THRESHOLD = 0.65` in `server/model.py` on load.
- After a Kaggle training run, `model_config.json` is at `/kaggle/working/model_config.json` in Output Files — it will not appear in the notebook viewer, only in the file list.
- Google Drive large-file download requires a confirm token for files >100 MB. `gdown` handles this automatically; the `model_loader.py` uses `gdown`.
