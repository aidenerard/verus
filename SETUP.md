# Verus — Supabase Setup Guide

## 1. Create a Supabase project

1. Go to [app.supabase.com](https://app.supabase.com) and create a new project.
2. Note your **Project URL** and two API keys:
   - **anon / public key** — safe to expose in the browser
   - **service_role / secret key** — server only, never expose to the browser

---

## 2. Run the database migrations

Open **SQL Editor** in your Supabase dashboard and run each migration **in order**:

| File | What it creates |
|---|---|
| `supabase/migrations/001_initial.sql` | `profiles` table, auth trigger, RLS policies; `analysis_jobs` table |
| `supabase/migrations/002_projects.sql` | `projects` table with `manufacturer`, `frequency_mhz`, `name`, `structure_name`; links `analysis_jobs.project_id` |
| `supabase/migrations/003_result_fields.sql` | Extra result columns on `analysis_jobs` (rebar, amplitude, grids, confidence metrics) |
| `supabase/migrations/004_project_equipment.sql` | Safety net: `add column if not exists` for all equipment fields + `inspection_method` on `projects` |

All migrations are idempotent (`if not exists`) — safe to re-run.

---

## 3. Create the Storage bucket

In **Storage → New bucket**:
- Name: `cscan-images`
- Public: ✅ (so the frontend can display the C-scan image via URL)

Or run in SQL Editor:
```sql
select storage.create_bucket('cscan-images', '{"public": true}');
```

---

## 4. Frontend environment variables

Add to `frontend/.env.local` (never commit this file):

```
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...your-anon-key...
```

---

## 5. Backend (Render) environment variables

In your Render service → Environment:

| Variable | Value |
|---|---|
| `SUPABASE_URL` | `https://your-project-ref.supabase.co` |
| `SUPABASE_SERVICE_KEY` | your **service_role** key (secret) |
| `MODEL_GDRIVE_URL` | Google Drive direct-download URL for `model.pth` |
| `MODEL_CONFIG_GDRIVE_URL` | Google Drive direct-download URL for `model_config.json` |
| `REBAR_MODEL_GDRIVE_URL` | Google Drive direct-download URL for `rebar_model.pth` |

The server uses the service role key to bypass RLS when writing job results.

### Getting `MODEL_CONFIG_GDRIVE_URL`

The model config file records the architecture used during training so the server
can reconstruct the exact network without hardcoding layer sizes.

1. After training, the Kaggle notebook saves `model_config.json` alongside `model.pth`.
2. Download `model_config.json` from Kaggle → Output Files.
3. Upload to Google Drive and set sharing to **Anyone with the link → Viewer**.
4. Right-click → **Get link** → copy the file ID (the long string after `/d/`).
5. Set the env var to `https://drive.google.com/uc?export=download&id=<FILE_ID>`

If `MODEL_CONFIG_GDRIVE_URL` is not set, the server tries a set of known fallback
architectures in order until one loads without error. No crash, but always set this
after retraining to guarantee the correct architecture is used.

### Getting `REBAR_MODEL_GDRIVE_URL`

1. Train the model using `rebar_training.ipynb` on Kaggle.
2. Download `rebar_model.pth` **and** `rebar_model_config.json` from Kaggle → Output Files.
3. Upload both files to Google Drive and set sharing to **Anyone with the link → Viewer**.
4. Right-click → **Get link** → copy each file ID (the long string after `/d/`).
5. Set `REBAR_MODEL_GDRIVE_URL` to `https://drive.google.com/uc?export=download&id=<FILE_ID>`
   (or paste the full share URL — `gdown` handles both formats).

If `REBAR_MODEL_GDRIVE_URL` is not set, the server starts normally and rebar depth
uses a physics-based fallback (peak amplitude arrival time). No crash, no config error.

---

## 6. Auth email settings (optional)

By default Supabase requires email confirmation on signup. During development
you can disable this:

**Authentication → Email → Confirm email → Off**

For production, leave it on and configure your SMTP provider.

---

## 7. Local development

```bash
# Terminal 1 — Python server
cd server
pip install -r requirements.txt
uvicorn server:app --reload --port 10000

# Terminal 2 — Vite frontend
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/analyze` and `/job/*` to `localhost:10000` via
the config in `vite.config.ts`.
