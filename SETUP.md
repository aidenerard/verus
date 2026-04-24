# Verus — Supabase Setup Guide

## 1. Create a Supabase project

1. Go to [app.supabase.com](https://app.supabase.com) and create a new project.
2. Note your **Project URL** and two API keys:
   - **anon / public key** — safe to expose in the browser
   - **service_role / secret key** — server only, never expose to the browser

---

## 2. Run the database migration

Open **SQL Editor** in your Supabase dashboard and paste the contents of:

```
supabase/migrations/001_initial.sql
```

This creates:
- `profiles` table (auto-populated on signup via trigger)
- `analysis_jobs` table (one row per analysis run)
- Row-level security policies so users only see their own rows

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

The server uses the service role key to bypass RLS when writing job results.

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
