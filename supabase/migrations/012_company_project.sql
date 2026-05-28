-- ── Verus GPR — company / project columns on analysis_jobs ────────────────
-- Captured from the upload card and used to organize cscan-images storage
-- keys into {company}/{project}/{job_id}{suffix}.png folders. Also surfaced
-- on the results header and dashboard.
--
-- Apply manually via Supabase SQL editor or `supabase db push`.

ALTER TABLE analysis_jobs
  ADD COLUMN IF NOT EXISTS company text DEFAULT '',
  ADD COLUMN IF NOT EXISTS project text DEFAULT '';
