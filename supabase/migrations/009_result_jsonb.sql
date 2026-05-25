-- ── Verus GPR — full result JSONB persistence ──────────────────────────────
-- Holds the complete job result dict so the frontend can reconstruct the
-- workspace from a saved job without round-tripping the in-memory _jobs cache.
-- Used as the canonical store for Proceq jobs (which have no per-field columns)
-- and as a defensive duplicate for analysis_jobs alongside the existing
-- structured columns.

alter table public.analysis_jobs
  add column if not exists result jsonb;
