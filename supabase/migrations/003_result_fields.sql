-- ── Verus GPR — full result persistence ──────────────────────────────────────
-- Adds all fields needed to reconstruct the workspace from a saved job.

alter table public.analysis_jobs
  add column if not exists rebar_cscan_image_url   text,
  add column if not exists amplitude_image_url     text,
  add column if not exists prob_grid               text,
  add column if not exists prob_grid_rows          integer,
  add column if not exists prob_grid_cols          integer,
  add column if not exists otsu_threshold          double precision,
  add column if not exists twt_grid                text,
  add column if not exists twt_grid_rows           integer,
  add column if not exists twt_grid_cols           integer,
  add column if not exists frequency_mhz           integer,
  add column if not exists manufacturer            text,
  add column if not exists rebar_model_used        boolean,
  add column if not exists model_confidence_pct    double precision,
  add column if not exists depth_accuracy_in       double precision,
  add column if not exists signal_quality          text;
