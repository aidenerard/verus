-- Add real-time progress tracking columns to analysis_jobs.
-- progress: 0–100 integer updated by the job runner at key checkpoints.
-- stage:    human-readable label for the current processing step.

alter table public.analysis_jobs
  add column if not exists progress smallint not null default 0,
  add column if not exists stage    text;
