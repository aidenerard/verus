-- 010_analysis_name_notes.sql
-- Add user-supplied analysis name + notes to the analysis_jobs row.
-- Filled at job creation time by the /analyze and /analyze-proceq handlers.

ALTER TABLE analysis_jobs
  ADD COLUMN IF NOT EXISTS analysis_name  text NOT NULL DEFAULT 'Untitled Analysis',
  ADD COLUMN IF NOT EXISTS analysis_notes text NOT NULL DEFAULT '';
