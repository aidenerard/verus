-- ── Verus GPR — project equipment fields ─────────────────────────────────────
-- Ensures projects table has all equipment + identity columns.
-- All are already present from 002_projects.sql; this is a safety net for
-- instances that may have skipped that migration.

alter table public.projects
  add column if not exists name               text,
  add column if not exists structure_name     text,
  add column if not exists manufacturer       text,
  add column if not exists frequency_mhz      integer,
  add column if not exists inspection_method  text;
