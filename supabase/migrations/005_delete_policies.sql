-- ── Delete policies ───────────────────────────────────────────────────────────
-- analysis_jobs was missing a DELETE policy (001_initial.sql only had SELECT
-- and INSERT). Without this, client-side deletes are blocked by RLS even when
-- the user owns the row.
--
-- projects already has a "for all" policy from 002_projects.sql that covers
-- DELETE, so no change is needed there.

drop policy if exists "Users can delete own jobs" on public.analysis_jobs;
create policy "Users can delete own jobs"
  on public.analysis_jobs for delete
  using (auth.uid() = user_id);
