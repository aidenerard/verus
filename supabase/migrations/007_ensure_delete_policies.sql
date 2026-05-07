-- Re-apply delete policies idempotently.
-- Run this in Supabase SQL Editor if project deletions silently fail.

drop policy if exists "Users can delete own jobs" on public.analysis_jobs;
create policy "Users can delete own jobs"
  on public.analysis_jobs for delete
  using (auth.uid() = user_id);

drop policy if exists "Users can manage own projects" on public.projects;
create policy "Users can manage own projects"
  on public.projects for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
