-- ── Verus GPR — projects table + project_settings ──────────────────────────────

create table if not exists public.projects (
  id               uuid primary key default gen_random_uuid(),
  user_id          uuid references auth.users(id) on delete cascade,
  name             text not null default 'New Project',
  structure_name   text,
  bridge_id        text,
  inspection_date  date default current_date,
  notes            text,
  manufacturer     text,
  frequency_mhz    integer default 1600,
  project_settings jsonb default '{}',
  created_at       timestamptz default now(),
  updated_at       timestamptz default now()
);

alter table public.projects enable row level security;

create policy "Users can manage own projects"
  on public.projects for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- Link analysis jobs to projects
alter table public.analysis_jobs
  add column if not exists project_id uuid references public.projects(id) on delete set null;

-- Auto-update updated_at on row change
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists projects_updated_at on public.projects;
create trigger projects_updated_at
  before update on public.projects
  for each row execute function public.set_updated_at();
