-- ── Verus GPR — picks table + processing_state for the Interactive view ─────

create extension if not exists btree_gist;

create table if not exists public.picks (
  id            uuid primary key default gen_random_uuid(),
  job_id        uuid not null references public.analysis_jobs(id) on delete cascade,
  scan_line_id  text not null,
  trace_index   integer not null,
  depth_in      double precision,
  amplitude     double precision,
  confidence    double precision,
  lat           double precision,
  lon           double precision,
  x_ft          double precision,
  y_ft          double precision,
  is_edited     boolean not null default false,
  is_deleted    boolean not null default false,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

-- Composite btree for job lookups; gist on (x_ft, y_ft) for viewport bbox queries.
create index if not exists picks_job_idx       on public.picks (job_id);
create index if not exists picks_scan_line_idx on public.picks (job_id, scan_line_id);
create index if not exists picks_xy_gist       on public.picks using gist (x_ft, y_ft);

drop trigger if exists picks_updated_at on public.picks;
create trigger picks_updated_at
  before update on public.picks
  for each row execute function public.set_updated_at();

alter table public.picks enable row level security;

create policy "Users can read own picks"
  on public.picks for select
  using (job_id in (select id from public.analysis_jobs where user_id = auth.uid()));

create policy "Users can update own picks"
  on public.picks for update
  using (job_id in (select id from public.analysis_jobs where user_id = auth.uid()))
  with check (job_id in (select id from public.analysis_jobs where user_id = auth.uid()));

-- Inserts happen server-side via the service role (which bypasses RLS); no user-insert policy needed.

-- processing_state on analysis_jobs. Defaults reflect the current pipeline
-- behavior so existing jobs continue to render unchanged.
alter table public.analysis_jobs
  add column if not exists processing_state jsonb not null default '{
    "time_zero_shifts": {},
    "filters": [],
    "gridding": {
      "algorithm":         "nearest",
      "search_radius_ft":  10.0,
      "edge_clip":         true,
      "anisotropy_angle":  0.0,
      "anisotropy_ratio":  1.0,
      "cell_size_ft":      0.5
    },
    "gps_latency_ms": 0,
    "needs_regrid":   false
  }'::jsonb;
