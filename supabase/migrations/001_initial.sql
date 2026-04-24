-- ── Verus GPR — initial schema ────────────────────────────────────────────────
-- Run this once in Supabase SQL Editor (or via supabase db push).
-- All tables use RLS so users can only see their own rows.


-- ── profiles ──────────────────────────────────────────────────────────────────
-- Created automatically on signup via trigger (and belt-and-suspenders upsert
-- in AuthContext.tsx).

create table if not exists public.profiles (
  id         uuid primary key references auth.users(id) on delete cascade,
  full_name  text,
  company    text,
  created_at timestamptz default now()
);

alter table public.profiles enable row level security;

create policy "Users can read own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can update own profile"
  on public.profiles for update
  using (auth.uid() = id);

create policy "Users can insert own profile"
  on public.profiles for insert
  with check (auth.uid() = id);

-- Trigger: auto-create profile row on new user signup
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer as $$
begin
  insert into public.profiles (id, full_name, company)
  values (
    new.id,
    new.raw_user_meta_data->>'full_name',
    new.raw_user_meta_data->>'company'
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();


-- ── analysis_jobs ─────────────────────────────────────────────────────────────
-- One row per /analyze call.  Created with status=pending, updated to
-- complete/failed by the background worker.

create table if not exists public.analysis_jobs (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid references auth.users(id) on delete set null,
  status             text not null default 'pending'
                       check (status in ('pending','processing','complete','failed')),
  created_at         timestamptz default now(),
  completed_at       timestamptz,

  -- result fields
  signals_analyzed   integer,
  delamination_pct   numeric(6,2),
  sound_pct          numeric(6,2),
  analysis_time_sec  numeric(8,3),
  cscan_url          text,
  file_names         text[],
  per_file_summary   jsonb,
  error_msg          text
);

alter table public.analysis_jobs enable row level security;

create policy "Users can read own jobs"
  on public.analysis_jobs for select
  using (auth.uid() = user_id);

create policy "Users can insert own jobs"
  on public.analysis_jobs for insert
  with check (auth.uid() = user_id);

-- Service role (server) can update any row (status changes, result writes).
-- No RLS policy needed for service role — it bypasses RLS by default.


-- ── Supabase Storage bucket ───────────────────────────────────────────────────
-- Create the bucket in the Supabase dashboard:
--   Storage → New bucket → Name: cscan-images → Public: true
--
-- Or run via supabase-js admin:
--   supabase.storage.createBucket('cscan-images', { public: true })
--
-- The server uploads to: {user_id}/{job_id}.png
-- The public URL is stored in analysis_jobs.cscan_url
