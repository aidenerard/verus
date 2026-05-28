-- ── Verus GPR — uploads storage bucket for large-dataset direct upload ─────
-- Used by POST /analyze-proceq-storage. Browser uploads Proceq dataset
-- files (PRC_*.scan, Swath_*.pos, CScan_*.CScan) directly to this bucket,
-- then sends the folder path to the backend. Backend uses the service
-- role key to list + download the files into a tmpdir for processing.
--
-- Apply manually via Supabase SQL editor or `supabase db push`.

insert into storage.buckets (id, name, public)
values ('uploads', 'uploads', false)
on conflict do nothing;

-- Authenticated users can upload files into folders prefixed with their
-- user uuid (i.e. uploads/{user.uid}/...). storage.foldername(name) returns
-- the path segments; [1] is the first folder.
drop policy if exists "Users can upload their own files" on storage.objects;
create policy "Users can upload their own files"
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'uploads'
  and auth.uid()::text = (storage.foldername(name))[1]
);

drop policy if exists "Users can read their own files" on storage.objects;
create policy "Users can read their own files"
on storage.objects for select
to authenticated
using (
  bucket_id = 'uploads'
  and auth.uid()::text = (storage.foldername(name))[1]
);

drop policy if exists "Users can delete their own files" on storage.objects;
create policy "Users can delete their own files"
on storage.objects for delete
to authenticated
using (
  bucket_id = 'uploads'
  and auth.uid()::text = (storage.foldername(name))[1]
);
