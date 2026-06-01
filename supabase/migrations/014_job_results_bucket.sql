-- ── Verus GPR — job-results storage bucket ─────────────────────────────────
-- Large job-result blobs (base64 PNGs, bscan trace JSON) are offloaded here
-- by jobs._upload_result_to_storage so the analysis_jobs.result JSONB row
-- stays small enough for Supabase to round-trip on read. Public bucket so
-- <img src> and the frontend fetch() can read directly.
--
-- Applied to project xbbtvtjnvveitfsnhihs via the Supabase MCP.

INSERT INTO storage.buckets (id, name, public)
VALUES ('job-results', 'job-results', true)
ON CONFLICT DO NOTHING;

DROP POLICY IF EXISTS "Public read job results" ON storage.objects;
CREATE POLICY "Public read job results"
ON storage.objects FOR SELECT
USING (bucket_id = 'job-results');

DROP POLICY IF EXISTS "Service role upload job results" ON storage.objects;
CREATE POLICY "Service role upload job results"
ON storage.objects FOR INSERT
TO service_role
WITH CHECK (bucket_id = 'job-results');
