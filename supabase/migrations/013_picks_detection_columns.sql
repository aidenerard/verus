-- ── Verus GPR — hyperbola pick-detection columns on the picks table ─────────
-- The picks table (migration 008) was built for the per-trace Interactive
-- regrid flow (trace_index / is_edited / scan_line_id). Hyperbola apex
-- detection + the analyst pick editor add four more fields:
--   sample_idx  — vertical (time/depth-sample) position of the apex on the
--                 B-scan canvas, needed to redraw the pick at the right Y
--   confidence  — 0-1 normalized Hilbert-envelope peak amplitude
--   is_manual   — true once the analyst drags/adds the pick (mirrors
--                 is_edited; kept distinct so the frontend JSON shape matches)
--   swath_idx   — integer swath index (mirrors scan_line_id as an int)
--
-- Written by pipeline._persist_hyperbola_picks, analysis_proceq's per-swath
-- detection, and the POST /job/{id}/redetect-picks + /picks endpoints.
--
-- Apply via Supabase SQL editor, `supabase db push`, or the Supabase MCP.

ALTER TABLE public.picks
  ADD COLUMN IF NOT EXISTS sample_idx integer,
  ADD COLUMN IF NOT EXISTS confidence double precision DEFAULT 1.0,
  ADD COLUMN IF NOT EXISTS is_manual  boolean DEFAULT false,
  ADD COLUMN IF NOT EXISTS swath_idx  integer DEFAULT 0;
