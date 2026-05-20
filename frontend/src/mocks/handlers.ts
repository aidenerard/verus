import { http, HttpResponse } from 'msw';
import scene      from '../__fixtures__/interactive/scene.json';
import picksFx    from '../__fixtures__/interactive/picks.json';
import processing from '../__fixtures__/interactive/processing.json';
import gridding   from '../__fixtures__/interactive/gridding.json';
import scanLineSl1 from '../__fixtures__/interactive/scan_line_sl-1.json';

interface Pick {
  id: string; scan_line_id: string; trace_idx: number;
  x_ft: number; y_ft: number; z_ft: number;
  depth_in: number; time_ns: number; sample_idx: number;
  amplitude: number; confidence: number;
  lat: number | null; lon: number | null;
  is_edited: boolean; is_deleted: boolean;
}

const picksState = new Map<string, Pick>(
  (picksFx as { picks: Pick[] }).picks.map(p => [p.id, { ...p }]),
);
let processingState = structuredClone(processing);
let griddingState   = structuredClone(gridding);

const matchJob = '*';

export const handlers = [
  http.get(`*/jobs/${matchJob}/scene`, () =>
    HttpResponse.json({ ...scene, picks: Array.from(picksState.values()).filter(p => !p.is_deleted) })),

  http.get(`*/jobs/${matchJob}/picks`, () =>
    HttpResponse.json({ picks: Array.from(picksState.values()).filter(p => !p.is_deleted) })),

  http.patch(`*/picks/:pickId`, async ({ params, request }) => {
    const id = params.pickId as string;
    const cur = picksState.get(id);
    if (!cur) return HttpResponse.json({ error: 'not_found' }, { status: 404 });
    const patch = await request.json() as Partial<Pick>;
    const next: Pick = { ...cur, ...patch, is_edited: true };
    picksState.set(id, next);
    return HttpResponse.json({ pick: next });
  }),

  http.get(`*/jobs/${matchJob}/scan_line/:scanLineId`, ({ params }) => {
    const slId = params.scanLineId as string;
    if (slId === 'sl-1') return HttpResponse.json(scanLineSl1);
    return HttpResponse.json({
      ...scanLineSl1, id: slId, label: scene.scan_lines.find(s => s.id === slId)?.label ?? slId,
      pick_ids: Array.from(picksState.values()).filter(p => p.scan_line_id === slId && !p.is_deleted).map(p => p.id),
    });
  }),

  http.get(`*/jobs/${matchJob}/processing`, () => HttpResponse.json(processingState)),
  http.post(`*/jobs/${matchJob}/processing`, async ({ request }) => {
    processingState = { ...processingState, ...(await request.json() as object) };
    return HttpResponse.json(processingState);
  }),

  http.get(`*/jobs/${matchJob}/gridding`, () => HttpResponse.json(griddingState)),
  http.post(`*/jobs/${matchJob}/gridding`, async ({ request }) => {
    griddingState = { ...griddingState, ...(await request.json() as object) };
    return HttpResponse.json(griddingState);
  }),

  http.post(`*/jobs/${matchJob}/reprocess`, () => {
    console.info('[msw] reprocess (mock) — depths would recompute on real backend');
    return HttpResponse.json({ job_id: 'reproc-' + Date.now(), status: 'queued' });
  }),
  http.post(`*/jobs/${matchJob}/regrid`, () => {
    console.info(
      '[msw] regrid (mock) — fixture returns the same surface, real backend would change visually',
      { algorithm: griddingState.algorithm },
    );
    return HttpResponse.json({ job_id: 'regrid-' + Date.now(), status: 'queued', texture_url: scene.surface.texture_url + '?v=' + Date.now() });
  }),
];
