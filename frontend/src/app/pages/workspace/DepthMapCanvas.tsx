import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Download, ImageOff } from 'lucide-react';
import { supabase } from '../../../lib/supabase';
import { BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3 } from './tokens';

interface DepthMapCanvasProps {
  jobId:           string;
  serverUrl:       string;
  analysisName:    string;
  staticImageB64?: string;          // fallback when picks list is empty
  needsRegen?:     boolean;          // parent flips after picks save → refetch
  onRegenerated?:  () => void;       // ack so parent can reset needsRegen
}

interface Pick {
  trace_idx:  number;
  sample_idx: number;
  depth_in:   number;
  confidence: number;
  is_manual:  boolean;
  swath_idx:  number;
}

// Viridis-inspired colormap: dark purple (shallow) → blue → teal → green →
// yellow (deep). Mapped over the job's *actual* depth range (not a fixed
// 3-8.5" scale), so a dataset whose depths cluster — or fall outside that
// fixed window — no longer collapses to a single solid colour.
const VIRIDIS: [number, number, number][] = [
  [68,   1,  84],
  [59,  82, 139],
  [33, 145, 140],
  [94, 201,  98],
  [253, 231, 37],
];

function depthToColor(t: number): [number, number, number] {
  const x   = Math.max(0, Math.min(1, t));
  const idx = x * (VIRIDIS.length - 1);
  const lo  = Math.floor(idx);
  const hi  = Math.min(lo + 1, VIRIDIS.length - 1);
  const f   = idx - lo;
  return [
    Math.round(VIRIDIS[lo][0] * (1 - f) + VIRIDIS[hi][0] * f),
    Math.round(VIRIDIS[lo][1] * (1 - f) + VIRIDIS[hi][1] * f),
    Math.round(VIRIDIS[lo][2] * (1 - f) + VIRIDIS[hi][2] * f),
  ];
}

function safeFilename(name: string): string {
  const cleaned = (name ?? '').replace(/[^\w\s-]/g, '').replace(/\s+/g, '_').trim();
  return cleaned || 'rebar_depth_map';
}

// Robust depth range from picks (ignores junk values outside 0–20").
function depthRangeOf(picks: Pick[]): [number, number] | null {
  const depths = picks.map(p => p.depth_in).filter(d => d > 0 && d < 20);
  if (depths.length === 0) return null;
  return [Math.min(...depths), Math.max(...depths)];
}

// Build a 2D depth grid (rows = swaths, cols = traces), linear-interpolate
// gaps within each row, bilinear-upsample to ≥40 rows, colour via the
// dynamic viridis scale. Preserves full 2D spatial variation.
function renderDepthMap(
  canvas: HTMLCanvasElement, picks: Pick[], minDepth: number, maxDepth: number,
): void {
  if (picks.length === 0) return;
  const range    = (maxDepth - minDepth) || 1;
  const swathIds = Array.from(new Set(picks.map(p => p.swath_idx))).sort((a, b) => a - b);
  const maxTrace = Math.max(...picks.map(p => p.trace_idx));
  const nRows    = swathIds.length;
  const nCols    = maxTrace + 1;
  if (nCols < 2 || nRows < 1) return;

  const grid: number[][] = Array.from({ length: nRows }, () => new Array<number>(nCols).fill(NaN));
  for (const p of picks) {
    const r = swathIds.indexOf(p.swath_idx);
    if (r < 0 || p.trace_idx < 0 || p.trace_idx >= nCols) continue;
    grid[r][p.trace_idx] = p.depth_in;
  }

  for (let r = 0; r < nRows; r++) {
    const row = grid[r];
    const valid: number[] = [];
    for (let i = 0; i < nCols; i++) if (!Number.isNaN(row[i])) valid.push(i);
    if (valid.length === 0) continue;
    if (valid.length === 1) { row.fill(row[valid[0]]); continue; }
    for (let i = 0; i < valid[0]; i++) row[i] = row[valid[0]];
    const last = valid[valid.length - 1];
    for (let i = last + 1; i < nCols; i++) row[i] = row[last];
    for (let k = 0; k < valid.length - 1; k++) {
      const a = valid[k], b = valid[k + 1], va = row[a], vb = row[b];
      for (let i = a + 1; i < b; i++) { const t = (i - a) / (b - a); row[i] = va * (1 - t) + vb * t; }
    }
  }

  const TARGET_ROWS = Math.max(nRows, 40);
  canvas.width  = nCols;
  canvas.height = TARGET_ROWS;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const imageData = ctx.createImageData(nCols, TARGET_ROWS);
  for (let r = 0; r < TARGET_ROWS; r++) {
    const src = (r / Math.max(1, TARGET_ROWS - 1)) * Math.max(0, nRows - 1);
    const r0  = Math.floor(src);
    const r1  = Math.min(nRows - 1, r0 + 1);
    const t   = src - r0;
    for (let c = 0; c < nCols; c++) {
      const v0 = grid[r0][c];
      const v1 = grid[r1][c];
      const v  = Number.isNaN(v0) ? v1 : Number.isNaN(v1) ? v0 : v0 * (1 - t) + v1 * t;
      const i  = (r * nCols + c) * 4;
      if (Number.isNaN(v)) {
        imageData.data[i] = imageData.data[i + 1] = imageData.data[i + 2] = 240;
      } else {
        const [R, G, B] = depthToColor((v - minDepth) / range);
        imageData.data[i] = R; imageData.data[i + 1] = G; imageData.data[i + 2] = B;
      }
      imageData.data[i + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

export default function DepthMapCanvas({
  jobId, serverUrl, analysisName, staticImageB64,
  needsRegen, onRegenerated,
}: DepthMapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [picks,   setPicks]   = useState<Pick[]>([]);
  const [loading, setLoading] = useState(false);

  const range = useMemo(() => depthRangeOf(picks), [picks]);

  // Fetch picks on mount + whenever parent flips needsRegen after a save.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        const token = session?.access_token;
        const res = await fetch(`${serverUrl}/job/${jobId}/picks`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (res.ok) {
          const d = await res.json();
          const list: Pick[] = d.picks ?? [];
          console.log('[DEPTH MAP] picks received:', JSON.stringify(list.slice(0, 3)));
          console.log('[DEPTH MAP] pick keys:', list[0] ? Object.keys(list[0]) : 'no picks');
          if (!cancelled) setPicks(list);
        }
        if (!cancelled) onRegenerated?.();
      } catch (e) {
        console.error('[DepthMapCanvas] picks fetch failed:', e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, serverUrl, needsRegen]);

  // Re-render canvas whenever picks (or the derived range) change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (picks.length === 0 || !range) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const [lo, hi] = range;
    console.log(`[DEPTH MAP] depth range: ${lo.toFixed(2)}" to ${hi.toFixed(2)}", ${picks.length} picks`);
    renderDepthMap(canvas, picks, lo, hi);
  }, [picks, range]);

  const downloadPng = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || picks.length === 0) return;
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = `${safeFilename(analysisName)}_rebar_depth.png`;
    a.click();
  }, [picks, analysisName]);

  const hasPicks    = picks.length > 0 && !!range;
  const fallbackSrc = !hasPicks && staticImageB64
    ? (staticImageB64.startsWith('data:') ? staticImageB64 : `data:image/png;base64,${staticImageB64}`)
    : null;

  const gradientCss = `linear-gradient(to right, ${
    Array.from({ length: 11 }, (_, i) => {
      const [R, G, B] = depthToColor(i / 10);
      return `rgb(${R},${G},${B}) ${i * 10}%`;
    }).join(', ')
  })`;

  return (
    <div style={{
      background: PANEL, border: `1px solid ${BORDER}`,
      display: 'flex', flexDirection: 'column',
      minHeight: 400, overflow: 'hidden', borderRadius: 8,
    }}>
      <div style={{
        flexShrink: 0,
        padding: '8px 14px', borderBottom: `1px solid ${BORDER}`,
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <span style={{
          fontSize: 10, fontWeight: 800, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: TEXT2,
        }}>
          Rebar Depth Map
        </span>
        <span style={{ flex: 1, fontSize: 11, color: TEXT3, fontStyle: 'italic' }}>
          {hasPicks
            ? `Live render from ${picks.length} pick${picks.length === 1 ? '' : 's'}`
            : (loading ? 'Loading picks…' : 'No picks yet — Detect Picks on B-scan')}
        </span>
        <button
          onClick={downloadPng}
          disabled={!hasPicks}
          style={iconBtn(!hasPicks)}
          title={hasPicks ? 'Download current map as PNG' : 'No picks to export'}
        >
          <Download size={11} /> PNG
        </button>
      </div>

      <div style={{
        flex: 1, minHeight: 0,
        position: 'relative', overflow: 'hidden',
        background: RAISED,
      }}>
        {hasPicks ? (
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%',
              imageRendering: 'auto', display: 'block',
            }}
          />
        ) : fallbackSrc ? (
          <img
            src={fallbackSrc}
            alt="Rebar Depth Map (static)"
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'fill', display: 'block' }}
          />
        ) : (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex',
            flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            gap: 8, color: TEXT3,
          }}>
            <ImageOff size={22} />
            <span style={{ fontSize: 11 }}>No depth map yet</span>
          </div>
        )}
      </div>

      <div style={{
        flexShrink: 0,
        padding: '8px 14px 10px', borderTop: `1px solid ${BORDER}`,
        display: 'flex', flexDirection: 'column', gap: 4,
      }}>
        <div style={{ height: 12, borderRadius: 2, background: gradientCss, border: '1px solid rgba(0,0,0,0.08)' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: TEXT3 }}>
          <span>{range ? `${range[0].toFixed(1)}" (shallow)` : '—'}</span>
          <span style={{ fontWeight: 700, color: TEXT2 }}>Rebar Depth (in)</span>
          <span>{range ? `${range[1].toFixed(1)}" (deep)` : '—'}</span>
        </div>
      </div>
    </div>
  );
}

function iconBtn(disabled: boolean): React.CSSProperties {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 4,
    padding: '5px 10px', fontSize: 10, fontWeight: 700,
    letterSpacing: '0.08em', textTransform: 'uppercase',
    border: `1px solid ${BORDER}`, borderRadius: 4,
    background: disabled ? RAISED : PANEL,
    color: disabled ? TEXT3 : TEXT,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontFamily: 'inherit',
  };
}
