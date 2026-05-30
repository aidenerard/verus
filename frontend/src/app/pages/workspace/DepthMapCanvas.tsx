import { useCallback, useEffect, useRef, useState } from 'react';
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

// ── Colormap: YlOrRd_r in 0.5" discrete bands (red shallow → yellow deep) ─
// Mirrors server/analysis.py build_unified_depth_map so the live JS canvas
// reads identically to the backend's high-quality PNG.
const VMIN = 3.0;
const VMAX = 8.5;
const PALETTE: [number, [number, number, number]][] = [
  [3.0, [165,   0,   0]],
  [3.5, [200,  20,   0]],
  [4.0, [220,  50,  20]],
  [4.5, [240,  90,  40]],
  [5.0, [250, 130,  60]],
  [5.5, [253, 160,  80]],
  [6.0, [254, 190, 100]],
  [6.5, [254, 220, 130]],
  [7.0, [254, 240, 160]],
  [7.5, [254, 245, 190]],
  [8.0, [255, 250, 220]],
  [8.5, [255, 255, 240]],
];

function colorForDepth(d: number): [number, number, number] {
  const clamped = Math.max(VMIN, Math.min(VMAX, d));
  // Quantize to 0.5" → discrete contour-style bands.
  const band = Math.round(clamped * 2) / 2;
  for (const [edge, rgb] of PALETTE) if (band <= edge) return rgb;
  return PALETTE[PALETTE.length - 1][1];
}

function safeFilename(name: string): string {
  const cleaned = (name ?? '').replace(/[^\w\s-]/g, '').replace(/\s+/g, '_').trim();
  return cleaned || 'rebar_depth_map';
}

// Build a 2D depth grid from picks, linear-interpolate gaps within each row,
// then bilinear-upsample to ≥40 rows so even sparse swath data has visual
// breathing room (matches the backend renderer's nd_zoom pass).
function renderDepthMap(canvas: HTMLCanvasElement, picks: Pick[]): void {
  if (picks.length === 0) return;
  const swathIds = Array.from(new Set(picks.map(p => p.swath_idx))).sort((a, b) => a - b);
  const maxTrace = Math.max(...picks.map(p => p.trace_idx));
  const nRows = swathIds.length;
  const nCols = maxTrace + 1;
  if (nCols < 2 || nRows < 1) return;

  const grid: number[][] = Array.from({ length: nRows }, () => new Array<number>(nCols).fill(NaN));
  for (const p of picks) {
    const r = swathIds.indexOf(p.swath_idx);
    if (r < 0) continue;
    if (p.trace_idx < 0 || p.trace_idx >= nCols) continue;
    grid[r][p.trace_idx] = p.depth_in;
  }

  for (let r = 0; r < nRows; r++) {
    const row = grid[r];
    const valid: number[] = [];
    for (let i = 0; i < nCols; i++) if (!Number.isNaN(row[i])) valid.push(i);
    if (valid.length === 0) continue;
    if (valid.length === 1) {
      const v = row[valid[0]];
      for (let i = 0; i < nCols; i++) row[i] = v;
      continue;
    }
    for (let i = 0; i < valid[0]; i++) row[i] = row[valid[0]];
    const last = valid[valid.length - 1];
    for (let i = last + 1; i < nCols; i++) row[i] = row[last];
    for (let k = 0; k < valid.length - 1; k++) {
      const a = valid[k], b = valid[k + 1];
      const va = row[a], vb = row[b];
      for (let i = a + 1; i < b; i++) {
        const t = (i - a) / (b - a);
        row[i] = va * (1 - t) + vb * t;
      }
    }
  }

  const TARGET_ROWS = Math.max(nRows, 40);
  canvas.width = nCols;
  canvas.height = TARGET_ROWS;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const imageData = ctx.createImageData(nCols, TARGET_ROWS);
  for (let r = 0; r < TARGET_ROWS; r++) {
    const src = (r / Math.max(1, TARGET_ROWS - 1)) * Math.max(0, nRows - 1);
    const r0 = Math.floor(src);
    const r1 = Math.min(nRows - 1, r0 + 1);
    const t = src - r0;
    for (let c = 0; c < nCols; c++) {
      const v0 = grid[r0][c];
      const v1 = grid[r1][c];
      const v = Number.isNaN(v0) ? v1 : Number.isNaN(v1) ? v0 : v0 * (1 - t) + v1 * t;
      const i = (r * nCols + c) * 4;
      if (Number.isNaN(v)) {
        imageData.data[i] = imageData.data[i + 1] = imageData.data[i + 2] = 240;
      } else {
        const [R, G, B] = colorForDepth(v);
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
          if (!cancelled) setPicks(d.picks ?? []);
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

  // Re-render canvas whenever picks change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (picks.length === 0) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    renderDepthMap(canvas, picks);
  }, [picks]);

  const downloadPng = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || picks.length === 0) return;
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = `${safeFilename(analysisName)}_rebar_depth.png`;
    a.click();
  }, [picks, analysisName]);

  const hasPicks    = picks.length > 0;
  const fallbackSrc = !hasPicks && staticImageB64
    ? (staticImageB64.startsWith('data:') ? staticImageB64 : `data:image/png;base64,${staticImageB64}`)
    : null;

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
        padding: '6px 14px 8px', borderTop: `1px solid ${BORDER}`,
        display: 'flex', alignItems: 'flex-end', gap: 2, fontSize: 9, color: TEXT3,
      }}>
        <span style={{ marginRight: 6, fontSize: 10, alignSelf: 'center' }}>Depth (in)</span>
        {PALETTE.map(([edge, [R, G, B]], i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <div style={{ width: 22, height: 11, background: `rgb(${R},${G},${B})`, border: '1px solid rgba(0,0,0,0.06)' }} />
            <span>{edge.toFixed(1)}</span>
          </div>
        ))}
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
