import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { supabase } from '../../../lib/supabase';

export interface Pick {
  id?:         string;
  trace_idx:   number;
  sample_idx:  number;
  depth_in:    number;
  confidence:  number;
  is_manual:   boolean;
  swath_idx:   number;
}

export interface BScanViewerHandle {
  savePicks:   () => Promise<void>;
  detectPicks: () => Promise<void>;
  getPicks:    () => Pick[];
  getSaving:   () => boolean;
}

interface TraceData {
  data:      string;
  n_traces:  number;
  n_samples: number;
  encoding:  string;
}

interface BScanViewerProps {
  bscanData:     TraceData[];
  jobId:         string;
  serverUrl:     string;
  onPicksSaved?: (picks: Pick[]) => void;
}

const ACCENT     = '#E8601C';
const ACCENT_DIM = '#374151';
const TIME_RANGE_NS = 16.0;
const EPSILON_R     = 9.0;

async function decodeTraces(encoded: TraceData): Promise<Int8Array | null> {
  try {
    const compressed = Uint8Array.from(atob(encoded.data), c => c.charCodeAt(0));
    const ds     = new DecompressionStream('deflate');
    const writer = ds.writable.getWriter();
    const reader = ds.readable.getReader();
    writer.write(compressed);
    writer.close();
    const chunks: Uint8Array[] = [];
    let done = false;
    while (!done) {
      const { value, done: d } = await reader.read();
      if (value) chunks.push(value);
      done = d;
    }
    const total  = chunks.reduce((s, c) => s + c.length, 0);
    const buffer = new Uint8Array(total);
    let offset   = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
    return new Int8Array(buffer.buffer);
  } catch (e) {
    console.error('Failed to decode traces:', e);
    return null;
  }
}

function sampleIdxToDepthIn(sample_idx: number, n_samples: number): number {
  const nsPerSample = TIME_RANGE_NS / n_samples;
  const velocity    = 0.15 / Math.sqrt(EPSILON_R);          // m/ns
  const depthM      = (sample_idx * nsPerSample * velocity) / 2;
  return Math.round(depthM * 39.3701 * 1000) / 1000;        // → inches, 3 dp
}

const BScanViewer = forwardRef<BScanViewerHandle, BScanViewerProps>(function BScanViewer({
  bscanData, jobId, serverUrl, onPicksSaved,
}, ref) {
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const scrollRef     = useRef<HTMLDivElement>(null);
  const isDragging    = useRef(false);
  const dragPickIdx   = useRef<number | null>(null);
  const isPanning     = useRef(false);
  const panStart      = useRef({ x: 0, scrollLeft: 0 });
  const didMoveOrPan  = useRef(false);

  const [activeSwath,    setActiveSwath]    = useState(0);
  const [zoom,           setZoom]           = useState(1.0);
  const [containerWidth, setContainerWidth] = useState(800);
  const [contrast,       setContrast]       = useState(1.5);
  const [colormap,       setColormap]       = useState<'gray' | 'seismic'>('gray');
  const [picks,       setPicks]       = useState<Pick[]>([]);
  const [loading,     setLoading]     = useState(false);
  const [saving,      setSaving]      = useState(false);
  const [traces,      setTraces]      = useState<Int8Array | null>(null);
  const [dims,        setDims]        = useState({ n_traces: 0, n_samples: 0 });
  const [saveMsg,     setSaveMsg]     = useState('');

  // Load picks from backend
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    (async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        const token = session?.access_token;
        const res = await fetch(`${serverUrl}/job/${jobId}/picks`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setPicks(data.picks ?? []);
      } catch (e) {
        console.error('Failed to load picks:', e);
      }
    })();
    return () => { cancelled = true; };
  }, [jobId, serverUrl]);

  // Decode the active swath's trace blob
  useEffect(() => {
    if (!bscanData || bscanData.length === 0) return;
    const swath = bscanData[activeSwath];
    if (!swath) return;
    setLoading(true);
    decodeTraces(swath).then(decoded => {
      if (decoded) {
        setTraces(decoded);
        setDims({ n_traces: swath.n_traces, n_samples: swath.n_samples });
      }
      setLoading(false);
    });
  }, [activeSwath, bscanData]);

  // Render B-scan + pick overlay to the canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !traces || dims.n_traces === 0) return;
    canvas.width  = dims.n_traces;
    canvas.height = dims.n_samples;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imageData = ctx.createImageData(dims.n_traces, dims.n_samples);
    const data      = imageData.data;
    for (let t = 0; t < dims.n_traces; t++) {
      for (let s = 0; s < dims.n_samples; s++) {
        const val   = traces[t * dims.n_samples + s] / 127;
        const boost = Math.max(-1, Math.min(1, val * contrast));
        const pixel = Math.round((boost + 1) / 2 * 255);
        const idx   = (s * dims.n_traces + t) * 4;
        if (colormap === 'seismic') {
          data[idx]     = boost > 0 ? 255 : Math.round(255 * (1 + boost));
          data[idx + 1] = Math.round(255 * (1 - Math.abs(boost)));
          data[idx + 2] = boost < 0 ? 255 : Math.round(255 * (1 - boost));
        } else {
          data[idx] = data[idx + 1] = data[idx + 2] = pixel;
        }
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);

    const swathPicks = picks.filter(p => p.swath_idx === activeSwath);
    ctx.fillStyle   = 'rgba(196, 181, 253, 0.85)';   // light purple — auto picks
    ctx.strokeStyle = 'rgba(124,  58, 237, 0.95)';   // darker purple ring
    ctx.lineWidth   = 1;
    for (const pick of swathPicks) {
      if (pick.trace_idx >= dims.n_traces || pick.sample_idx >= dims.n_samples) continue;
      const r = pick.is_manual ? 5 : 4;
      if (pick.is_manual) {
        ctx.fillStyle   = 'rgba(167, 139, 250, 0.95)';
        ctx.strokeStyle = 'rgba(91,  33, 182, 1)';
      } else {
        ctx.fillStyle   = 'rgba(196, 181, 253, 0.85)';
        ctx.strokeStyle = 'rgba(124,  58, 237, 0.95)';
      }
      ctx.beginPath();
      ctx.arc(pick.trace_idx, pick.sample_idx, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [traces, dims, contrast, colormap, picks, activeSwath]);

  const canvasToPickCoords = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { trace_idx: 0, sample_idx: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      trace_idx:  Math.round((clientX - rect.left) / zoom),
      sample_idx: Math.round((clientY - rect.top)  / zoom),
    };
  }, [zoom]);

  const findNearestPick = useCallback((
    trace_idx: number, sample_idx: number, threshold = 10,
  ): number | null => {
    let nearest: number | null = null;
    let minDist = Infinity;
    picks.forEach((p, i) => {
      if (p.swath_idx !== activeSwath) return;
      const dist = Math.hypot(p.trace_idx - trace_idx, p.sample_idx - sample_idx);
      if (dist < minDist && dist < threshold / zoom) {
        minDist = dist;
        nearest = i;
      }
    });
    return nearest;
  }, [picks, activeSwath, zoom]);

  // Measure the scroll container so zoom can be floored at "fills the width".
  useEffect(() => {
    const measure = () => {
      if (scrollRef.current) setContainerWidth(scrollRef.current.clientWidth);
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  // Minimum zoom that still fills the container width. Below this the scan
  // would leave black gutters, so we never let the user (or wheel) go under it.
  const minZoom = dims.n_traces > 0
    ? Math.max(0.1, containerWidth / dims.n_traces)
    : 0.1;
  const minZoomRef = useRef(minZoom);
  useEffect(() => { minZoomRef.current = minZoom; }, [minZoom]);

  const setZoomClamped = useCallback((next: number | ((z: number) => number)) => {
    setZoom(prev => {
      const v = typeof next === 'function' ? next(prev) : next;
      return Math.max(minZoomRef.current, Math.min(8, v));
    });
  }, []);

  // Native non-passive wheel listener — React's onWheel is passive, which
  // logs "Unable to preventDefault inside passive event listener" any time
  // we swallow the page scroll for Ctrl-wheel zoom or horizontal panning.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        setZoom(z => Math.max(minZoomRef.current, Math.min(8, z + (e.deltaY < 0 ? 0.15 : -0.15))));
      } else {
        el.scrollLeft += e.deltaY;
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // On load / resize / swath change, snap zoom to fit the container width.
  useEffect(() => {
    if (dims.n_traces > 0 && containerWidth > 0) setZoomClamped(minZoom);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dims.n_traces, containerWidth]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    didMoveOrPan.current = false;
    const { trace_idx, sample_idx } = canvasToPickCoords(e.clientX, e.clientY);

    if (e.button === 2) {
      e.preventDefault();
      const idx = findNearestPick(trace_idx, sample_idx);
      if (idx !== null) setPicks(prev => prev.filter((_, i) => i !== idx));
      return;
    }

    const nearIdx = findNearestPick(trace_idx, sample_idx);
    if (nearIdx !== null) {
      isDragging.current  = true;
      dragPickIdx.current = nearIdx;
    } else {
      isPanning.current = true;
      panStart.current  = { x: e.clientX, scrollLeft: scrollRef.current?.scrollLeft ?? 0 };
    }
  }, [canvasToPickCoords, findNearestPick]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging.current && dragPickIdx.current !== null) {
      didMoveOrPan.current = true;
      const { trace_idx, sample_idx } = canvasToPickCoords(e.clientX, e.clientY);
      const ti = Math.max(0, Math.min(dims.n_traces - 1, trace_idx));
      const si = Math.max(0, Math.min(dims.n_samples - 1, sample_idx));
      setPicks(prev => prev.map((p, i) => i !== dragPickIdx.current ? p : {
        ...p,
        trace_idx:  ti,
        sample_idx: si,
        depth_in:   sampleIdxToDepthIn(si, dims.n_samples),
        is_manual:  true,
      }));
    } else if (isPanning.current && scrollRef.current) {
      didMoveOrPan.current = true;
      scrollRef.current.scrollLeft = panStart.current.scrollLeft - (e.clientX - panStart.current.x);
    }
  }, [canvasToPickCoords, dims]);

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    const wasInteraction = isDragging.current || isPanning.current;
    isDragging.current  = false;
    dragPickIdx.current = null;
    isPanning.current   = false;
    if (wasInteraction && didMoveOrPan.current) return;
    if (e.button !== 0) return;

    // Left click without drag/pan: add a new pick if not on an existing one
    const { trace_idx, sample_idx } = canvasToPickCoords(e.clientX, e.clientY);
    if (trace_idx < 0 || trace_idx >= dims.n_traces) return;
    if (sample_idx < 0 || sample_idx >= dims.n_samples) return;
    if (findNearestPick(trace_idx, sample_idx) !== null) return;
    setPicks(prev => [...prev, {
      trace_idx, sample_idx,
      depth_in:   sampleIdxToDepthIn(sample_idx, dims.n_samples),
      confidence: 1.0,
      is_manual:  true,
      swath_idx:  activeSwath,
    }]);
  }, [canvasToPickCoords, findNearestPick, dims, activeSwath]);

  const handleMouseLeave = useCallback(() => {
    isDragging.current = false; isPanning.current = false; dragPickIdx.current = null;
  }, []);

  const savePicks = useCallback(async () => {
    if (!jobId) { setSaveMsg('No job id'); return; }
    setSaving(true); setSaveMsg('');
    try {
      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token;
      const res = await fetch(`${serverUrl}/job/${jobId}/picks`, {
        method:  'POST',
        headers: {
          'Content-Type':  'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ picks }),
      });
      if (!res.ok) throw new Error(`Save failed: ${res.status}`);
      setSaveMsg(`✓ ${picks.length} picks saved`);
      onPicksSaved?.(picks);
    } catch (e) {
      setSaveMsg(`✗ ${e instanceof Error ? e.message : 'Save failed'}`);
    } finally {
      setSaving(false);
    }
  }, [picks, jobId, serverUrl, onPicksSaved]);

  // Re-run hyperbola detection server-side on the job's stored bscan_data
  // and reload picks. Useful for jobs created before detection shipped, or
  // to wipe manual edits and return to the auto-pick set.
  const detectPicks = useCallback(async () => {
    if (!jobId) { setSaveMsg('No job id'); return; }
    setSaving(true); setSaveMsg('Detecting…');
    try {
      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token;
      const res = await fetch(`${serverUrl}/job/${jobId}/redetect-picks`, {
        method: 'POST',
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error(`Detect failed: ${res.status}`);
      const data = await res.json();
      // Reload picks from server now that they've been inserted.
      const r = await fetch(`${serverUrl}/job/${jobId}/picks`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (r.ok) {
        const d = await r.json();
        setPicks(d.picks ?? []);
      }
      setSaveMsg(`✓ Detected ${data.detected ?? 0} picks across ${data.swaths_processed ?? 0} swaths`);
      onPicksSaved?.(picks);
    } catch (e) {
      setSaveMsg(`✗ ${e instanceof Error ? e.message : 'Detect failed'}`);
    } finally {
      setSaving(false);
    }
  }, [jobId, serverUrl, picks, onPicksSaved]);

  useImperativeHandle(ref, () => ({
    savePicks,
    detectPicks,
    getPicks:  () => picks,
    getSaving: () => saving,
  }), [savePicks, detectPicks, picks, saving]);

  if (!bscanData || bscanData.length === 0) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#9ca3af', background: '#111', borderRadius: 8 }}>
        <p style={{ margin: 0 }}>No B-scan data available.</p>
        <p style={{ fontSize: 12, marginTop: 8 }}>
          Upload Proceq .scan or GSSI .dzt files to enable the interactive B-scan viewer.
        </p>
      </div>
    );
  }

  const swathPicks = picks.filter(p => p.swath_idx === activeSwath);

  return (
    // Fill the available height of the InteractiveTabBody container so the
    // B-scan dominates the viewport like Proceq OneVision / GSSI RADAN.
    // Toolbar + hint row stay fixed at the top via flex-shrink:0; the canvas
    // container is flex:1 so it grows to consume the rest.
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 8,
      background: '#111', borderRadius: 8, padding: 12,
      height: '100%', minHeight: 0,
    }}>

      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', paddingBottom: 8, borderBottom: '1px solid #374151', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: '#9ca3af', fontSize: 12 }}>Swath:</span>
          <select value={activeSwath} onChange={e => setActiveSwath(Number(e.target.value))} style={selectStyle}>
            {bscanData.map((_, i) => <option key={i} value={i}>Swath {i + 1}</option>)}
          </select>
        </div>

        <div style={divider} />

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button onClick={() => setZoomClamped(z => z - 0.25)} style={btnStyle}>−</button>
          <span style={{ color: '#f9fafb', fontSize: 12, minWidth: 40, textAlign: 'center' }}>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoomClamped(z => z + 0.25)} style={btnStyle}>+</button>
          <button onClick={() => setZoomClamped(1)} style={btnStyle}>1:1</button>
          <button onClick={() => setZoomClamped(minZoom)} style={btnStyle}>Fit</button>
        </div>

        <div style={divider} />

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: '#9ca3af', fontSize: 12 }}>Gain:</span>
          <input type="range" min={0.5} max={4} step={0.1} value={contrast}
            onChange={e => setContrast(Number(e.target.value))}
            style={{ width: 80, accentColor: ACCENT }} />
          <span style={{ color: '#f9fafb', fontSize: 11, minWidth: 24 }}>{contrast.toFixed(1)}×</span>
        </div>

        <div style={divider} />

        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={() => setColormap('gray')}    style={{ ...btnStyle, background: colormap === 'gray'    ? ACCENT : ACCENT_DIM }}>Gray</button>
          <button onClick={() => setColormap('seismic')} style={{ ...btnStyle, background: colormap === 'seismic' ? ACCENT : ACCENT_DIM }}>Seismic</button>
        </div>

        <div style={{ flex: 1 }} />

        <span style={{ color: '#9ca3af', fontSize: 11 }}>
          {swathPicks.length} pick{swathPicks.length === 1 ? '' : 's'} on swath {activeSwath + 1}
        </span>

        <button
          onClick={detectPicks}
          disabled={saving || !jobId}
          style={{ ...btnStyle, background: '#7c3aed', padding: '4px 14px', fontWeight: 600, opacity: !jobId ? 0.5 : 1 }}
          title={!jobId ? 'Save the project first' : 'Re-run hyperbola detection on the stored B-scan (wipes manual edits)'}
        >
          Detect Picks
        </button>

        <button
          onClick={savePicks}
          disabled={saving || !jobId}
          style={{ ...btnStyle, background: '#16a34a', padding: '4px 16px', fontWeight: 600, opacity: !jobId ? 0.5 : 1 }}
          title={!jobId ? 'Save the project first to persist picks' : 'Save picks to the server'}
        >
          {saving ? 'Saving…' : 'Save Picks'}
        </button>

        {saveMsg && (
          <span style={{ fontSize: 11, color: saveMsg.startsWith('✓') ? '#4ade80' : '#f87171' }}>
            {saveMsg}
          </span>
        )}
      </div>

      <div style={{ fontSize: 11, color: '#6b7280', display: 'flex', gap: 16, flexWrap: 'wrap', flexShrink: 0 }}>
        <span>• Click to add pick</span>
        <span>• Drag to move pick</span>
        <span>• Right-click to remove</span>
        <span>• Ctrl+scroll to zoom</span>
        <span style={{ color: '#c4b5fd' }}>● Automated picks</span>
        <span style={{ color: '#a78bfa' }}>● Manual picks</span>
      </div>

      <div
        ref={scrollRef}
        style={{
          overflow: 'auto', width: '100%', background: '#000', borderRadius: 4, border: '1px solid #374151',
          flex: 1, minHeight: 0, position: 'relative', touchAction: 'none',
        }}
      >
        {loading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.7)', color: '#fff', zIndex: 10, fontSize: 13,
          }}>
            Loading B-scan…
          </div>
        )}
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onContextMenu={e => e.preventDefault()}
          style={{
            display: 'block',
            imageRendering: zoom > 2 ? 'pixelated' : 'auto',
            width:  dims.n_traces > 0
              ? `${Math.max(containerWidth, dims.n_traces * zoom)}px`
              : '100%',
            height: dims.n_samples > 0 ? `${dims.n_samples * zoom}px` : '300px',
            minWidth: '100%',
            cursor: 'crosshair',
            userSelect: 'none',
          }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#6b7280', padding: '0 4px', flexShrink: 0 }}>
        <span>← Distance along scan (traces) →</span>
        <span>{dims.n_traces} × {dims.n_samples} · Swath {activeSwath + 1}/{bscanData.length}</span>
        <span>↕ Depth (in)</span>
      </div>
    </div>
  );
});

export default BScanViewer;

const btnStyle: React.CSSProperties = {
  padding: '3px 10px', fontSize: 12,
  border: '1px solid #4b5563', borderRadius: 4,
  background: '#374151', color: '#f9fafb', cursor: 'pointer',
};
const selectStyle: React.CSSProperties = {
  padding: '3px 8px', fontSize: 12,
  border: '1px solid #4b5563', borderRadius: 4,
  background: '#374151', color: '#f9fafb',
};
const divider: React.CSSProperties = {
  width: 1, height: 20, background: '#374151',
};
