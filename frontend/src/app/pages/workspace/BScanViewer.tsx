import { useEffect, useRef, useState, useCallback } from 'react';

interface TraceData {
  data:      string;   // base64 + zlib + int8
  n_traces:  number;
  n_samples: number;
  encoding:  string;
}

interface BScanViewerProps {
  bscanData:  TraceData[];
  picks_in?:  number[][];   // optional per-swath horizon picks in inches
  epsr?:      number;
  timeRangeNs?: number;
}

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
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    return new Int8Array(buffer.buffer);
  } catch (e) {
    console.error('Failed to decode traces:', e);
    return null;
  }
}

function renderBScan(
  canvas: HTMLCanvasElement,
  traces: Int8Array,
  n_traces: number,
  n_samples: number,
  options: {
    contrast?: number;
    colormap?: 'gray' | 'seismic';
    picks?:    number[];
  } = {},
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const { contrast = 1.5, colormap = 'gray', picks } = options;
  const imageData = ctx.createImageData(n_traces, n_samples);
  const data      = imageData.data;

  for (let t = 0; t < n_traces; t++) {
    for (let s = 0; s < n_samples; s++) {
      const val     = traces[t * n_samples + s] / 127;
      const boosted = Math.max(-1, Math.min(1, val * contrast));
      const pixel   = Math.round((boosted + 1) / 2 * 255);
      const idx     = (s * n_traces + t) * 4;

      if (colormap === 'seismic') {
        if (boosted > 0) {
          data[idx]     = 255;
          data[idx + 1] = Math.round(255 * (1 - boosted));
          data[idx + 2] = Math.round(255 * (1 - boosted));
        } else {
          data[idx]     = Math.round(255 * (1 + boosted));
          data[idx + 1] = Math.round(255 * (1 + boosted));
          data[idx + 2] = 255;
        }
      } else {
        data[idx]     = pixel;
        data[idx + 1] = pixel;
        data[idx + 2] = pixel;
      }
      data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  if (picks && picks.length > 0) {
    ctx.strokeStyle = '#FF4400';
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    for (let t = 0; t < Math.min(picks.length, n_traces); t++) {
      const s = picks[t];
      if (t === 0) ctx.moveTo(t, s);
      else ctx.lineTo(t, s);
    }
    ctx.stroke();
  }
}

export default function BScanViewer({
  bscanData,
  picks_in,
  epsr = 9.0,
  timeRangeNs = 16.0,
}: BScanViewerProps) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollRef    = useRef<HTMLDivElement>(null);
  const isPanning    = useRef(false);
  const panStart     = useRef({ x: 0, scrollLeft: 0 });

  const [activeSwath, setActiveSwath] = useState(0);
  const [zoom,        setZoom]        = useState(1.0);
  const [contrast,    setContrast]    = useState(1.5);
  const [colormap,    setColormap]    = useState<'gray' | 'seismic'>('gray');
  const [loading,     setLoading]     = useState(false);
  const [traces,      setTraces]      = useState<Int8Array | null>(null);
  const [dims,        setDims]        = useState({ n_traces: 0, n_samples: 0 });

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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !traces || dims.n_traces === 0) return;
    canvas.width  = dims.n_traces;
    canvas.height = dims.n_samples;

    let pickSamples: number[] | undefined;
    if (picks_in && picks_in[activeSwath]) {
      const velocity    = 0.15 / Math.sqrt(epsr);
      const nsPerSample = timeRangeNs / dims.n_samples;
      pickSamples = picks_in[activeSwath].map(depthIn => {
        const depthM = depthIn / 39.3701;
        const tNs    = depthM / velocity * 2;
        return Math.round(tNs / nsPerSample);
      });
    }

    renderBScan(canvas, traces, dims.n_traces, dims.n_samples, {
      contrast, colormap, picks: pickSamples,
    });
  }, [traces, dims, contrast, colormap, picks_in, activeSwath, epsr, timeRangeNs]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      setZoom(z => Math.min(8, Math.max(0.25, z + (e.deltaY < 0 ? 0.15 : -0.15))));
    } else if (scrollRef.current) {
      scrollRef.current.scrollLeft += e.deltaY;
    }
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isPanning.current = true;
    panStart.current  = { x: e.clientX, scrollLeft: scrollRef.current?.scrollLeft ?? 0 };
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current || !scrollRef.current) return;
    const dx = e.clientX - panStart.current.x;
    scrollRef.current.scrollLeft = panStart.current.scrollLeft - dx;
  }, []);

  const handleMouseUp = useCallback(() => { isPanning.current = false; }, []);

  if (!bscanData || bscanData.length === 0) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#9ca3af', background: '#111', borderRadius: '8px' }}>
        <p>No B-scan data available.</p>
        <p style={{ fontSize: '12px', marginTop: '8px' }}>
          Upload Proceq .scan files to enable the interactive B-scan viewer.
        </p>
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{
      display: 'flex', flexDirection: 'column', gap: '8px',
      background: '#111', borderRadius: '8px', padding: '12px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap', paddingBottom: '8px', borderBottom: '1px solid #374151' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: '#9ca3af', fontSize: '12px' }}>Swath:</span>
          <select value={activeSwath} onChange={e => setActiveSwath(Number(e.target.value))} style={selectStyle}>
            {bscanData.map((_, i) => <option key={i} value={i}>Swath {i + 1}</option>)}
          </select>
        </div>
        <div style={{ width: '1px', height: '20px', background: '#374151' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: '#9ca3af', fontSize: '12px' }}>Zoom:</span>
          <button onClick={() => setZoom(z => Math.max(0.25, z - 0.25))} style={btnStyle}>−</button>
          <span style={{ color: '#f9fafb', fontSize: '12px', minWidth: '40px', textAlign: 'center' }}>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(8, z + 0.25))} style={btnStyle}>+</button>
          <button onClick={() => setZoom(1)} style={btnStyle}>1:1</button>
          <button onClick={() => setZoom(0.5)} style={btnStyle}>Fit</button>
        </div>
        <div style={{ width: '1px', height: '20px', background: '#374151' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: '#9ca3af', fontSize: '12px' }}>Gain:</span>
          <input type="range" min="0.5" max="4" step="0.1" value={contrast} onChange={e => setContrast(Number(e.target.value))} style={{ width: '80px', accentColor: '#E8572A' }} />
          <span style={{ color: '#f9fafb', fontSize: '11px', minWidth: '24px' }}>{contrast.toFixed(1)}×</span>
        </div>
        <div style={{ width: '1px', height: '20px', background: '#374151' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: '#9ca3af', fontSize: '12px' }}>Color:</span>
          <button onClick={() => setColormap('gray')} style={{ ...btnStyle, background: colormap === 'gray' ? '#E8572A' : '#374151' }}>Gray</button>
          <button onClick={() => setColormap('seismic')} style={{ ...btnStyle, background: colormap === 'seismic' ? '#E8572A' : '#374151' }}>Seismic</button>
        </div>
        <div style={{ flex: 1 }} />
        <span style={{ color: '#6b7280', fontSize: '11px' }}>Drag to pan · Ctrl+scroll to zoom</span>
      </div>

      <div
        ref={scrollRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          overflow: 'auto', cursor: isPanning.current ? 'grabbing' : 'crosshair',
          background: '#000', borderRadius: '4px', minHeight: '300px', maxHeight: '65vh', position: 'relative',
        }}
      >
        {loading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.7)', color: '#fff', zIndex: 10 }}>
            Loading B-scan data...
          </div>
        )}
        <canvas
          ref={canvasRef}
          style={{
            display: 'block',
            imageRendering: zoom > 2 ? 'pixelated' : 'auto',
            width:  dims.n_traces  > 0 ? `${dims.n_traces  * zoom}px` : '100%',
            height: dims.n_samples > 0 ? `${dims.n_samples * zoom}px` : 'auto',
          }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#6b7280', padding: '0 4px' }}>
        <span>← Distance along scan (m) →</span>
        <span>{dims.n_traces} traces × {dims.n_samples} samples | Swath {activeSwath + 1} of {bscanData.length}</span>
        <span>↕ Two-way travel time (ns) / Depth (in)</span>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: '3px 10px', fontSize: '12px',
  border: '1px solid #4b5563', borderRadius: '4px',
  background: '#374151', color: '#f9fafb', cursor: 'pointer',
};

const selectStyle: React.CSSProperties = {
  padding: '3px 8px', fontSize: '12px',
  border: '1px solid #4b5563', borderRadius: '4px',
  background: '#374151', color: '#f9fafb', cursor: 'pointer',
};
