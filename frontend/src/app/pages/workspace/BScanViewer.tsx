import { useCallback, useRef, useState } from 'react';

interface BScanViewerProps {
  bscanImages: string[];
}

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4.0;

export default function BScanViewer({ bscanImages }: BScanViewerProps) {
  const [zoom, setZoom]   = useState(1.0);
  const scrollRef         = useRef<HTMLDivElement>(null);

  const resolveUrl = (b64: string) =>
    b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;

  const handleZoom = useCallback((delta: number) => {
    setZoom(z => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z + delta)));
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      handleZoom(e.deltaY < 0 ? 0.1 : -0.1);
    } else if (scrollRef.current) {
      e.preventDefault();
      scrollRef.current.scrollLeft += e.deltaY;
    }
  }, [handleZoom]);

  if (!bscanImages || bscanImages.length === 0) {
    return (
      <div style={{
        padding: '40px', textAlign: 'center', color: '#9ca3af',
        background: '#111', borderRadius: 8,
      }}>
        <p style={{ margin: 0 }}>No B-scan data available.</p>
        <p style={{ fontSize: 12, marginTop: 8 }}>
          B-scans are generated from Proceq .scan files.
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '8px 12px', background: '#1f2937', borderRadius: 6,
        color: '#f9fafb', fontSize: 13,
      }}>
        <span style={{ fontWeight: 600 }}>
          B-Scan — {bscanImages.length} swath{bscanImages.length > 1 ? 's' : ''}
        </span>

        <div style={{ flex: 1 }} />

        <span style={{ color: '#9ca3af', fontSize: 11 }}>
          Scroll to pan · Ctrl+scroll to zoom
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button
            onClick={() => handleZoom(-0.25)}
            disabled={zoom <= MIN_ZOOM}
            title="Zoom out"
            style={btnStyle}
          >−</button>
          <span style={{ minWidth: 40, textAlign: 'center', fontSize: 12 }}>
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => handleZoom(0.25)}
            disabled={zoom >= MAX_ZOOM}
            title="Zoom in"
            style={btnStyle}
          >+</button>
          <button
            onClick={() => setZoom(1.0)}
            title="Reset zoom"
            style={{ ...btnStyle, marginLeft: 4 }}
          >Reset</button>
        </div>

        <button
          onClick={() => setZoom(0.5)}
          title="Fit all swaths"
          style={btnStyle}
        >Fit</button>
      </div>

      <div
        ref={scrollRef}
        onWheel={handleWheel}
        style={{
          overflowX: 'auto', overflowY: 'hidden',
          background: '#000', borderRadius: 6,
          border: '1px solid #374151',
          cursor: 'grab', userSelect: 'none',
          minHeight: 300, maxHeight: '60vh',
        }}
      >
        <div style={{
          display: 'inline-flex', flexDirection: 'row',
          alignItems: 'stretch', height: '100%',
          gap: 2, padding: 0,
        }}>
          {bscanImages.map((img, idx) => (
            <div key={idx} style={{ position: 'relative', flexShrink: 0 }}>
              <div style={{
                position: 'absolute', top: 4, left: 4, zIndex: 10,
                background: 'rgba(0,0,0,0.6)', color: '#fff',
                fontSize: 10, padding: '1px 5px', borderRadius: 3,
                pointerEvents: 'none',
              }}>
                Swath {idx + 1}
              </div>

              <img
                src={resolveUrl(img)}
                alt={`B-scan swath ${idx + 1}`}
                style={{
                  height: `${zoom * 400}px`,
                  width: 'auto',
                  display: 'block',
                  imageRendering: zoom > 2 ? 'pixelated' : 'auto',
                }}
                draggable={false}
              />
            </div>
          ))}
        </div>
      </div>

      <div style={{
        display: 'flex', gap: 16, fontSize: 11, color: '#6b7280',
        alignItems: 'center', padding: '0 4px', flexWrap: 'wrap',
      }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          <span style={{
            display: 'inline-block', width: 20, height: 2,
            background: '#FF4400', verticalAlign: 'middle',
          }} />
          Rebar horizon pick
        </span>
        <span>Y = two-way travel time (ns) / depth (in)</span>
        <span>X = distance along scan (m)</span>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: '3px 10px', fontSize: 12,
  border: '1px solid #4b5563', borderRadius: 4,
  background: '#374151', color: '#f9fafb', cursor: 'pointer',
};
