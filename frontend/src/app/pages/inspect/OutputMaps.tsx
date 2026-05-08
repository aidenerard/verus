import { useEffect, useRef, type RefObject } from 'react';
import { Download } from 'lucide-react';
import { RAISED, BORDER, TEXT, TEXT2 } from './constants';
import type { AnalysisResult, OutputTab } from './types';
import MapColorBar from './MapColorBar';

interface Badge { text: string; color: string }

interface Props {
  outputTab: OutputTab;
  hasResult: boolean;
  analysisResult: AnalysisResult | null;
  condCanvasRef: RefObject<HTMLCanvasElement>;
  rebarCanvasRef: RefObject<HTMLCanvasElement>;
  rebarColorbarRef: RefObject<HTMLCanvasElement>;
  ampCanvasRef: RefObject<HTMLCanvasElement>;
  onExport: () => void;
  condBadge: () => Badge | null;
  depthBadge: () => Badge | null;
  ampBadge: () => Badge | null;
}

export default function OutputMaps({
  outputTab, hasResult, analysisResult,
  condCanvasRef, rebarCanvasRef, rebarColorbarRef, ampCanvasRef,
  onExport, condBadge, depthBadge, ampBadge,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          console.log('[OutputMaps] container sized:', Math.round(width), 'x', Math.round(height));
        }
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  if (!hasResult) {
    return (
      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', padding: 32 }}>
          <p style={{ color: TEXT2, fontSize: 13, marginBottom: 8 }}>Run an analysis to generate output maps.</p>
          <p style={{ color: TEXT2, fontSize: 11 }}>GPS map available immediately — switch to the GPS tab above.</p>
        </div>
      </div>
    );
  }

  const r = analysisResult!;

  const pd = r.prob_grid_data;
  console.log(
    '[OutputMaps] prob_grid_data:',
    pd ? `${pd.length} rows × ${pd[0]?.length ?? 0} cols, sample[0][0]=` + pd[0]?.[0] : 'missing',
    '| prob_grid (b64):', r.prob_grid ? `${r.prob_grid.length} chars` : 'missing',
  );

  const canvasStyle: React.CSSProperties = {
    position: 'absolute', inset: 0, width: '100%', height: '100%',
    display: 'block', imageRendering: 'pixelated',
  };
  const imgStyle: React.CSSProperties = {
    width: '100%', height: '100%', objectFit: 'contain', display: 'block',
  };
  const naStyle: React.CSSProperties = {
    padding: 48, textAlign: 'center', color: TEXT2, fontSize: 12,
  };

  const getBadge = (): Badge | null => {
    if (outputTab === 'condition')   return condBadge();
    if (outputTab === 'rebar_depth') return depthBadge();
    if (outputTab === 'amplitude')   return ampBadge();
    return null;
  };
  const badge = getBadge();

  const hasCondCanvas  = !!(
    (r.prob_grid_data  && r.prob_grid_data.length  > 0) || r.prob_grid
  );
  const hasRebarCanvas = !!(r.rebar_depth_grid && r.rebar_depth_grid.length > 0);
  const hasAmpCanvas   = !!(r.amplitude_grid_data && r.amplitude_grid_data.length > 0);

  // Rebar tab shows colorbar: main canvas occupies top, colorbar occupies bottom 44px
  const COLORBAR_H = 44;
  const showRebarColorbar = outputTab === 'rebar_depth' && hasRebarCanvas;

  const colorBar = (() => {
    if (outputTab === 'condition') {
      return <MapColorBar gradient="spectral" min={0} max={100} label="Delamination probability (%)" formatValue={v => `${v}%`} />;
    }
    if (outputTab === 'amplitude') {
      return <MapColorBar gradient="yellow-red" label="Amplitude" />;
    }
    return null;
  })();

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div ref={containerRef} style={{ flex: 1, position: 'relative', overflow: 'hidden', background: RAISED, minHeight: 0 }}>
        {badge && (
          <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 5, fontSize: 10, fontWeight: 700, padding: '3px 10px', borderRadius: 20, background: badge.color + '20', color: badge.color, backdropFilter: 'blur(4px)' }}>
            {badge.text}
          </div>
        )}
        <button onClick={onExport} style={{ position: 'absolute', bottom: showRebarColorbar ? COLORBAR_H + 8 : 12, right: 12, zIndex: 5, display: 'flex', alignItems: 'center', gap: 5, padding: '6px 14px', background: 'rgba(255,255,255,0.92)', color: TEXT, border: `1px solid ${BORDER}`, cursor: 'pointer', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: 'Inter, sans-serif', backdropFilter: 'blur(4px)' }}>
          <Download size={11} /> Export
        </button>

        {/* ── Condition canvas ── */}
        <canvas
          ref={condCanvasRef}
          style={{ ...canvasStyle, display: outputTab === 'condition' && hasCondCanvas ? 'block' : 'none' }}
        />
        {outputTab === 'condition' && !hasCondCanvas && (
          r.cscan_url
            ? <img src={r.cscan_url} alt="Condition map" style={imgStyle} />
            : <div style={naStyle}>C-scan not available — re-run analysis.</div>
        )}

        {/* ── Rebar depth canvas + colorbar ── */}
        {outputTab === 'rebar_depth' && r.rebar_model_used !== undefined && (
          <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 5 }}>
            <span style={{ fontSize: 10, fontWeight: 700, padding: '3px 10px', borderRadius: 20, background: r.rebar_model_used ? '#22c55e20' : '#f59e0b20', color: r.rebar_model_used ? '#16a34a' : '#d97706' }}>
              {r.rebar_model_used ? 'AI Model' : 'Physics Estimate'}
            </span>
          </div>
        )}
        <canvas
          ref={rebarCanvasRef}
          style={{
            position: 'absolute',
            top: 0, left: 0, right: 0,
            bottom: showRebarColorbar ? COLORBAR_H : 0,
            width: '100%',
            display: outputTab === 'rebar_depth' && hasRebarCanvas ? 'block' : 'none',
            imageRendering: 'pixelated',
          }}
        />
        {showRebarColorbar && (
          <canvas
            ref={rebarColorbarRef}
            style={{
              position: 'absolute', bottom: 0, left: 0, right: 0,
              width: '100%', height: COLORBAR_H,
              display: 'block', background: '#fff',
              borderTop: `1px solid ${BORDER}`,
            }}
          />
        )}
        {outputTab === 'rebar_depth' && !hasRebarCanvas && (
          r.rebar_cscan_image_url
            ? <img src={r.rebar_cscan_image_url} alt="Rebar depth map" style={imgStyle} />
            : <div style={naStyle}>Rebar depth map not available — re-run analysis.</div>
        )}

        {/* ── Amplitude canvas ── */}
        <canvas
          ref={ampCanvasRef}
          style={{ ...canvasStyle, display: outputTab === 'amplitude' && hasAmpCanvas ? 'block' : 'none' }}
        />
        {outputTab === 'amplitude' && !hasAmpCanvas && (
          r.amplitude_image_url
            ? <img src={r.amplitude_image_url} alt="Amplitude map" style={imgStyle} />
            : r.amplitude_image
              ? <img src={`data:image/png;base64,${r.amplitude_image}`} alt="Amplitude map" style={imgStyle} />
              : <div style={naStyle}>Amplitude map not available — re-run analysis.</div>
        )}
      </div>

      {colorBar}
    </div>
  );
}
