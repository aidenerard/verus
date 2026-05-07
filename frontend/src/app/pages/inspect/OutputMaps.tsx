import type { RefObject } from 'react';
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
  ampCanvasRef: RefObject<HTMLCanvasElement>;
  onExport: () => void;
  condBadge: () => Badge | null;
  depthBadge: () => Badge | null;
  ampBadge: () => Badge | null;
}

export default function OutputMaps({
  outputTab, hasResult, analysisResult,
  condCanvasRef, rebarCanvasRef, ampCanvasRef,
  onExport, condBadge, depthBadge, ampBadge,
}: Props) {
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
  const canvasStyle: React.CSSProperties = {
    width: '100%', height: '100%', display: 'block', imageRendering: 'pixelated',
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

  const rebarFiles = r.per_file_summary.filter(
    f => f.rebar_depth_min != null && f.rebar_depth_max != null,
  );
  const rebarMin = rebarFiles.length ? Math.min(...rebarFiles.map(f => f.rebar_depth_min!)) : undefined;
  const rebarMax = rebarFiles.length ? Math.max(...rebarFiles.map(f => f.rebar_depth_max!)) : undefined;
  const fmtIn = (v: number) => `${v % 1 === 0 ? v : v.toFixed(1)}"`;

  const colorBar = (() => {
    if (outputTab === 'condition') {
      return <MapColorBar gradient="spectral" min={0} max={100} label="Delamination probability (%)" formatValue={v => `${v}%`} />;
    }
    if (outputTab === 'rebar_depth') {
      return <MapColorBar gradient="yellow-red" min={rebarMin} max={rebarMax} label="Cover depth (in)" formatValue={fmtIn} />;
    }
    if (outputTab === 'amplitude') {
      return <MapColorBar gradient="yellow-red" label="Amplitude" />;
    }
    return null;
  })();

  // Canvas is used when we have JSON grid data (new) or base64 grid data (old)
  const hasCondCanvas  = !!(r.prob_grid_data || r.prob_grid);
  const hasRebarCanvas = !!(r.rebar_depth_grid);
  const hasAmpCanvas   = !!(r.amplitude_grid_data);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden', background: RAISED, minHeight: 0 }}>
        {badge && (
          <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 5, fontSize: 10, fontWeight: 700, padding: '3px 10px', borderRadius: 20, background: badge.color + '20', color: badge.color, backdropFilter: 'blur(4px)' }}>
            {badge.text}
          </div>
        )}
        <button onClick={onExport} style={{ position: 'absolute', bottom: 12, right: 12, zIndex: 5, display: 'flex', alignItems: 'center', gap: 5, padding: '6px 14px', background: 'rgba(255,255,255,0.92)', color: TEXT, border: `1px solid ${BORDER}`, cursor: 'pointer', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: 'Inter, sans-serif', backdropFilter: 'blur(4px)' }}>
          <Download size={11} /> Export
        </button>

        {outputTab === 'condition' && (
          hasCondCanvas
            ? <canvas ref={condCanvasRef} style={canvasStyle} />
            : r.cscan_url
              ? <img src={r.cscan_url} alt="Condition map" style={imgStyle} />
              : <div style={naStyle}>C-scan not available — re-run analysis.</div>
        )}

        {outputTab === 'rebar_depth' && (
          <>
            {r.rebar_model_used !== undefined && (
              <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 5 }}>
                <span style={{ fontSize: 10, fontWeight: 700, padding: '3px 10px', borderRadius: 20, background: r.rebar_model_used ? '#22c55e20' : '#f59e0b20', color: r.rebar_model_used ? '#16a34a' : '#d97706' }}>
                  {r.rebar_model_used ? 'AI Model' : 'Physics Estimate'}
                </span>
              </div>
            )}
            {hasRebarCanvas
              ? <canvas ref={rebarCanvasRef} style={canvasStyle} />
              : r.rebar_cscan_image_url
                ? <img src={r.rebar_cscan_image_url} alt="Rebar depth map" style={imgStyle} />
                : <div style={naStyle}>Rebar depth map not available — re-run analysis.</div>
            }
          </>
        )}

        {outputTab === 'amplitude' && (
          hasAmpCanvas
            ? <canvas ref={ampCanvasRef} style={canvasStyle} />
            : r.amplitude_image_url
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
