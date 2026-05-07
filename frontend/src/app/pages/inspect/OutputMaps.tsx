import type { RefObject } from 'react';
import { Download } from 'lucide-react';
import { RAISED, BORDER, TEXT, TEXT2, ACCENT } from './constants';
import type { AnalysisResult, OutputTab } from './types';
import { delamColor } from './utils';
import MapColorBar from './MapColorBar';

interface Badge { text: string; color: string }

interface Props {
  outputTab: OutputTab;
  hasResult: boolean;
  analysisResult: AnalysisResult | null;
  useCondCanvas: boolean;
  condCanvasRef: RefObject<HTMLCanvasElement>;
  useRebarCanvas: boolean;
  rebarCanvasRef: RefObject<HTMLCanvasElement>;
  useAmpCanvas: boolean;
  ampCanvasRef: RefObject<HTMLCanvasElement>;
  onExport: () => void;
  condBadge: () => Badge | null;
  depthBadge: () => Badge | null;
  ampBadge: () => Badge | null;
}

export default function OutputMaps({
  outputTab, hasResult, analysisResult,
  useCondCanvas, condCanvasRef,
  useRebarCanvas, rebarCanvasRef,
  useAmpCanvas, ampCanvasRef,
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
  const imgStyle: React.CSSProperties = { width: '100%', height: 'auto', display: 'block' };
  const canvasStyle: React.CSSProperties = { width: '100%', height: 'auto', display: 'block', imageRendering: 'pixelated' };
  const naStyle: React.CSSProperties = { padding: 48, textAlign: 'center', color: TEXT2, fontSize: 12 };

  const getBadge = (): Badge | null => {
    if (outputTab === 'condition') return condBadge();
    if (outputTab === 'rebar_depth') return depthBadge();
    if (outputTab === 'amplitude') return ampBadge();
    return null;
  };
  const badge = getBadge();

  // Rebar depth range from per-file summaries
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

  const statsBar = outputTab === 'condition' ? (
    <div style={{ flexShrink: 0, display: 'flex', gap: 40, padding: '10px 20px', borderTop: `1px solid ${BORDER}` }}>
      {[
        { label: 'Signals',       value: r.signals_analyzed  != null ? r.signals_analyzed.toLocaleString()         : '--' },
        { label: 'Delamination',  value: r.delamination_pct  != null ? `${r.delamination_pct.toFixed(1)}%`          : '--', color: r.delamination_pct != null ? delamColor(r.delamination_pct) : TEXT },
        { label: 'Sound',         value: r.sound_pct         != null ? `${r.sound_pct.toFixed(1)}%`                 : '--' },
        { label: 'Analysis Time', value: r.analysis_time_sec != null ? `${r.analysis_time_sec.toFixed(1)}s`         : '--' },
      ].map(({ label, value, color }) => (
        <div key={label}>
          <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.09em', textTransform: 'uppercase', color: TEXT2, marginBottom: 2 }}>{label}</div>
          <div style={{ fontSize: 15, fontWeight: 700, color: color || TEXT }}>{value}</div>
        </div>
      ))}
    </div>
  ) : null;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Map area */}
      <div style={{ flex: 1, position: 'relative', overflow: 'auto', background: RAISED, minHeight: 0 }}>
        {badge && (
          <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 5, fontSize: 10, fontWeight: 700, padding: '3px 10px', borderRadius: 20, background: badge.color + '20', color: badge.color, backdropFilter: 'blur(4px)' }}>
            {badge.text}
          </div>
        )}
        <button onClick={onExport} style={{ position: 'absolute', bottom: 12, right: 12, zIndex: 5, display: 'flex', alignItems: 'center', gap: 5, padding: '6px 14px', background: 'rgba(255,255,255,0.92)', color: TEXT, border: `1px solid ${BORDER}`, cursor: 'pointer', fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: 'Inter, sans-serif', backdropFilter: 'blur(4px)' }}>
          <Download size={11} /> Export
        </button>

        {outputTab === 'condition' && (
          useCondCanvas
            ? <canvas ref={condCanvasRef} style={canvasStyle} />
            : r.cscan_image
              ? <img src={`data:image/png;base64,${r.cscan_image}`} alt="Condition map" style={imgStyle} />
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
            {useRebarCanvas
              ? <canvas ref={rebarCanvasRef} style={canvasStyle} />
              : (r.rebar_cscan_image || r.rebar_depth_image)
                ? <img src={`data:image/png;base64,${r.rebar_cscan_image || r.rebar_depth_image}`} alt="Rebar depth map" style={imgStyle} />
                : r.rebar_cscan_image_url
                  ? <img src={r.rebar_cscan_image_url} alt="Rebar depth map" style={imgStyle} />
                  : <div style={naStyle}>Rebar depth map not available — re-run analysis.</div>
            }
          </>
        )}

        {outputTab === 'amplitude' && (
          useAmpCanvas
            ? <canvas ref={ampCanvasRef} style={canvasStyle} />
            : r.amplitude_image
              ? <img src={`data:image/png;base64,${r.amplitude_image}`} alt="Amplitude map" style={imgStyle} />
              : r.amplitude_image_url
                ? <img src={r.amplitude_image_url} alt="Amplitude map" style={imgStyle} />
                : <div style={naStyle}>Amplitude map not available — re-run analysis.</div>
        )}
      </div>

      {statsBar}
      {colorBar}
    </div>
  );
}
