import type { RefObject } from 'react';
import { Download } from 'lucide-react';
import { PANEL, RAISED, BORDER, TEXT, TEXT2, ACCENT } from './constants';
import type { AnalysisResult, OutputTab } from './types';
import { delamColor } from './utils';

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

function MapHeader({ title, subtitle, badge, onExport }: {
  title: string; subtitle: string; badge: Badge | null; onExport: () => void;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <div>
        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2, marginBottom: 4 }}>{title}</div>
        <div style={{ fontSize: 11, color: TEXT2 }}>{subtitle}</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {badge && (
          <span style={{ fontSize: 10, fontWeight: 700, padding: '3px 10px', borderRadius: 20, background: badge.color + '20', color: badge.color }}>
            {badge.text}
          </span>
        )}
        <button onClick={onExport}
          style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '7px 16px', background: ACCENT, color: '#fff', border: 'none', cursor: 'pointer', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: 'Inter, sans-serif' }}>
          <Download size={13} /> Export
        </button>
      </div>
    </div>
  );
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
  const canvasStyle: React.CSSProperties = { width: '100%', height: 'auto', display: 'block', imageRendering: 'pixelated' };
  const imgStyle: React.CSSProperties = { width: '100%', height: 'auto', display: 'block' };
  const naStyle: React.CSSProperties = { padding: 32, textAlign: 'center', color: TEXT2, fontSize: 12 };
  const legendStyle: React.CSSProperties = { display: 'flex', justifyContent: 'space-between', padding: '6px 12px', background: RAISED, border: `1px solid ${BORDER}`, fontSize: 10, color: TEXT2 };

  return (
    <div style={{ padding: 24, minHeight: '100%', display: 'flex', flexDirection: 'column', gap: 20 }}>

      {outputTab === 'condition' && (
        <>
          <MapHeader
            title="C-Scan Condition Map"
            subtitle={`ASTM D6087 · ${r.signals_analyzed.toLocaleString()} signals`}
            badge={condBadge() ? { text: `Model Confidence: ${condBadge()!.text}`, color: condBadge()!.color } : null}
            onExport={onExport}
          />
          <div style={{ border: `1px solid ${BORDER}`, overflow: 'hidden', background: RAISED }}>
            {useCondCanvas
              ? <canvas ref={condCanvasRef} style={canvasStyle} />
              : r.cscan_image
                ? <img src={`data:image/png;base64,${r.cscan_image}`} alt="Condition map" style={imgStyle} />
                : r.cscan_url
                  ? <img src={r.cscan_url} alt="Condition map" style={imgStyle} />
                  : <div style={naStyle}>C-scan not available — re-run analysis.</div>
            }
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            {[
              { label: 'Signals Analyzed', value: r.signals_analyzed.toLocaleString() },
              { label: 'Delamination', value: `${r.delamination_pct.toFixed(1)}%`, hi: true },
              { label: 'Sound', value: `${r.sound_pct.toFixed(1)}%` },
              { label: 'Analysis Time', value: `${r.analysis_time_sec.toFixed(1)}s` },
            ].map(({ label, value, hi }) => (
              <div key={label} style={{ background: RAISED, border: `1px solid ${BORDER}`, padding: '14px 16px' }}>
                <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: TEXT2, marginBottom: 6 }}>{label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: hi ? delamColor(r.delamination_pct) : TEXT }}>{value}</div>
              </div>
            ))}
          </div>
        </>
      )}

      {outputTab === 'rebar_depth' && (
        <>
          <MapHeader
            title="Rebar Depth Map"
            subtitle="Estimated cover depth in inches"
            badge={depthBadge() ? { text: `Depth Accuracy: ${depthBadge()!.text}`, color: depthBadge()!.color } : null}
            onExport={onExport}
          />
          {r.rebar_model_used !== undefined && (
            <div>
              <span style={{
                fontSize: 10, fontWeight: 700, padding: '3px 10px', borderRadius: 20,
                background: r.rebar_model_used ? '#22c55e20' : '#f59e0b20',
                color: r.rebar_model_used ? '#16a34a' : '#d97706',
              }}>
                {r.rebar_model_used ? 'AI Model' : 'Physics Estimate'}
              </span>
            </div>
          )}
          <div style={{ border: `1px solid ${BORDER}`, overflow: 'hidden', background: RAISED }}>
            {useRebarCanvas
              ? <canvas ref={rebarCanvasRef} style={canvasStyle} />
              : (r.rebar_cscan_image || r.rebar_depth_image)
                ? <img src={`data:image/png;base64,${r.rebar_cscan_image || r.rebar_depth_image}`} alt="Rebar depth map" style={imgStyle} />
                : r.rebar_cscan_image_url
                  ? <img src={r.rebar_cscan_image_url} alt="Rebar depth map" style={imgStyle} />
                  : <div style={naStyle}>Rebar depth map not available — re-run analysis.</div>
            }
          </div>
          <div style={legendStyle}>
            <span>■ Blue — Shallow (0.5")</span>
            <span>■ Green/Yellow — Moderate (1–3")</span>
            <span>■ Red — Deep (&gt;4")</span>
          </div>
        </>
      )}

      {outputTab === 'amplitude' && (
        <>
          <MapHeader
            title="Amplitude Map"
            subtitle="Peak signal amplitude per trace — raw GPR reflection strength"
            badge={ampBadge() ? { text: `Signal Quality: ${ampBadge()!.text}`, color: ampBadge()!.color } : null}
            onExport={onExport}
          />
          <div style={{ border: `1px solid ${BORDER}`, overflow: 'hidden', background: RAISED }}>
            {useAmpCanvas
              ? <canvas ref={ampCanvasRef} style={canvasStyle} />
              : r.amplitude_image
                ? <img src={`data:image/png;base64,${r.amplitude_image}`} alt="Amplitude map" style={imgStyle} />
                : r.amplitude_image_url
                  ? <img src={r.amplitude_image_url} alt="Amplitude map" style={imgStyle} />
                  : <div style={naStyle}>Amplitude map not available — re-run analysis.</div>
            }
          </div>
          <div style={legendStyle}>
            <span>■ Red — Low Amplitude (Deteriorated)</span>
            <span>■ Blue — High Amplitude (Sound)</span>
          </div>
        </>
      )}

    </div>
  );
}
