import { useMemo } from 'react';
import { ImageOff } from 'lucide-react';
import type { AnalysisResult } from '../inspect/types';
import { SERVER } from '../inspect/constants';
import { useMapbox } from '../inspect/useMapbox';
import { BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3, ACCENT, ACCENT_SOFT } from './tokens';

interface Props {
  result: AnalysisResult;
}

interface PanelSpec { title: string; src: string | undefined }

function resolveUrl(src: string | undefined): string | undefined {
  if (!src) return undefined;
  if (src.startsWith('http') || src.startsWith('data:')) return src;
  return `${SERVER}${src.startsWith('/') ? '' : '/'}${src}`;
}

function meanRebarDepth(result: AnalysisResult): number | undefined {
  const samples = result.per_file_summary
    .map(f => f.rebar_depth_mean)
    .filter((n): n is number => typeof n === 'number');
  if (!samples.length) return undefined;
  return samples.reduce((a, b) => a + b, 0) / samples.length;
}

export default function GPRResults({ result }: Props) {
  const panels: PanelSpec[] = useMemo(() => [
    { title: 'Horizon Picks', src: resolveUrl(result.horizon_picks ?? result.cscan_url ?? result.cscan_image) },
    { title: 'Rebar Depth Map', src: resolveUrl(result.rebar_depth_map ?? result.rebar_depth_image) },
    { title: 'Corrosion Risk', src: resolveUrl(result.corrosion_map ?? result.amplitude_image) },
  ], [result]);

  const stats = useMemo(() => {
    const mean = result.mean_depth_inches ?? meanRebarDepth(result);
    const thick = result.deck_thickness_inches;
    const risk = result.high_risk_pct ?? result.delamination_pct;
    return { mean, thick, risk };
  }, [result]);

  const { mapContainerRef } = useMapbox({
    analysisResult: result,
    layerVis:       { gpr: true, condition: true, amplitude: false, satellite: true, annotations: true },
    conditionOpacity: 80,
  });

  const hasGps = result.per_file_summary.some(f => f.gps);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="gpr-panel-grid">
        {panels.map(p => <ResultPanel key={p.title} title={p.title} src={p.src} />)}
      </div>

      <div style={{ background: PANEL, border: `1px solid ${BORDER}`, padding: '14px 20px', display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'center' }}>
        <Stat label="Mean Depth"      value={stats.mean !== undefined ? `${stats.mean.toFixed(2)}"` : '—'} />
        <Stat label="Deck Thickness"  value={stats.thick !== undefined ? `${stats.thick.toFixed(2)}"` : '—'} />
        <Stat label="Risk"            value={stats.risk !== undefined ? `${stats.risk.toFixed(0)}%` : '—'} accent={typeof stats.risk === 'number' && stats.risk >= 30} />
      </div>

      <div style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
        <div style={{ padding: '12px 20px', borderBottom: `1px solid ${BORDER}`, fontSize: 11, fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: TEXT2 }}>
          Scan Location
        </div>
        <div style={{ position: 'relative', height: 320 }}>
          <div ref={mapContainerRef} style={{ position: 'absolute', inset: 0 }} />
          {!hasGps && (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: RAISED, color: TEXT3, fontSize: 12, pointerEvents: 'none' }}>
              No GPS data in scan
            </div>
          )}
        </div>
      </div>

      <style>{`
        .gpr-panel-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 16px;
        }
        @media (max-width: 900px) {
          .gpr-panel-grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  );
}

function ResultPanel({ title, src }: PanelSpec) {
  return (
    <div style={{ background: PANEL, border: `1px solid ${BORDER}`, display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px 14px', borderBottom: `1px solid ${BORDER}`, fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT2 }}>
        {title}
      </div>
      <div style={{ flex: 1, aspectRatio: '4 / 3', background: RAISED, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
        {src ? (
          <img src={src} alt={title} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, color: TEXT3 }}>
            <ImageOff size={22} />
            <span style={{ fontSize: 11 }}>No data</span>
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{ fontSize: 9, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT3 }}>{label}</span>
      <span style={{
        fontSize: 16, fontWeight: 700, color: accent ? ACCENT : TEXT,
        padding: accent ? '0 6px' : 0, background: accent ? ACCENT_SOFT : 'transparent',
      }}>{value}</span>
    </div>
  );
}
