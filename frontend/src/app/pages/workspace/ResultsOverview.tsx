import { useMemo } from 'react';
import { ImageOff } from 'lucide-react';
import type { AnalysisResult } from '../inspect/types';
import { useMapbox } from '../inspect/useMapbox';
import { PlaceholderPanel, QuantitiesTable } from './ResultsQuantities';
import { BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3, ACCENT, ACCENT_SOFT } from './tokens';

interface PanelSpec { title: string; src: string | undefined }

function resolveImageSrc(urlField?: string | null, b64Field?: string | null): string | undefined {
  // Prefer a storage URL (large blobs are offloaded to the job-results
  // bucket); fall back to an inline base64 PNG for older/local results.
  if (urlField) return urlField;
  if (!b64Field) return undefined;
  if (b64Field.startsWith('http') || b64Field.startsWith('data:')) return b64Field;
  return `data:image/png;base64,${b64Field}`;
}

function meanRebarDepth(result: AnalysisResult): number | undefined {
  const samples = (result.per_file_summary ?? [])
    .map(f => f.rebar_depth_mean)
    .filter((n): n is number => typeof n === 'number');
  if (!samples.length) return undefined;
  return samples.reduce((a, b) => a + b, 0) / samples.length;
}

export default function OverviewTab({ result }: { result: AnalysisResult }) {
  // Maps come from the SAME canonical backend renderers so output is fixed and
  // identical across datasets (depth: build_unified_depth_map YlOrRd_r 3.0–8.5";
  // corrosion: render_corrosion_db_map). No client-side auto-scaling — see
  // docs/DEPTH_MAP_SPEC.md.
  const depthSrc = useMemo(
    () => resolveImageSrc(
      result.rebar_depth_map_url ?? result.rebar_depth_image_url,
      result.rebar_depth_map ?? result.rebar_depth_image,
    ),
    [result],
  );
  const corrosionSrc = useMemo(
    () => resolveImageSrc(result.corrosion_map_url, result.corrosion_map),
    [result],
  );
  const dielectricSrc = useMemo(
    () => resolveImageSrc(result.dielectric_map_url, result.dielectric_map),
    [result],
  );

  const stats = useMemo(() => {
    const mean     = result.mean_depth_inches ?? meanRebarDepth(result);
    const risk     = result.high_risk_pct     ?? result.delamination_pct;
    const moisture = result.high_moisture_pct ?? undefined;
    return { mean, risk, moisture };
  }, [result]);

  const { mapContainerRef } = useMapbox({
    analysisResult: result,
    layerVis:       { gpr: true, condition: true, amplitude: false, satellite: true, annotations: true },
    conditionOpacity: 80,
  });

  // per_file_summary is DZT-only; Proceq results don't carry it.
  const hasGps = (result.per_file_summary ?? []).some(f => f.gps);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="gpr-panel-grid">
        <ResultPanel title="Rebar Depth Map" src={depthSrc} />
        <ResultPanel title="Corrosion Risk Map" src={corrosionSrc} />
        {dielectricSrc
          ? <ResultPanel title="Dielectric / Moisture Map" src={dielectricSrc} />
          : <PlaceholderPanel
              title="Dielectric / Moisture Map"
              reason={result.dielectric_map_unavailable_reason
                ?? 'Requires a metal-plate calibration scan at data collection to compute per-trace dielectric.'}
            />}
        <QuantitiesTable
          quantities={result.quantities ?? result.stats?.quantities}
          meanDepth={stats.mean}
          highRisk={stats.risk}
        />
      </div>

      <div style={{ background: PANEL, border: `1px solid ${BORDER}`, padding: '14px 20px', display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'center' }}>
        <Stat label="Mean Depth"    value={stats.mean !== undefined ? `${stats.mean.toFixed(2)}"` : '—'} />
        <Stat label="High Risk"     value={stats.risk !== undefined ? `${stats.risk.toFixed(0)}%` : '—'} accent={typeof stats.risk === 'number' && stats.risk >= 30} />
        <Stat label="High Moisture" value={stats.moisture !== undefined && stats.moisture !== null ? `${stats.moisture.toFixed(0)}%` : 'N/A'} />
        <Stat label="Standard"      value={result.astm_method ?? 'ASTM D6087-22'} />
      </div>

      {hasGps && (
        <div style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div style={{ padding: '12px 20px', borderBottom: `1px solid ${BORDER}`, fontSize: 11, fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: TEXT2 }}>
            Scan Location
          </div>
          <div style={{ position: 'relative', height: 320 }}>
            <div ref={mapContainerRef} style={{ position: 'absolute', inset: 0 }} />
          </div>
        </div>
      )}

      <style>{`
        .gpr-panel-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
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
    <div style={{
      background: PANEL, border: `1px solid ${BORDER}`,
      display: 'flex', flexDirection: 'column',
      minHeight: 400, overflow: 'hidden', borderRadius: 8,
    }}>
      <div style={{
        flexShrink: 0,
        padding: '10px 14px', borderBottom: `1px solid ${BORDER}`,
        fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT2,
      }}>
        {title}
      </div>
      <div style={{ flex: 1, minHeight: 0, position: 'relative', overflow: 'hidden', background: RAISED }}>
        {src ? (
          <img
            src={src}
            alt={title}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'fill', display: 'block' }}
          />
        ) : (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, color: TEXT3 }}>
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
