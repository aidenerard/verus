import { Lock } from 'lucide-react';
import type { DeckQuantities } from '../inspect/types';
import { BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3, ACCENT } from './tokens';

const PANEL_STYLE: React.CSSProperties = {
  background: PANEL, border: `1px solid ${BORDER}`,
  display: 'flex', flexDirection: 'column',
  minHeight: 400, overflow: 'hidden', borderRadius: 8,
};
const HEAD_STYLE: React.CSSProperties = {
  flexShrink: 0, padding: '10px 14px', borderBottom: `1px solid ${BORDER}`,
  fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT2,
};

// Designed "locked feature" panel — not an error state. Tells the inspector
// exactly what extra data collection unlocks the analysis.
export function PlaceholderPanel({ title, reason, badge = 'Coming Soon' }: {
  title: string; reason: string; badge?: string;
}) {
  return (
    <div style={PANEL_STYLE}>
      <div style={HEAD_STYLE}>{title}</div>
      <div style={{
        flex: 1, minHeight: 0, position: 'relative', background: RAISED,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        gap: 12, padding: 24, textAlign: 'center',
      }}>
        <div style={{
          width: 44, height: 44, borderRadius: '50%', background: 'rgba(232,96,28,0.10)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', color: ACCENT,
        }}>
          <Lock size={20} />
        </div>
        <span style={{
          fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase',
          color: ACCENT, background: 'rgba(232,96,28,0.10)', padding: '4px 10px', borderRadius: 2,
        }}>
          {badge}
        </span>
        <p style={{ margin: 0, maxWidth: 320, fontSize: 13, lineHeight: 1.6, color: TEXT2 }}>
          {reason}
        </p>
      </div>
    </div>
  );
}

export function QuantitiesTable({ quantities, meanDepth, highRisk }: {
  quantities?: DeckQuantities | null;
  meanDepth?: number;
  highRisk?: number;
}) {
  const q = quantities;
  const cover = q
    ? `${q.min_cover_in.toFixed(2)}" – ${q.max_cover_in.toFixed(2)}"`
    : '—';
  const rows: Array<[string, string]> = [
    ['Total Picks', q ? q.n_picks.toLocaleString() : '—'],
    ['Mean Cover', q ? `${q.mean_cover_in.toFixed(2)}"` : (meanDepth !== undefined ? `${meanDepth.toFixed(2)}"` : '—')],
    ['Cover Range', cover],
    ['Deterioration', q ? `${q.deteriorated_pct.toFixed(1)}%` : (highRisk !== undefined ? `${highRisk.toFixed(1)}%` : '—')],
    ['Sound Concrete', q ? `${q.sound_pct.toFixed(1)}%` : '—'],
    ['High Moisture Risk', q?.high_moisture_pct !== undefined ? `${q.high_moisture_pct.toFixed(1)}%` : 'N/A — needs plate cal'],
    ['Analysis Standard', q?.astm_method ?? 'ASTM D6087-22'],
    ['Method', q?.deterioration_method ?? 'Depth-corrected amplitude'],
    ['Deterioration Threshold', q?.threshold_note ?? '-8.0 dB default — verify with cores'],
  ];
  return (
    <div style={PANEL_STYLE}>
      <div style={HEAD_STYLE}>Quantities — ASTM D6087</div>
      <div style={{ flex: 1, minHeight: 0, overflow: 'auto', background: RAISED }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <tbody>
            {rows.map(([k, v], i) => (
              <tr key={k} style={{ background: i % 2 ? 'transparent' : 'rgba(0,0,0,0.015)' }}>
                <td style={{ padding: '10px 16px', color: TEXT2, borderBottom: `1px solid ${BORDER}`, whiteSpace: 'nowrap' }}>{k}</td>
                <td style={{ padding: '10px 16px', color: TEXT, fontWeight: 600, textAlign: 'right', borderBottom: `1px solid ${BORDER}`, fontVariantNumeric: 'tabular-nums' }}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!q && (
          <p style={{ margin: 0, padding: '12px 16px', fontSize: 11, color: TEXT3, fontStyle: 'italic' }}>
            Showing available stats — full quantities populate once the analysis completes.
          </p>
        )}
      </div>
    </div>
  );
}
