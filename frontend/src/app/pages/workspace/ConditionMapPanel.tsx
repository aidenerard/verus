import { ImageOff } from 'lucide-react';
import type { AnalysisResult } from '../inspect/types';
import {
  BORDER, PANEL, RAISED, TEXT2, TEXT3,
  COND_SOUND, COND_MONITOR, COND_ELEVATED, COND_HIGH,
} from './tokens';

interface Props {
  result: AnalysisResult;
  src:    string | undefined;
}

interface ClassSpec {
  color:  string;
  label:  string;
  range:  string;
  pctKey: keyof NonNullable<AnalysisResult['condition_class_pcts']>;
}

const CLASSES: ClassSpec[] = [
  { color: COND_SOUND,    label: 'Sound',               range: '<35%',   pctKey: 'sound' },
  { color: COND_MONITOR,  label: 'Monitor',              range: '35–55%', pctKey: 'monitor' },
  { color: COND_ELEVATED, label: 'Anomalous Response',   range: '55–75%', pctKey: 'anomalous_response' },
  { color: COND_HIGH,     label: 'Significant Anomaly',  range: '>75%',   pctKey: 'significant_anomaly' },
];

export default function ConditionMapPanel({ result, src }: Props) {
  const pcts = result.condition_class_pcts;

  return (
    <div style={{
      background: PANEL, border: `1px solid ${BORDER}`,
      display: 'flex', flexDirection: 'column', borderRadius: 8, overflow: 'hidden',
    }}>
      <div style={{
        padding: '10px 14px', borderBottom: `1px solid ${BORDER}`,
        fontSize: 10, fontWeight: 800, letterSpacing: '0.12em',
        textTransform: 'uppercase', color: TEXT2,
      }}>
        Condition Map
      </div>

      <div style={{ position: 'relative', minHeight: 260, background: RAISED, overflow: 'hidden' }}>
        {src ? (
          <img src={src} alt="Condition Map"
            style={{ display: 'block', width: '100%', height: '100%', objectFit: 'fill' }} />
        ) : (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 8, color: TEXT3,
          }}>
            <ImageOff size={22} />
            <span style={{ fontSize: 11 }}>No data</span>
          </div>
        )}
      </div>

      <div style={{
        padding: '10px 14px', borderTop: `1px solid ${BORDER}`,
        display: 'flex', flexWrap: 'wrap', gap: '8px 24px', alignItems: 'flex-start',
      }}>
        {CLASSES.map(({ color, label, range, pctKey }) => (
          <div key={pctKey} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: 2, background: color, flexShrink: 0 }} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: TEXT2 }}>
                {label}
              </span>
              <span style={{ fontSize: 9, color: TEXT3, letterSpacing: '0.04em' }}>
                P(det.) {range}{pcts != null ? `  ·  ${pcts[pctKey]}% of deck` : ''}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
