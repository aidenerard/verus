import { BORDER, TEXT2 } from './constants';

const GRADIENTS = {
  'yellow-red': 'linear-gradient(to right, #ffffd4, #fed98e, #fe9929, #d95f0e, #993404)',
  'spectral':   'linear-gradient(to right, #2b83ba, #abdda4, #ffffbf, #fdae61, #d7191c)',
};

interface Props {
  gradient: keyof typeof GRADIENTS;
  min?: number;
  max?: number;
  label: string;
  formatValue?: (v: number) => string;
}

export default function MapColorBar({ gradient, min, max, label, formatValue }: Props) {
  const hasTicks = min != null && max != null;
  const fmt = formatValue ?? ((v: number) => Number.isInteger(v) ? String(v) : v.toFixed(1));
  const mid = hasTicks ? (min! + max!) / 2 : 0;

  return (
    <div style={{ flexShrink: 0, padding: '8px 20px 10px', borderTop: `1px solid ${BORDER}` }}>
      <div style={{ height: 13, background: GRADIENTS[gradient], borderRadius: 2 }} />
      {hasTicks && (
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
          <span style={{ fontSize: 9, color: TEXT2 }}>{fmt(min!)}</span>
          <span style={{ fontSize: 9, color: TEXT2 }}>{fmt(mid)}</span>
          <span style={{ fontSize: 9, color: TEXT2 }}>{fmt(max!)}</span>
        </div>
      )}
      <div style={{ textAlign: 'center', fontSize: 9, fontWeight: 600, letterSpacing: '0.06em', color: TEXT2, textTransform: 'uppercase', marginTop: hasTicks ? 2 : 4 }}>
        {label}
      </div>
    </div>
  );
}
