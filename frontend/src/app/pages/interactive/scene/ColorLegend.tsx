import { spectralStops } from './colormap';
import { BORDER, PANEL, TEXT, TEXT2, TEXT3 } from '../tokens';

interface Props {
  range: [number, number];
  units?: string;
}

export default function ColorLegend({ range, units = 'in' }: Props) {
  const stops = spectralStops(11, range);
  const gradient = stops
    .map((s, i) => `${s.color} ${((i / (stops.length - 1)) * 100).toFixed(1)}%`)
    .join(', ');

  return (
    <div style={{
      position: 'absolute', bottom: 14, left: 14, zIndex: 5,
      background: PANEL, border: `1px solid ${BORDER}`,
      padding: '10px 12px', minWidth: 180,
    }}>
      <div style={{ fontSize: 9, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT3, marginBottom: 6 }}>
        Rebar Depth ({units})
      </div>
      <div style={{
        height: 8, width: '100%',
        background: `linear-gradient(to right, ${gradient})`,
      }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 10, color: TEXT2 }}>
        <span style={{ color: TEXT }}>{range[0].toFixed(1)}</span>
        <span>{((range[0] + range[1]) / 2).toFixed(1)}</span>
        <span style={{ color: TEXT }}>{range[1].toFixed(1)}</span>
      </div>
    </div>
  );
}
