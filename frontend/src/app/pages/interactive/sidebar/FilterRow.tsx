import { GripVertical } from 'lucide-react';
import type { FilterStep } from '../state/types';
import { Toggle } from './fields';
import { BORDER, PANEL, TEXT, TEXT2, TEXT3 } from '../tokens';

const FILTER_LABELS: Record<FilterStep['type'], string> = {
  bandpass:           'Bandpass',
  background_removal: 'Background Removal',
  gain:               'Gain',
  agc:                'AGC',
  hilbert:            'Hilbert Envelope',
};

interface Props {
  step:        FilterStep;
  onChange:    (next: FilterStep) => void;
  onDragStart: (id: string) => void;
  onDragOver:  (id: string) => void;
  onDrop:      () => void;
}

export default function FilterRow({ step, onChange, onDragStart, onDragOver, onDrop }: Props) {
  return (
    <div
      draggable
      onDragStart={() => onDragStart(step.id)}
      onDragOver={e => { e.preventDefault(); onDragOver(step.id); }}
      onDrop={e => { e.preventDefault(); onDrop(); }}
      style={{
        display: 'grid', gridTemplateColumns: 'auto auto 1fr auto', alignItems: 'center', gap: 8,
        padding: '8px 10px', background: PANEL, border: `1px solid ${BORDER}`,
        cursor: 'grab', userSelect: 'none',
      }}
    >
      <span style={{ color: TEXT3, display: 'flex', cursor: 'grab' }}>
        <GripVertical size={14} />
      </span>
      <Toggle checked={step.enabled} onChange={v => onChange({ ...step, enabled: v })} />
      <span style={{ fontSize: 12, color: step.enabled ? TEXT : TEXT2 }}>
        {FILTER_LABELS[step.type]}
      </span>
      <ParamsSummary step={step} />
    </div>
  );
}

function ParamsSummary({ step }: { step: FilterStep }) {
  const text = describeParams(step);
  return (
    <span style={{ fontSize: 10, color: TEXT3, fontVariantNumeric: 'tabular-nums', textAlign: 'right' }}>
      {text}
    </span>
  );
}

function describeParams(step: FilterStep): string {
  if (step.type === 'bandpass') {
    const lo = step.params.low_mhz  as number | undefined;
    const hi = step.params.high_mhz as number | undefined;
    return lo && hi ? `${lo}-${hi} MHz` : '';
  }
  if (step.type === 'background_removal') {
    return `w=${step.params.window ?? '—'}`;
  }
  if (step.type === 'gain') {
    return `${step.params.gain_db ?? 0} dB`;
  }
  if (step.type === 'agc') {
    return `${step.params.window_ns ?? '—'} ns`;
  }
  return '';
}
