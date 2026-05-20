import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useProcessingOptions } from './ProcessingOptionsContext';
import type { GriddingAlgorithm, ProcessingFilters } from './types';
import { ACCENT, BORDER, RAISED, TEXT, TEXT2, TEXT3 } from './tokens';

const GRIDDING_CHOICES: { value: GriddingAlgorithm; label: string }[] = [
  { value: 'linear',             label: 'Linear' },
  { value: 'kriging',            label: 'Kriging' },
  { value: 'natural_neighbor',   label: 'Natural Neighbor' },
  { value: 'minimum_curvature',  label: 'Min. Curvature' },
];

const FILTER_LABELS: { key: keyof ProcessingFilters; label: string }[] = [
  { key: 'backgroundRemoval', label: 'Background Removal' },
  { key: 'gain',              label: 'Gain' },
  { key: 'bandpass',          label: 'Bandpass' },
];

export default function ProcessingOptionsPanel() {
  const { options, setGridding, setSearchRadius, setEdgeClipping, toggleFilter } = useProcessingOptions();
  const [open, setOpen] = useState(true);

  return (
    <div>
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          width: '100%', background: 'none', border: 'none', padding: '4px 0', cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: 4,
          fontSize: 11, fontWeight: 800, letterSpacing: '0.10em', textTransform: 'uppercase',
          color: TEXT2, fontFamily: 'inherit',
        }}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        Processing Options
      </button>

      {open && (
        <div style={{ padding: '8px 0 12px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          <Field label="Gridding Algorithm">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {GRIDDING_CHOICES.map(c => {
                const active = options.gridding === c.value;
                return (
                  <label key={c.value} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: active ? TEXT : TEXT2, cursor: 'pointer' }}>
                    <input
                      type="radio"
                      name="gridding"
                      value={c.value}
                      checked={active}
                      onChange={() => setGridding(c.value)}
                      style={{ accentColor: ACCENT }}
                    />
                    {c.label}
                  </label>
                );
              })}
            </div>
          </Field>

          <Field label="Search Radius (m)">
            <input
              type="number"
              min={0}
              step={0.1}
              value={options.searchRadius}
              onChange={e => setSearchRadius(parseFloat(e.target.value) || 0)}
              style={{ width: '100%', padding: '6px 8px', background: RAISED, border: `1px solid ${BORDER}`, fontSize: 12, color: TEXT, fontFamily: 'inherit', outline: 'none' }}
            />
          </Field>

          <Field label="Edge Clipping">
            <Toggle checked={options.edgeClipping} onChange={setEdgeClipping} />
          </Field>

          <div>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT3, marginBottom: 8 }}>
              Filters
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {FILTER_LABELS.map(f => (
                <div key={f.key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12, color: TEXT2 }}>
                  <span>{f.label}</span>
                  <Toggle checked={options.filters[f.key]} onChange={() => toggleFilter(f.key)} />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: TEXT3, marginBottom: 6 }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      role="switch"
      aria-checked={checked}
      style={{
        width: 32, height: 18, border: 'none', cursor: 'pointer', padding: 0,
        background: checked ? ACCENT : BORDER, borderRadius: 999, position: 'relative',
        transition: 'background 0.15s',
      }}
    >
      <span style={{
        position: 'absolute', top: 2, left: checked ? 16 : 2, width: 14, height: 14,
        background: '#fff', borderRadius: '50%', transition: 'left 0.15s',
      }} />
    </button>
  );
}
