import type { ReactNode } from 'react';
import { BORDER, RAISED, TEXT, TEXT2, TEXT3 } from '../tokens';

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ marginBottom: 22 }}>
      <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase', color: TEXT3, marginBottom: 10 }}>
        {title}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>{children}</div>
    </div>
  );
}

export function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(80px, 110px) 1fr', alignItems: 'center', gap: 10 }}>
      <span style={{ fontSize: 11, color: TEXT2 }}>{label}</span>
      <div style={{ minWidth: 0 }}>{children}</div>
    </div>
  );
}

interface NumberFieldProps {
  value:     number;
  onChange:  (v: number) => void;
  step?:     number;
  min?:      number;
  max?:      number;
  unit?:     string;
  readOnly?: boolean;
}

export function NumberField({ value, onChange, step = 0.01, min, max, unit, readOnly }: NumberFieldProps) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
        min={min}
        max={max}
        readOnly={readOnly}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{
          flex: 1, minWidth: 0, padding: '6px 8px',
          background: RAISED, border: `1px solid ${BORDER}`,
          color: TEXT, fontSize: 12, fontFamily: 'inherit', outline: 'none',
        }}
      />
      {unit && <span style={{ fontSize: 11, color: TEXT3 }}>{unit}</span>}
    </div>
  );
}

interface SliderProps {
  value:    number;
  onChange: (v: number) => void;
  min:      number;
  max:      number;
  step?:    number;
}

export function Slider({ value, onChange, min, max, step = 0.01 }: SliderProps) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 60px', gap: 8, alignItems: 'center' }}>
      <input
        type="range"
        value={value} min={min} max={max} step={step}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: '#E8601C' }}
      />
      <span style={{ fontSize: 11, color: TEXT, fontVariantNumeric: 'tabular-nums', textAlign: 'right' }}>
        {value.toFixed(step >= 1 ? 0 : 2)}
      </span>
    </div>
  );
}

export function Select<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value as T)}
      style={{ width: '100%', padding: '6px 8px', background: RAISED, border: `1px solid ${BORDER}`, color: TEXT, fontSize: 12, fontFamily: 'inherit', outline: 'none' }}
    >
      {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

export function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      role="switch"
      aria-checked={checked}
      style={{
        width: 32, height: 18, border: 'none', cursor: 'pointer', padding: 0,
        background: checked ? '#E8601C' : BORDER, borderRadius: 999, position: 'relative',
      }}
    >
      <span style={{
        position: 'absolute', top: 2, left: checked ? 16 : 2, width: 14, height: 14,
        background: '#fff', borderRadius: '50%', transition: 'left 0.15s',
      }} />
    </button>
  );
}

interface ButtonProps {
  onClick:  () => void;
  children: ReactNode;
  variant?: 'primary' | 'ghost' | 'danger';
  disabled?:boolean;
}

export function Button({ onClick, children, variant = 'ghost', disabled }: ButtonProps) {
  const bg     = variant === 'primary' ? '#E8601C' : variant === 'danger' ? '#b91c1c' : 'transparent';
  const color  = variant === 'ghost' ? TEXT : '#fff';
  const border = variant === 'ghost' ? `1px solid ${BORDER}` : 'none';
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        background: disabled ? '#D4CFC9' : bg, color: disabled ? '#7A7470' : color,
        border, padding: '8px 14px',
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase',
        fontFamily: 'inherit',
        opacity: disabled ? 0.8 : 1,
      }}
    >
      {children}
    </button>
  );
}
