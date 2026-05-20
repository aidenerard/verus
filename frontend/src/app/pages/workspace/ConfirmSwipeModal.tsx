import { useEffect } from 'react';
import { X } from 'lucide-react';
import SwipeToConfirm from './SwipeToConfirm';
import { BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3 } from './tokens';

export interface SummaryRow {
  label: string;
  value: string;
}

interface Props {
  open:      boolean;
  onCancel:  () => void;
  onConfirm: () => void;
  rows?:     SummaryRow[];
  title?:    string;
  label?:    string;
}

export default function ConfirmSwipeModal({
  open, onCancel, onConfirm,
  rows  = [],
  title = 'Ready to Analyze',
  label = 'Slide to begin',
}: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onCancel(); };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-swipe-title"
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(10,10,10,0.65)', backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 24, fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 460, background: PANEL,
          border: `1px solid ${BORDER}`,
          boxShadow: '0 24px 64px rgba(0,0,0,0.28)',
        }}
      >
        <div style={{ padding: '18px 24px 14px', borderBottom: `1px solid ${BORDER}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span id="confirm-swipe-title" style={{ fontSize: 15, fontWeight: 700, color: TEXT }}>
            {title}
          </span>
          <button
            onClick={onCancel}
            aria-label="Cancel"
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', padding: 2 }}
          >
            <X size={16} />
          </button>
        </div>

        <div style={{ padding: '22px 24px 24px' }}>
          {rows.length > 0 && (
            <dl
              style={{
                margin: '0 0 22px', padding: '14px 16px',
                background: RAISED, border: `1px solid ${BORDER}`,
                display: 'grid', gridTemplateColumns: 'auto 1fr', columnGap: 24, rowGap: 8,
                fontSize: 12,
              }}
            >
              {rows.map(r => (
                <Row key={r.label} label={r.label} value={r.value} />
              ))}
            </dl>
          )}

          <SwipeToConfirm onConfirm={onConfirm} label={label} />

          <div style={{ marginTop: 16, textAlign: 'center' }}>
            <button
              onClick={onCancel}
              style={{
                background: 'none', border: 'none', padding: '6px 10px', cursor: 'pointer',
                fontSize: 12, fontWeight: 600, color: TEXT2, fontFamily: 'inherit',
                textDecoration: 'underline', textUnderlineOffset: 3,
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: SummaryRow) {
  return (
    <>
      <dt style={{ color: TEXT3, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', fontSize: 10 }}>
        {label}
      </dt>
      <dd style={{ margin: 0, color: TEXT, fontWeight: 600, textAlign: 'right' }}>
        {value}
      </dd>
    </>
  );
}
