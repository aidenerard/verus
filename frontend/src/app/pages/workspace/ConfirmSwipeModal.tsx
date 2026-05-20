import { useEffect } from 'react';
import { X } from 'lucide-react';
import SwipeToConfirm from './SwipeToConfirm';
import { BORDER, PANEL, TEXT, TEXT2 } from './tokens';

interface Props {
  open:      boolean;
  onCancel:  () => void;
  onConfirm: () => void;
  title?:    string;
  context?:  string;
  label?:    string;
}

export default function ConfirmSwipeModal({
  open, onCancel, onConfirm,
  title   = 'Confirm Analysis',
  context = 'This will upload your files and begin processing.',
  label   = 'Slide to begin analysis',
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

  const handleConfirm = () => {
    onConfirm();
  };

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

        <div style={{ padding: '20px 24px 24px' }}>
          <p style={{ margin: '0 0 22px', fontSize: 13, color: TEXT2, lineHeight: 1.55 }}>
            {context}
          </p>

          <SwipeToConfirm onConfirm={handleConfirm} label={label} />

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
