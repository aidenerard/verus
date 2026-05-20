import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Check } from 'lucide-react';
import { ACCENT, BORDER, BORDER2, RAISED, TEXT, TEXT2, TEXT3 } from './tokens';

interface SwipeToConfirmProps {
  onConfirm:  () => void;
  disabled?:  boolean;
  label?:     string;
}

const HEIGHT       = 56;
const PAD          = 4;
const THUMB        = HEIGHT - PAD * 2;
const CONFIRM_PCT  = 0.85;
const SPRING_OUT   = 'transform 0.32s cubic-bezier(0.34, 1.56, 0.64, 1), width 0.32s cubic-bezier(0.34, 1.56, 0.64, 1)';
const SNAP_FORWARD = 'transform 0.22s cubic-bezier(0.22, 1, 0.36, 1), width 0.22s cubic-bezier(0.22, 1, 0.36, 1)';

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export default function SwipeToConfirm({
  onConfirm,
  disabled = false,
  label = 'Slide to begin analysis',
}: SwipeToConfirmProps) {
  const trackRef    = useRef<HTMLDivElement>(null);
  const dragXRef    = useRef(0);
  const startX      = useRef(0);
  const startDragX  = useRef(0);
  const maxXRef     = useRef(0);

  const [dragX,     setDragX]     = useState(0);
  const [maxX,      setMaxX]      = useState(0);
  const [dragging,  setDragging]  = useState(false);
  const [confirmed, setConfirmed] = useState(false);

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const update = () => {
      const next = Math.max(0, el.clientWidth - THUMB - PAD * 2);
      maxXRef.current = next;
      setMaxX(next);
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (disabled && !dragging) {
      dragXRef.current = 0;
      setDragX(0);
      setConfirmed(false);
    }
  }, [disabled, dragging]);

  const blocked = disabled || confirmed;

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (blocked) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    startX.current = e.clientX;
    startDragX.current = dragXRef.current;
    setDragging(true);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging || blocked) return;
    const next = clamp(startDragX.current + (e.clientX - startX.current), 0, maxXRef.current);
    dragXRef.current = next;
    setDragX(next);
  };

  const settle = () => {
    if (!dragging) return;
    setDragging(false);
    const x = dragXRef.current;
    const m = maxXRef.current;
    if (m > 0 && x / m >= CONFIRM_PCT) {
      dragXRef.current = m;
      setDragX(m);
      setConfirmed(true);
      onConfirm();
    } else {
      dragXRef.current = 0;
      setDragX(0);
    }
  };

  const onPointerEnd = (e: React.PointerEvent<HTMLDivElement>) => {
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    settle();
  };

  const reachedConfirm = confirmed || (maxX > 0 && dragX / maxX >= CONFIRM_PCT);
  const fillWidth = dragX + THUMB + PAD * 2;
  const transition = dragging ? 'none' : (confirmed ? SNAP_FORWARD : SPRING_OUT);
  const trackBg = blocked && !confirmed ? RAISED : '#FFFFFF';
  const labelColor = reachedConfirm ? '#fff' : (blocked ? TEXT3 : TEXT2);
  const trackBorder = blocked && !confirmed ? BORDER : BORDER2;

  return (
    <div
      ref={trackRef}
      style={{
        position: 'relative', width: '100%', height: HEIGHT,
        background: trackBg, border: `1px solid ${trackBorder}`,
        overflow: 'hidden', userSelect: 'none', touchAction: 'none',
        opacity: disabled && !confirmed ? 0.55 : 1,
        transition: 'opacity 0.15s, border-color 0.15s, background 0.15s',
      }}
    >
      <div
        aria-hidden
        style={{
          position: 'absolute', top: 0, left: 0, height: '100%',
          width: fillWidth, background: ACCENT,
          transition,
        }}
      />

      <div
        aria-hidden
        style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
          color: labelColor, paddingLeft: THUMB,
          transition: 'color 0.18s', pointerEvents: 'none',
          fontFamily: 'Inter, sans-serif',
        }}
      >
        {confirmed ? 'Analysis Started' : label}
      </div>

      <div
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
        role="slider"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={maxX > 0 ? Math.round((dragX / maxX) * 100) : 0}
        aria-disabled={blocked}
        style={{
          position: 'absolute', top: PAD, left: PAD,
          width: THUMB, height: THUMB, borderRadius: '50%',
          background: confirmed ? '#FFFFFF' : ACCENT,
          color: confirmed ? ACCENT : '#FFFFFF',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: blocked ? 'not-allowed' : (dragging ? 'grabbing' : 'grab'),
          transform: `translateX(${dragX}px)`,
          transition,
          boxShadow: '0 4px 12px rgba(10,10,10,0.18)',
        }}
      >
        {confirmed ? <Check size={20} strokeWidth={3} /> : <ArrowRight size={20} strokeWidth={2.5} />}
      </div>

      {!confirmed && !dragging && maxX > 0 && !disabled && (
        <Hint />
      )}
    </div>
  );
}

function Hint() {
  return (
    <div
      aria-hidden
      style={{
        position: 'absolute', top: 0, right: 16, bottom: 0,
        display: 'flex', alignItems: 'center', pointerEvents: 'none',
        color: TEXT3, opacity: 0.55,
      }}
    >
      <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: TEXT }}>
        Swipe
      </span>
    </div>
  );
}
