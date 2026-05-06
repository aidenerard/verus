import { Link } from 'react-router';
import { C } from './tokens';
import { useReveal } from './useReveal';

const METHODS = [
  { label: 'Ground-Penetrating Radar',               short: 'GPR',  active: true  },
  { label: 'Multichannel Analysis of Surface Waves', short: 'MASW', active: false },
  { label: 'Infrared Thermography',                  short: 'IR',   active: false },
];

export default function OurPlatform() {
  const ref     = useReveal(0);
  const listRef = useReveal(120);

  return (
    <section style={{ background: C.offWhite, padding: '96px 48px' }}>
      <div style={{ maxWidth: 1080, margin: '0 auto' }}>
        <div ref={ref} className="vr-reveal" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80, alignItems: 'start' }}>
          <div>
            <p className="vr-platform-label">Our Platform</p>
            <h2 style={{ fontSize: 'clamp(24px, 3vw, 38px)', fontWeight: 800, color: C.black, margin: '0 0 20px', letterSpacing: '-0.025em' }}>
              One Platform.<br />Every Inspection Method.
            </h2>
            <p style={{ fontSize: 14, color: C.textGray, lineHeight: 1.75, margin: '0 0 36px', maxWidth: 400 }}>
              Verus is built to support the full range of non-destructive evaluation methods.
              Start with what you need today and expand as your program grows.
            </p>
            <Link to="/signup" className="vr-btn-primary">Get Early Access</Link>
          </div>
          <div ref={listRef} className="vr-reveal">
            <div style={{ borderTop: `2px solid ${C.black}` }}>
              {METHODS.map(m => (
                <div key={m.label} className={`vr-method-row${m.active ? ' vr-method-row-active' : ''}`}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <span style={{ fontSize: 10, fontWeight: 800, color: m.active ? C.orange : C.textGray, letterSpacing: '0.08em', width: 36, flexShrink: 0, transition: 'color 0.2s' }}>{m.short}</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: m.active ? C.black : C.textGray, transition: 'color 0.2s' }}>{m.label}</span>
                  </div>
                  <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.09em', textTransform: 'uppercase', padding: '4px 10px', background: m.active ? C.orange : 'transparent', color: m.active ? '#fff' : C.textGray, border: `1px solid ${m.active ? C.orange : C.border}`, flexShrink: 0, transition: 'background 0.2s, border-color 0.2s, color 0.2s' }}>
                    {m.active ? 'Live' : 'Soon'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
