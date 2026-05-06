import { useRef, useEffect } from 'react';
import { C } from './tokens';
import { useReveal } from './useReveal';

const WHY_ITEMS = [
  { num: '01', title: 'Fully Automated', body: 'No manual data processing. Upload any inspection file and get results instantly. Our AI handles format detection, signal processing, and classification.' },
  { num: '02', title: 'Any Equipment', body: 'Works with data from all major inspection equipment manufacturers and file formats. If your device can export it, Verus can analyze it.' },
  { num: '03', title: 'Same-Day Reports', body: 'Standards-compliant condition reports generated in minutes, not weeks. Export C-scan maps and share results the same day as your inspection.' },
  { num: '04', title: 'Built to Scale', body: 'Designed for inspection teams managing hundreds of structures across large networks. Batch upload, multi-file analysis, and GPS-referenced output.' },
];

export default function WhyVerus() {
  const headRef = useReveal(0);
  const r0 = useRef<HTMLDivElement>(null);
  const r1 = useRef<HTMLDivElement>(null);
  const r2 = useRef<HTMLDivElement>(null);
  const r3 = useRef<HTMLDivElement>(null);
  const rows = [r0, r1, r2, r3];

  useEffect(() => {
    rows.forEach((ref, i) => {
      const el = ref.current;
      if (!el) return;
      const obs = new IntersectionObserver(
        ([entry]) => { if (entry.isIntersecting) { setTimeout(() => el.classList.add('revealed'), i * 80); obs.disconnect(); } },
        { threshold: 0.1 }
      );
      obs.observe(el);
      return () => obs.disconnect();
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section id="why-verus" style={{ background: C.offWhite, padding: '96px 48px' }}>
      <div style={{ maxWidth: 1080, margin: '0 auto' }}>
        <div ref={headRef} className="vr-reveal" style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: 40, marginBottom: 16, paddingBottom: 28, borderBottom: `2px solid ${C.black}` }}>
          <h2 style={{ fontSize: 'clamp(26px, 3.5vw, 42px)', fontWeight: 800, color: C.black, margin: 0, letterSpacing: '-0.025em', lineHeight: 1.1 }}>
            Automated Analysis.<br />Zero Manual Steps.
          </h2>
          <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.orange, margin: 0, flexShrink: 0 }}>Why Verus</p>
        </div>
        <div>
          {WHY_ITEMS.map((item, i) => (
            <div key={item.num} ref={rows[i]} className="vr-why-row">
              <div style={{ display: 'grid', gridTemplateColumns: '64px 1fr 2fr', gap: 32, alignItems: 'start' }}>
                <span className="vr-why-num" style={{ fontSize: 11, fontWeight: 800, color: C.orange, letterSpacing: '0.04em', fontVariantNumeric: 'tabular-nums', paddingTop: 2 }}>{item.num}</span>
                <h3 style={{ fontSize: 17, fontWeight: 700, color: C.black, margin: 0, letterSpacing: '-0.01em' }}>{item.title}</h3>
                <p style={{ fontSize: 14, color: C.textGray, lineHeight: 1.75, margin: 0 }}>{item.body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
