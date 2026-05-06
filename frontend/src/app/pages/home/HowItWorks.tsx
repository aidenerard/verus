import { useRef, useEffect } from 'react';
import { Link } from 'react-router';
import { C } from './tokens';
import { useReveal } from './useReveal';

const STEPS = [
  { num: '01', title: 'Upload', body: 'Drop your inspection data files in any supported format. Verus detects the format automatically and prepares it for analysis.' },
  { num: '02', title: 'Analyze', body: 'Our AI models process every signal and classify anomalies automatically, with no manual configuration required.' },
  { num: '03', title: 'Report', body: 'Download standards-compliant condition reports, same day. GPS-tagged data renders on an interactive satellite map.' },
];

export default function HowItWorks() {
  const headRef = useReveal(0);
  const ctaRef  = useReveal(200);
  const s0 = useRef<HTMLDivElement>(null);
  const s1 = useRef<HTMLDivElement>(null);
  const s2 = useRef<HTMLDivElement>(null);
  const stepRefs = [s0, s1, s2];

  useEffect(() => {
    stepRefs.forEach((ref, i) => {
      const el = ref.current;
      if (!el) return;
      const obs = new IntersectionObserver(
        ([entry]) => { if (entry.isIntersecting) { setTimeout(() => el.classList.add('revealed'), i * 110); obs.disconnect(); } },
        { threshold: 0.1 }
      );
      obs.observe(el);
      return () => obs.disconnect();
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section id="how-it-works" style={{ background: C.offWhite, padding: '96px 48px' }}>
      <div style={{ maxWidth: 1080, margin: '0 auto' }}>
        <div ref={headRef} className="vr-reveal" style={{ marginBottom: 72 }}>
          <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', color: C.orange, margin: '0 0 14px' }}>How It Works</p>
          <h2 style={{ fontSize: 'clamp(26px, 3.5vw, 42px)', fontWeight: 800, color: C.black, margin: 0, letterSpacing: '-0.025em' }}>
            From Raw Data to Report<br />in Minutes
          </h2>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 40, marginBottom: 64 }}>
          {STEPS.map((step, i) => (
            <div key={step.num} ref={stepRefs[i]} className="vr-step-card">
              <div style={{ fontSize: 10, fontWeight: 800, color: C.orange, letterSpacing: '0.1em', marginBottom: 16, fontVariantNumeric: 'tabular-nums' }}>{step.num}</div>
              <h3 style={{ fontSize: 20, fontWeight: 700, color: C.black, margin: '0 0 12px', letterSpacing: '-0.015em' }}>{step.title}</h3>
              <p style={{ fontSize: 14, color: C.textGray, lineHeight: 1.75, margin: 0 }}>{step.body}</p>
            </div>
          ))}
        </div>
        <div ref={ctaRef} className="vr-reveal">
          <Link to="/signup" className="vr-btn-primary">Start Analyzing Today</Link>
        </div>
      </div>
    </section>
  );
}
