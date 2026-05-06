import { Link } from 'react-router';
import { C } from './tokens';

export default function Hero() {
  return (
    <section style={{ background: C.black, minHeight: '100vh', display: 'flex', alignItems: 'center', position: 'relative', overflow: 'hidden', padding: '120px 48px 80px' }}>
      {/* Dot grid */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 1, backgroundImage: `radial-gradient(circle, rgba(255,255,255,0.04) 1px, transparent 1px)`, backgroundSize: '32px 32px', animation: 'drift 26s ease-in-out infinite' }} />
      {/* Edge vignette */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 2, background: 'radial-gradient(ellipse 100% 80% at 50% 50%, transparent 20%, rgba(10,10,10,0.75) 100%)' }} />

      <div style={{ maxWidth: 1080, margin: '0 auto', width: '100%', position: 'relative', zIndex: 3 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 72, alignItems: 'center' }}>

          {/* Left — copy */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 32 }}>
              <div style={{ width: 28, height: 1.5, background: C.orange }} />
              <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.16em', textTransform: 'uppercase', color: C.orange }}>Now in Beta</span>
            </div>
            <h1 style={{ fontSize: 'clamp(36px, 5.5vw, 68px)', fontWeight: 800, color: '#FFFFFF', lineHeight: 1.07, margin: '0 0 22px', letterSpacing: '-0.03em' }}>
              The Future of<br />
              <span style={{ color: C.orange }}>Infrastructure</span><br />
              Inspection
            </h1>
            <p style={{ fontSize: 15, color: C.textMuted, lineHeight: 1.7, margin: '0 0 40px', maxWidth: 420 }}>
              AI-powered analysis for every inspection method. Upload your data.
              Get same-day reports. No manual processing required.
            </p>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 44 }}>
              <Link to="/signup" className="vr-btn-primary">Get Started</Link>
              <a href="#why-verus" className="vr-btn-ghost">Learn More</a>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {['600,000+ US Bridges', 'Same-Day Reporting', 'Multi-Standard Compliant'].map(s => (
                <div key={s} className="vr-stat-pill">{s}</div>
              ))}
            </div>
          </div>

          {/* Right — abstract interface panel */}
          <div style={{ position: 'relative' }}>
            <div style={{ border: `1px solid rgba(255,255,255,0.1)`, background: 'rgba(255,255,255,0.025)', backdropFilter: 'blur(6px)', padding: 2 }}>
              {/* Title bar */}
              <div style={{ background: 'rgba(255,255,255,0.04)', borderBottom: `1px solid rgba(255,255,255,0.07)`, padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ display: 'flex', gap: 6 }}>
                  {[C.orange, 'rgba(255,255,255,0.2)', 'rgba(255,255,255,0.2)'].map((bg, i) => (
                    <div key={i} style={{ width: 7, height: 7, borderRadius: '50%', background: bg }} />
                  ))}
                </div>
                <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.08em', marginLeft: 8 }}>VERUS ANALYSIS — GPR C-SCAN</span>
              </div>

              {/* C-scan placeholder grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 2, padding: 16, aspectRatio: '4/3', background: 'rgba(10,10,10,0.6)' }}>
                {Array.from({ length: 96 }).map((_, i) => {
                  const row = Math.floor(i / 12);
                  const col = i % 12;
                  const isHot = (row >= 2 && row <= 4 && col >= 3 && col <= 6) || (row >= 5 && row <= 6 && col >= 7 && col <= 10);
                  const isMid = (row >= 1 && row <= 5 && col >= 2 && col <= 7 && !isHot);
                  return (
                    <div key={i} style={{ aspectRatio: '1', background: isHot ? `rgba(232,96,28,${0.55 + Math.random() * 0.35})` : isMid ? `rgba(232,96,28,${0.12 + Math.random() * 0.18})` : `rgba(46,204,113,${0.18 + Math.random() * 0.22})`, borderRadius: 1 }} />
                  );
                })}
              </div>

              {/* Bottom status bar */}
              <div style={{ borderTop: `1px solid rgba(255,255,255,0.06)`, background: 'rgba(255,255,255,0.02)', padding: '10px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: 20 }}>
                  {[['Delamination', '18.4%'], ['Sound', '81.6%']].map(([k, v]) => (
                    <div key={k}>
                      <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>{k}&nbsp;</span>
                      <span style={{ fontSize: 11, fontWeight: 700, color: k === 'Delamination' ? C.orange : 'rgba(46,204,113,0.8)' }}>{v}</span>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 9, color: C.orange, background: 'rgba(232,96,28,0.1)', border: '1px solid rgba(232,96,28,0.25)', padding: '3px 8px', letterSpacing: '0.08em' }}>ASTM D6087</div>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Scroll indicator */}
      <div style={{ position: 'absolute', bottom: 28, left: '50%', transform: 'translateX(-50%)', zIndex: 3 }}>
        <div style={{ width: 1, height: 40, background: 'rgba(255,255,255,0.15)', animation: 'pulseBar 2.5s ease-in-out infinite', transformOrigin: 'top' }} />
      </div>
    </section>
  );
}
