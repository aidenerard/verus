import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router';
import VerusLogo from '../components/VerusLogo';

// ── Design tokens ─────────────────────────────────────────────────────────────
const C = {
  black:    '#0A0A0A',
  orange:   '#E8601C',
  orangeDk: '#D4521A',
  offWhite: '#F5F3EF',
  card:     '#FDFCFA',
  textDark: '#0A0A0A',
  textLight:'#F0EDE8',
  textMuted:'rgba(240,237,232,0.55)',
  textGray: '#7A7470',
  border:   '#E2DED9',
  borderDk: 'rgba(255,255,255,0.08)',
} as const;

// ── Global CSS ────────────────────────────────────────────────────────────────
const PAGE_CSS = `
  @keyframes drift {
    0%   { transform: translate(0,0); }
    33%  { transform: translate(12px,-8px); }
    66%  { transform: translate(-6px,10px); }
    100% { transform: translate(0,0); }
  }
  @keyframes marquee {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
  }
  @keyframes pulseBar {
    0%,100% { transform: scaleY(0.85); opacity: 0.12; }
    50%     { transform: scaleY(1.2);  opacity: 0.55; }
  }
  @keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
  }

  /* Nav */
  .vr-nav-link {
    position: relative; text-decoration: none;
    color: rgba(240,237,232,0.68); font-size: 13px;
    font-weight: 500; letter-spacing: 0.01em;
    transition: color 0.2s;
  }
  .vr-nav-link::after {
    content:''; position:absolute; bottom:-3px; left:0;
    width:0; height:1px; background:#E8601C;
    transition: width 0.25s ease;
  }
  .vr-nav-link:hover { color:#F0EDE8; }
  .vr-nav-link:hover::after { width:100%; }

  .vr-nav-login {
    font-size:12px; font-weight:600; color:rgba(240,237,232,0.68);
    text-decoration:none; letter-spacing:0.03em; transition:color 0.2s;
  }
  .vr-nav-login:hover { color:#F0EDE8; }

  .vr-nav-signup {
    display:inline-block; padding:8px 20px;
    background:#E8601C; color:#fff;
    font-size:11px; font-weight:700;
    letter-spacing:0.09em; text-transform:uppercase;
    text-decoration:none; border:none;
    transition: background 0.18s, transform 0.18s;
  }
  .vr-nav-signup:hover { background:#D4521A; transform:translateY(-1px); }

  /* Why Verus row hover */
  .vr-why-row:hover {
    background: rgba(232,96,28,0.03);
  }
  .vr-why-row:hover .vr-why-num {
    color: #D4521A;
    transform: scale(1.08);
  }
  .vr-why-num {
    display: inline-block;
    transition: color 0.2s, transform 0.2s;
  }

  /* Platform method row */
  .vr-method-row {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 18px 12px 18px 0;
    border-bottom: 1px solid #E2DED9;
    border-left: 2px solid transparent;
    padding-left: 14px;
    gap: 16px;
    transition: border-left-color 0.22s, background 0.22s, padding-left 0.22s;
    cursor: default;
  }
  .vr-method-row:hover {
    border-left-color: #E8601C;
    background: rgba(232,96,28,0.035);
    padding-left: 20px;
  }
  .vr-method-row-active {
    border-left-color: #E8601C;
  }

  /* Platform section label glow */
  .vr-platform-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #E8601C; margin: 0 0 14px;
    position: relative; display: inline-block;
  }
  .vr-platform-label::after {
    content: '';
    position: absolute;
    bottom: -6px; left: 0;
    width: 24px; height: 2px;
    background: #E8601C;
    box-shadow: 0 0 8px rgba(232,96,28,0.6);
  }

  @keyframes marqueeReverse {
    from { transform: translateX(-50%); }
    to   { transform: translateX(0); }
  }

  /* Hero CTAs */
  .vr-btn-primary {
    display:inline-block; padding:13px 32px;
    background:#E8601C; color:#FFFFFF;
    font-weight:700; font-size:12px;
    letter-spacing:0.09em; text-transform:uppercase;
    text-decoration:none; border:2px solid #E8601C;
    transition: background 0.2s, transform 0.2s;
  }
  .vr-btn-primary:hover { background:#D4521A; border-color:#D4521A; transform:translateY(-2px); }

  .vr-btn-ghost {
    display:inline-block; padding:13px 32px;
    background:transparent; color:#F0EDE8;
    font-weight:700; font-size:12px;
    letter-spacing:0.09em; text-transform:uppercase;
    text-decoration:none; border:2px solid rgba(240,237,232,0.25);
    transition: border-color 0.2s, color 0.2s, transform 0.2s;
  }
  .vr-btn-ghost:hover { border-color:rgba(240,237,232,0.6); color:#fff; transform:translateY(-2px); }

  /* Stat pills in hero */
  .vr-stat-pill {
    padding:7px 16px;
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    font-size:11px; font-weight:600;
    color:rgba(240,237,232,0.6);
    letter-spacing:0.05em;
    transition:transform 0.2s;
    cursor:default;
  }
  .vr-stat-pill:hover { transform:scale(1.04); }

  /* Why cards (now list rows) */
  .vr-why-row {
    border-bottom:1px solid #E2DED9;
    padding:32px 0;
    opacity:0; transform:translateY(20px);
    transition: opacity 0.55s ease, transform 0.55s ease;
  }
  .vr-why-row.revealed { opacity:1; transform:translateY(0); }
  .vr-why-row:last-child { border-bottom:none; }

  /* Step cards */
  .vr-step-card {
    border-left:2px solid #E2DED9;
    padding-left:28px;
    opacity:0; transform:translateX(-16px);
    transition: opacity 0.55s ease, transform 0.55s ease, border-color 0.25s;
  }
  .vr-step-card.revealed { opacity:1; transform:translateX(0); }
  .vr-step-card:hover { border-left-color:#E8601C; }

  /* Footer links */
  .vr-footer-link {
    font-size:12px; color:rgba(240,237,232,0.78);
    text-decoration:none; transition:color 0.2s;
  }
  .vr-footer-link:hover { color:#E8601C; }

  /* Reveal utility */
  .vr-reveal {
    opacity:0; transform:translateY(24px);
    transition: opacity 0.6s ease, transform 0.6s ease;
  }
  .vr-reveal.revealed { opacity:1; transform:translateY(0); }
`;

// ── useReveal hook ────────────────────────────────────────────────────────────
function useReveal(delay = 0) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => el.classList.add('revealed'), delay);
          obs.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [delay]);
  return ref as React.RefObject<HTMLDivElement>;
}

// ── Navbar ────────────────────────────────────────────────────────────────────
function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', h, { passive: true });
    return () => window.removeEventListener('scroll', h);
  }, []);

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
      height: 58,
      background: C.black,
      display: 'flex', alignItems: 'center',
      padding: '0 48px',
      borderBottom: `1px solid ${scrolled ? 'rgba(255,255,255,0.08)' : 'transparent'}`,
      boxShadow: scrolled ? '0 4px 24px rgba(0,0,0,0.5)' : 'none',
      transition: 'border-color 0.3s, box-shadow 0.3s',
    }}>
      {/* Logo — left */}
      <Link to="/" style={{ textDecoration: 'none', flexShrink: 0 }}>
        <VerusLogo size={30} wordmarkColor="#F0EDE8" />
      </Link>

      {/* Nav links — absolutely centered */}
      <div style={{
        position: 'absolute', left: '50%', top: '50%',
        transform: 'translate(-50%, -50%)',
        display: 'flex', alignItems: 'center', gap: 32,
      }}>
        <Link to="/" className="vr-nav-link">Home</Link>
        <Link to="/team" className="vr-nav-link">Team</Link>
      </div>

      {/* Auth — right */}
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 22, flexShrink: 0 }}>
        <Link to="/login" className="vr-nav-login">Log In</Link>
        <Link to="/signup" className="vr-nav-signup">Sign Up</Link>
      </div>
    </nav>
  );
}

// ── Hero — left-aligned split layout ─────────────────────────────────────────
function Hero() {
  return (
    <section style={{
      background: C.black,
      minHeight: '100vh',
      display: 'flex', alignItems: 'center',
      position: 'relative', overflow: 'hidden',
      padding: '120px 48px 80px',
    }}>
      {/* Dot grid */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 1,
        backgroundImage: `radial-gradient(circle, rgba(255,255,255,0.04) 1px, transparent 1px)`,
        backgroundSize: '32px 32px',
        animation: 'drift 26s ease-in-out infinite',
      }} />
      {/* Edge vignette */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 2,
        background: 'radial-gradient(ellipse 100% 80% at 50% 50%, transparent 20%, rgba(10,10,10,0.75) 100%)',
      }} />

      <div style={{ maxWidth: 1080, margin: '0 auto', width: '100%', position: 'relative', zIndex: 3 }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 72,
          alignItems: 'center',
        }}>
          {/* Left — copy */}
          <div>
            {/* Eyebrow */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 12,
              marginBottom: 32,
            }}>
              <div style={{ width: 28, height: 1.5, background: C.orange }} />
              <span style={{
                fontSize: 10, fontWeight: 700, letterSpacing: '0.16em',
                textTransform: 'uppercase', color: C.orange,
              }}>
                Now in Beta
              </span>
            </div>

            {/* Headline */}
            <h1 style={{
              fontSize: 'clamp(36px, 5.5vw, 68px)',
              fontWeight: 800, color: '#FFFFFF',
              lineHeight: 1.07, margin: '0 0 22px',
              letterSpacing: '-0.03em',
            }}>
              The Future of<br />
              <span style={{ color: C.orange }}>Infrastructure</span><br />
              Inspection
            </h1>

            <p style={{
              fontSize: 15, color: C.textMuted,
              lineHeight: 1.7, margin: '0 0 40px',
              maxWidth: 420,
            }}>
              AI-powered analysis for every inspection method. Upload your data.
              Get same-day reports. No manual processing required.
            </p>

            {/* CTAs */}
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 44 }}>
              <Link to="/signup" className="vr-btn-primary">Get Started</Link>
              <a href="#why-verus" className="vr-btn-ghost">Learn More</a>
            </div>

            {/* Stat pills */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {['600,000+ US Bridges', 'Same-Day Reporting', 'Multi-Standard Compliant'].map(s => (
                <div key={s} className="vr-stat-pill">{s}</div>
              ))}
            </div>
          </div>

          {/* Right — abstract interface panel */}
          <div style={{ position: 'relative' }}>
            {/* Outer frame */}
            <div style={{
              border: `1px solid rgba(255,255,255,0.1)`,
              background: 'rgba(255,255,255,0.025)',
              backdropFilter: 'blur(6px)',
              padding: 2,
            }}>
              {/* Title bar */}
              <div style={{
                background: 'rgba(255,255,255,0.04)',
                borderBottom: `1px solid rgba(255,255,255,0.07)`,
                padding: '10px 16px',
                display: 'flex', alignItems: 'center', gap: 10,
              }}>
                <div style={{ display: 'flex', gap: 6 }}>
                  {[C.orange, 'rgba(255,255,255,0.2)', 'rgba(255,255,255,0.2)'].map((bg, i) => (
                    <div key={i} style={{ width: 7, height: 7, borderRadius: '50%', background: bg }} />
                  ))}
                </div>
                <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.08em', marginLeft: 8 }}>
                  VERUS ANALYSIS — GPR C-SCAN
                </span>
              </div>

              {/* C-scan placeholder grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(12, 1fr)',
                gap: 2,
                padding: 16,
                aspectRatio: '4/3',
                background: 'rgba(10,10,10,0.6)',
              }}>
                {Array.from({ length: 96 }).map((_, i) => {
                  const row = Math.floor(i / 12);
                  const col = i % 12;
                  // Fake delamination pattern
                  const isHot = (row >= 2 && row <= 4 && col >= 3 && col <= 6)
                              || (row >= 5 && row <= 6 && col >= 7 && col <= 10);
                  const isMid = (row >= 1 && row <= 5 && col >= 2 && col <= 7 && !isHot);
                  return (
                    <div key={i} style={{
                      aspectRatio: '1',
                      background: isHot
                        ? `rgba(232,96,28,${0.55 + Math.random() * 0.35})`
                        : isMid
                        ? `rgba(232,96,28,${0.12 + Math.random() * 0.18})`
                        : `rgba(46,204,113,${0.18 + Math.random() * 0.22})`,
                      borderRadius: 1,
                    }} />
                  );
                })}
              </div>

              {/* Bottom status bar */}
              <div style={{
                borderTop: `1px solid rgba(255,255,255,0.06)`,
                background: 'rgba(255,255,255,0.02)',
                padding: '10px 16px',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <div style={{ display: 'flex', gap: 20 }}>
                  {[['Delamination', '18.4%'], ['Sound', '81.6%']].map(([k, v]) => (
                    <div key={k}>
                      <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                        {k}&nbsp;
                      </span>
                      <span style={{ fontSize: 11, fontWeight: 700, color: k === 'Delamination' ? C.orange : 'rgba(46,204,113,0.8)' }}>
                        {v}
                      </span>
                    </div>
                  ))}
                </div>
                <div style={{
                  fontSize: 9, color: C.orange,
                  background: 'rgba(232,96,28,0.1)',
                  border: '1px solid rgba(232,96,28,0.25)',
                  padding: '3px 8px', letterSpacing: '0.08em',
                }}>
                  ASTM D6087
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>

      {/* Scroll indicator */}
      <div style={{
        position: 'absolute', bottom: 28, left: '50%', transform: 'translateX(-50%)',
        zIndex: 3,
      }}>
        <div style={{
          width: 1, height: 40,
          background: 'rgba(255,255,255,0.15)',
          animation: 'pulseBar 2.5s ease-in-out infinite',
          transformOrigin: 'top',
        }} />
      </div>
    </section>
  );
}

// ── Why Verus — numbered list rows ────────────────────────────────────────────
const WHY_ITEMS = [
  {
    num: '01',
    title: 'Fully Automated',
    body: 'No manual data processing. Upload any inspection file and get results instantly. Our AI handles format detection, signal processing, and classification.',
  },
  {
    num: '02',
    title: 'Any Equipment',
    body: 'Works with data from all major inspection equipment manufacturers and file formats. If your device can export it, Verus can analyze it.',
  },
  {
    num: '03',
    title: 'Same-Day Reports',
    body: 'Standards-compliant condition reports generated in minutes, not weeks. Export C-scan maps and share results the same day as your inspection.',
  },
  {
    num: '04',
    title: 'Built to Scale',
    body: 'Designed for inspection teams managing hundreds of structures across large networks. Batch upload, multi-file analysis, and GPS-referenced output.',
  },
];

function WhyVerus() {
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
        ([entry]) => {
          if (entry.isIntersecting) {
            setTimeout(() => el.classList.add('revealed'), i * 80);
            obs.disconnect();
          }
        },
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

        {/* Heading row */}
        <div ref={headRef} className="vr-reveal" style={{
          display: 'flex', alignItems: 'flex-end',
          justifyContent: 'space-between', gap: 40,
          marginBottom: 16,
          paddingBottom: 28,
          borderBottom: `2px solid ${C.black}`,
        }}>
          <h2 style={{
            fontSize: 'clamp(26px, 3.5vw, 42px)',
            fontWeight: 800, color: C.black, margin: 0,
            letterSpacing: '-0.025em', lineHeight: 1.1,
          }}>
            Automated Analysis.<br />Zero Manual Steps.
          </h2>
          <p style={{
            fontSize: 10, fontWeight: 700, letterSpacing: '0.14em',
            textTransform: 'uppercase', color: C.orange,
            margin: 0, flexShrink: 0,
          }}>
            Why Verus
          </p>
        </div>

        {/* Numbered rows */}
        <div>
          {WHY_ITEMS.map((item, i) => (
            <div key={item.num} ref={rows[i]} className="vr-why-row">
              <div style={{
                display: 'grid',
                gridTemplateColumns: '64px 1fr 2fr',
                gap: 32,
                alignItems: 'start',
              }}>
                <span className="vr-why-num" style={{
                  fontSize: 11, fontWeight: 800,
                  color: C.orange, letterSpacing: '0.04em',
                  fontVariantNumeric: 'tabular-nums',
                  paddingTop: 2,
                }}>
                  {item.num}
                </span>
                <h3 style={{
                  fontSize: 17, fontWeight: 700, color: C.black,
                  margin: 0, letterSpacing: '-0.01em',
                }}>
                  {item.title}
                </h3>
                <p style={{ fontSize: 14, color: C.textGray, lineHeight: 1.75, margin: 0 }}>
                  {item.body}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── Black Stripe: Ticker ───────────────────────────────────────────────────────
function TickerStripe() {
  const items = [
    'INFRASTRUCTURE INSPECTION',
    'AI-POWERED ANALYSIS',
    'SAME-DAY REPORTS',
    'MULTI-SENSOR PLATFORM',
    'AUTOMATED REPORTING',
  ];

  const renderItems = () =>
    items.map((item, i) => (
      <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '1em' }}>
        <span style={{ color: C.textLight, letterSpacing: '0.13em' }}>{item}</span>
        <span style={{ color: C.orange, fontWeight: 700, fontSize: '0.7em' }}>◆</span>
      </span>
    ));

  return (
    <div style={{
      background: C.black,
      /* Diagonal hairline pattern */
      backgroundImage: `repeating-linear-gradient(
        -55deg,
        transparent,
        transparent 18px,
        rgba(255,255,255,0.013) 18px,
        rgba(255,255,255,0.013) 19px
      )`,
      borderTop: `2px solid ${C.orange}`,
      borderBottom: `1px solid rgba(255,255,255,0.07)`,
      padding: '20px 0',
      overflow: 'hidden',
      whiteSpace: 'nowrap',
      userSelect: 'none',
    }}>
      <div style={{
        display: 'inline-flex', gap: '1em',
        fontSize: 13, fontWeight: 600,
        animation: 'marquee 38s linear infinite',
        willChange: 'transform',
      }}>
        {renderItems()}{renderItems()}
        {renderItems()}{renderItems()}
      </div>
    </div>
  );
}

// ── How It Works — vertical steps on off-white ────────────────────────────────
const STEPS = [
  {
    num: '01',
    title: 'Upload',
    body: 'Drop your inspection data files in any supported format. Verus detects the format automatically and prepares it for analysis.',
  },
  {
    num: '02',
    title: 'Analyze',
    body: 'Our AI models process every signal and classify anomalies automatically, with no manual configuration required.',
  },
  {
    num: '03',
    title: 'Report',
    body: 'Download standards-compliant condition reports, same day. GPS-tagged data renders on an interactive satellite map.',
  },
];

function HowItWorks() {
  const headRef = useReveal(0);
  const s0 = useRef<HTMLDivElement>(null);
  const s1 = useRef<HTMLDivElement>(null);
  const s2 = useRef<HTMLDivElement>(null);
  const stepRefs = [s0, s1, s2];
  const ctaRef = useReveal(200);

  useEffect(() => {
    stepRefs.forEach((ref, i) => {
      const el = ref.current;
      if (!el) return;
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setTimeout(() => el.classList.add('revealed'), i * 110);
            obs.disconnect();
          }
        },
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
          <p style={{
            fontSize: 10, fontWeight: 700, letterSpacing: '0.14em',
            textTransform: 'uppercase', color: C.orange, margin: '0 0 14px',
          }}>
            How It Works
          </p>
          <h2 style={{
            fontSize: 'clamp(26px, 3.5vw, 42px)',
            fontWeight: 800, color: C.black,
            margin: 0, letterSpacing: '-0.025em',
          }}>
            From Raw Data to Report<br />in Minutes
          </h2>
        </div>

        {/* Vertical steps */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: 40,
          marginBottom: 64,
        }}>
          {STEPS.map((step, i) => (
            <div key={step.num} ref={stepRefs[i]} className="vr-step-card">
              <div style={{
                fontSize: 10, fontWeight: 800,
                color: C.orange, letterSpacing: '0.1em',
                marginBottom: 16, fontVariantNumeric: 'tabular-nums',
              }}>
                {step.num}
              </div>
              <h3 style={{
                fontSize: 20, fontWeight: 700, color: C.black,
                margin: '0 0 12px', letterSpacing: '-0.015em',
              }}>
                {step.title}
              </h3>
              <p style={{ fontSize: 14, color: C.textGray, lineHeight: 1.75, margin: 0 }}>
                {step.body}
              </p>
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

// ── Method marquee divider ────────────────────────────────────────────────────
function MethodSlider() {
  const items = [
    { name: 'GROUND-PENETRATING RADAR', standard: 'ASTM D6087' },
    { name: 'IMPACT-ECHO',              standard: 'ASTM C1383' },
    { name: 'INFRARED THERMOGRAPHY',    standard: 'ASTM D4788' },
    { name: 'AUTOMATED SOUNDING',       standard: 'ASTM D4580' },
  ];

  const renderItems = () =>
    items.map((item, i) => (
      <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.9em' }}>
        <span style={{ color: C.textLight, letterSpacing: '0.12em' }}>{item.name}</span>
        <span style={{ color: 'rgba(240,237,232,0.3)', fontSize: '0.8em', letterSpacing: '0.08em' }}>
          {item.standard}
        </span>
        <span style={{ color: C.orange, fontWeight: 700, fontSize: '0.7em' }}>◆</span>
      </span>
    ));

  return (
    <div style={{
      background: C.black,
      backgroundImage: `repeating-linear-gradient(
        -55deg,
        transparent,
        transparent 18px,
        rgba(255,255,255,0.013) 18px,
        rgba(255,255,255,0.013) 19px
      )`,
      borderTop: `2px solid ${C.orange}`,
      borderBottom: `1px solid rgba(255,255,255,0.07)`,
      padding: '20px 0',
      overflow: 'hidden',
      whiteSpace: 'nowrap',
      userSelect: 'none',
    }}>
      <div style={{
        display: 'inline-flex', gap: '0.9em',
        fontSize: 13, fontWeight: 600,
        animation: 'marquee 32s linear infinite',
        willChange: 'transform',
      }}>
        {renderItems()}{renderItems()}
        {renderItems()}{renderItems()}
      </div>
    </div>
  );
}

// ── Our Platform ──────────────────────────────────────────────────────────────
const METHODS = [
  { label: 'Ground-Penetrating Radar', short: 'GPR',    active: true  },
  { label: 'Impact-Echo',              short: 'IE',     active: false },
  { label: 'Infrared Thermography',    short: 'IR',     active: false },
  { label: 'Automated Sounding',       short: 'RAS',    active: false },
];

function OurPlatform() {
  const ref = useReveal(0);
  const listRef = useReveal(120);

  return (
    <section style={{ background: C.offWhite, padding: '96px 48px' }}>
      <div style={{ maxWidth: 1080, margin: '0 auto' }}>
        <div ref={ref} className="vr-reveal" style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 80, alignItems: 'start',
        }}>
          {/* Left */}
          <div>
            <p className="vr-platform-label">Our Platform</p>
            <h2 style={{
              fontSize: 'clamp(24px, 3vw, 38px)',
              fontWeight: 800, color: C.black,
              margin: '0 0 20px', letterSpacing: '-0.025em',
            }}>
              One Platform.<br />Every Inspection Method.
            </h2>
            <p style={{
              fontSize: 14, color: C.textGray, lineHeight: 1.75,
              margin: '0 0 36px', maxWidth: 400,
            }}>
              Verus is built to support the full range of non-destructive evaluation methods.
              Start with what you need today and expand as your program grows.
            </p>
            <Link to="/signup" className="vr-btn-primary">Get Early Access</Link>
          </div>

          {/* Right — method table */}
          <div ref={listRef} className="vr-reveal">
            <div style={{ borderTop: `2px solid ${C.black}` }}>
              {METHODS.map(m => (
                <div
                  key={m.label}
                  className={`vr-method-row${m.active ? ' vr-method-row-active' : ''}`}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <span style={{
                      fontSize: 10, fontWeight: 800,
                      color: m.active ? C.orange : C.textGray,
                      letterSpacing: '0.08em', width: 36, flexShrink: 0,
                      transition: 'color 0.2s',
                    }}>
                      {m.short}
                    </span>
                    <span style={{
                      fontSize: 14, fontWeight: 600,
                      color: m.active ? C.black : C.textGray,
                      transition: 'color 0.2s',
                    }}>
                      {m.label}
                    </span>
                  </div>
                  <span style={{
                    fontSize: 9, fontWeight: 700,
                    letterSpacing: '0.09em', textTransform: 'uppercase',
                    padding: '4px 10px',
                    background: m.active ? C.orange : 'transparent',
                    color: m.active ? '#fff' : C.textGray,
                    border: `1px solid ${m.active ? C.orange : C.border}`,
                    flexShrink: 0,
                    transition: 'background 0.2s, border-color 0.2s, color 0.2s',
                  }}>
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

// ── Footer — compact single band ──────────────────────────────────────────────
function Footer() {
  return (
    <footer style={{
      background: C.black,
      borderTop: `1px solid rgba(255,255,255,0.07)`,
      padding: '24px 48px',
    }}>
      <div style={{
        maxWidth: 1080, margin: '0 auto',
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', gap: 24, flexWrap: 'wrap',
      }}>
        <Link to="/" style={{ textDecoration: 'none' }}>
          <VerusLogo size={26} wordmarkColor="rgba(240,237,232,0.8)" />
        </Link>

        <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
          <Link to="/" className="vr-footer-link">Home</Link>
          <Link to="/team" className="vr-footer-link">Team</Link>
          <Link to="/login" className="vr-footer-link">Log In</Link>
          <Link to="/signup" className="vr-footer-link">Sign Up</Link>
          <a href="mailto:hello@verus.ai" className="vr-footer-link">hello@verus.ai</a>
        </div>

        <p style={{ fontSize: 11, color: 'rgba(240,237,232,0.55)', margin: 0 }}>
          © {new Date().getFullYear()} Verus Technologies, Inc.
        </p>
      </div>
    </footer>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function HomePage() {
  return (
    <div style={{ fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', overflowX: 'hidden' }}>
      <style>{PAGE_CSS}</style>
      <Navbar />
      <Hero />
      <WhyVerus />
      <TickerStripe />
      <HowItWorks />
      <MethodSlider />
      <OurPlatform />
      <Footer />
    </div>
  );
}
