import { useState } from 'react';
import { Link } from 'react-router';
import { C } from './team/tokens';
import FounderCard, { FOUNDERS } from './team/FounderCard';
import { useReveal } from './home/useReveal';

const PAGE_CSS = `
  .tm-nav-link {
    position: relative;
    color: rgba(240,237,232,0.7);
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.03em;
    padding-bottom: 2px;
    transition: color 0.2s;
  }
  .tm-nav-link::after {
    content: '';
    position: absolute;
    left: 0; bottom: 0;
    height: 1.5px; width: 0;
    background: ${C.orange};
    transition: width 0.25s ease;
  }
  .tm-nav-link:hover { color: #F0EDE8; }
  .tm-nav-link:hover::after { width: 100%; }

  .tm-nav-signup {
    display: inline-block;
    padding: 8px 20px;
    background: ${C.orange};
    color: #fff;
    font-size: 12px; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    text-decoration: none; border: none;
    transition: transform 0.18s, background 0.18s;
  }
  .tm-nav-signup:hover { background: #D4521A; transform: translateY(-2px); }

  .tm-card {
    background: ${C.card};
    border: 1.5px solid ${C.border};
    transition: border-color 0.25s;
    padding: 28px;
    display: flex;
    flex-direction: row;
    gap: 24px;
    align-items: flex-start;
  }
  .tm-card:hover { border-color: ${C.orange}; }

  .tm-footer-link {
    color: rgba(240,237,232,0.5);
    text-decoration: none;
    font-size: 13px;
    transition: color 0.2s;
  }
  .tm-footer-link:hover { color: ${C.orange}; }

  .tm-reveal {
    opacity: 0;
    transform: translateY(24px);
    transition: opacity 0.55s ease, transform 0.55s ease;
  }
  .tm-reveal.revealed { opacity: 1; transform: translateY(0); }

  /* Nav desktop/mobile split */
  .tm-nav-inner {
    display: flex; align-items: center;
    width: 100%; height: 58px;
    padding: 0 48px;
  }
  .tm-nav-desktop-links {
    position: absolute; left: 50%; top: 50%;
    transform: translate(-50%, -50%);
    display: flex; align-items: center; gap: 32px;
  }
  .tm-nav-desktop-cta {
    margin-left: auto;
    display: flex; align-items: center; gap: 22px; flex-shrink: 0;
  }
  .tm-hamburger {
    display: none;
    flex-direction: column;
    justify-content: center;
    gap: 5px;
    cursor: pointer;
    background: none; border: none;
    padding: 6px; margin-left: auto; flex-shrink: 0;
  }
  .tm-hamburger span {
    display: block;
    width: 20px; height: 2px;
    background: rgba(240,237,232,0.85);
    border-radius: 1px;
    transition: transform 0.22s, opacity 0.22s;
  }
  .tm-mobile-menu {
    display: none;
    position: fixed;
    top: 58px; left: 0; right: 0;
    background: #0A0A0A;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 4px 24px 20px;
    flex-direction: column;
    z-index: 99;
  }
  .tm-mobile-menu.open { display: flex; }
  .tm-mobile-menu-link {
    padding: 14px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    color: rgba(240,237,232,0.75);
    font-size: 15px; font-weight: 500;
    text-decoration: none;
    transition: color 0.2s;
  }
  .tm-mobile-menu-link:hover { color: #F0EDE8; }

  /* Layout classes */
  .tm-hero-section { padding: 100px 48px 80px; }
  .tm-founders-section { padding: 80px 48px 96px; }
  .tm-founders-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 24px;
  }
  .tm-footer-section { padding: 52px 48px; }
  .tm-footer-inner {
    display: flex; align-items: center;
    justify-content: space-between;
    flex-wrap: wrap; gap: 24px;
  }

  /* ── Tablet ≤ 768px ─── */
  @media (max-width: 768px) {
    .tm-nav-inner         { padding: 0 20px; }
    .tm-hamburger         { display: flex; }
    .tm-nav-desktop-links { display: none; }
    .tm-nav-desktop-cta   { display: none; }

    .tm-hero-section     { padding: 80px 24px 60px; }
    .tm-founders-section { padding: 56px 24px 72px; }
    .tm-footer-section   { padding: 36px 24px; }

    .tm-founders-grid { grid-template-columns: 1fr; }
  }

  /* ── Mobile ≤ 640px ─── */
  @media (max-width: 640px) {
    .tm-hero-section     { padding: 72px 20px 52px; }
    .tm-founders-section { padding: 48px 20px 64px; }
    .tm-footer-section   { padding: 28px 20px; }

    .tm-card { padding: 20px; gap: 16px; }
    .tm-footer-inner { flex-direction: column; align-items: flex-start; }
  }
`;


export default function TeamPage() {
  const heroRef    = useReveal(0);
  const labelRef   = useReveal(80);
  const taglineRef = useReveal(160);
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div style={{ fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', overflowX: 'hidden' }}>
      <style>{PAGE_CSS}</style>

      {/* ── Navbar ── */}
      <nav style={{
        position: 'sticky', top: 0, zIndex: 100,
        background: C.black,
        borderBottom: `1px solid ${C.borderDk}`,
      }}>
        <div className="tm-nav-inner">
          <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
            <img src="/verus-logo.png" alt="Verus"
              style={{ width: 30, height: 30, objectFit: 'contain', display: 'block' }} />
            <span style={{ color: '#F0EDE8', fontSize: 14, fontWeight: 700, letterSpacing: '0.04em' }}>
              VERUS
            </span>
          </Link>

          <div className="tm-nav-desktop-links">
            <Link to="/" className="tm-nav-link">Home</Link>
            <Link to="/team" className="tm-nav-link" style={{ color: '#F0EDE8' }}>Team</Link>
          </div>

          <div className="tm-nav-desktop-cta">
            <Link to="/login" className="tm-nav-link">Log In</Link>
            <Link to="/signup" className="tm-nav-signup">Sign Up</Link>
          </div>

          <button className="tm-hamburger" onClick={() => setMenuOpen(o => !o)} aria-label="Toggle menu">
            <span style={{ transform: menuOpen ? 'translateY(7px) rotate(45deg)' : 'none' }} />
            <span style={{ opacity: menuOpen ? 0 : 1 }} />
            <span style={{ transform: menuOpen ? 'translateY(-7px) rotate(-45deg)' : 'none' }} />
          </button>
        </div>
      </nav>

      <div className={`tm-mobile-menu${menuOpen ? ' open' : ''}`}>
        <Link to="/" className="tm-mobile-menu-link" onClick={() => setMenuOpen(false)}>Home</Link>
        <Link to="/team" className="tm-mobile-menu-link" onClick={() => setMenuOpen(false)}>Team</Link>
        <Link to="/login" className="tm-mobile-menu-link" onClick={() => setMenuOpen(false)}>Log In</Link>
        <Link to="/signup" className="tm-nav-signup" style={{ marginTop: 16, textAlign: 'center' }} onClick={() => setMenuOpen(false)}>
          Sign Up
        </Link>
      </div>

      {/* ── Hero ── */}
      <section className="tm-hero-section" style={{ background: C.black, textAlign: 'center' }}>
        <div ref={labelRef} className="tm-reveal" style={{
          display: 'inline-block',
          fontSize: 10, fontWeight: 700, letterSpacing: '0.14em',
          textTransform: 'uppercase', color: C.orange,
          marginBottom: 24,
        }}>
          The People Behind Verus
        </div>

        <div ref={heroRef} className="tm-reveal" style={{ transitionDelay: '80ms' }}>
          <h1 style={{
            fontSize: 'clamp(36px, 6vw, 64px)',
            fontWeight: 800, color: '#F0EDE8',
            margin: '0 0 20px',
            lineHeight: 1.1,
            letterSpacing: '-0.03em',
          }}>
            The Team
          </h1>
        </div>

        <div ref={taglineRef} className="tm-reveal" style={{ transitionDelay: '160ms' }}>
          <p style={{
            fontSize: 17, color: 'rgba(240,237,232,0.6)',
            maxWidth: 560, margin: '0 auto',
            lineHeight: 1.65, fontStyle: 'italic',
          }}>
            Met at Georgia Tech building an inspection drone. Realized the real problem was the data analysis.
          </p>
        </div>
      </section>

      {/* ── Founders ── */}
      <section className="tm-founders-section" style={{ background: C.offWhite }}>
        <div style={{ maxWidth: 1100, margin: '0 auto' }}>
          <div className="tm-founders-grid">
            {FOUNDERS.map((founder, i) => (
              <FounderCard key={founder.name} founder={founder} delay={i * 120} />
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="tm-footer-section" style={{
        background: C.black,
        borderTop: `1px solid ${C.borderDk}`,
      }}>
        <div style={{ maxWidth: 1080, margin: '0 auto' }}>
          <div className="tm-footer-inner">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <img src="/verus-logo.png" alt="Verus"
                style={{ width: 28, height: 28, objectFit: 'contain', opacity: 0.8 }} />
              <span style={{ color: 'rgba(240,237,232,0.4)', fontSize: 12 }}>
                © {new Date().getFullYear()} Verus
              </span>
            </div>

            <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap' }}>
              <Link to="/" className="tm-footer-link">Home</Link>
              <Link to="/team" className="tm-footer-link">Team</Link>
              <Link to="/login" className="tm-footer-link">Log In</Link>
              <Link to="/signup" className="tm-footer-link">Sign Up</Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
