import { C } from './team/tokens';
import FounderCard, { FOUNDERS } from './team/FounderCard';
import { useReveal } from './home/useReveal';
import Navbar, { NAVBAR_HEIGHT } from '../components/Navbar';
import Footer from '../components/Footer';

const PAGE_CSS = `
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

  .tm-reveal {
    opacity: 0;
    transform: translateY(24px);
    transition: opacity 0.55s ease, transform 0.55s ease;
  }
  .tm-reveal.revealed { opacity: 1; transform: translateY(0); }

  .tm-hero-section { padding: ${NAVBAR_HEIGHT + 72}px 48px 80px; }
  .tm-founders-section { padding: 80px 48px 96px; }
  .tm-founders-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 24px;
  }

  @media (max-width: 768px) {
    .tm-hero-section     { padding: ${NAVBAR_HEIGHT + 48}px 24px 60px; }
    .tm-founders-section { padding: 56px 24px 72px; }
    .tm-founders-grid    { grid-template-columns: 1fr; }
  }

  @media (max-width: 640px) {
    .tm-hero-section     { padding: ${NAVBAR_HEIGHT + 36}px 20px 52px; }
    .tm-founders-section { padding: 48px 20px 64px; }
    .tm-card { padding: 20px; gap: 16px; }
  }
`;

export default function TeamPage() {
  const heroRef    = useReveal(0);
  const labelRef   = useReveal(80);
  const taglineRef = useReveal(160);

  return (
    <div style={{ fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', overflowX: 'hidden' }}>
      <style>{PAGE_CSS}</style>

      <Navbar />

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

      <Footer />
    </div>
  );
}
