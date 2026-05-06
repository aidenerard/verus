import { C } from './tokens';
import { useReveal } from '../home/useReveal';

export interface Founder {
  name: string;
  title: string;
  initials: string;
  photo?: string;
  bullets: string[];
}

export const FOUNDERS: Founder[] = [
  {
    name: 'Aiden Erard',
    title: 'CEO & Co-Founder',
    initials: 'AE',
    photo: '/aiden.png',
    bullets: [
      'Computer Engineering at Georgia Tech',
      'Researching bipedal robotics and autonomous navigation',
      'Won multiple hackathons — built real-time telemetry systems and multi-API integrations',
      'Experience scaling businesses, marketing, and customer discovery',
    ],
  },
  {
    name: 'Taran Govindu',
    title: 'CTO & Co-Founder',
    initials: 'TG',
    photo: '/taran.png',
    bullets: [
      'Aerospace Engineering at Georgia Tech',
      'Researching AI-accelerated simulation',
      'Built neural networks for exoplanet detection and medical diagnostics (98%+ accuracy)',
      'Published peer-reviewed research (5,000+ reads)',
      'Designed rocket propulsion systems and simulations',
    ],
  },
];

export default function FounderCard({ founder, delay }: { founder: Founder; delay: number }) {
  const ref = useReveal(delay);
  return (
    <div ref={ref} className="tm-reveal tm-card">
      <img src={founder.photo} alt={founder.name}
        style={{ height: 160, width: 'auto', flexShrink: 0, display: 'block', borderRadius: 4, border: `2px solid ${C.orange}` }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <h3 style={{ margin: '0 0 4px', fontSize: 17, fontWeight: 700, color: C.black }}>{founder.name}</h3>
        <p style={{ margin: '0 0 14px', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: C.orange }}>
          {founder.title}
        </p>
        <p style={{ margin: 0, fontSize: 13, color: C.textGray, lineHeight: 1.75 }}>
          {founder.bullets.join('. ')}.
        </p>
      </div>
    </div>
  );
}
