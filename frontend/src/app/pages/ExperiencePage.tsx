import { useEffect, useRef } from 'react';
import { Link } from 'react-router';
import { Play, Pause } from 'lucide-react';
import VerusLogo from '../components/VerusLogo';
import { useDeckExperience } from './experience/useDeckExperience';
import { EXPERIENCE_CSS } from './experience/styles';
import { STOPS, RULER_TICKS, GEO } from './experience/constants';

export default function ExperiencePage() {
  const rootRef = useRef<HTMLDivElement>(null);
  useDeckExperience(rootRef);

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = prev; };
  }, []);

  return (
    <div className="exp-root" ref={rootRef}>
      <style>{EXPERIENCE_CSS}</style>

      <div className="exp-stage">
        <div className="exp-layer-wrap"><div className="exp-layer exp-lfar" /></div>
        <div className="exp-layer-wrap"><div className="exp-layer exp-lback" /></div>
        <div className="exp-layer-wrap"><div className="exp-layer exp-lworld" /></div>

        <div className="exp-cart">
          <svg width="120" height="78" viewBox="0 0 120 78" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M40 54 L18 230 L102 230 L80 54 Z" fill="rgba(232,96,28,0.10)" />
            <line x1="60" y1="54" x2="60" y2="240" stroke="rgba(232,96,28,0.5)" strokeWidth="1.5" strokeDasharray="3 4" />
            <line x1="78" y1="14" x2="96" y2="46" stroke="#F0EDE8" strokeWidth="3" />
            <rect x="30" y="34" width="56" height="20" rx="2" fill="#1a1a1a" stroke="#E8601C" strokeWidth="2" />
            <rect x="36" y="39" width="18" height="10" fill="#E8601C" />
            <circle cx="42" cy="62" r="9" fill="#0A0A0A" stroke="#F0EDE8" strokeWidth="2.5" />
            <circle cx="74" cy="62" r="9" fill="#0A0A0A" stroke="#F0EDE8" strokeWidth="2.5" />
            <circle cx="42" cy="62" r="2" fill="#E8601C" />
            <circle cx="74" cy="62" r="2" fill="#E8601C" />
          </svg>
        </div>

        <div className="exp-ruler">
          {RULER_TICKS.map((t) => <div className="tick" key={t}>{t}</div>)}
        </div>
        <div className="exp-lens" />
      </div>

      <header className="exp-header">
        <Link to="/" style={{ textDecoration: 'none' }}><VerusLogo size={28} wordmarkColor="#F0EDE8" /></Link>
        <span className="exp-htag">Interactive · Inside the Deck</span>
        <Link to="/" className="exp-hback">← Back to site</Link>
      </header>

      <nav className="exp-pills">
        {STOPS.map((st, i) => (
          <button className={`exp-pill${i === 0 ? ' on' : ''}`} key={st.tag}>
            <i>0{i + 1}</i>{st.tag}
          </button>
        ))}
      </nav>

      <div className="exp-card">
        <span className="exp-card-num">01</span>
        <span className="exp-card-tag">{STOPS[0].tag}</span>
        <h2 className="exp-card-title">{STOPS[0].title}</h2>
        <p className="exp-card-body">{STOPS[0].body}</p>
      </div>

      <div className="exp-transport">
        <button className="exp-playbtn" aria-label="Play / pause">
          <Play className="exp-icon-play" size={16} fill="currentColor" strokeWidth={0} />
          <Pause className="exp-icon-pause" size={16} fill="currentColor" strokeWidth={0} />
        </button>
        <div className="exp-track">
          <div className="exp-bar" />
          <div className="exp-barfill" />
          <div className="exp-tickrow">
            {STOPS.map((st) => (
              <div className="exp-stick" key={st.tag} style={{ left: `${(st.x / GEO.W) * 100}%` }} />
            ))}
          </div>
          <div className="exp-playhead" />
        </div>
        <span className="exp-hint">Scroll, drag the timeline, or tap a chip to explore</span>
      </div>

      <div className="exp-tip">
        <div className="exp-tip-t" />
        <div className="exp-tip-s" />
      </div>
    </div>
  );
}
