import { ReactNode } from 'react';
import { ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router';
import VerusLogo from '../../components/VerusLogo';
import { BG, BORDER, PANEL, TEXT, TEXT2, TEXT3, FONT_FAMILY, TOPBAR_HEIGHT } from './tokens';

interface Props {
  eyebrow:   string;
  title:     string;
  subtitle?: string;
  backTo:    string;
  backLabel: string;
  children:  ReactNode;
}

export default function SelectPageShell({ eyebrow, title, subtitle, backTo, backLabel, children }: Props) {
  const navigate = useNavigate();
  return (
    <div style={{ minHeight: '100vh', background: BG, color: TEXT, fontFamily: FONT_FAMILY, display: 'flex', flexDirection: 'column' }}>
      <header style={{
        height: TOPBAR_HEIGHT, flexShrink: 0, background: PANEL,
        borderBottom: `1px solid ${BORDER}`, padding: '0 24px',
        display: 'flex', alignItems: 'center', gap: 16,
      }}>
        <button
          onClick={() => navigate(backTo)}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: TEXT2, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, padding: '4px 8px', fontFamily: 'inherit' }}
        >
          <ArrowLeft size={14} /> {backLabel}
        </button>
        <div style={{ width: 1, height: 20, background: BORDER }} />
        <VerusLogo size={22} wordmarkColor={TEXT} />
      </header>

      <main style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px 32px' }}>
        <div style={{ width: '100%', maxWidth: 1000 }}>
          <div style={{ textAlign: 'center', marginBottom: 48 }}>
            <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.16em', textTransform: 'uppercase', color: TEXT3, marginBottom: 10 }}>
              {eyebrow}
            </div>
            <h1 style={{ margin: '0 0 8px', fontSize: 28, fontWeight: 800, letterSpacing: '-0.02em', color: TEXT }}>
              {title}
            </h1>
            {subtitle && (
              <p style={{ margin: 0, fontSize: 14, color: TEXT2 }}>{subtitle}</p>
            )}
          </div>
          {children}
        </div>
      </main>
    </div>
  );
}
