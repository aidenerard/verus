import { useState } from 'react';
import { useNavigate } from 'react-router';
import { ArrowLeft, Waves, Thermometer, Speaker } from 'lucide-react';
import VerusLogo from '../../components/VerusLogo';

const BG     = '#F5F3EF';
const PANEL  = '#FFFFFF';
const BORDER = '#E2DED9';
const TEXT   = '#0A0A0A';
const TEXT2  = '#7A7470';
const ACCENT = '#E8601C';

interface MethodConfig {
  name: string;
  fullName: string;
  standard: string;
  icon: React.ComponentType<{ size?: number; style?: React.CSSProperties }>;
}

const METHODS: Record<string, MethodConfig> = {
  'impact-echo': {
    name: 'Impact-Echo',
    fullName: 'Impact-Echo Acoustic Testing',
    standard: 'ASTM C1383',
    icon: Waves,
  },
  ir: {
    name: 'Infrared',
    fullName: 'Infrared Thermography',
    standard: 'ASTM D4788',
    icon: Thermometer,
  },
  ras: {
    name: 'Automated Sounding',
    fullName: 'Rapid Automated Sounding (RAS)',
    standard: 'ASTM D4580',
    icon: Speaker,
  },
};

export default function ComingSoonWorkspace({ method }: { method: string }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const cfg = METHODS[method] ?? METHODS['impact-echo'];
  const Icon = cfg.icon;

  return (
    <div style={{
      height: '100vh', display: 'flex', flexDirection: 'column',
      background: BG, color: TEXT,
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    }}>
      {/* Toolbar */}
      <div style={{
        height: 48, flexShrink: 0,
        display: 'flex', alignItems: 'center',
        padding: '0 16px', gap: 12,
        background: PANEL,
        borderBottom: `1px solid ${BORDER}`,
      }}>
        <button
          onClick={() => navigate('/dashboard')}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: TEXT2, display: 'flex', alignItems: 'center', gap: 6,
            fontSize: 12, fontWeight: 600, padding: '4px 8px',
          }}
          onMouseEnter={e => (e.currentTarget.style.color = TEXT)}
          onMouseLeave={e => (e.currentTarget.style.color = TEXT2)}
        >
          <ArrowLeft size={14} /> Dashboard
        </button>
        <div style={{ width: 1, height: 20, background: BORDER }} />
        <VerusLogo size={22} wordmarkColor="#0A0A0A" />
      </div>

      {/* Body */}
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 24,
      }}>
        <div style={{
          background: PANEL, border: `1px solid ${BORDER}`,
          maxWidth: 520, width: '100%', padding: '48px 40px',
          textAlign: 'center',
        }}>
          <div style={{
            width: 64, height: 64, borderRadius: '50%',
            background: 'rgba(232,96,28,0.12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 24px',
          }}>
            <Icon size={28} style={{ color: ACCENT }} />
          </div>

          <div style={{
            display: 'inline-block',
            background: 'rgba(232,96,28,0.12)', color: ACCENT,
            padding: '4px 12px', fontSize: 10, fontWeight: 700,
            letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 16,
          }}>
            In Development
          </div>

          <h1 style={{ margin: '0 0 12px', fontSize: 22, fontWeight: 700, color: TEXT }}>
            {cfg.fullName}
          </h1>

          <p style={{ margin: '0 0 8px', fontSize: 14, color: TEXT2, lineHeight: 1.6 }}>
            Verus is training a dedicated AI model for {cfg.name} analysis.
            You'll be notified when it launches.
          </p>

          <p style={{ margin: '0 0 32px', fontSize: 12, color: TEXT2, opacity: 0.6 }}>
            Standard: {cfg.standard}
          </p>

          {!submitted ? (
            <div style={{ display: 'flex', gap: 10 }}>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="your@email.com"
                style={{
                  flex: 1, padding: '10px 14px',
                  background: '#F5F3EF',
                  border: `1px solid ${BORDER}`, borderRadius: 0,
                  color: TEXT, fontSize: 13, fontFamily: 'Inter, sans-serif',
                  outline: 'none',
                }}
              />
              <button
                onClick={() => { if (email.includes('@')) setSubmitted(true); }}
                style={{
                  padding: '10px 20px',
                  background: ACCENT, color: '#fff',
                  border: 'none', cursor: 'pointer',
                  fontSize: 12, fontWeight: 700,
                  letterSpacing: '0.06em', textTransform: 'uppercase',
                  fontFamily: 'Inter, sans-serif',
                }}
              >
                Notify Me
              </button>
            </div>
          ) : (
            <p style={{ fontSize: 13, color: '#22c55e', fontWeight: 600 }}>
              ✓ We'll notify you at {email} when {cfg.name} launches.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
