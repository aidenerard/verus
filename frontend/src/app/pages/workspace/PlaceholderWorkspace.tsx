import { useState } from 'react';
import type { ComponentType } from 'react';
import { Mail } from 'lucide-react';
import type { PlaceholderContent } from './types';
import { ACCENT, ACCENT_SOFT, BORDER, PANEL, RAISED, TEXT, TEXT2, TEXT3 } from './tokens';

interface Props {
  content: PlaceholderContent;
  Icon:    ComponentType<{ size?: number; style?: React.CSSProperties }>;
}

export default function PlaceholderWorkspace({ content, Icon }: Props) {
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);

  return (
    <div style={{ padding: '48px 32px', display: 'flex', justifyContent: 'center' }}>
      <div style={{ background: PANEL, border: `1px solid ${BORDER}`, maxWidth: 640, width: '100%', padding: '44px 40px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 6 }}>
          <div style={{
            width: 52, height: 52, borderRadius: '50%', background: ACCENT_SOFT,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={24} style={{ color: ACCENT }} />
          </div>
          <div>
            <div style={{ fontSize: 22, fontWeight: 800, color: TEXT, letterSpacing: '-0.01em' }}>{content.name}</div>
            {content.standard && (
              <div style={{ fontSize: 11, color: TEXT3, fontWeight: 600, letterSpacing: '0.04em', marginTop: 2 }}>
                {content.standard}
              </div>
            )}
          </div>
        </div>

        <p style={{ fontSize: 14, lineHeight: 1.65, color: TEXT2, margin: '20px 0 24px' }}>
          {content.description}
        </p>

        <div style={{ background: RAISED, border: `1px solid ${BORDER}`, padding: '16px 20px', marginBottom: 28 }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', color: TEXT3, marginBottom: 10 }}>
            Use Cases
          </div>
          <ul style={{ margin: 0, paddingLeft: 18, color: TEXT, fontSize: 13, lineHeight: 1.7 }}>
            {content.useCases.map(uc => <li key={uc}>{uc}</li>)}
          </ul>
        </div>

        <div style={{ borderTop: `1px solid ${BORDER}`, paddingTop: 24 }}>
          <div style={{ display: 'inline-block', background: ACCENT_SOFT, color: ACCENT, padding: '4px 12px', fontSize: 10, fontWeight: 700, letterSpacing: '0.10em', textTransform: 'uppercase', marginBottom: 12 }}>
            Coming Soon
          </div>
          <div style={{ fontSize: 13, color: TEXT2, marginBottom: 16 }}>
            We're training a dedicated AI model for {content.name}. Drop your email to be notified at launch.
          </div>

          {!submitted ? (
            <div style={{ display: 'flex', gap: 10 }}>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="your@email.com"
                style={{ flex: 1, padding: '10px 14px', background: RAISED, border: `1px solid ${BORDER}`, color: TEXT, fontSize: 13, fontFamily: 'inherit', outline: 'none' }}
              />
              <button
                onClick={() => { if (email.includes('@')) setSubmitted(true); }}
                style={{ padding: '10px 18px', background: ACCENT, color: '#fff', border: 'none', cursor: 'pointer', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 6 }}
              >
                <Mail size={12} /> Notify
              </button>
            </div>
          ) : (
            <p style={{ fontSize: 13, color: '#22c55e', fontWeight: 600, margin: 0 }}>
              ✓ We'll notify you at {email} when {content.name} launches.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
