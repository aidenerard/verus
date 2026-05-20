import { useNavigate } from 'react-router';
import { ArrowRight } from 'lucide-react';
import { MODULES } from './modules';
import SelectPageShell from './SelectPageShell';
import { ACCENT, ACCENT_SOFT, BORDER, BORDER2, PANEL, TEXT, TEXT2, TEXT3 } from './tokens';

export default function ModuleSelectPage() {
  const navigate = useNavigate();

  return (
    <SelectPageShell
      eyebrow="Step 1 of 2"
      title="Choose an Inspection Module"
      subtitle="Pick a sensing modality. You'll choose a specific method on the next screen."
      backTo="/dashboard"
      backLabel="Dashboard"
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 20 }}>
        {MODULES.map(mod => {
          const methodPreview = mod.methods.map(m => m.name).join(' · ');
          const anyAvailable = mod.methods.some(m => m.status === 'available');
          const path = `/workspace/${mod.id}`;
          return (
            <button
              key={mod.id}
              onClick={() => navigate(path)}
              className="module-card"
              style={{
                background: PANEL, border: `1px solid ${BORDER}`,
                padding: '32px 32px 28px', textAlign: 'left', cursor: 'pointer',
                display: 'flex', flexDirection: 'column', gap: 14,
                fontFamily: 'inherit', transition: 'border-color 0.15s, box-shadow 0.15s, transform 0.15s',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: '0.14em', textTransform: 'uppercase', color: TEXT3 }}>
                  Module
                </span>
                {anyAvailable && (
                  <span style={{ fontSize: 9, fontWeight: 800, letterSpacing: '0.08em', textTransform: 'uppercase', padding: '3px 8px', background: ACCENT_SOFT, color: ACCENT }}>
                    Available
                  </span>
                )}
              </div>

              <div style={{ fontSize: 26, fontWeight: 800, color: TEXT, letterSpacing: '-0.02em', textTransform: 'uppercase' }}>
                {mod.label}
              </div>

              <div style={{ fontSize: 13, color: TEXT2, lineHeight: 1.55 }}>
                {methodPreview}
              </div>

              <div style={{ flex: 1 }} />

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: `1px solid ${BORDER}`, paddingTop: 16, marginTop: 8 }}>
                <span style={{ fontSize: 11, color: TEXT3 }}>{mod.methods.length} method{mod.methods.length !== 1 ? 's' : ''}</span>
                <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: ACCENT, display: 'flex', alignItems: 'center', gap: 6 }}>
                  Select <ArrowRight size={12} />
                </span>
              </div>
            </button>
          );
        })}
      </div>

      <style>{`
        .module-card:hover {
          border-color: ${BORDER2} !important;
          box-shadow: 0 10px 30px rgba(10,10,10,0.06);
          transform: translateY(-2px);
        }
      `}</style>
    </SelectPageShell>
  );
}
