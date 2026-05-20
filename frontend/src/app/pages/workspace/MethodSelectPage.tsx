import { useNavigate, Navigate } from 'react-router';
import { ArrowRight } from 'lucide-react';
import { MODULES } from './modules';
import SelectPageShell from './SelectPageShell';
import { ACCENT, ACCENT_SOFT, BORDER, BORDER2, PANEL, RAISED, TEXT, TEXT2, TEXT3 } from './tokens';

const METHOD_BLURBS: Record<string, string[]> = {
  gpr:          ['Rebar depth mapping', 'Corrosion-risk heatmaps', 'Horizon-pick interpretation'],
  fdem:         ['Apparent conductivity', 'Buried metallic + saline plumes', 'Wide-area survey coverage'],
  magnetometer: ['Ferrous-object detection', 'UXO & buried tanks', 'Archaeological prospection'],
  masw:         ['Shear-wave velocity profiles', 'Foundation stiffness', 'Bedrock + void mapping'],
  'impact-echo':['Plate thickness verification', 'Delamination detection', 'Crack-depth estimation'],
};

interface Props { moduleId: string }

export default function MethodSelectPage({ moduleId }: Props) {
  const navigate = useNavigate();
  const mod = MODULES.find(m => m.id === moduleId);
  if (!mod) return <Navigate to="/workspace" replace />;

  return (
    <SelectPageShell
      eyebrow={`Step 2 of 2 · ${mod.label}`}
      title="Choose an Inspection Method"
      subtitle="Pick a specific technique. Setup and analysis happen on the next screen."
      backTo="/workspace"
      backLabel="Modules"
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16 }}>
        {mod.methods.map(m => {
          const Icon = m.Icon;
          const available = m.status === 'available';
          const blurbs = METHOD_BLURBS[m.id] ?? [];

          const cardStyle: React.CSSProperties = {
            background: PANEL, border: `1px solid ${BORDER}`,
            padding: '22px 22px 18px', textAlign: 'left',
            display: 'flex', flexDirection: 'column', gap: 12, minHeight: 260,
            fontFamily: 'inherit',
            cursor: available ? 'pointer' : 'not-allowed',
            opacity: available ? 1 : 0.55,
            transition: 'border-color 0.15s, box-shadow 0.15s, transform 0.15s',
          };

          const inner = (
            <>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ width: 36, height: 36, borderRadius: '50%', background: available ? ACCENT_SOFT : RAISED, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Icon size={16} style={{ color: available ? ACCENT : TEXT3 }} />
                </div>
                {!available && (
                  <span style={{ fontSize: 9, fontWeight: 800, letterSpacing: '0.10em', textTransform: 'uppercase', padding: '3px 8px', background: RAISED, color: TEXT3, border: `1px solid ${BORDER}` }}>
                    Soon
                  </span>
                )}
              </div>

              <div>
                <div style={{ fontSize: 18, fontWeight: 800, color: TEXT, letterSpacing: '-0.01em' }}>{m.name}</div>
                <div style={{ fontSize: 11, color: TEXT3, marginTop: 2 }}>{m.fullName}</div>
              </div>

              {blurbs.length > 0 && (
                <ul style={{ margin: 0, paddingLeft: 16, color: TEXT2, fontSize: 12, lineHeight: 1.6, flex: 1 }}>
                  {blurbs.map(b => <li key={b}>{b}</li>)}
                </ul>
              )}

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', borderTop: `1px solid ${BORDER}`, paddingTop: 12, marginTop: 4 }}>
                <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: available ? ACCENT : TEXT3, display: 'flex', alignItems: 'center', gap: 6 }}>
                  {available ? <>Select <ArrowRight size={12} /></> : 'Coming Soon'}
                </span>
              </div>
            </>
          );

          return available ? (
            <button key={m.id} onClick={() => navigate(m.path)} className="method-card" style={cardStyle}>
              {inner}
            </button>
          ) : (
            <div key={m.id} style={cardStyle} aria-disabled>{inner}</div>
          );
        })}
      </div>

      <style>{`
        .method-card:hover {
          border-color: ${BORDER2} !important;
          box-shadow: 0 10px 30px rgba(10,10,10,0.06);
          transform: translateY(-2px);
        }
      `}</style>
    </SelectPageShell>
  );
}
