import { X } from 'lucide-react';
import type { InspectionModule } from './types';

interface Props {
  module: InspectionModule;
  onClose: () => void;
}

export default function ComingSoonModal({ module, onClose }: Props) {
  const Icon = module.icon;
  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(10,10,10,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }} onClick={onClose}>
      <div style={{ background: '#FFFFFF', border: '2px solid #E2DED9', maxWidth: 440, width: '100%' }} onClick={e => e.stopPropagation()}>
        <div style={{ padding: '16px 24px', borderBottom: '2px solid #E2DED9', background: '#F5F3EF', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Icon style={{ color: '#0A0A0A' }} className="w-4 h-4" />
            <span style={{ fontSize: 13, fontWeight: 700, color: '#0A0A0A', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{module.fullName}</span>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 4 }}>
            <X className="w-4 h-4" style={{ color: '#7A7470' }} />
          </button>
        </div>
        <div style={{ padding: '28px 24px' }}>
          <div style={{ display: 'inline-block', background: '#F5F3EF', color: '#0A0A0A', padding: '4px 10px', marginBottom: 16, fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            In Development
          </div>
          <p style={{ fontSize: 14, color: '#0A0A0A', lineHeight: 1.6, margin: '0 0 16px' }}>
            This module is coming soon. Verus is training a dedicated AI model for <strong>{module.fullName}</strong> analysis.
          </p>
          <p style={{ fontSize: 13, color: '#7A7470', lineHeight: 1.6, margin: '0 0 24px' }}>
            {module.description} You'll be notified when it launches.
          </p>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: '#B0A9A4' }}>Standard: {module.standard}</span>
            <button onClick={onClose} style={{ padding: '9px 22px', background: '#E8601C', color: '#FFFFFF', border: '2px solid #E8601C', fontWeight: 700, fontSize: 11, letterSpacing: '0.07em', textTransform: 'uppercase', cursor: 'pointer', fontFamily: 'Inter, sans-serif' }}>
              Got It
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
