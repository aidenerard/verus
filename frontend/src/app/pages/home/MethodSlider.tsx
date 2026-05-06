import { C } from './tokens';

const ITEMS = [
  { name: 'GROUND-PENETRATING RADAR',               standard: 'ASTM D6087' },
  { name: 'MULTICHANNEL ANALYSIS OF SURFACE WAVES', standard: 'ASTM D7400' },
  { name: 'INFRARED THERMOGRAPHY',                  standard: 'ASTM D4788' },
];

function renderItems() {
  return ITEMS.map((item, i) => (
    <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.9em' }}>
      <span style={{ color: C.textLight, letterSpacing: '0.12em' }}>{item.name}</span>
      <span style={{ color: 'rgba(240,237,232,0.3)', fontSize: '0.8em', letterSpacing: '0.08em' }}>{item.standard}</span>
      <span style={{ color: C.orange, fontWeight: 700, fontSize: '0.7em' }}>◆</span>
    </span>
  ));
}

export default function MethodSlider() {
  return (
    <div style={{ background: C.black, backgroundImage: `repeating-linear-gradient(-55deg, transparent, transparent 18px, rgba(255,255,255,0.013) 18px, rgba(255,255,255,0.013) 19px)`, borderTop: `2px solid ${C.orange}`, borderBottom: `1px solid rgba(255,255,255,0.07)`, padding: '20px 0', overflow: 'hidden', whiteSpace: 'nowrap', userSelect: 'none' }}>
      <div style={{ display: 'inline-flex', gap: '0.9em', fontSize: 13, fontWeight: 600, animation: 'marquee 32s linear infinite', willChange: 'transform' }}>
        {renderItems()}{renderItems()}{renderItems()}{renderItems()}
      </div>
    </div>
  );
}
