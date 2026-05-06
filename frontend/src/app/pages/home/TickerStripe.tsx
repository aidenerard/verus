import { C } from './tokens';

const ITEMS = [
  'INFRASTRUCTURE INSPECTION', 'AI-POWERED ANALYSIS',
  'SAME-DAY REPORTS', 'MULTI-SENSOR PLATFORM', 'AUTOMATED REPORTING',
];

function renderItems() {
  return ITEMS.map((item, i) => (
    <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '1em' }}>
      <span style={{ color: C.textLight, letterSpacing: '0.13em' }}>{item}</span>
      <span style={{ color: C.orange, fontWeight: 700, fontSize: '0.7em' }}>◆</span>
    </span>
  ));
}

export default function TickerStripe() {
  return (
    <div style={{ background: C.black, backgroundImage: `repeating-linear-gradient(-55deg, transparent, transparent 18px, rgba(255,255,255,0.013) 18px, rgba(255,255,255,0.013) 19px)`, borderTop: `2px solid ${C.orange}`, borderBottom: `1px solid rgba(255,255,255,0.07)`, padding: '20px 0', overflow: 'hidden', whiteSpace: 'nowrap', userSelect: 'none' }}>
      <div style={{ display: 'inline-flex', gap: '1em', fontSize: 13, fontWeight: 600, animation: 'marquee 38s linear infinite', willChange: 'transform' }}>
        {renderItems()}{renderItems()}{renderItems()}{renderItems()}
      </div>
    </div>
  );
}
