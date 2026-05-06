export function delamColor(pct: number): string {
  const t = Math.min(1, Math.max(0, pct / 100));
  if (t <= 0.5) {
    const s = t * 2;
    const r = Math.round(0x22 + (0xf5 - 0x22) * s);
    const g = Math.round(0xc5 + (0x9e - 0xc5) * s);
    const b = Math.round(0x5e + (0x0b - 0x5e) * s);
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
  }
  const s = (t - 0.5) * 2;
  const r = Math.round(0xf5 + (0xef - 0xf5) * s);
  const g = Math.round(0x9e + (0x44 - 0x9e) * s);
  const b = Math.round(0x0b + (0x44 - 0x0b) * s);
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

export function badgeColor(good: boolean, ok: boolean): string {
  return good ? '#22c55e' : ok ? '#f59e0b' : '#ef4444';
}
