export function estimateAnalysisSeconds(files: File[], manufacturer: string): number {
  let total = 0;
  for (const file of files) {
    const sizeMB = file.size / (1024 * 1024);
    const ext = file.name.split('.').pop()?.toLowerCase() ?? 'csv';
    const formatBase: Record<string, number> = {
      dzt: 8, rd3: 10, rd7: 12, dt1: 7, segy: 9, sgy: 9, csv: 3,
    };
    const base = formatBase[ext] ?? 8;
    const sizeMultiplier = Math.max(1, sizeMB / 2);
    const manufacturerMultiplier: Record<string, number> = {
      gssi: 1.0, mala: 1.2, sensors_software: 1.1, ids: 1.3, impulseradar: 1.2, segy: 1.0, csv: 0.8,
    };
    const mfgMult = manufacturerMultiplier[manufacturer] ?? 1.0;
    total += base * sizeMultiplier * mfgMult;
  }
  const overhead = 20;
  const inferenceTime = files.length * 3;
  return Math.round(total + overhead + inferenceTime);
}

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
