export const GEO = {
  W: 5200,
  H: 800,
  DECK_Y: 250,
  SLAB_TOP: 274,
  SLAB_BOT: 600,
  REBAR_Y: 340,
} as const;

export const ORANGE = '#E8601C';
export const RED = '#C0392B';

export const TEXT_LIGHT = '#F0EDE8';
export const TEXT_MUTED = 'rgba(240,237,232,0.6)';

export interface Stop {
  x: number;
  tag: string;
  title: string;
  body: string;
}

export const STOPS: Stop[] = [
  { x: 640, tag: 'Survey', title: 'Roll. Scan. Capture.', body: 'A GPR cart rolls the deck, firing radar pulses on a survey grid. No lane closures for coring — just a continuous, GPS-referenced pass.' },
  { x: 1850, tag: 'Subsurface', title: 'See through concrete.', body: 'Every reflection becomes a hyperbola. Verus picks the rebar horizon automatically and maps cover depth across the whole deck.' },
  { x: 3010, tag: 'Delamination', title: 'Find what’s failing.', body: 'Corrosion-driven delamination shows up as bright, shallow returns. The model flags and grades each zone — before it becomes a pothole.' },
  { x: 4140, tag: 'Condition Map', title: 'Every defect, mapped.', body: 'Signals collapse into a false-color C-scan: a same-day, ASTM D6087 condition map of the entire deck, sound to high-risk.' },
  { x: 4880, tag: 'Report', title: 'From raw data to report.', body: 'Export a standards-compliant report the same day as the inspection. Share it, archive it, schedule the repair.' },
];

export interface Delam {
  x: number;
  y: number;
  rx: number;
  ry: number;
  risk: 'High' | 'Elevated' | 'Monitor';
  pct: number;
}

export const DELAMS: Delam[] = [
  { x: 2880, y: GEO.REBAR_Y - 6, rx: 95, ry: 26, risk: 'High', pct: 34 },
  { x: 3120, y: GEO.REBAR_Y + 10, rx: 64, ry: 18, risk: 'Elevated', pct: 22 },
  { x: 3300, y: GEO.REBAR_Y - 2, rx: 50, ry: 15, risk: 'Monitor', pct: 12 },
];

export const REBAR = { start: 420, end: 3620, step: 142 } as const;
export const JOINTS = [520, 1560, 2600, 3640];
export const GIRDER = { start: 380, step: 520 } as const;
export const RULER_TICKS = ['0″', '2″', '4″', '6″'];

export const PARALLAX = { far: 0.35, back: 0.7, world: 1 } as const;
export const SWEEP_SECONDS = 46;
export const LENS_R = 135;

export const COND: Array<[number, number, number]> = [
  [39, 174, 96],
  [241, 196, 15],
  [230, 126, 34],
  [192, 57, 43],
];
