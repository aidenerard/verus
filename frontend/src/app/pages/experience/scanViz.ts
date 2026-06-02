import { COND } from './constants';

function mulberry32(a: number) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function rampColor(v: number): string {
  const x = Math.max(0, Math.min(0.9999, v)) * 3;
  const i = Math.floor(x);
  const f = x - i;
  const c0 = COND[i];
  const c1 = COND[Math.min(3, i + 1)];
  return `rgb(${Math.round(lerp(c0[0], c1[0], f))},${Math.round(lerp(c0[1], c1[1], f))},${Math.round(lerp(c0[2], c1[2], f))})`;
}

interface ConditionOpts {
  seed?: number;
  cols?: number;
  rows?: number;
}

export function drawConditionMap(canvas: HTMLCanvasElement, opts: ConditionOpts = {}): void {
  const seed = opts.seed ?? 7;
  const cols = opts.cols ?? 48;
  const rows = opts.rows ?? 16;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const cw = W / cols;
  const ch = H / rows;
  const rnd = mulberry32(seed);

  const blobs: Array<{ x: number; y: number; r: number; s: number }> = [];
  const nb = 3 + Math.floor(rnd() * 3);
  for (let i = 0; i < nb; i++) {
    blobs.push({ x: rnd() * cols, y: rnd() * rows, r: 2 + rnd() * 5, s: 0.6 + rnd() * 0.4 });
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let v = 0.04 + rnd() * 0.1;
      for (const b of blobs) {
        const d = Math.hypot(c - b.x, r - b.y);
        v += b.s * Math.exp(-(d * d) / (2 * b.r * b.r));
      }
      ctx.fillStyle = rampColor(v);
      ctx.fillRect(Math.floor(c * cw), Math.floor(r * ch), Math.ceil(cw) + 1, Math.ceil(ch) + 1);
    }
  }
}
