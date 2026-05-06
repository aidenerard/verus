export type RGB = [number, number, number];

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function lerpRGB(c1: RGB, c2: RGB, t: number): RGB {
  return [Math.round(lerp(c1[0], c2[0], t)), Math.round(lerp(c1[1], c2[1], t)), Math.round(lerp(c1[2], c2[2], t))];
}

export function applyStops(t: number, stops: RGB[]): RGB {
  const n  = stops.length - 1;
  const s  = Math.max(0, Math.min(1, t)) * n;
  const lo = Math.min(n - 1, Math.floor(s));
  return lerpRGB(stops[lo], stops[lo + 1], s - lo);
}

// red → orange → yellow → green → blue  (deteriorated → sound)
export const COND_STOPS: RGB[] = [[192,57,43],[230,126,34],[241,196,15],[39,174,96],[41,128,185]];
// blue → green → yellow → red  (shallow → deep)
export const DEPTH_STOPS: RGB[] = [[37,99,235],[16,185,129],[251,191,36],[239,68,68]];

export function decodeF32(b64: string): Float32Array {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Float32Array(buf.buffer);
}

const BG_FILL: RGB = [240, 239, 236];

export function renderConditionToCanvas(
  canvas: HTMLCanvasElement,
  data: Float32Array, rows: number, cols: number,
  threshold: number,
): void {
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i++) {
    const p   = data[i];
    const idx = i * 4;
    if (isNaN(p)) {
      [img.data[idx], img.data[idx+1], img.data[idx+2]] = BG_FILL;
    } else {
      const d = p <= threshold
        ? 0.5 * p / threshold
        : 0.5 + 0.5 * (p - threshold) / Math.max(0.001, 1 - threshold);
      const [r,g,b] = applyStops(d, COND_STOPS);
      img.data[idx] = r; img.data[idx+1] = g; img.data[idx+2] = b;
    }
    img.data[idx+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function renderDepthToCanvas(
  canvas: HTMLCanvasElement,
  twtData: Float32Array, rows: number, cols: number,
  er: number,
): void {
  canvas.width = cols; canvas.height = rows;
  const ctx      = canvas.getContext('2d')!;
  const img      = ctx.createImageData(cols, rows);
  const velocity = 0.3 / Math.sqrt(er);
  const IN_PER_M = 39.3701;
  for (let i = 0; i < rows * cols; i++) {
    const twt = twtData[i];
    const idx = i * 4;
    if (isNaN(twt)) {
      [img.data[idx], img.data[idx+1], img.data[idx+2]] = BG_FILL;
    } else {
      const depth_in = velocity * twt / 2 * IN_PER_M;
      const t        = Math.max(0, Math.min(1, (depth_in - 1.0) / 3.0));
      const [r,g,b]  = applyStops(t, DEPTH_STOPS);
      img.data[idx] = r; img.data[idx+1] = g; img.data[idx+2] = b;
    }
    img.data[idx+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function renderAmpToCanvas(
  canvas: HTMLCanvasElement,
  ampData: Float32Array, rows: number, cols: number,
  ampMin: number, ampMax: number,
): void {
  canvas.width = cols; canvas.height = rows;
  const ctx   = canvas.getContext('2d')!;
  const img   = ctx.createImageData(cols, rows);
  const range = Math.max(0.001, ampMax - ampMin);
  for (let i = 0; i < rows * cols; i++) {
    const a   = ampData[i];
    const idx = i * 4;
    if (isNaN(a)) {
      [img.data[idx], img.data[idx+1], img.data[idx+2]] = BG_FILL;
    } else {
      const t       = Math.max(0, Math.min(1, (a - ampMin) / range));
      const [r,g,b] = applyStops(t, COND_STOPS);
      img.data[idx] = r; img.data[idx+1] = g; img.data[idx+2] = b;
    }
    img.data[idx+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
