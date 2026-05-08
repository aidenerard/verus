export type RGB = [number, number, number];

// ── Rebar depth: discrete 0.5-inch band color scale ─────────────────────────
export const DEPTH_BANDS: { max: number; color: RGB }[] = [
  { max: 3.0, color: [255, 255, 150] },
  { max: 3.5, color: [255, 240,  50] },
  { max: 4.0, color: [255, 210,   0] },
  { max: 4.5, color: [255, 180,   0] },
  { max: 5.0, color: [255, 140,   0] },
  { max: 5.5, color: [240, 100,   0] },
  { max: 6.0, color: [220,  60,   0] },
  { max: 6.5, color: [200,  30,   0] },
  { max: 7.0, color: [170,   0,   0] },
  { max: 7.5, color: [130,   0,   0] },
  { max: 999, color: [ 90,   0,   0] },
];

export function getDepthColor(depthInches: number): RGB {
  for (const band of DEPTH_BANDS) {
    if (depthInches <= band.max) return band.color;
  }
  return [90, 0, 0];
}

function drawRebarLines(
  ctx: CanvasRenderingContext2D,
  peakGrid: (number | null)[][],
  W: number,
  H: number,
): void {
  const rows = peakGrid.length;
  const cols = peakGrid[0]?.length ?? 0;
  if (!rows || !cols) return;

  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 1.0;
  ctx.globalAlpha = 0.65;

  for (let row = 0; row < rows; row++) {
    const stripMid = ((row + 0.5) / rows) * H;
    const stripH   = H / rows;
    ctx.beginPath();
    let started = false;
    for (let col = 0; col < cols; col++) {
      const peak = peakGrid[row]?.[col];
      if (peak == null || isNaN(peak)) continue;
      const x = ((col + 0.5) / cols) * W;
      const y = stripMid + (peak / 512 - 0.5) * stripH * 0.8;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    if (started) ctx.stroke();
  }
  ctx.globalAlpha = 1.0;
}

export function renderRebarDepthCanvas(
  canvas: HTMLCanvasElement,
  depthGrid: (number | null)[][],
  peakGrid: (number | null)[][] | null,
): void {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (W === 0 || H === 0) return;
  canvas.width  = W;
  canvas.height = H;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const rows = depthGrid.length;
  const cols = depthGrid[0]?.length ?? 0;
  if (!rows || !cols) return;

  const imageData = ctx.createImageData(W, H);
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const gridCol = Math.min(cols - 1, Math.floor((px / W) * cols));
      const gridRow = Math.min(rows - 1, Math.floor((py / H) * rows));
      const depth   = depthGrid[gridRow]?.[gridCol];
      const [r, g, b] = depth != null && !isNaN(depth) ? getDepthColor(depth) : [200, 200, 200] as RGB;
      const idx = (py * W + px) * 4;
      imageData.data[idx]     = r;
      imageData.data[idx + 1] = g;
      imageData.data[idx + 2] = b;
      imageData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);

  if (peakGrid && peakGrid.length > 0) drawRebarLines(ctx, peakGrid, W, H);
}

export function renderDepthColorbar(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  canvas.width  = canvas.clientWidth;
  canvas.height = canvas.clientHeight || 44;
  if (canvas.width === 0) return;

  const bands  = DEPTH_BANDS.slice(0, -1);
  const blockW = canvas.width / bands.length;
  const barH   = 20;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  bands.forEach((band, i) => {
    const [r, g, b] = band.color;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(Math.floor(i * blockW), 0, Math.ceil(blockW) + 1, barH);

    ctx.fillStyle  = '#222';
    ctx.font       = '9px Inter, sans-serif';
    ctx.textAlign  = 'center';
    const prev     = i === 0 ? 0 : DEPTH_BANDS[i - 1].max;
    const label    = i === 0 ? `≤${band.max}"` : `${prev}–${band.max}"`;
    ctx.fillText(label, i * blockW + blockW / 2, barH + 11);
  });

  ctx.fillStyle = '#222';
  ctx.font      = '9px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`>${DEPTH_BANDS[bands.length - 1].max}"`, bands.length * blockW - 28, barH + 11);
}

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

// ── JSON-array canvas renderers ───────────────────────────────────────────────

type Grid = (number | null)[][];

function spectralColor(t: number): RGB {
  const stops: RGB[] = [
    [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 128, 0], [255, 0, 0],
  ];
  const n = stops.length - 1;
  const s = Math.max(0, Math.min(1, t)) * n;
  const lo = Math.min(n - 1, Math.floor(s));
  return lerpRGB(stops[lo], stops[lo + 1], s - lo);
}

function rebarColor(t: number): RGB {
  const stops: RGB[] = [
    [255, 255, 100], [255, 200, 0], [255, 140, 0], [220, 60, 0], [180, 0, 0],
  ];
  const n = stops.length - 1;
  const s = Math.max(0, Math.min(1, t)) * n;
  const lo = Math.min(n - 1, Math.floor(s));
  return lerpRGB(stops[lo], stops[lo + 1], s - lo);
}

function drawGrid(
  canvas: HTMLCanvasElement,
  data: Grid,
  colorFn: (t: number) => RGB,
  toT: (v: number) => number,
): void {
  if (!data.length || !data[0].length) return;
  const rows = data.length, cols = data[0].length;
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = data[r][c];
      const idx = (r * cols + c) * 4;
      if (v === null || v !== v) {
        img.data[idx] = BG_FILL[0]; img.data[idx+1] = BG_FILL[1]; img.data[idx+2] = BG_FILL[2];
      } else {
        const [R, G, B] = colorFn(Math.max(0, Math.min(1, toT(v))));
        img.data[idx] = R; img.data[idx+1] = G; img.data[idx+2] = B;
      }
      img.data[idx+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function renderConditionGrid(
  canvas: HTMLCanvasElement, data: Grid, _threshold: number,
): void {
  drawGrid(canvas, data, spectralColor, v => 1 - v);
}

export function renderRebarGrid(canvas: HTMLCanvasElement, data: Grid): void {
  let min = Infinity, max = -Infinity;
  for (const row of data) for (const v of row) {
    if (v !== null && v === v) { if (v < min) min = v; if (v > max) max = v; }
  }
  if (min === Infinity) { min = 1; max = 5; }
  const range = Math.max(0.001, max - min);
  drawGrid(canvas, data, rebarColor, v => (v - min) / range);
}

export function renderAmpGrid(
  canvas: HTMLCanvasElement, data: Grid, ampMin: number, ampMax: number,
): void {
  const range = Math.max(0.001, ampMax - ampMin);
  drawGrid(canvas, data, t => applyStops(t, COND_STOPS), v => (v - ampMin) / range);
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
