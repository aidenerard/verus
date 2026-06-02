const NS = 'http://www.w3.org/2000/svg';

export type Attrs = Record<string, string | number>;

export function el(tag: string, attrs: Attrs = {}): SVGElement {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  return e;
}

export const rect = (x: number, y: number, w: number, h: number, a: Attrs = {}) =>
  el('rect', { x, y, width: w, height: h, ...a });

export const line = (x1: number, y1: number, x2: number, y2: number, a: Attrs = {}) =>
  el('line', { x1, y1, x2, y2, ...a });

export const circle = (cx: number, cy: number, r: number, a: Attrs = {}) =>
  el('circle', { cx, cy, r, ...a });

export const ellipse = (cx: number, cy: number, rx: number, ry: number, a: Attrs = {}) =>
  el('ellipse', { cx, cy, rx, ry, ...a });

export const path = (d: string, a: Attrs = {}) => el('path', { d, ...a });

export function svgText(x: number, y: number, str: string, a: Attrs = {}): SVGElement {
  const t = el('text', { x, y, ...a });
  t.textContent = str;
  return t;
}
