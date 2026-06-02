import { el, rect, line, circle, ellipse, path, svgText, Attrs } from './svgUtil';
import { drawConditionMap } from './scanViz';
import { GEO, ORANGE, RED, REBAR, DELAMS, JOINTS, GIRDER } from './constants';

const { W, H, DECK_Y, SLAB_TOP, SLAB_BOT, REBAR_Y } = GEO;

function hyperbola(cx: number, cy: number, half: number, k: number): string {
  let d = '';
  for (let dx = -half; dx <= half; dx += 4) {
    const y = cy + k * dx * dx;
    d += (dx === -half ? 'M' : 'L') + ` ${(cx + dx).toFixed(1)} ${y.toFixed(1)}`;
  }
  return d;
}

function defs(): SVGElement {
  const d = el('defs');
  d.innerHTML = `
    <linearGradient id="expConcrete" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#17130f"/><stop offset="0.5" stop-color="#100d0a"/><stop offset="1" stop-color="#0c0a08"/>
    </linearGradient>
    <radialGradient id="expLensGrad" gradientUnits="objectBoundingBox" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#fff"/><stop offset="0.55" stop-color="#fff"/><stop offset="1" stop-color="#000"/>
    </radialGradient>
    <mask id="expLensMask" maskUnits="userSpaceOnUse" x="0" y="0" width="${W}" height="${H}">
      <rect x="0" y="0" width="${W}" height="${H}" fill="#000"/>
      <ellipse class="exp-cart-reveal" cx="640" cy="380" rx="250" ry="205" fill="url(#expLensGrad)"/>
      <circle class="exp-cursor-reveal" cx="-9999" cy="-9999" r="135" fill="url(#expLensGrad)"/>
    </mask>`;
  return d;
}

export function buildBackLayer(): SVGElement {
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'exp-dio' });
  for (let x = 0, m = 0; x < W; x += 200, m += 5) {
    svg.appendChild(line(x, 60, x, H - 60, { stroke: 'rgba(255,255,255,0.045)', 'stroke-width': 1 }));
    svg.appendChild(svgText(x + 8, 78, `${m}m`, { fill: 'rgba(240,237,232,0.18)', 'font-size': 13, 'letter-spacing': '0.12em', 'font-weight': 700 }));
  }
  return svg;
}

export function buildWorld(): SVGElement {
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'exp-dio' });
  svg.appendChild(defs());

  svg.appendChild(rect(0, SLAB_TOP, W, SLAB_BOT - SLAB_TOP, { fill: 'url(#expConcrete)', stroke: 'rgba(255,255,255,0.10)', 'stroke-width': 1 }));
  svg.appendChild(rect(0, SLAB_TOP, W, 3, { fill: 'rgba(255,255,255,0.18)' }));
  svg.appendChild(line(0, DECK_Y, W, DECK_Y, { stroke: 'rgba(255,255,255,0.22)', 'stroke-width': 2 }));
  for (let x = 0; x < W; x += 26) {
    svg.appendChild(line(x, DECK_Y, x + 14, SLAB_TOP, { stroke: 'rgba(255,255,255,0.04)', 'stroke-width': 1 }));
  }

  for (let x = GIRDER.start; x < W; x += GIRDER.step) {
    svg.appendChild(path(`M ${x - 70} ${SLAB_BOT} L ${x} ${H - 40} L ${x + 70} ${SLAB_BOT} Z`, { fill: 'none', stroke: 'rgba(255,255,255,0.09)', 'stroke-width': 2 }));
    svg.appendChild(line(x, SLAB_BOT, x, H - 40, { stroke: 'rgba(255,255,255,0.06)', 'stroke-width': 1 }));
  }
  svg.appendChild(line(0, H - 40, W, H - 40, { stroke: 'rgba(255,255,255,0.10)', 'stroke-width': 2 }));

  const gR = el('g', { mask: 'url(#expLensMask)' });
  const ghost = el('g', { opacity: 0.05 });

  for (let x = REBAR.start; x <= REBAR.end; x += REBAR.step) {
    gR.appendChild(path(hyperbola(x, REBAR_Y, 42, 0.05), { fill: 'none', stroke: 'rgba(232,96,28,0.55)', 'stroke-width': 2 }));
    const depth = (2.6 + ((x * 13) % 60) / 100).toFixed(2);
    gR.appendChild(circle(x, REBAR_Y, 5.5, { fill: '#0A0A0A', stroke: ORANGE, 'stroke-width': 2.5, class: 'exp-hot', 'data-tip': `REBAR · cover ${depth}″`, 'data-sub': 'Auto-picked horizon' }));
    ghost.appendChild(circle(x, REBAR_Y, 4, { fill: 'none', stroke: '#fff', 'stroke-width': 1.5 }));
  }
  for (let x = REBAR.start + 70; x <= REBAR.end; x += REBAR.step) {
    gR.appendChild(circle(x, REBAR_Y + 120, 4, { fill: 'none', stroke: 'rgba(255,255,255,0.32)', 'stroke-width': 1.5 }));
  }

  DELAMS.forEach((d) => {
    const g = el('g', { class: 'exp-hot', 'data-tip': `DELAMINATION · ${d.risk}`, 'data-sub': `${d.pct}% area · ASTM D6087` });
    const isHigh = d.risk === 'High';
    g.appendChild(ellipse(d.x, d.y, d.rx + 14, d.ry + 10, { fill: isHigh ? 'rgba(192,57,43,0.22)' : 'rgba(232,96,28,0.16)' }));
    g.appendChild(ellipse(d.x, d.y, d.rx, d.ry, { fill: 'none', stroke: isHigh ? RED : ORANGE, 'stroke-width': 2, 'stroke-dasharray': '4 4' }));
    g.appendChild(ellipse(d.x, d.y, d.rx * 0.5, d.ry * 0.5, { fill: isHigh ? 'rgba(192,57,43,0.5)' : 'rgba(232,96,28,0.45)' }));
    gR.appendChild(g);
  });

  svg.appendChild(ghost);
  svg.appendChild(gR);

  const joints: Attrs = { fill: 'rgba(255,255,255,0.0)', stroke: 'rgba(255,255,255,0.25)', 'stroke-width': 1.5, class: 'exp-hot', 'data-tip': 'EXPANSION JOINT', 'data-sub': 'GPS waypoint logged' };
  JOINTS.forEach((x) => svg.appendChild(rect(x - 3, DECK_Y - 18, 6, 18, joints)));

  return svg;
}

export function mountCanvases(world: HTMLElement): void {
  const cw = 560, ch = 300, cx = 3860, cy = 250;
  const wrap = document.createElement('div');
  wrap.className = 'exp-wpanel';
  wrap.style.cssText = `left:${cx}px;top:${cy}px;width:${cw}px;height:${ch}px`;
  const cv = document.createElement('canvas');
  cv.width = cw;
  cv.height = ch;
  wrap.appendChild(cv);
  const lab = document.createElement('div');
  lab.className = 'exp-wpanel-lab';
  lab.textContent = 'C-SCAN · CONDITION MAP — ASTM D6087';
  wrap.appendChild(lab);
  world.appendChild(wrap);
  drawConditionMap(cv, { seed: 7, cols: 70, rows: 24 });

  const rep = document.createElement('div');
  rep.className = 'exp-wreport';
  rep.style.cssText = 'left:4640px;top:300px;width:430px';
  rep.innerHTML = `
    <div class="exp-wr-eyebrow">SAME-DAY REPORT</div>
    <div class="exp-wr-title">Deck B440029</div>
    <div class="exp-wr-row"><span>Mean rebar cover</span><b>2.84″</b></div>
    <div class="exp-wr-row"><span>Deck thickness</span><b>8.20″</b></div>
    <div class="exp-wr-row"><span>Delaminated area</span><b class="risk">34%</b></div>
    <div class="exp-wr-row"><span>Standard</span><b>ASTM D6087</b></div>
    <div class="exp-wr-cta">Start Your Inspection</div>`;
  world.appendChild(rep);
}
