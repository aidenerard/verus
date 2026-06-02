import { RefObject, useEffect } from 'react';
import { buildWorld, buildBackLayer, mountCanvases } from './diorama';
import { GEO, STOPS, PARALLAX, SWEEP_SECONDS, LENS_R } from './constants';

const { H, DECK_Y, REBAR_Y, SLAB_TOP, SLAB_BOT } = GEO;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

export function useDeckExperience(rootRef: RefObject<HTMLDivElement | null>): void {
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const q = <T extends Element>(sel: string) => root.querySelector(sel) as T;

    const stage = q<HTMLElement>('.exp-stage');
    const lFar = q<HTMLElement>('.exp-lfar');
    const lBack = q<HTMLElement>('.exp-lback');
    const lWorld = q<HTMLElement>('.exp-lworld');
    const cart = q<HTMLElement>('.exp-cart');
    const ruler = q<HTMLElement>('.exp-ruler');
    const ring = q<HTMLElement>('.exp-lens');
    const card = q<HTMLElement>('.exp-card');
    const playBtn = q<HTMLElement>('.exp-playbtn');
    const track = q<HTMLElement>('.exp-track');
    const barFill = q<HTMLElement>('.exp-barfill');
    const playhead = q<HTMLElement>('.exp-playhead');
    const tip = q<HTMLElement>('.exp-tip');

    lBack.appendChild(buildBackLayer());
    lWorld.appendChild(buildWorld());
    mountCanvases(lWorld);

    STOPS.forEach((st) => {
      const pin = document.createElement('div');
      pin.className = 'exp-pin';
      pin.style.left = st.x + 'px';
      pin.style.top = DECK_Y - 64 + 'px';
      pin.innerHTML = `<span class="exp-pin-dot"></span><span class="exp-pin-lab">${st.tag}</span>`;
      pin.addEventListener('click', () => tweenTo(st.x));
      lWorld.appendChild(pin);
    });

    const layers = [
      { el: lFar, f: PARALLAX.far },
      { el: lBack, f: PARALLAX.back },
      { el: lWorld, f: PARALLAX.world },
    ];

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let s = 1, vw = 0, vh = 0, travel = 0, camX = 0;
    let playing = !reduced, dragging = false;
    let target: number | null = null;
    let lastT = 0, activeStop = -1, rafId = 0;

    function setPlayBtn() {
      playBtn.classList.toggle('playing', playing);
    }

    function positionStatic() {
      const top = (vh - H * s) / 2;
      layers.forEach((L) => {
        L.el.style.transformOrigin = 'top left';
        (L.el.parentElement as HTMLElement).style.top = top + 'px';
      });
      ruler.style.top = top + SLAB_TOP * s + 'px';
      ruler.style.height = (SLAB_BOT - SLAB_TOP) * s + 'px';
    }

    function updateActive(centerX: number) {
      let best = 0, bd = 1e9;
      STOPS.forEach((st, i) => {
        const d = Math.abs(st.x - centerX);
        if (d < bd) { bd = d; best = i; }
      });
      if (best === activeStop) return;
      activeStop = best;
      const st = STOPS[best];
      card.classList.remove('in');
      void card.offsetWidth;
      card.classList.add('in');
      q<HTMLElement>('.exp-card-tag').textContent = st.tag;
      q<HTMLElement>('.exp-card-title').textContent = st.title;
      q<HTMLElement>('.exp-card-body').textContent = st.body;
      q<HTMLElement>('.exp-card-num').textContent = '0' + (best + 1);
      root.querySelectorAll('.exp-pill').forEach((p, i) => p.classList.toggle('on', i === best));
    }

    function render() {
      const camPx = camX * s;
      layers.forEach((L) => {
        L.el.style.transform = `translateX(${(-camPx * L.f).toFixed(2)}px) scale(${s})`;
      });
      const cartDesignX = camX + vw / s / 2;
      const top = (vh - H * s) / 2;
      cart.style.left = vw / 2 + 'px';
      cart.style.top = top + DECK_Y * s + 'px';
      cart.style.transform = `translate(-50%, -100%) scale(${s})`;
      const cr = root.querySelector('.exp-cart-reveal');
      if (cr) { cr.setAttribute('cx', cartDesignX.toFixed(1)); cr.setAttribute('cy', String(REBAR_Y + 40)); }
      const p = travel ? camX / travel : 0;
      barFill.style.width = p * 100 + '%';
      playhead.style.left = p * 100 + '%';
      updateActive(cartDesignX);
    }

    function layout() {
      vw = window.innerWidth;
      vh = window.innerHeight;
      s = vh / H;
      if (GEO.W * s < vw) s = vw / GEO.W;
      travel = Math.max(1, GEO.W - vw / s);
      camX = clamp(camX, 0, travel);
      positionStatic();
      render();
    }

    function tweenTo(designCenter: number) {
      target = clamp(designCenter - vw / s / 2, 0, travel);
      playing = false;
      setPlayBtn();
    }

    function loop(t: number) {
      const dt = Math.min(0.05, (t - lastT) / 1000 || 0);
      lastT = t;
      if (target !== null) {
        camX += (target - camX) * Math.min(1, dt * 6);
        if (Math.abs(target - camX) < 0.5) { camX = target; target = null; }
        render();
      } else if (playing && !dragging) {
        camX += (travel / SWEEP_SECONDS) * dt;
        if (camX >= travel) { camX = travel; playing = false; setPlayBtn(); }
        render();
      }
      rafId = requestAnimationFrame(loop);
    }

    function onMove(e: MouseEvent) {
      const h = (e.target as Element).closest('.exp-hot');
      if (h) {
        q<HTMLElement>('.exp-tip-t').textContent = h.getAttribute('data-tip') || '';
        q<HTMLElement>('.exp-tip-s').textContent = h.getAttribute('data-sub') || '';
        tip.style.left = e.clientX + 'px';
        tip.style.top = e.clientY + 'px';
        tip.classList.add('show');
      } else tip.classList.remove('show');
      const top = (vh - H * s) / 2;
      const dx = camX + e.clientX / s;
      const dy = (e.clientY - top) / s;
      const lensC = root.querySelector('.exp-cursor-reveal');
      if (lensC) { lensC.setAttribute('cx', dx.toFixed(1)); lensC.setAttribute('cy', dy.toFixed(1)); }
      ring.style.left = e.clientX + 'px';
      ring.style.top = e.clientY + 'px';
      ring.style.width = ring.style.height = LENS_R * 2 * s + 'px';
      ring.classList.add('on');
    }
    function onLeave() {
      tip.classList.remove('show');
      ring.classList.remove('on');
      const lensC = root.querySelector('.exp-cursor-reveal');
      if (lensC) { lensC.setAttribute('cx', '-9999'); lensC.setAttribute('cy', '-9999'); }
    }

    const setFromX = (clientX: number) => {
      const r = track.getBoundingClientRect();
      const p = clamp((clientX - r.left) / r.width, 0, 1);
      camX = p * travel;
      target = null;
      render();
    };
    const onDown = (e: PointerEvent) => {
      dragging = true; playing = false; setPlayBtn();
      track.setPointerCapture(e.pointerId);
      setFromX(e.clientX);
    };
    const onDrag = (e: PointerEvent) => { if (dragging) setFromX(e.clientX); };
    const onUp = (e: PointerEvent) => { dragging = false; track.releasePointerCapture(e.pointerId); };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      playing = false; setPlayBtn(); target = null;
      camX = clamp(camX + ((e.deltaY + e.deltaX) * 2.2) / s, 0, travel);
      render();
    };
    const onPlay = () => {
      if (camX >= travel - 1) camX = 0;
      playing = !playing; target = null; setPlayBtn();
    };

    const pillEls = Array.from(root.querySelectorAll('.exp-pill')) as HTMLElement[];
    const pillHandlers = pillEls.map((p, i) => {
      const fn = () => tweenTo(STOPS[i].x);
      p.addEventListener('click', fn);
      return { p, fn };
    });

    stage.addEventListener('mousemove', onMove);
    stage.addEventListener('mouseleave', onLeave);
    stage.addEventListener('wheel', onWheel, { passive: false });
    track.addEventListener('pointerdown', onDown);
    track.addEventListener('pointermove', onDrag);
    track.addEventListener('pointerup', onUp);
    playBtn.addEventListener('click', onPlay);
    window.addEventListener('resize', layout);

    setPlayBtn();
    layout();
    updateActive(camX + vw / s / 2);
    lastT = performance.now();
    rafId = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', layout);
      stage.removeEventListener('mousemove', onMove);
      stage.removeEventListener('mouseleave', onLeave);
      stage.removeEventListener('wheel', onWheel);
      track.removeEventListener('pointerdown', onDown);
      track.removeEventListener('pointermove', onDrag);
      track.removeEventListener('pointerup', onUp);
      playBtn.removeEventListener('click', onPlay);
      pillHandlers.forEach(({ p, fn }) => p.removeEventListener('click', fn));
      lBack.replaceChildren();
      lWorld.replaceChildren();
    };
  }, [rootRef]);
}
