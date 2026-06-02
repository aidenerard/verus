export const EXPERIENCE_CSS = `
  .exp-root { position: fixed; inset: 0; overflow: hidden;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
  .exp-stage { position: fixed; inset: 0; overflow: hidden; cursor: crosshair;
    background:
      radial-gradient(ellipse 120% 90% at 50% 40%, rgba(232,96,28,0.05), transparent 60%),
      radial-gradient(ellipse 100% 80% at 50% 50%, transparent 25%, rgba(10,10,10,0.85) 100%),
      #0A0A0A; }

  .exp-lens { position: fixed; z-index: 12; pointer-events: none; transform: translate(-50%, -50%);
    border: 1.5px solid rgba(232,96,28,0.6); border-radius: 50%; opacity: 0; transition: opacity 0.25s;
    box-shadow: inset 0 0 40px rgba(232,96,28,0.16), 0 0 14px rgba(232,96,28,0.12); }
  .exp-lens.on { opacity: 1; }
  .exp-lens::after { content: ''; position: absolute; left: 50%; top: 50%; width: 5px; height: 5px;
    border-radius: 50%; background: #E8601C; transform: translate(-50%, -50%);
    box-shadow: 0 0 8px rgba(232,96,28,0.9); }

  .exp-layer-wrap { position: absolute; left: 0; right: 0; height: 0; }
  .exp-layer { position: absolute; left: 0; top: 0; will-change: transform; }
  .exp-lfar { width: 6200px; height: 800px;
    background-image: radial-gradient(circle, rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 34px 34px; }
  .exp-dio { display: block; overflow: visible; }

  .exp-cart { position: absolute; z-index: 6; pointer-events: none; transform-origin: bottom center; }

  .exp-pin { position: absolute; z-index: 5; transform: translate(-50%, 0); cursor: pointer;
    display: flex; flex-direction: column; align-items: center; gap: 6px; }
  .exp-pin-dot { width: 13px; height: 13px; border-radius: 50%; background: #E8601C;
    box-shadow: 0 0 0 5px rgba(232,96,28,0.18), 0 0 14px rgba(232,96,28,0.6); position: relative; }
  .exp-pin-dot::after { content: ''; position: absolute; inset: -5px; border-radius: 50%;
    border: 1px solid rgba(232,96,28,0.5); animation: exp-pinpulse 2.4s ease-out infinite; }
  @keyframes exp-pinpulse { 0% { transform: scale(0.7); opacity: 0.9; } 100% { transform: scale(2.4); opacity: 0; } }
  .exp-pin-lab { font-size: 11px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
    color: #F0EDE8; white-space: nowrap; background: rgba(10,10,10,0.6); padding: 3px 8px; border: 1px solid rgba(255,255,255,0.12); }
  .exp-hot { cursor: help; }

  .exp-wpanel { position: absolute; z-index: 5; border: 1px solid rgba(255,255,255,0.14);
    background: rgba(10,10,10,0.5); padding: 30px 0 0; }
  .exp-wpanel canvas { display: block; width: 100%; height: calc(100% - 30px); }
  .exp-wpanel-lab { position: absolute; top: 0; left: 0; right: 0; height: 30px; display: flex; align-items: center;
    padding: 0 12px; font-size: 11px; font-weight: 700; letter-spacing: 0.12em; color: rgba(240,237,232,0.55);
    border-bottom: 1px solid rgba(255,255,255,0.1); }
  .exp-wreport { position: absolute; z-index: 5; background: rgba(20,16,13,0.66); border: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(4px); padding: 26px 28px; }
  .exp-wr-eyebrow { font-size: 12px; font-weight: 800; letter-spacing: 0.16em; color: #E8601C; }
  .exp-wr-title { font-size: 30px; font-weight: 800; color: #fff; margin: 8px 0 18px; letter-spacing: -0.02em; }
  .exp-wr-row { display: flex; justify-content: space-between; align-items: baseline; padding: 9px 0;
    border-top: 1px solid rgba(255,255,255,0.08); font-size: 15px; color: rgba(240,237,232,0.6); }
  .exp-wr-row b { color: #F0EDE8; font-weight: 700; font-variant-numeric: tabular-nums; }
  .exp-wr-row b.risk { color: #E8601C; }
  .exp-wr-cta { margin-top: 22px; text-align: center; background: #E8601C; color: #fff; font-weight: 800;
    font-size: 14px; letter-spacing: 0.08em; text-transform: uppercase; padding: 14px; }

  .exp-ruler { position: fixed; left: 22px; z-index: 10; width: 44px; pointer-events: none;
    border-left: 1px solid rgba(255,255,255,0.18); display: flex; flex-direction: column; justify-content: space-between; }
  .exp-ruler .tick { font-size: 10px; font-weight: 700; letter-spacing: 0.06em; color: rgba(240,237,232,0.4);
    padding-left: 6px; position: relative; }
  .exp-ruler .tick::before { content: ''; position: absolute; left: -1px; top: 50%; width: 8px; height: 1px; background: rgba(255,255,255,0.25); }

  .exp-header { position: fixed; top: 0; left: 0; right: 0; z-index: 20; height: 60px; display: flex; align-items: center;
    justify-content: space-between; padding: 0 26px; }
  .exp-htag { font-size: 10px; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: #E8601C;
    border: 1px solid rgba(232,96,28,0.4); padding: 5px 10px; white-space: nowrap; }
  .exp-hback { font-size: 12px; font-weight: 600; color: rgba(240,237,232,0.6); text-decoration: none; letter-spacing: 0.03em; }
  .exp-hback:hover { color: #F0EDE8; }

  .exp-card { position: fixed; left: 40px; bottom: 122px; z-index: 15; width: 380px;
    background: rgba(16,12,9,0.72); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.12);
    padding: 26px 28px 28px; box-shadow: 0 18px 50px rgba(0,0,0,0.5); }
  .exp-card-num { position: absolute; top: 18px; right: 22px; font-size: 12px; font-weight: 800; color: rgba(240,237,232,0.3);
    letter-spacing: 0.1em; font-variant-numeric: tabular-nums; }
  .exp-card-tag { font-size: 11px; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; color: #E8601C;
    display: inline-flex; align-items: center; gap: 10px; }
  .exp-card-tag::before { content: ''; width: 22px; height: 2px; background: #E8601C; display: inline-block; }
  .exp-card-title { font-size: 30px; font-weight: 800; color: #fff; letter-spacing: -0.025em; line-height: 1.08; margin: 16px 0 12px; }
  .exp-card-body { font-size: 14px; line-height: 1.7; color: rgba(240,237,232,0.62); margin: 0; }
  .exp-card.in { animation: exp-cardIn 0.5s cubic-bezier(0.2,0.7,0.2,1); }
  @keyframes exp-cardIn { from { transform: translateY(14px); } to { transform: none; } }

  .exp-pills { position: fixed; top: 70px; left: 50%; transform: translateX(-50%); z-index: 20;
    display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; max-width: 92vw; }
  .exp-pill { display: inline-flex; align-items: center; gap: 7px; padding: 7px 13px; cursor: pointer;
    font-family: inherit; font-size: 10px; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
    color: rgba(240,237,232,0.6); background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    transition: color 0.2s; white-space: nowrap; }
  .exp-pill i { font-style: normal; font-size: 9px; font-weight: 800; color: rgba(240,237,232,0.35); }
  .exp-pill:hover { color: #F0EDE8; border-color: rgba(255,255,255,0.25); }
  .exp-pill.on { color: #fff; background: #E8601C; border-color: #E8601C; }
  .exp-pill.on i { color: rgba(255,255,255,0.7); }

  .exp-transport { position: fixed; left: 0; right: 0; bottom: 0; z-index: 20; height: 86px;
    display: flex; align-items: center; gap: 20px; padding: 0 40px;
    background: linear-gradient(to top, rgba(10,10,10,0.92), rgba(10,10,10,0)); }
  .exp-playbtn { flex-shrink: 0; width: 46px; height: 46px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.05); color: #F0EDE8; cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background 0.2s, border-color 0.2s; }
  .exp-playbtn:hover { background: #E8601C; border-color: #E8601C; color: #fff; }
  .exp-playbtn .exp-icon-pause { display: none; }
  .exp-playbtn .exp-icon-play { display: flex; }
  .exp-playbtn.playing .exp-icon-pause { display: flex; }
  .exp-playbtn.playing .exp-icon-play { display: none; }
  .exp-track { position: relative; flex: 1; height: 40px; display: flex; align-items: center; cursor: pointer; }
  .exp-bar { position: absolute; left: 0; right: 0; height: 3px; background: rgba(255,255,255,0.12); }
  .exp-barfill { position: absolute; left: 0; height: 3px; background: #E8601C; width: 0; }
  .exp-playhead { position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; background: #fff;
    transform: translate(-50%, -50%); box-shadow: 0 0 0 4px rgba(232,96,28,0.3); pointer-events: none; }
  .exp-tickrow { position: absolute; left: 0; right: 0; top: 50%; transform: translateY(-50%); pointer-events: none; }
  .exp-stick { position: absolute; width: 2px; height: 12px; background: rgba(255,255,255,0.25); transform: translate(-50%,-50%); top: 50%; }
  .exp-hint { flex-shrink: 0; font-size: 11px; color: rgba(240,237,232,0.4); letter-spacing: 0.04em; white-space: nowrap; }

  .exp-tip { position: fixed; z-index: 40; pointer-events: none; transform: translate(14px, -50%);
    background: #0A0A0A; border: 1px solid #E8601C; padding: 8px 12px; opacity: 0; transition: opacity 0.12s;
    max-width: 240px; }
  .exp-tip.show { opacity: 1; }
  .exp-tip-t { font-size: 11px; font-weight: 800; letter-spacing: 0.08em; color: #E8601C; text-transform: uppercase; }
  .exp-tip-s { font-size: 11px; color: rgba(240,237,232,0.6); margin-top: 2px; }

  @media (max-width: 720px) {
    .exp-card { left: 16px; right: 16px; width: auto; bottom: 110px; padding: 18px 20px; }
    .exp-card-title { font-size: 24px; }
    .exp-pills { gap: 5px; } .exp-pill { padding: 6px 10px; font-size: 10px; }
    .exp-hint { display: none; }
  }
  @media (prefers-reduced-motion: reduce) { .exp-pin-dot::after { animation: none; } }
`;
