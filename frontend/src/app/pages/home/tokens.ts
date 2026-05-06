export const C = {
  black:    '#0A0A0A',
  orange:   '#E8601C',
  orangeDk: '#D4521A',
  offWhite: '#F5F3EF',
  card:     '#FDFCFA',
  textDark: '#0A0A0A',
  textLight:'#F0EDE8',
  textMuted:'rgba(240,237,232,0.55)',
  textGray: '#7A7470',
  border:   '#E2DED9',
  borderDk: 'rgba(255,255,255,0.08)',
} as const;

export const PAGE_CSS = `
  @keyframes drift {
    0%   { transform: translate(0,0); }
    33%  { transform: translate(12px,-8px); }
    66%  { transform: translate(-6px,10px); }
    100% { transform: translate(0,0); }
  }
  @keyframes marquee {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
  }
  @keyframes pulseBar {
    0%,100% { transform: scaleY(0.85); opacity: 0.12; }
    50%     { transform: scaleY(1.2);  opacity: 0.55; }
  }
  @keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
  }
  .vr-nav-link {
    position: relative; text-decoration: none;
    color: rgba(240,237,232,0.68); font-size: 13px;
    font-weight: 500; letter-spacing: 0.01em; transition: color 0.2s;
  }
  .vr-nav-link::after {
    content:''; position:absolute; bottom:-3px; left:0;
    width:0; height:1px; background:#E8601C; transition: width 0.25s ease;
  }
  .vr-nav-link:hover { color:#F0EDE8; }
  .vr-nav-link:hover::after { width:100%; }
  .vr-nav-login {
    font-size:12px; font-weight:600; color:rgba(240,237,232,0.68);
    text-decoration:none; letter-spacing:0.03em; transition:color 0.2s;
  }
  .vr-nav-login:hover { color:#F0EDE8; }
  .vr-nav-signup {
    display:inline-block; padding:8px 20px; background:#E8601C; color:#fff;
    font-size:11px; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;
    text-decoration:none; border:none; transition: background 0.18s, transform 0.18s;
  }
  .vr-nav-signup:hover { background:#D4521A; transform:translateY(-1px); }
  .vr-why-row:hover { background: rgba(232,96,28,0.03); }
  .vr-why-row:hover .vr-why-num { color: #D4521A; transform: scale(1.08); }
  .vr-why-num { display: inline-block; transition: color 0.2s, transform 0.2s; }
  .vr-method-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 12px 18px 0; border-bottom: 1px solid #E2DED9;
    border-left: 2px solid transparent; padding-left: 14px; gap: 16px;
    transition: border-left-color 0.22s, background 0.22s, padding-left 0.22s; cursor: default;
  }
  .vr-method-row:hover { border-left-color: #E8601C; background: rgba(232,96,28,0.035); padding-left: 20px; }
  .vr-method-row-active { border-left-color: #E8601C; }
  .vr-platform-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: #E8601C; margin: 0 0 14px; position: relative; display: inline-block;
  }
  .vr-platform-label::after {
    content: ''; position: absolute; bottom: -6px; left: 0;
    width: 24px; height: 2px; background: #E8601C; box-shadow: 0 0 8px rgba(232,96,28,0.6);
  }
  @keyframes marqueeReverse {
    from { transform: translateX(-50%); }
    to   { transform: translateX(0); }
  }
  .vr-btn-primary {
    display:inline-block; padding:13px 32px; background:#E8601C; color:#FFFFFF;
    font-weight:700; font-size:12px; letter-spacing:0.09em; text-transform:uppercase;
    text-decoration:none; border:2px solid #E8601C; transition: background 0.2s, transform 0.2s;
  }
  .vr-btn-primary:hover { background:#D4521A; border-color:#D4521A; transform:translateY(-2px); }
  .vr-btn-ghost {
    display:inline-block; padding:13px 32px; background:transparent; color:#F0EDE8;
    font-weight:700; font-size:12px; letter-spacing:0.09em; text-transform:uppercase;
    text-decoration:none; border:2px solid rgba(240,237,232,0.25);
    transition: border-color 0.2s, color 0.2s, transform 0.2s;
  }
  .vr-btn-ghost:hover { border-color:rgba(240,237,232,0.6); color:#fff; transform:translateY(-2px); }
  .vr-stat-pill {
    padding:7px 16px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
    font-size:11px; font-weight:600; color:rgba(240,237,232,0.6); letter-spacing:0.05em;
    transition:transform 0.2s; cursor:default;
  }
  .vr-stat-pill:hover { transform:scale(1.04); }
  .vr-why-row {
    border-bottom:1px solid #E2DED9; padding:32px 0;
    opacity:0; transform:translateY(20px);
    transition: opacity 0.55s ease, transform 0.55s ease;
  }
  .vr-why-row.revealed { opacity:1; transform:translateY(0); }
  .vr-why-row:last-child { border-bottom:none; }
  .vr-step-card {
    border-left:2px solid #E2DED9; padding-left:28px;
    opacity:0; transform:translateX(-16px);
    transition: opacity 0.55s ease, transform 0.55s ease, border-color 0.25s;
  }
  .vr-step-card.revealed { opacity:1; transform:translateX(0); }
  .vr-step-card:hover { border-left-color:#E8601C; }
  .vr-footer-link {
    font-size:12px; color:rgba(240,237,232,0.78); text-decoration:none; transition:color 0.2s;
  }
  .vr-footer-link:hover { color:#E8601C; }
  .vr-reveal { opacity:0; transform:translateY(24px); transition: opacity 0.6s ease, transform 0.6s ease; }
  .vr-reveal.revealed { opacity:1; transform:translateY(0); }
`;
