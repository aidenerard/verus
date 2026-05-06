# CLAUDE.md — Standing Instructions for Claude Code

## File Size Limit

**300 lines maximum per file.** If a file approaches 300 lines, split it into co-located sub-modules before adding more code. Each file must have a single clear responsibility.

The precedent is already set: GPRWorkspace (1659→940), HomePage (897→20), DashboardPage (527→130).

---

## Folder Structure

### Frontend (`frontend/src/`)

| What | Where |
|---|---|
| New top-level pages | `app/pages/PageName.tsx` |
| Page sub-components | `app/pages/[page]/ComponentName.tsx` (co-located) |
| Page design tokens | `app/pages/[page]/tokens.ts` or inline in `constants.ts` |
| Shared components | `app/components/ComponentName.tsx` |
| shadcn/ui primitives | `app/components/ui/` ← never hand-edit |
| Figma-generated assets | `app/components/figma/` ← never hand-edit |
| Auth context | `context/AuthContext.tsx` |
| Third-party client init | `lib/` (e.g., `lib/supabase.ts`) |
| Route definitions | `app/Router.tsx` (only file that lists routes) |
| Shared scroll-reveal hook | `app/pages/home/useReveal.ts` |

### Backend (`server/`)

Flat module layout — no subdirectories. Each file has one responsibility:

| File | Responsibility |
|---|---|
| `server.py` | FastAPI app, all route handlers |
| `auth.py` | JWT decode, `verify_token` dependency |
| `jobs.py` | ThreadPoolExecutor job queue, `run_analysis_job` |
| `model.py` | CNN1D architecture definition |
| `run.py` | `run_inference` + thin re-export shim for model |
| `inference.py` | Inference logic (signal → prob grid) |
| `data.py` | CSV loading and preprocessing |
| `ingest.py` | Public re-export surface; format routing |
| `ingest_converters.py` | Per-format converters (DZT, SEG-Y, DT1, RD3/RD7) |
| `ingest_gps.py` | GPS/NMEA extraction |
| `ingest_utils.py` | Shared ingest helpers |
| `grids.py` | C-scan grid construction |
| `render.py` | Image/colormap rendering |

New Python modules go in `server/`. No subdirectories.

### Database

Migrations: `supabase/migrations/NNN_description.sql` (sequential numeric prefix).
Tables in use: `analysis_jobs`, `projects`.

### Project root

ML training notebooks (`.ipynb`) and large data directories stay at project root. Never put training artifacts inside `frontend/` or `server/`.

---

## Code Style Rules

### Comments
Write **no comments** by default. Only add one when the WHY is non-obvious: a hidden constraint, a subtle invariant, or a workaround for a specific bug. Never describe what the code does — well-named identifiers do that.

### TypeScript
- Use explicit types. Avoid `any`; use `unknown` + narrowing instead.
- No `as unknown as X` casts unless genuinely unavoidable.
- Prefer `interface` for object shapes, `type` for unions/aliases.
- There is no `tsconfig.json` — Vite handles transpilation. Do not add one.

### Environment variables
**Never hardcode URLs or keys.** All external references must go through:
- Frontend: `import.meta.env.VITE_*` (set in `.env.local` for dev, Vercel dashboard for prod)
- Backend: `os.environ` / `os.getenv()` (set in Render dashboard)

The only hardcoded URL permitted is the fallback in `inspect/constants.ts`:
```ts
export const SERVER = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : 'https://verus-server.onrender.com';
```

### CSS and design tokens
- Define CSS class rules as template literal strings (e.g., `PAGE_CSS`) injected via `<style>`.
- Always use token variables from the co-located `tokens.ts` or `constants.ts`, never raw hex strings inline.
- Global design tokens: Black `#0A0A0A` · Orange `#E8601C` · Off-white `#F5F3EF` · Border `#E2DED9` · Text-gray `#7A7470`
- Inspect workspace tokens live in `inspect/constants.ts` as `BG`, `PANEL`, `RAISED`, `BORDER`, `BORDER2`, `TEXT`, `TEXT2`, `ACCENT`.

### Python
- Public API surfaces through `ingest.py` and `run.py` re-exports. Callers import from those, not from sub-modules directly.
- No `import *`.
- No blocking synchronous HTTP calls inside async FastAPI handlers (the Supabase auth bug: `auth.py` uses local JWT decode exactly to avoid this).

---

## Files Never To Modify Directly

| Path | Reason |
|---|---|
| `frontend/src/app/components/ui/` | shadcn/ui auto-generated — re-run `npx shadcn-ui add` to update |
| `frontend/src/app/components/figma/` | Figma Make output — regenerate from Figma instead |
| `server/model.pth` | Trained PyTorch weights — replace by retraining, never hand-edit |
| `server/models/` | Archived model checkpoints — read-only |

---

## Tech Stack

### Frontend
- **React 18** + **TypeScript** via **Vite 6** (`@vitejs/plugin-react`)
- **React Router 7** — `BrowserRouter` with `ProtectedRoute` wrapper in `Router.tsx`
- **Tailwind CSS 4** (`@tailwindcss/vite`) — utility classes for layout; inline style objects for design system colors
- **shadcn/ui** (Radix UI primitives) — `components/ui/`
- **Supabase JS** (`@supabase/supabase-js`) — auth sessions + database reads/writes
- **Mapbox GL** — GPS trace overlay in GPR workspace
- **Three.js** — 3D bridge deck visualization
- **Lucide React** — icons throughout

### Backend
- **Python 3.11**, **FastAPI 0.109**, **uvicorn** on port 10000
- **PyTorch 2.2 (CPU-only)** — CNN1D for delamination detection
- **numpy / pandas / scipy** — signal processing
- **readgssi** + **segyio** — GPR file format parsers
- **Supabase Python client** — job result persistence
- Hosted on **Render** free tier (Docker, `render.yaml`)

### Database / Auth
- **Supabase** (PostgreSQL + Row Level Security + Auth)
- Auth: JWT sub-claim decoded locally in `server/auth.py` — no Supabase network call on every request

### Hosting
- **Vercel** — frontend (auto-deploys on push to `main`)
- **Render** — backend (Docker, auto-deploys on push to `main`)

---

## Routes

| Path | Component | Auth |
|---|---|---|
| `/` | `HomePage` | public |
| `/login` | `LoginPage` | public |
| `/signup` | `SignupPage` | public |
| `/team` | `TeamPage` | public |
| `/dashboard` | `DashboardPage` | protected |
| `/inspect/gpr` | `GPRWorkspace` | protected |
| `/inspect/masw` | `ComingSoonWorkspace` | protected |
| `/inspect/ir` | `ComingSoonWorkspace` | protected |
| `/analyze` | → redirects to `/inspect/gpr` | — |

---

## Dev Commands

```bash
# Frontend
cd frontend && npm run dev        # Vite dev server — proxies /health /analyze etc. to localhost:10000
cd frontend && npm run build      # production build

# Backend
cd server && uvicorn server:app --port 10000

# Deploy
git push origin main              # triggers both Vercel + Render auto-deploy
```
