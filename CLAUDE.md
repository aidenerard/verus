# CLAUDE.md — Standing Instructions for Claude Code

## File Size Limit

**300 lines maximum per file.** If a file approaches 300 lines, split it into co-located sub-modules before adding more code. Each file must have a single clear responsibility.

The precedent is already set: GPRWorkspace (1659→940→493→277), HomePage (897→20), DashboardPage (527→130).

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
| `jobs.py` | ThreadPoolExecutor job queue, `run_analysis_job` orchestrator |
| `model_loader.py` | Model download + background loading (`load_models_background`) |
| `pipeline.py` | Per-file inference loop (`process_files`) + result assembly (`build_result_payload`) |
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

#### Model architecture is config-driven

`CNN1D` accepts `in_channels`, `conv_channels`, and `head_hidden` params. On startup,
`server.py` tries to download `model_config.json` (via `MODEL_CONFIG_GDRIVE_URL`),
instantiates `CNN1D(**config)`, then falls back to probing known architectures if
no config is found. **After every retraining run, upload both `model.pth` and
`model_config.json` to Google Drive and update both `MODEL_GDRIVE_URL` and
`MODEL_CONFIG_GDRIVE_URL` in Render.** Skipping the config file forces the fallback
probe, which may silently load the wrong architecture.

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

## Parallel Worktree Sessions

Two local worktrees are used for parallel Claude Code sessions. **Do not edit files outside your worktree's scope** — the whole point is zero overlap so there are no merge conflicts.

| Worktree path | Branch | Scope |
|---|---|---|
| `~/Desktop/verus` | `main` | Merges only — no feature commits here |
| `~/Desktop/verus-backend` | `feature/backend-render` | `server/` only: `render.py`, `jobs.py`, `inference.py`, `ingest.py` |
| `~/Desktop/verus-frontend` | `feature/frontend-ux` | `frontend/src/**` only |

**Workflow:** edit locally in the worktree → commit to the feature branch → merge into `main` in `verus/` → `git push origin main`. The feature branches never need to be pushed to GitHub. To clean up after merging:

```bash
git worktree remove ../verus-backend
git worktree remove ../verus-frontend
git branch -d feature/backend-render feature/frontend-ux
```

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

---

## File Map

### Project Root

| File | Purpose |
|---|---|
| `CLAUDE.md` | Standing instructions for Claude Code |
| `SETUP.md` | Supabase, env vars, and deployment guide |
| `render.yaml` | Render service config |
| `Dockerfile` | Root Docker stub (Render uses `server/Dockerfile`) |
| `rebar_training.ipynb` | Local copy of rebar model training notebook |
| `data/` | Bridge DZT scan datasets — never commit large files |
| `media/` | Marketing videos and screenshots |
| `kaggle_push/` | Kaggle notebook + submitted output for rebar training |

### Backend (`server/`)

| File | Purpose |
|---|---|
| `server.py` | FastAPI app — route handlers `/analyze`, `/job/*`, `/health` |
| `auth.py` | Local JWT decode, `verify_token` dependency |
| `jobs.py` | ThreadPoolExecutor runner, `run_analysis_job` thin orchestrator |
| `model_loader.py` | Model download (`download_file`) + background loading (`load_models_background`) |
| `pipeline.py` | `process_files` per-file inference loop + `build_result_payload` result assembly |
| `model.py` | `CNN1D` + `RebarDepthCNN` PyTorch architectures |
| `run.py` | Public re-export shim — import from here, not sub-modules |
| `inference.py` | Signal → prob grid + `run_rebar_inference` |
| `data.py` | CSV loading and signal preprocessing |
| `ingest.py` | Format router (public ingestion API) |
| `ingest_converters.py` | Per-format converters: DZT, SEG-Y, DT1, RD3/RD7 |
| `ingest_gps.py` | GPS/NMEA coordinate extraction |
| `ingest_utils.py` | Shared helpers for ingest modules |
| `grids.py` | C-scan + rebar grid construction and downsampling |
| `render.py` | PNG rendering — condition, rebar, amplitude colormaps |
| `model.pth` | CNN1D delamination weights (binary, read-only) |
| `models/` | Archived checkpoint versions (read-only) |
| `requirements.txt` | Production pip dependencies |
| `Dockerfile` | Docker image for Render deployment |
| `test_csv_format.py` | Dev utility: validate CSV signal format |

### Frontend (`frontend/src/`)

| File | Purpose |
|---|---|
| `main.tsx` | React entry point |
| `app/App.tsx` | Root component — wraps Router + AuthProvider |
| `app/Router.tsx` | All route definitions + `ProtectedRoute` |
| `lib/supabase.ts` | Supabase client singleton |
| `context/AuthContext.tsx` | Auth state + `useAuth` hook |
| `styles/` | Global CSS: reset, Tailwind directives, fonts, theme tokens |
| `app/components/PlanView.tsx` | Mapbox satellite map with GPR trace overlay |
| `app/components/ThreeDView.tsx` | Three.js 3D bridge deck visualization |
| `app/components/ThreeDTypes.ts` | Types and constants for ThreeDView (FileResult, TooltipState, grid dims) |
| `app/components/VerusLogo.tsx` | SVG logo component |
| `app/components/ui/` | shadcn/ui Radix primitives — never hand-edit |
| `app/components/figma/` | Figma Make output — never hand-edit |
| `app/pages/HomePage.tsx` | Public marketing landing shell |
| `app/pages/LoginPage.tsx` | Email/password sign-in |
| `app/pages/SignupPage.tsx` | New account creation |
| `app/pages/TeamPage.tsx` | Public team bios |
| `app/pages/DashboardPage.tsx` | Project list + job history (protected) |
| `app/pages/home/` | Sub-components: Hero, Navbar, HowItWorks, MethodSlider, OurPlatform, WhyVerus, TickerStripe, Footer, tokens, useReveal |
| `app/pages/dashboard/` | Sub-components: JobTable, ComingSoonModal, types |
| `app/pages/inspect/GPRWorkspace.tsx` | GPR workspace shell — state, hook composition, top-level layout |
| `app/pages/inspect/CentralColumn.tsx` | Vertical PanelGroup: viewport + icon strip + RightSlidePanel + BScanPanel |
| `app/pages/inspect/WorkspaceToolbar.tsx` | 44px top toolbar: project title, view toggle, export, settings |
| `app/pages/inspect/LayersSidebar.tsx` | Left 220px panel: layer list, file list, add/export menus |
| `app/pages/inspect/RightSlidePanel.tsx` | Slide-out right panel: Properties / Analysis / Adjust tabs |
| `app/pages/inspect/BScanPanel.tsx` | Resizable bottom B-scan panel |
| `app/pages/inspect/ProjectsDrawer.tsx` | "My Projects" overlay drawer with recent job list |
| `app/pages/inspect/OutputMaps.tsx` | Output tab canvases: condition, rebar depth, amplitude |
| `app/pages/inspect/AdjustPanel.tsx` | Sliders: detection threshold, dielectric εr, amp clamp |
| `app/pages/inspect/SetupWizard.tsx` | Project setup wizard for new analysis runs |
| `app/pages/inspect/ComingSoonWorkspace.tsx` | Placeholder for MASW / IR workspaces |
| `app/pages/inspect/useAnalysisJob.ts` | Analysis job hook: submit, poll, progress overlay state |
| `app/pages/inspect/useCanvasRenderers.ts` | Canvas render effects: b-scan, condition, rebar, amplitude |
| `app/pages/inspect/useMapbox.ts` | Mapbox init, GPS layer, layer visibility + opacity effects |
| `app/pages/inspect/useViewJobLoader.ts` | Startup effects: restore from localStorage + load job from URL param |
| `app/pages/inspect/useSetupCallbacks.ts` | `completeSetup` (Supabase upsert) and `newProject` (full reset) |
| `app/pages/inspect/constants.ts` | Inspect design tokens + `SERVER` URL constant |
| `app/pages/inspect/types.ts` | `AnalysisResult`, `FileResult`, `OutputTab` types |
| `app/pages/inspect/utils.ts` | `badgeColor` and shared view helpers |
| `app/pages/inspect/colormaps.ts` | Canvas colormap LUTs: condition, rebar, amplitude |
| `app/pages/team/` | Sub-components: FounderCard, tokens |
| `vite.config.ts` | Vite + Tailwind config, dev proxy to `:10000` |

### Database (`supabase/migrations/`)

| File | Purpose |
|---|---|
| `001_initial.sql` | `profiles` table, auth trigger, RLS policies |
| `002_projects.sql` | `projects` + `analysis_jobs` tables |
