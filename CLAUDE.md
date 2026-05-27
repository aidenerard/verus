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
| Inspection workspaces | `app/pages/workspace/[Method]Workspace.tsx` |
| Workspace shell | `app/pages/workspace/WorkspaceLayout.tsx` (analysis view) |
| Pre-workspace flow | `app/pages/workspace/ModuleSelectPage.tsx`, `MethodSelectPage.tsx`, `SelectPageShell.tsx` |
| Interactive (3D) view | `app/pages/interactive/` (scene, sidebar, bscan + state subdirs) |
| MSW fixtures | `__fixtures__/interactive/*.json` (checked-in JSON used by both dev and tests) |
| MSW handlers + bootstrap | `mocks/handlers.ts`, `mocks/browser.ts` (lazy-loaded by `main.tsx` when `VITE_USE_MOCKS=true`) |
| Processing options context | `app/pages/workspace/ProcessingOptionsContext.tsx` |
| Shared analysis primitives | `app/pages/inspect/` (polling hook, Mapbox hook, types, constants) |
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
- **Three.js** + **@react-three/fiber 8.x** + **@react-three/drei 9.x** — 3D bridge deck visualization (fiber pinned to v8 for React-18 compatibility; v9 requires React 19)
- **Zustand 4.x** — interactive view UI state
- **SWR 2.x** — interactive view data fetching + cache invalidation
- **MSW 2.x (devDependency)** — interactive endpoints mocked behind `VITE_USE_MOCKS=true` while the backend is being built in parallel
- **Lucide React** — icons throughout

### Mocks & fixtures (interactive view only)
- `src/__fixtures__/interactive/*.json` are checked in and hand-curated.
- `src/mocks/{handlers,browser}.ts` register MSW handlers for `/jobs/{id}/{scene,picks,scan_line/{id},processing,gridding}`, `PATCH /picks/{id}`, and `POST /jobs/{id}/{processing,gridding,reprocess,regrid}`.
- `src/main.tsx` dynamically imports `mocks/browser.ts` only when `import.meta.env.VITE_USE_MOCKS` is truthy. Add `VITE_USE_MOCKS=true` to `frontend/.env.local` to enable; leave it unset in production so the worker is never registered and the bundle never loads the msw chunk.
- `public/mockServiceWorker.js` is the worker shipped by `msw init` — do not hand-edit. Re-run `npx msw init public/` after upgrading msw.

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
| `/workspace` | `ModuleSelectPage` (Electromagnetic vs Seismic) | protected |
| `/workspace/em` | `MethodSelectPage` (GPR · FDEM · Magnetometer) | protected |
| `/workspace/seismic` | `MethodSelectPage` (MASW · Impact Echo) | protected |
| `/workspace/em/gpr` | `GPRWorkspace` (inside `WorkspaceLayout`) | protected |
| `/workspace/em/fdem` | `FDEMWorkspace` (placeholder) | protected |
| `/workspace/em/magnetometer` | `MagWorkspace` (placeholder) | protected |
| `/workspace/seismic/masw` | `MASWWorkspace` (placeholder) | protected |
| `/workspace/seismic/impact-echo` | `ImpactEchoWorkspace` (placeholder) | protected |
| `/analyze` | → redirects to `/workspace/em/gpr` (preserves query) | — |

The **interactive view** is a tab inside `GPRResults` (Overview / Interactive), not a standalone route. No URL change happens when the user opens it — `?project_id=…` is still the only param.
| `/inspect/gpr` | → redirects to `/workspace/em/gpr` (preserves query) | — |
| `/inspect/masw` | → redirects to `/workspace/seismic/masw` | — |
| `/inspect/ir` | → redirects to `/workspace/seismic/impact-echo` | — |

Inspection is a guided two-step flow: `/workspace` (module) → `/workspace/<module>` (method) → `/workspace/<module>/<method>` (analysis). The two select pages share `SelectPageShell` (logo + back button, no sidebar). Once the user picks a method, `WorkspaceLayout` wraps the analysis view with a minimal top bar (back, Verus logo, breadcrumb `Workspace › Module › Method`, user avatar) — no sidebar. `ProcessingOptionsProvider` lives on `WorkspaceLayout` and is consumed by the GPR upload card's collapsible "Advanced Options" section. Legacy `/inspect/*` redirects use `RedirectKeepQuery` so `?project_id=…` deep links from the dashboard survive.

The **interactive view** lives as an "Interactive" tab inside `GPRResults` (sibling of the default "Overview" tab — no separate route, no extra chrome). When the user selects the Interactive tab, `GPRResults` mounts `InteractiveView` in the same content area: a three-pane grid (3D scene top-left, B-scan bottom-left, sidebar right) in the same light Verus palette as the rest of the workspace (tokens at `app/pages/interactive/tokens.ts` mirror the workspace tokens). Data fetching uses **SWR**; UI state uses **Zustand**; camera state persists in `localStorage` keyed per project. The mock service worker (see "Mocks & fixtures") supplies all `/jobs/{id}/*` and `/picks/{id}` endpoints behind `VITE_USE_MOCKS=true` while the real backend is being built in a parallel branch. Pick depth/position are **read-only in the inspector** — they're derived from survey + velocity; adjust velocity globally in the Processing tab to recompute depths.

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
| `app/pages/workspace/ModuleSelectPage.tsx` | Step 1: full-screen card picker for Electromagnetic vs Seismic |
| `app/pages/workspace/MethodSelectPage.tsx` | Step 2: card picker for methods within a module; coming-soon methods rendered as disabled |
| `app/pages/workspace/SelectPageShell.tsx` | Shared shell for the two select pages (back button, Verus logo, eyebrow + heading) |
| `app/pages/workspace/WorkspaceLayout.tsx` | Analysis-view shell: minimal top bar + breadcrumb + `<Outlet />`; provides `ProcessingOptionsProvider` |
| `app/pages/workspace/ProcessingOptionsPanel.tsx` | Pure-content form (gridding, search radius, edge clipping, filters) — embedded into GPR upload card |
| `app/pages/workspace/ProcessingOptionsContext.tsx` | `ProcessingOptionsProvider` + `useProcessingOptions` hook + `toFormData` serializer |
| `app/pages/workspace/PlaceholderWorkspace.tsx` | Reusable Coming Soon screen (description, use cases, email capture) |
| `app/pages/workspace/GPRWorkspace.tsx` | GPR shell: upload + polling + result; loads saved job via `?project_id=` |
| `app/pages/workspace/GPRUploadCard.tsx` | Drag-drop file list + equipment/frequency selects + Start button |
| `app/pages/workspace/GPRResults.tsx` | Three-panel result view (Horizon Picks / Rebar Depth / Corrosion Risk) + stats bar + Mapbox GPS panel |
| `app/pages/workspace/FDEMWorkspace.tsx` | Placeholder workspace for Frequency-Domain EM |
| `app/pages/workspace/MagWorkspace.tsx` | Placeholder workspace for Magnetometry |
| `app/pages/workspace/MASWWorkspace.tsx` | Placeholder workspace for MASW |
| `app/pages/workspace/ImpactEchoWorkspace.tsx` | Placeholder workspace for Impact Echo |
| `app/pages/workspace/modules.ts` | Module catalog (Electromagnetic + Seismic) driving sidebar + breadcrumb |
| `app/pages/workspace/tokens.ts` | Workspace design tokens (BG/PANEL/BORDER/TEXT/ACCENT + layout constants) |
| `app/pages/workspace/types.ts` | `ProcessingOptions`, `MethodMeta`, `ModuleMeta`, defaults |
| `app/pages/interactive/InteractiveView.tsx` | Embedded 3-pane orchestrator — mounted by the Interactive tab in `GPRResults` |
| `app/pages/interactive/tokens.ts` | Light Verus palette + layout constants (mirrors workspace tokens) |
| `app/pages/interactive/state/types.ts` | `Pick`, `Scene`, `ScanLineTraces`, `ProcessingConfig`, `GriddingConfig`, `ViewMode`, `CameraState` |
| `app/pages/interactive/state/api.ts` | Typed `fetch` wrappers + SWR cache keys |
| `app/pages/interactive/state/hooks.ts` | `useScene`, `usePicks`, `useScanLine`, `useProcessing`, `useGridding` + mutate helpers |
| `app/pages/interactive/state/useInteractiveStore.ts` | Zustand store: selection, pick map, viewMode, surface cache-bust + localStorage camera state |
| `app/pages/interactive/scene/SceneCanvas.tsx` | R3F `<Canvas>` shell, `OrbitControls`, camera rig + persistence |
| `app/pages/interactive/scene/BridgeDeckSurface.tsx` | Semi-transparent ground plane with client-generated Spectral `CanvasTexture` |
| `app/pages/interactive/scene/RebarPicks.tsx` | `<Instances>` of depth-colored spheres + selected-pick ring + halo |
| `app/pages/interactive/scene/ScanLines.tsx` | Per-scan-line drei `<Line>` polylines |
| `app/pages/interactive/scene/ColorLegend.tsx` | Bottom-left overlay legend for the depth colormap |
| `app/pages/interactive/scene/colormap.ts` | 11-stop Spectral colormap + `spectral01`, `depthToColor`, `spectralStops` |
| `app/pages/interactive/sidebar/SidebarTabs.tsx` | Tab strip for Inspector / Processing / Gridding |
| `app/pages/interactive/sidebar/InspectorTab.tsx` | Read-only pick metadata + delete + "Add pick" stub; position/depth not user-editable |
| `app/pages/interactive/sidebar/ProcessingTab.tsx` | Velocity (global, debounced regrid) + time-zero slider + filter chain + GPS latency |
| `app/pages/interactive/sidebar/VelocityControl.tsx` | Velocity (m/ns) slider, 0.05–0.15, default 0.10; debounced reprocess + revalidate |
| `app/pages/interactive/sidebar/FilterRow.tsx` | Drag-to-reorder filter row with enabled toggle + param summary |
| `app/pages/interactive/sidebar/GriddingTab.tsx` | Algorithm (Min Curvature default · Kriging · Natural · Nearest), radius, edge clip, cell size, anisotropy (Kriging/Min-curve) |
| `app/pages/interactive/sidebar/fields.tsx` | Shared form atoms (Section, Row, NumberField, Slider, Select, Toggle, Button) |
| `app/pages/interactive/bscan/BScanPanel.tsx` | Bottom panel — header, scrolling, Y-axis (ns + depth in via ε_r) |
| `app/pages/interactive/bscan/BScanCanvas.tsx` | Canvas trace renderer (int8 → grayscale ImageData) |
| `app/pages/interactive/bscan/PickDots.tsx` | SVG overlay; click selects a pick (sync with 3D scene + sidebar) |
| `mocks/handlers.ts` + `mocks/browser.ts` | MSW handlers + worker bootstrap (lazy imported behind `VITE_USE_MOCKS`) |
| `__fixtures__/interactive/*.json` | scene + picks + sl-1 trace data + processing/gridding defaults |
| `app/pages/inspect/useAnalysisJob.ts` | Analysis job hook: wake, submit, poll; accepts `extraFormData` for processing options |
| `app/pages/inspect/useMapbox.ts` | Mapbox init, GPS layer, visibility + opacity effects |
| `app/pages/inspect/ConfirmAnalysisModal.tsx` | "Ready to analyze" confirmation dialog |
| `app/pages/inspect/AnalysisProgressOverlay.tsx` | Full-screen progress overlay during job execution |
| `app/pages/inspect/constants.ts` | `SERVER` URL, Mapbox token, manufacturer + frequency catalogs, shared tokens |
| `app/pages/inspect/types.ts` | `AnalysisResult` (with v5 keys `horizon_picks`, `rebar_depth_map`, `corrosion_map`, `mean_depth_inches`, etc.), `FileResult`, `UploadedFile` |
| `app/pages/inspect/utils.ts` | `estimateAnalysisSeconds`, `delamColor`, `badgeColor` |
| `app/pages/team/` | Sub-components: FounderCard, tokens |
| `vite.config.ts` | Vite + Tailwind config, dev proxy to `:10000` |

### Database (`supabase/migrations/`)

| File | Purpose |
|---|---|
| `001_initial.sql` | `profiles` table, auth trigger, RLS policies |
| `002_projects.sql` | `projects` + `analysis_jobs` tables |

---

## Machine Learning

### Models

| Model | File | Task | Input | Output |
|---|---|---|---|---|
| CNN1D (V17) | `server/model.pth` | Delamination classification | (n, 2, 250) dual-channel | P(sound) via sigmoid |
| HorizonCNN | `server/models/horizon_model.pth` | Rebar depth regression | (n, 1, 512) | depth ∈ [0,1] × 300 mm |
| RebarDepthCNN | `server/models/rebar_model.pth` | Legacy rebar depth | (n, 2, 256) | depth in inches |
| Corrosion model | `server/models/corrosion_model.pth` | Corrosion risk | (n, 1, 512) | risk score |
| Deck thickness | `server/models/thickness_model.pth` | Deck thickness | (n, 1, 512) | thickness |

**Label convention (critical):** `model output = P(sound)` via sigmoid. `1 = sound`, `0 = delaminated`. Positive class for all inspection metrics = delaminated (label 0). This is inverted from sklearn's convention — watch for it in every metric call.

**Production threshold:** `THRESHOLD = 0.65` in `server/model.py`. `P(sound) < 0.65 → delaminated`. The val-selected best-F1 threshold from training is written into `model_config.json` and picked up by the server on load.

### Data Pipeline

```
Raw DZT/SEG-Y/RD3 files
      ↓ server/ingest_converters.py
SDNET2021 CSV format  (FILE____*.csv)
      ↓ kaggle_push/cnn.py: load_csv()
(n, 512) float32  DC-remove (offset 32768) + taper + per-signal z-score
      ↓ Hilbert envelope
(n, 2, 512)  [channel 0 = raw z-scored, channel 1 = envelope]
      ↓ crop samples 200:450
(n, 2, 250) → CNN1D input
```

**SDNET2021 CSV row layout:**
- Row 0, col 4: `n_signals`
- Row 7, cols 1…n: labels (1=sound, 0=delaminated)
- Rows 9–13: amplitude data start (512 rows × n_signals columns)

### Raw Data Directories (outside repo)

All raw field data lives at `C:\Users\quack\Documents\Projects\Verus\Data\` — not inside the repo. Never commit large data files.

| Directory | Format | Count | Contents |
|---|---|---|---|
| `Data/SDNET 2021 GPR Data/` | `.xlsx` (`FILE____*.xlsx`) | 206 files, 5 bridges | Delamination-labelled A-scans. Bridges: Park River Median (1–49), Forest River NB (50–77), Park River NB (78–102+), Park River SB, Forest River SB |
| `Data/Ken Infrasense Data #1/` | GSSI `.DZT` | 9 DZT, 2 bridges | B170020 (4 files, scans 95–98), B440029 (5 files, scans 799–807). Ground truth: `B170020/B440029 Rebar Depth Report.csv` — cols `preProcessedFileName`, `scanNumber`, `L2Depth_inches` |
| `Data/Stephen Terracon Cornbread/Data/` | Proceq `.scan` + `TS_NNNN_1/2.txt` | 168 scan files, 14 swaths | `PRC_000001–000168.scan` (~12/swath). Ground truth: `TS_NNNN_1.txt` and `TS_NNNN_2.txt` — one float per line, rebar depth in **mm**. GPS: `GPSLog_Scan_N.nmea`. Processed C-scans in `swath_NNNN/` subdirs |

**Processed training-ready data (inside repo, gitignored):**

```
data/
  csv/
    sdnet2021/       5 bridges, ~657,938 signals — FILE____.csv format (converted from xlsx)
    synthetic_numpy/ 50,000 fast numpy Ricker-wavelet signals
    synthetic_gprmax/ ~50,000 gprMax physics-simulation signals
    gatech/          GT bridge data after running ingest_gpr_data.py
```

### Training Commands

**Delamination model (CNN1D V17) — runs on Kaggle:**
1. Upload `kaggle_push/cnn.py` as a Kaggle notebook script.
2. Attach datasets: `aidenerard/all-bridges-csv` + `aidenerard/synthetic-data`.
3. Run. Training branch executes when no `model.pth` is found at `MODEL_PATH`.
4. Evaluation branch executes when `model.pth` already exists (drop in existing weights to eval only).
5. Download `model.pth` + `model_config.json` from Kaggle → Output Files.

Key hyperparameters in `kaggle_push/cnn.py`:
- `FocalLoss(alpha=0.75, gamma=2.0)` — no `pos_weight` needed
- `WeightedRandomSampler` — 50/50 batches regardless of class imbalance
- `CosineAnnealingLR(T_max=60, eta_min=1e-6)`, early-stop patience=20

**Rebar depth model (HorizonCNN) — runs on Colab:**
```bash
# Adapt notebook paths for local execution
python adapt_notebooks.py          # rewrites data/model paths in train_rebar_horizon.ipynb
jupyter notebook train_rebar_horizon.ipynb
# OR paste colab_train_horizon.py as a single Colab cell after mounting Drive
```

**Deck thickness / Corrosion models:**
```bash
jupyter notebook notebooks/train_deck_thickness.ipynb
jupyter notebook notebooks/train_corrosion.ipynb
# OR notebooks/train_rebar_universal.ipynb for the combined rebar training
```

### Evaluation Commands

```bash
# Full eval from repo root — outputs to eval_results/<timestamp>/
python scripts/eval_model.py --model server/model.pth --data data/csv/sdnet2021/

# With a specific threshold (use the value from model_config.json)
python scripts/eval_model.py --model server/model.pth --data data/csv/ --threshold 0.65

# Rebar depth validation against Infrasense ground truth
cd server && python test_rebar_validation.py
```

`scripts/eval_model.py` outputs: precision, recall, F1, FNR, confusion matrix, PR-AUC, threshold sweep, per-bridge breakdown, and saved plots. No manual metric recalculation needed after tuning.

### Post-Training Deployment

After every retraining run:
1. Download `model.pth` **and** `model_config.json` from Kaggle/Colab output.
2. Upload both to Google Drive → share as "Anyone with link (Viewer)".
3. Copy each file's Drive ID (`/d/<FILE_ID>` from the share URL).
4. Update **both** env vars in the Render dashboard:
   - `MODEL_GDRIVE_URL` → `https://drive.google.com/uc?export=download&id=<ID>`
   - `MODEL_CONFIG_GDRIVE_URL` → same pattern for the config file
5. Trigger a Render redeploy (or push to `main`).

Skipping `model_config.json` causes the server to probe fallback architectures — it may load the wrong layer sizes silently.

### Scripts

| Script | Run from | Purpose |
|---|---|---|
| `scripts/eval_model.py` | repo root | Full eval: metrics + plots + per-bridge CSV |
| `kaggle_push/cnn.py` | Kaggle | Train/eval CNN1D delamination model |
| `colab_train_horizon.py` | Colab | Train HorizonCNN rebar depth model |
| `adapt_notebooks.py` | repo root | Rewrite notebook paths for local execution |
| `server/test_rebar_validation.py` | `server/` | Rebar depth MAE vs Infrasense ground truth |
| `server/test_csv_format.py` | `server/` | Validate CSV signal format before training |
