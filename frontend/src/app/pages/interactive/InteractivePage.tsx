import { Suspense } from 'react';
import { useParams } from 'react-router';
import { useScene } from './state/hooks';
import InteractiveTopBar from './InteractiveTopBar';
import SceneCanvas from './scene/SceneCanvas';
import ColorLegend from './scene/ColorLegend';
import SidebarTabs from './sidebar/SidebarTabs';
import BScanPanel from './bscan/BScanPanel';
import {
  BG, BORDER, PANEL, PANEL_LIGHT, TEXT, TEXT2, TEXT3,
  FONT_FAMILY, SIDEBAR_WIDTH, BSCAN_HEIGHT_PCT, ACCENT,
} from './tokens';

export default function InteractivePage() {
  const { projectId = 'demo-job-001' } = useParams<{ projectId: string }>();
  const { scene, error, isLoading } = useScene(projectId);

  const backHref = `/workspace/em/gpr?project_id=${projectId}`;

  return (
    <div style={{
      height: '100vh', width: '100vw', overflow: 'hidden',
      background: BG, color: TEXT, fontFamily: FONT_FAMILY,
      display: 'flex', flexDirection: 'column',
    }}>
      <InteractiveTopBar
        projectId={projectId}
        projectName={scene?.project_name}
        backHref={backHref}
      />

      <div style={{
        flex: 1, minHeight: 0,
        display: 'grid',
        gridTemplateColumns: `1fr ${SIDEBAR_WIDTH}px`,
        gridTemplateRows:    `${100 - BSCAN_HEIGHT_PCT}fr ${BSCAN_HEIGHT_PCT}fr`,
      }}>
        <section style={{ background: PANEL, borderRight: `1px solid ${BORDER}`, borderBottom: `1px solid ${BORDER}`, position: 'relative' }}>
          {scene ? (
            <>
              <Suspense fallback={<SceneStatus message="Loading 3D scene…" />}>
                <SceneCanvas projectId={projectId} scene={scene} />
              </Suspense>
              <ColorLegend range={scene.surface.depth_range_in} units="in" />
            </>
          ) : (
            <ScenePlaceholder loading={isLoading} error={error?.message} />
          )}
        </section>

        <aside style={{ background: PANEL_LIGHT, borderBottom: `1px solid ${BORDER}`, gridRow: '1 / span 2', overflow: 'hidden' }}>
          <SidebarTabs projectId={projectId} scene={scene} />
        </aside>

        <section style={{ background: PANEL, gridColumn: '1', borderRight: `1px solid ${BORDER}`, position: 'relative' }}>
          {scene && <BScanPanel projectId={projectId} scene={scene} />}
        </section>
      </div>
    </div>
  );
}

function SceneStatus({ message }: { message: string }) {
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: TEXT2, fontSize: 13 }}>
      {message}
    </div>
  );
}

function ScenePlaceholder({ loading, error }: { loading: boolean; error?: string }) {
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', textAlign: 'center', padding: 24 }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.14em', textTransform: 'uppercase', color: TEXT3, marginBottom: 12 }}>3D Scene</div>
        {loading && <div style={{ color: TEXT2, fontSize: 13 }}>Loading scene…</div>}
        {error   && <div style={{ color: ACCENT, fontSize: 13 }}>Failed to load: {error}</div>}
        {!loading && !error && (
          <div style={{ color: TEXT2, fontSize: 13, maxWidth: 360 }}>
            Three.js viewport renders here in the next commit (BridgeDeckSurface · RebarPicks · ScanLines).
          </div>
        )}
      </div>
    </div>
  );
}

