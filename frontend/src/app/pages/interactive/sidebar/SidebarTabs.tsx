import { useState } from 'react';
import type { Scene } from '../state/types';
import InspectorTab from './InspectorTab';
import ProcessingTab from './ProcessingTab';
import GriddingTab from './GriddingTab';
import { ACCENT, BORDER, PANEL_LIGHT, TEXT, TEXT2, TEXT3 } from '../tokens';

type TabId = 'inspector' | 'processing' | 'gridding';

interface Props {
  projectId: string;
  scene:     Scene | undefined;
}

const TABS: { id: TabId; label: string }[] = [
  { id: 'inspector',  label: 'Inspector' },
  { id: 'processing', label: 'Processing' },
  { id: 'gridding',   label: 'Gridding' },
];

export default function SidebarTabs({ projectId, scene }: Props) {
  const [active, setActive] = useState<TabId>('inspector');

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: PANEL_LIGHT }}>
      <div role="tablist" style={{ display: 'flex', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
        {TABS.map(t => {
          const isActive = t.id === active;
          return (
            <button
              key={t.id}
              role="tab"
              aria-selected={isActive}
              onClick={() => setActive(t.id)}
              style={{
                flex: 1, padding: '12px 8px', border: 'none', cursor: 'pointer',
                background: 'transparent',
                color: isActive ? TEXT : TEXT2,
                fontSize: 10, fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase',
                borderBottom: `2px solid ${isActive ? ACCENT : 'transparent'}`,
                fontFamily: 'inherit',
                transition: 'color 0.15s, border-color 0.15s',
              }}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '16px 16px 24px', color: TEXT3 }}>
        {!scene && <div style={{ fontSize: 12, color: TEXT3 }}>Loading…</div>}
        {scene && active === 'inspector'  && <InspectorTab  projectId={projectId} scene={scene} />}
        {scene && active === 'processing' && <ProcessingTab projectId={projectId} scene={scene} />}
        {scene && active === 'gridding'   && <GriddingTab   projectId={projectId} />}
      </div>
    </div>
  );
}
