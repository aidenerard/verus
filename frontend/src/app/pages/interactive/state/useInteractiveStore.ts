import { create } from 'zustand';
import type { Pick, ViewMode, CameraState } from './types';

interface InteractiveState {
  selectedPickIds: string[];
  picks:           Map<string, Pick>;
  viewMode:        ViewMode;
  surfaceTextureCacheBust: number;

  setPicks:         (picks: Pick[]) => void;
  upsertPick:       (pick: Pick) => void;
  selectPick:       (id: string, shift?: boolean) => void;
  clearSelection:   () => void;
  setViewMode:      (mode: ViewMode) => void;
  bumpSurfaceCache: () => void;
}

export const useInteractiveStore = create<InteractiveState>((set) => ({
  selectedPickIds: [],
  picks:           new Map(),
  viewMode:        'top',
  surfaceTextureCacheBust: 0,

  setPicks: (picks) => set(() => ({
    picks: new Map(picks.filter(p => !p.is_deleted).map(p => [p.id, p])),
  })),

  upsertPick: (pick) => set((s) => {
    const next = new Map(s.picks);
    if (pick.is_deleted) next.delete(pick.id);
    else next.set(pick.id, pick);
    return { picks: next };
  }),

  selectPick: (id, shift = false) => set((s) => {
    if (!shift) return { selectedPickIds: [id] };
    if (s.selectedPickIds.includes(id))
      return { selectedPickIds: s.selectedPickIds.filter(x => x !== id) };
    return { selectedPickIds: [...s.selectedPickIds, id] };
  }),

  clearSelection: () => set({ selectedPickIds: [] }),
  setViewMode:    (viewMode) => set({ viewMode }),
  bumpSurfaceCache: () => set((s) => ({ surfaceTextureCacheBust: s.surfaceTextureCacheBust + 1 })),
}));

const CAMERA_KEY = (projectId: string) => `verus_interactive_camera_${projectId}`;

export function loadCameraState(projectId: string): CameraState | null {
  try {
    const raw = localStorage.getItem(CAMERA_KEY(projectId));
    return raw ? JSON.parse(raw) as CameraState : null;
  } catch { return null; }
}

export function saveCameraState(projectId: string, state: CameraState): void {
  try { localStorage.setItem(CAMERA_KEY(projectId), JSON.stringify(state)); }
  catch { /* quota; ignore */ }
}
