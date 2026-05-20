import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  DEFAULT_PROCESSING_OPTIONS,
  type GriddingAlgorithm,
  type ProcessingFilters,
  type ProcessingOptions,
} from './types';

interface ProcessingOptionsContextValue {
  options:           ProcessingOptions;
  setGridding:       (g: GriddingAlgorithm) => void;
  setSearchRadius:   (r: number) => void;
  setEdgeClipping:   (v: boolean) => void;
  toggleFilter:      (key: keyof ProcessingFilters) => void;
  reset:             () => void;
  toFormData:        (fd: FormData) => void;
}

const Ctx = createContext<ProcessingOptionsContextValue | null>(null);

export function ProcessingOptionsProvider({ children }: { children: ReactNode }) {
  const [options, setOptions] = useState<ProcessingOptions>(DEFAULT_PROCESSING_OPTIONS);

  const setGridding = useCallback(
    (gridding: GriddingAlgorithm) => setOptions(o => ({ ...o, gridding })),
    [],
  );
  const setSearchRadius = useCallback(
    (searchRadius: number) => setOptions(o => ({ ...o, searchRadius })),
    [],
  );
  const setEdgeClipping = useCallback(
    (edgeClipping: boolean) => setOptions(o => ({ ...o, edgeClipping })),
    [],
  );
  const toggleFilter = useCallback(
    (key: keyof ProcessingFilters) =>
      setOptions(o => ({ ...o, filters: { ...o.filters, [key]: !o.filters[key] } })),
    [],
  );
  const reset = useCallback(() => setOptions(DEFAULT_PROCESSING_OPTIONS), []);

  const toFormData = useCallback((fd: FormData) => {
    fd.append('gridding', options.gridding);
    fd.append('search_radius_m', String(options.searchRadius));
    fd.append('edge_clipping', String(options.edgeClipping));
    fd.append('filter_background_removal', String(options.filters.backgroundRemoval));
    fd.append('filter_gain',               String(options.filters.gain));
    fd.append('filter_bandpass',           String(options.filters.bandpass));
  }, [options]);

  const value = useMemo(
    () => ({ options, setGridding, setSearchRadius, setEdgeClipping, toggleFilter, reset, toFormData }),
    [options, setGridding, setSearchRadius, setEdgeClipping, toggleFilter, reset, toFormData],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useProcessingOptions(): ProcessingOptionsContextValue {
  const v = useContext(Ctx);
  if (!v) throw new Error('useProcessingOptions must be used inside ProcessingOptionsProvider');
  return v;
}
