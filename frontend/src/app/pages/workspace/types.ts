import type { ComponentType } from 'react';

export type GriddingAlgorithm =
  | 'linear'
  | 'kriging'
  | 'natural_neighbor'
  | 'minimum_curvature';

export interface ProcessingFilters {
  backgroundRemoval: boolean;
  gain:              boolean;
  bandpass:          boolean;
}

export interface ProcessingOptions {
  gridding:      GriddingAlgorithm;
  searchRadius:  number;
  edgeClipping:  boolean;
  filters:       ProcessingFilters;
  // Real deck dimensions (feet) — used to scale the depth-map axes. 0 = unknown.
  bridgeLengthFt: number;
  bridgeWidthFt:  number;
}

export interface MethodMeta {
  id:       string;
  name:     string;
  fullName: string;
  path:     string;
  status:   'available' | 'coming-soon';
  Icon:     ComponentType<{ size?: number; style?: React.CSSProperties }>;
}

export interface ModuleMeta {
  id:      string;
  label:   string;
  methods: MethodMeta[];
}

export interface PlaceholderContent {
  name:        string;
  description: string;
  useCases:    string[];
  standard?:   string;
}

export const DEFAULT_PROCESSING_OPTIONS: ProcessingOptions = {
  gridding:     'linear',
  searchRadius: 1.0,
  edgeClipping: true,
  filters: {
    backgroundRemoval: true,
    gain:              false,
    bandpass:          false,
  },
  bridgeLengthFt: 0,
  bridgeWidthFt:  0,
};
