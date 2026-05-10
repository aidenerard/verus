/**
 * useMapbox.ts
 * Manages Mapbox GL map lifecycle: initialisation, GPS layer, layer visibility,
 * and opacity. Exposes refs so the parent can export the canvas.
 *
 * Does NOT: handle file uploads, analysis jobs, canvas renderers, or UI state
 * beyond map-related mouse coordinates.
 */

import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

import { MAPBOX_TOKEN, DEFAULT_CENTER } from './constants';
import type { LayerId } from './constants';
import type { AnalysisResult } from './types';

interface UseMapboxProps {
  analysisResult: AnalysisResult | null;
  layerVis: Record<LayerId, boolean>;
  conditionOpacity: number;
}

interface UseMapboxReturn {
  mapContainerRef: React.RefObject<HTMLDivElement>;
  mapRef: React.MutableRefObject<mapboxgl.Map | null>;
  mouseCoords: { x: number; y: number } | null;
}

export function useMapbox({
  analysisResult,
  layerVis,
  conditionOpacity,
}: UseMapboxProps): UseMapboxReturn {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef          = useRef<mapboxgl.Map | null>(null);
  const [mouseCoords, setMouseCoords] = useState<{ x: number; y: number } | null>(null);

  // Map initialisation
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current || !MAPBOX_TOKEN) return;
    mapboxgl.accessToken = MAPBOX_TOKEN;
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: DEFAULT_CENTER, zoom: 5,
    });
    map.addControl(new mapboxgl.NavigationControl({ visualizePitch: false }), 'bottom-right');
    map.addControl(new mapboxgl.ScaleControl({ unit: 'imperial' }), 'bottom-left');
    map.on('mousemove', (e) => {
      const c  = map.getCenter();
      const dx = (e.lngLat.lng - c.lng) * 111320 * Math.cos(c.lat * Math.PI / 180) * 3.28084;
      const dy = (e.lngLat.lat - c.lat) * 110540 * 3.28084;
      setMouseCoords({ x: Math.round(dx), y: Math.round(dy) });
    });
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  // GPS layer
  useEffect(() => {
    const map = analysisResult && mapRef.current ? mapRef.current : null;
    if (!map || !analysisResult) return;
    const add = () => {
      ['condition-fill'].forEach(id => { if (map.getLayer(id)) map.removeLayer(id); });
      if (map.getSource('condition')) map.removeSource('condition');
      const gpsFiles = analysisResult.per_file_summary.filter(f => f.gps);
      if (!gpsFiles.length) return;
      const features: GeoJSON.Feature<GeoJSON.LineString>[] = gpsFiles.map(f => ({
        type: 'Feature',
        properties: { filename: f.filename, delam_pct: f.delam_pct },
        geometry: { type: 'LineString', coordinates: f.gps!.coordinates.map(([lat, lon]) => [lon, lat]) },
      }));
      map.addSource('condition', { type: 'geojson', data: { type: 'FeatureCollection', features } });
      map.addLayer({
        id: 'condition-fill', type: 'line', source: 'condition',
        layout: { visibility: layerVis.condition ? 'visible' : 'none' },
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'delam_pct'], 0, '#22c55e', 50, '#f59e0b', 100, '#ef4444'],
          'line-width': 5, 'line-opacity': conditionOpacity / 100,
        },
      });
      const first = gpsFiles[0].gps!;
      map.flyTo({ center: [first.lon_start, first.lat_start], zoom: 17, duration: 1500 });
    };
    if (map.isStyleLoaded()) add(); else map.once('load', add);
  }, [analysisResult]); // eslint-disable-line

  // Layer visibility toggle
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (map.getLayer('condition-fill'))
      map.setLayoutProperty('condition-fill', 'visibility', layerVis.condition ? 'visible' : 'none');
  }, [layerVis.condition]);

  // Opacity change
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getLayer('condition-fill')) return;
    map.setPaintProperty('condition-fill', 'line-opacity', conditionOpacity / 100);
  }, [conditionOpacity]);

  return { mapContainerRef, mapRef, mouseCoords };
}
