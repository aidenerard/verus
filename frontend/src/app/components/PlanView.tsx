import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

interface GpsData {
  lat_start: number;
  lon_start: number;
  lat_end: number;
  lon_end: number;
  coordinates: [number, number][];  // [lat, lon] pairs
}

interface FileResult {
  filename: string;
  signals: number;
  delam_pct: number;
  gps?: GpsData | null;
}

interface PlanViewProps {
  perFileSummary: FileResult[];
}

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? '';

// Default center: George Washington Bridge, NJ/NY
const DEFAULT_CENTER: [number, number] = [-73.9519, 40.8517];

function delamToHex(pct: number): string {
  const t = Math.min(1, Math.max(0, pct / 100));
  let r: number, g: number, b: number;
  if (t <= 0.5) {
    // green (#2ECC71) → orange (#F39C12)
    const s = t * 2;
    r = Math.round(0x2e + (0xf3 - 0x2e) * s);
    g = Math.round(0xcc + (0x9c - 0xcc) * s);
    b = Math.round(0x71 + (0x12 - 0x71) * s);
  } else {
    // orange (#F39C12) → red (#E74C3C)
    const s = (t - 0.5) * 2;
    r = Math.round(0xf3 + (0xe7 - 0xf3) * s);
    g = Math.round(0x9c + (0x4c - 0x9c) * s);
    b = Math.round(0x12 + (0x3c - 0x12) * s);
  }
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

export default function PlanView({ perFileSummary }: PlanViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [overlayVisible, setOverlayVisible] = useState(true);

  const gpsFiles = perFileSummary.filter(f => f.gps != null);
  const hasGPS   = gpsFiles.length > 0;

  const center: [number, number] = hasGPS
    ? [gpsFiles[0].gps!.lon_start, gpsFiles[0].gps!.lat_start]
    : DEFAULT_CENTER;

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    mapboxgl.accessToken = MAPBOX_TOKEN;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center,
      zoom: hasGPS ? 17 : 5,
    });

    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    map.current.on('load', () => {
      const m = map.current;
      if (!m) return;

      if (!hasGPS) return;

      // Build GeoJSON: each GPS-tagged scan line → LineString from coordinates array
      // coordinates are [lat, lon] pairs from the backend; Mapbox needs [lon, lat]
      const features: GeoJSON.Feature<GeoJSON.LineString>[] = gpsFiles.map((f) => ({
        type: 'Feature',
        properties: {
          filename:  f.filename,
          delam_pct: f.delam_pct,
        },
        geometry: {
          type: 'LineString',
          coordinates: f.gps!.coordinates.map(([lat, lon]) => [lon, lat]),
        },
      }));

      m.addSource('condition', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features },
      });

      // Scan-line color driven by delam_pct interpolation
      m.addLayer({
        id: 'condition-fill',
        type: 'line',
        source: 'condition',
        paint: {
          'line-color': [
            'interpolate', ['linear'], ['get', 'delam_pct'],
            0,   '#2ECC71',
            50,  '#F39C12',
            100, '#E74C3C',
          ],
          'line-width': 4,
          'line-opacity': 0.9,
        },
      });

      // Wider transparent hit-area layer for hover popup
      m.addLayer({
        id: 'condition-outline',
        type: 'line',
        source: 'condition',
        paint: {
          'line-color': 'transparent',
          'line-width': 16,
          'line-opacity': 0,
        },
      });

      // Hover popup
      const popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: false,
      });

      // Use the wide transparent hit-area for hover detection
      m.on('mouseenter', 'condition-outline', (e) => {
        m.getCanvas().style.cursor = 'pointer';
        if (!e.features?.length) return;
        const props = e.features[0].properties!;
        popup
          .setLngLat(e.lngLat)
          .setHTML(
            `<div style="font-family:Inter,sans-serif;font-size:12px;color:#2C3E50;padding:2px 4px">
              <strong>${props.filename}</strong><br/>
              Delamination: <strong style="color:${delamToHex(props.delam_pct)}">${(+props.delam_pct).toFixed(1)}%</strong>
            </div>`
          )
          .addTo(m);
      });

      m.on('mouseleave', 'condition-outline', () => {
        m.getCanvas().style.cursor = '';
        popup.remove();
      });
    });

    return () => {
      map.current?.remove();
      map.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync overlay visibility toggle
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const vis = overlayVisible ? 'visible' : 'none';
    if (m.getLayer('condition-fill'))   m.setLayoutProperty('condition-fill',   'visibility', vis);
    if (m.getLayer('condition-outline')) m.setLayoutProperty('condition-outline', 'visibility', vis);
  }, [overlayVisible]);

  if (!MAPBOX_TOKEN) {
    return (
      <div style={{ padding: 40, textAlign: 'center' }}>
        <p style={{ fontSize: 13, color: '#6C757D' }}>
          Mapbox token not configured.{' '}
          Set <code style={{ background: '#F8F9FA', padding: '1px 4px' }}>VITE_MAPBOX_TOKEN</code>{' '}
          in <code style={{ background: '#F8F9FA', padding: '1px 4px' }}>.env.local</code> and restart the dev server.
        </p>
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      {/* No-GPS banner */}
      {!hasGPS && (
        <div style={{
          position: 'absolute', top: 12, left: 12, right: 12, zIndex: 10,
          background: '#FFF8E1', border: '1.5px solid #F39C12',
          padding: '10px 14px', borderRadius: 4,
          pointerEvents: 'none',
        }}>
          <p style={{ fontSize: 12, color: '#7D5A00', margin: 0 }}>
            No GPS data detected in uploaded files. Map overlay requires GPS-tagged scan data.
          </p>
        </div>
      )}

      {/* Controls */}
      <div style={{
        position: 'absolute', top: 12, left: 12, zIndex: 10,
        display: 'flex', gap: 8,
      }}>
        {hasGPS && (
          <button
            onClick={() => setOverlayVisible(v => !v)}
            style={{
              background: '#0D47A1', color: '#FFFFFF',
              border: 'none', padding: '6px 14px',
              fontSize: 11, fontWeight: 700, letterSpacing: '0.05em',
              textTransform: 'uppercase', cursor: 'pointer',
              borderRadius: 3,
            }}
          >
            {overlayVisible ? 'Hide Overlay' : 'Show Overlay'}
          </button>
        )}
      </div>

      {/* Map container */}
      <div ref={mapContainer} style={{ width: '100%', height: 500 }} />

      {/* Color legend (only when GPS overlay is active) */}
      {hasGPS && overlayVisible && (
        <div style={{
          position: 'absolute', bottom: 32, left: 12, zIndex: 10,
          background: 'rgba(255,255,255,0.92)', border: '1px solid #DEE2E6',
          padding: '8px 12px', borderRadius: 4, fontSize: 10,
          fontWeight: 600, color: '#2C3E50',
        }}>
          <div style={{ marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#6C757D' }}>
            Delamination
          </div>
          <div style={{
            width: 140, height: 10, borderRadius: 2,
            background: 'linear-gradient(to right, #2ECC71, #F39C12, #E74C3C)',
            marginBottom: 4,
          }} />
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>0% (Sound)</span>
            <span>100% (Deteriorated)</span>
          </div>
        </div>
      )}
    </div>
  );
}
