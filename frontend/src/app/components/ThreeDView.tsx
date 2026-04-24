import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface FileResult {
  filename: string;
  signals: number;
  delam_pct: number;
}

interface ThreeDViewProps {
  perFileSummary: FileResult[];
}

interface TooltipState {
  x: number;
  y: number;
  lines: string[];
}

const BG_COLOR   = '#0a1628';
const CELL_W     = 1.0;
const CELL_H     = 0.18;
const CELL_D     = 1.0;
const GAP        = 0.08;
const GRID_ROWS  = 24;  // visual rows per scan line (depth axis)
const HEIGHT_PX  = 500;

function delamToThreeColor(pct: number): THREE.Color {
  const t = Math.min(1, Math.max(0, pct / 100));
  let r: number, g: number, b: number;
  if (t <= 0.5) {
    const s = t * 2;
    r = (0x2e + (0xf3 - 0x2e) * s) / 255;
    g = (0xcc + (0x9c - 0xcc) * s) / 255;
    b = (0x71 + (0x12 - 0x71) * s) / 255;
  } else {
    const s = (t - 0.5) * 2;
    r = (0xf3 + (0xe7 - 0xf3) * s) / 255;
    g = (0x9c + (0x4c - 0x9c) * s) / 255;
    b = (0x12 + (0x3c - 0x12) * s) / 255;
  }
  return new THREE.Color(r, g, b);
}

export default function ThreeDView({ perFileSummary }: ThreeDViewProps) {
  const wrapperRef   = useRef<HTMLDivElement>(null);
  const canvasRef    = useRef<HTMLDivElement>(null);
  const rendererRef  = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef    = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef  = useRef<OrbitControls | null>(null);
  const meshesRef    = useRef<THREE.Mesh[]>([]);
  const rafRef       = useRef<number>(0);

  const defaultCamPos    = useRef(new THREE.Vector3());
  const defaultCamTarget = useRef(new THREE.Vector3());

  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  useEffect(() => {
    const container = canvasRef.current;
    if (!container) return;

    const W = container.clientWidth || 800;
    const H = HEIGHT_PX;
    const N = perFileSummary.length;
    const M = GRID_ROWS;

    // ── Scene ───────────────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BG_COLOR);

    // subtle grid helper on the ground plane
    const gridSize   = Math.max(N, M) * (CELL_W + GAP) + 2;
    const gridHelper = new THREE.GridHelper(gridSize, 20, 0x1a2f5a, 0x1a2f5a);
    gridHelper.position.set(
      (N * (CELL_W + GAP)) / 2,
      -CELL_H / 2 - 0.01,
      (M * (CELL_D + GAP)) / 2
    );
    scene.add(gridHelper);

    // ── Camera ──────────────────────────────────────────────────────────────
    const camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 2000);
    const cx = (N * (CELL_W + GAP)) / 2;
    const cz = (M * (CELL_D + GAP)) / 2;
    const dist = Math.max(N, M) * 1.6;
    camera.position.set(cx, dist * 0.9, cz + dist * 1.1);
    camera.lookAt(cx, 0, cz);
    defaultCamPos.current.copy(camera.position);
    defaultCamTarget.current.set(cx, 0, cz);
    cameraRef.current = camera;

    // ── Renderer ────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // ── Lights ──────────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(cx + 10, 30, cz - 10);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x8899cc, 0.3);
    fill.position.set(cx - 10, 10, cz + 20);
    scene.add(fill);

    // ── Controls ────────────────────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(cx, 0, cz);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();
    controlsRef.current = controls;

    // ── Meshes (shared geometry, per-file material) ──────────────────────────
    const geo = new THREE.BoxGeometry(CELL_W, CELL_H, CELL_D);
    const meshes: THREE.Mesh[] = [];

    perFileSummary.forEach((file, fi) => {
      const color = delamToThreeColor(file.delam_pct);
      const mat   = new THREE.MeshLambertMaterial({ color });

      for (let mi = 0; mi < M; mi++) {
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(
          fi * (CELL_W + GAP) + CELL_W / 2,
          0,
          mi * (CELL_D + GAP) + CELL_D / 2,
        );
        mesh.userData = {
          filename:  file.filename,
          scanLine:  fi + 1,
          signalIdx: mi + 1,
          delam_pct: file.delam_pct,
        };
        scene.add(mesh);
        meshes.push(mesh);
      }
    });

    meshesRef.current = meshes;

    // ── Animate ─────────────────────────────────────────────────────────────
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // ── Resize ──────────────────────────────────────────────────────────────
    const onResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      camera.aspect = w / H;
      camera.updateProjectionMatrix();
      renderer.setSize(w, H);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', onResize);
      controls.dispose();
      renderer.dispose();
      geo.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  // Run once on mount; perFileSummary won't change for a given result set.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Hover raycasting ──────────────────────────────────────────────────────
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const camera   = cameraRef.current;
    const renderer = rendererRef.current;
    const meshes   = meshesRef.current;
    if (!camera || !renderer || meshes.length === 0) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width)  *  2 - 1,
      ((e.clientY - rect.top)  / rect.height) * -2 + 1,
    );

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(meshes);

    if (hits.length > 0) {
      const { filename, scanLine, signalIdx, delam_pct } = hits[0].object.userData;
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        lines: [
          filename,
          `Scan line: ${scanLine}`,
          `Signal: ${signalIdx}`,
          `Delamination: ${(+delam_pct).toFixed(1)}%`,
        ],
      });
    } else {
      setTooltip(null);
    }
  }, []);

  const resetView = useCallback(() => {
    const camera   = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    camera.position.copy(defaultCamPos.current);
    controls.target.copy(defaultCamTarget.current);
    controls.update();
  }, []);

  return (
    <div
      ref={wrapperRef}
      style={{ position: 'relative', background: BG_COLOR, width: '100%', height: HEIGHT_PX }}
    >
      {/* Three.js canvas target */}
      <div
        ref={canvasRef}
        style={{ width: '100%', height: HEIGHT_PX }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />

      {/* Reset View button */}
      <button
        onClick={resetView}
        style={{
          position: 'absolute', top: 12, right: 12,
          background: '#0D47A1', color: '#FFFFFF',
          border: 'none', padding: '6px 14px',
          fontSize: 11, fontWeight: 700,
          letterSpacing: '0.05em', textTransform: 'uppercase',
          cursor: 'pointer', borderRadius: 3, zIndex: 10,
        }}
      >
        Reset View
      </button>

      {/* Hover tooltip */}
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: tooltip.x + 14,
            top:  tooltip.y - 14,
            background: 'rgba(10,22,40,0.92)',
            color: '#FFFFFF',
            padding: '6px 10px',
            fontSize: 11,
            lineHeight: 1.6,
            borderRadius: 3,
            pointerEvents: 'none',
            zIndex: 20,
            border: '1px solid rgba(255,255,255,0.12)',
            maxWidth: 220,
          }}
        >
          {tooltip.lines.map((line, i) => (
            <div key={i} style={{ fontWeight: i === 0 ? 700 : 400 }}>{line}</div>
          ))}
        </div>
      )}

      {/* Color legend */}
      <div style={{
        position: 'absolute', bottom: 16, left: 12, zIndex: 10,
        background: 'rgba(10,22,40,0.85)', border: '1px solid rgba(255,255,255,0.12)',
        padding: '8px 12px', borderRadius: 3,
        fontSize: 10, fontWeight: 600, color: '#aab4c8',
      }}>
        <div style={{ marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Delamination
        </div>
        <div style={{
          width: 130, height: 8, borderRadius: 2,
          background: 'linear-gradient(to right, #2ECC71, #F39C12, #E74C3C)',
          marginBottom: 4,
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9 }}>
          <span>0% Sound</span>
          <span>100% Delam</span>
        </div>
      </div>

      {/* Instructions */}
      <div style={{
        position: 'absolute', bottom: 16, right: 12, zIndex: 10,
        fontSize: 9, color: 'rgba(170,180,200,0.7)',
        textAlign: 'right', lineHeight: 1.7,
      }}>
        Drag to rotate · Scroll to zoom · Right-drag to pan
      </div>
    </div>
  );
}
