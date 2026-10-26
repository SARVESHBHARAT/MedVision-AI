import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface MedicalScene3DProps {
  className?: string;
}

/**
 * Cinematic 3D medical imaging hero scene.
 *
 * Renders layered scan planes, a volumetric brain hemisphere,
 * floating diagnostic data particles, and orbital scan rings —
 * all with midnight-navy fog, cyan/teal accent lighting,
 * and smooth mouse-parallax camera. Includes graceful WebGL fallback.
 */
export const MedicalScene3D: React.FC<MedicalScene3DProps> = ({ className = '' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [webglSupported, setWebglSupported] = useState<boolean>(true);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Check WebGL availability safely
    try {
      const testCanvas = document.createElement('canvas');
      const hasWebGL = !!(
        window.WebGLRenderingContext &&
        (testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl'))
      );
      if (!hasWebGL) {
        setWebglSupported(false);
        return;
      }
    } catch {
      setWebglSupported(false);
      return;
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const isMobile = window.innerWidth < 768;

    // ── Scene ──
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050811, 0.06);

    const camera = new THREE.PerspectiveCamera(
      40,
      container.clientWidth / container.clientHeight,
      0.1,
      100
    );
    camera.position.set(0, 0.5, 8);

    // ── Renderer with error boundary ──
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: !isMobile,
        alpha: true,
        powerPreference: 'high-performance',
      });
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, isMobile ? 1.5 : 2));
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.2;
      container.appendChild(renderer.domElement);
    } catch {
      setWebglSupported(false);
      return;
    }

    // ── Lighting: cinematic medical ──
    const ambientLight = new THREE.AmbientLight(0x0a1628, 3.0);
    scene.add(ambientLight);

    // Key light: cool white from upper right
    const keyLight = new THREE.PointLight(0xd4e5ff, 2.5, 25);
    keyLight.position.set(5, 6, 5);
    scene.add(keyLight);

    // Accent light: cyan fill from lower left
    const cyanLight = new THREE.PointLight(0x00f2fe, 1.8, 20);
    cyanLight.position.set(-5, -2, 4);
    scene.add(cyanLight);

    // Rim light: teal from behind
    const tealRim = new THREE.PointLight(0x00d4aa, 0.8, 18);
    tealRim.position.set(0, 3, -5);
    scene.add(tealRim);

    // ── Root group ──
    const group = new THREE.Group();
    scene.add(group);

    // ────────────────────────────────────────
    //  1. VOLUMETRIC BRAIN HEMISPHERE
    //  Procedural neural lattice with
    //  spatially-distributed particles and synaptic connections
    // ────────────────────────────────────────
    const nodeCount = isMobile ? 140 : 300;
    const nodePositions: THREE.Vector3[] = [];
    const positionsFloat = new Float32Array(nodeCount * 3);
    const colorsFloat = new Float32Array(nodeCount * 3);
    const sizesFloat = new Float32Array(nodeCount);

    const colorCyan = new THREE.Color(0x00f2fe);
    const colorTeal = new THREE.Color(0x00d4aa);
    const colorWhite = new THREE.Color(0xd4e5ff);
    const colorDim = new THREE.Color(0x1e3a5f);

    for (let i = 0; i < nodeCount; i++) {
      const isLeft = i % 2 === 0;
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);

      // Brain-like ellipsoid with cortical folds
      const r = 1.5 + Math.sin(phi * 4) * 0.15 + (Math.random() - 0.5) * 0.25;

      const x = (r * Math.sin(phi) * Math.cos(theta)) * 0.8 + (isLeft ? -0.35 : 0.35);
      const y = (r * Math.sin(phi) * Math.sin(theta)) * 0.9;
      const z = (r * Math.cos(phi)) * 1.3;

      const pos = new THREE.Vector3(x, y, z);
      nodePositions.push(pos);

      positionsFloat[i * 3] = x;
      positionsFloat[i * 3 + 1] = y;
      positionsFloat[i * 3 + 2] = z;

      // Depth-based color: cyan near front, teal at sides, dim deep
      const depth = (z + 2) / 4;
      const rand = Math.random();
      let mixedColor: THREE.Color;
      if (rand > 0.92) mixedColor = colorWhite;
      else if (rand > 0.7) mixedColor = colorCyan;
      else if (depth > 0.6) mixedColor = colorTeal;
      else mixedColor = colorDim;

      colorsFloat[i * 3] = mixedColor.r;
      colorsFloat[i * 3 + 1] = mixedColor.g;
      colorsFloat[i * 3 + 2] = mixedColor.b;

      sizesFloat[i] = 0.04 + Math.random() * 0.06;
    }

    // Neural point cloud
    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positionsFloat, 3));
    pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colorsFloat, 3));

    const pointsMaterial = new THREE.PointsMaterial({
      size: isMobile ? 0.06 : 0.08,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    const pointsMesh = new THREE.Points(pointsGeometry, pointsMaterial);
    group.add(pointsMesh);

    // Synaptic connections
    const lineIndices: number[] = [];
    const maxDist = isMobile ? 0.85 : 1.1;
    for (let i = 0; i < nodeCount; i++) {
      let connections = 0;
      for (let j = i + 1; j < nodeCount; j++) {
        if (connections >= (isMobile ? 2 : 3)) break;
        const d = nodePositions[i].distanceTo(nodePositions[j]);
        if (d < maxDist) {
          lineIndices.push(i, j);
          connections++;
        }
      }
    }

    const linesGeometry = new THREE.BufferGeometry();
    linesGeometry.setAttribute('position', new THREE.BufferAttribute(positionsFloat, 3));
    linesGeometry.setIndex(lineIndices);

    const linesMaterial = new THREE.LineBasicMaterial({
      color: 0x00f2fe,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending,
    });
    const linesMesh = new THREE.LineSegments(linesGeometry, linesMaterial);
    group.add(linesMesh);

    // ────────────────────────────────────────
    //  2. LAYERED SCAN PLANES
    //  Transparent cross-section slices
    //  giving volumetric depth to the brain
    // ────────────────────────────────────────
    const scanPlanes: THREE.Mesh[] = [];
    const planeCount = isMobile ? 3 : 5;
    for (let i = 0; i < planeCount; i++) {
      const geo = new THREE.PlaneGeometry(3.5, 3.5);
      const mat = new THREE.MeshBasicMaterial({
        color: 0x00f2fe,
        transparent: true,
        opacity: 0.025 + i * 0.005,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
      });
      const plane = new THREE.Mesh(geo, mat);
      const offset = (i - Math.floor(planeCount / 2)) * 0.5;
      plane.position.z = offset;
      plane.rotation.y = Math.PI * 0.02 * i;
      scanPlanes.push(plane);
      group.add(plane);
    }

    // ────────────────────────────────────────
    //  3. ORBITAL SCAN RINGS
    //  Thin diagnostic rings at different angles
    // ────────────────────────────────────────
    const rings: THREE.Mesh[] = [];
    const ringData = [
      { radius: 2.6, tube: 0.005, rx: Math.PI / 3.5, ry: 0, color: 0x00f2fe, op: 0.2 },
      { radius: 2.9, tube: 0.004, rx: 0, ry: Math.PI / 4, color: 0x00d4aa, op: 0.15 },
      { radius: 3.2, tube: 0.003, rx: Math.PI / 2.5, ry: Math.PI / 6, color: 0x0891b2, op: 0.1 },
    ];

    for (const rd of ringData) {
      const geo = new THREE.TorusGeometry(rd.radius, rd.tube, 16, 120);
      const mat = new THREE.MeshBasicMaterial({
        color: rd.color,
        transparent: true,
        opacity: rd.op,
        blending: THREE.AdditiveBlending,
      });
      const ring = new THREE.Mesh(geo, mat);
      ring.rotation.x = rd.rx;
      ring.rotation.y = rd.ry;
      rings.push(ring);
      group.add(ring);
    }

    // ────────────────────────────────────────
    //  4. AMBIENT MEDICAL PARTICLES
    //  Slow-drifting fine data points
    // ────────────────────────────────────────
    const ambientCount = isMobile ? 60 : 150;
    const ambientPos = new Float32Array(ambientCount * 3);
    const ambientColors = new Float32Array(ambientCount * 3);
    const initialAmbientY = new Float32Array(ambientCount);
    for (let i = 0; i < ambientCount; i++) {
      ambientPos[i * 3]     = (Math.random() - 0.5) * 12;
      const yVal = (Math.random() - 0.5) * 8;
      ambientPos[i * 3 + 1] = yVal;
      initialAmbientY[i]    = yVal;
      ambientPos[i * 3 + 2] = (Math.random() - 0.5) * 8;
      const c = Math.random() > 0.5 ? colorCyan : colorTeal;
      ambientColors[i * 3] = c.r;
      ambientColors[i * 3 + 1] = c.g;
      ambientColors[i * 3 + 2] = c.b;
    }
    const ambientGeo = new THREE.BufferGeometry();
    ambientGeo.setAttribute('position', new THREE.BufferAttribute(ambientPos, 3));
    ambientGeo.setAttribute('color', new THREE.BufferAttribute(ambientColors, 3));
    const ambientMat = new THREE.PointsMaterial({
      size: 0.025,
      vertexColors: true,
      transparent: true,
      opacity: 0.4,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    const ambientMesh = new THREE.Points(ambientGeo, ambientMat);
    scene.add(ambientMesh);

    // ────────────────────────────────────────
    //  5. MOUSE PARALLAX + ANIMATION
    // ────────────────────────────────────────
    let mouseX = 0;
    let mouseY = 0;
    let currentX = 0;
    let currentY = 0;

    const handleMouseMove = (e: MouseEvent) => {
      if (prefersReducedMotion) return;
      const rect = container.getBoundingClientRect();
      mouseX = ((e.clientX - rect.left) / rect.width - 0.5) * 1.5;
      mouseY = ((e.clientY - rect.top) / rect.height - 0.5) * 1.0;
    };
    window.addEventListener('mousemove', handleMouseMove);

    let animationFrameId: number;
    const startTime = performance.now();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsed = (performance.now() - startTime) / 1000;

      if (!prefersReducedMotion) {
        // Continuous smooth base rotation
        const baseSpeed = isMobile ? 0.0015 : 0.0022;
        group.rotation.y += baseSpeed;

        // Smooth camera parallax without angle wrapping artifacts
        currentX += (mouseX - currentX) * 0.04;
        currentY += (mouseY - currentY) * 0.04;
        camera.position.x = currentX * 1.8;
        camera.position.y = 0.5 - currentY * 1.2;
        camera.lookAt(0, 0, 0);

        // Gentle vertical floating
        group.position.y = Math.sin(elapsed * 0.7) * 0.08;

        // Orbital ring drift
        rings.forEach((ring, i) => {
          ring.rotation.z += 0.0006 * (i % 2 === 0 ? 1 : -1);
        });

        // Scan plane breathing
        scanPlanes.forEach((plane, i) => {
          const phase = elapsed * 0.4 + i * 0.8;
          (plane.material as THREE.MeshBasicMaterial).opacity =
            0.025 + Math.sin(phase) * 0.015;
        });

        // Ambient particle drift (stable periodic oscillation, no runaway drift)
        const ambPositions = ambientGeo.attributes.position as THREE.BufferAttribute;
        for (let i = 0; i < ambientCount; i++) {
          ambPositions.array[i * 3 + 1] = initialAmbientY[i] + Math.sin(elapsed * 0.5 + i) * 0.15;
        }
        ambPositions.needsUpdate = true;

        // Dynamic light subtle sway
        cyanLight.position.x = -5 + Math.sin(elapsed * 0.4) * 1.5;
        cyanLight.position.y = -2 + Math.cos(elapsed * 0.3) * 1;
      }

      renderer.render(scene, camera);
    };
    animate();

    // ── Resize ──
    const handleResize = () => {
      if (!container) return;
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // ── Cleanup ──
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);

      pointsGeometry.dispose();
      pointsMaterial.dispose();
      linesGeometry.dispose();
      linesMaterial.dispose();
      ambientGeo.dispose();
      ambientMat.dispose();
      scanPlanes.forEach(p => {
        p.geometry.dispose();
        (p.material as THREE.MeshBasicMaterial).dispose();
      });
      rings.forEach(r => {
        r.geometry.dispose();
        (r.material as THREE.MeshBasicMaterial).dispose();
      });
      renderer.dispose();

      if (container && renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  if (!webglSupported) {
    return (
      <div
        className={className}
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          minHeight: '420px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
          borderRadius: '16px',
          background: 'radial-gradient(ellipse at center, rgba(0, 242, 254, 0.05) 0%, rgba(5, 8, 17, 0.8) 70%)',
          border: '1px solid var(--border)',
        }}
      >
        <div style={{
          position: 'relative',
          width: '280px',
          height: '280px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <div style={{
            position: 'absolute',
            inset: 0,
            borderRadius: '50%',
            border: '1px solid rgba(0, 242, 254, 0.25)',
            boxShadow: '0 0 25px rgba(0, 242, 254, 0.15)',
            animation: 'pulse-ring 3s ease-in-out infinite',
          }} />
          <div style={{
            position: 'absolute',
            inset: '30px',
            borderRadius: '50%',
            border: '1px dashed rgba(0, 212, 170, 0.3)',
          }} />
          <div style={{
            position: 'absolute',
            inset: '60px',
            borderRadius: '50%',
            border: '1px solid rgba(0, 242, 254, 0.4)',
          }} />
          <div style={{
            textAlign: 'center',
            zIndex: 1,
            padding: '1rem',
          }}>
            <div style={{
              fontSize: '2.5rem',
              marginBottom: '0.5rem',
              filter: 'drop-shadow(0 0 12px rgba(0,242,254,0.4))',
            }}>
              🧠
            </div>
            <div style={{
              fontSize: '0.65rem',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: 'var(--accent)',
              fontWeight: 600,
            }}>
              Neural Core Active
            </div>
            <div style={{
              fontSize: '0.55rem',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              marginTop: '4px',
            }}>
              Diagnostic Cross-Section
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        minHeight: '420px',
        pointerEvents: 'none',
      }}
      aria-hidden="true"
    />
  );
};
