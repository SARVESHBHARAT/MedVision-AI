import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import { motion } from 'motion/react';
import {
  CheckCircle2,
  AlertTriangle,
  Clock,
  Activity,
  ArrowDown,
  RefreshCw,
} from 'lucide-react';
import type { PipelineStage } from '../App';

interface Diagnostic3DLoaderProps {
  isAnalyzing: boolean;
  pipelineStages: PipelineStage[];
  selectedModality: string;
  imagePreviewUrl?: string | null;
  executionError?: string | null;
  onRetry?: () => void;
  onScrollToResults?: () => void;
  generateReport?: boolean;
}

/**
 * Stage weights for computing real, deterministic progress.
 * If Gemma report is disabled by user, weights re-normalize cleanly to 100%.
 */
const BASE_WEIGHTS: Record<string, number> = {
  upload: 15, // Image Normalization & Tensor Ingest
  cnn: 30,    // ResNet50 Inference
  gradcam: 20,// Grad-CAM Activation Mapping
  shap: 15,   // SHAP Regional Attributions
  lime: 10,   // LIME Superpixel Surrogate
  gemma: 10,  // Gemma 4 Synthesis
};

export const Diagnostic3DLoader: React.FC<Diagnostic3DLoaderProps> = ({
  isAnalyzing,
  pipelineStages,
  selectedModality,
  imagePreviewUrl,
  executionError,
  onRetry,
  onScrollToResults,
  generateReport = true,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [webglSupported, setWebglSupported] = useState<boolean>(true);
  const [elapsedSec, setElapsedSec] = useState<number>(0);

  // Measure elapsed time during analysis
  useEffect(() => {
    if (!isAnalyzing) return;
    setElapsedSec(0);
    const t0 = Date.now();
    const interval = window.setInterval(() => {
      setElapsedSec(Math.round((Date.now() - t0) / 100) / 10);
    }, 100);
    return () => window.clearInterval(interval);
  }, [isAnalyzing]);

  // Compute actual, verified progress from real stage states
  const { progressPct, currentStage, isFailed, isComplete } = useMemo(() => {
    // Filter out unrequested or skipped stages
    const activeStages = pipelineStages.filter(
      (s) => (s.id !== 'gemma' || generateReport) && s.status !== 'skipped'
    );

    const totalWeight = activeStages.reduce(
      (acc, s) => acc + (BASE_WEIGHTS[s.id] || 15),
      0
    );

    let banked = 0;
    let runningStage: PipelineStage | null = null;
    let failedStage: PipelineStage | null = null;

    activeStages.forEach((stage) => {
      const weight = (BASE_WEIGHTS[stage.id] || 15) / (totalWeight || 1);
      if (stage.status === 'done') {
        banked += weight * 100;
      } else if (stage.status === 'running') {
        runningStage = stage;
        // Sub-progress within the active stage (steady deterministic advance)
        banked += weight * 50;
      } else if (stage.status === 'error' || stage.status === 'timeout') {
        failedStage = stage;
      }
    });

    const failed = Boolean(failedStage || executionError);
    const complete = !failed && !isAnalyzing && activeStages.length > 0 && activeStages.every((s) => s.status === 'done');

    const roundedPct = complete
      ? 100
      : failed
      ? Math.min(99, Math.round(banked))
      : Math.min(98, Math.max(isAnalyzing ? 8 : 0, Math.round(banked)));

    return {
      progressPct: roundedPct,
      currentStage: runningStage || failedStage || activeStages.find((s) => s.status === 'pending') || null,
      isFailed: failed,
      isComplete: complete,
    };
  }, [pipelineStages, isAnalyzing, executionError, generateReport]);

  // Three.js Volumetric Holographic Medical Scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // Test WebGL availability safely
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) {
        setWebglSupported(false);
        return;
      }
    } catch {
      setWebglSupported(false);
      return;
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const isMobile = window.innerWidth < 768;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.05);

    const width = container.clientWidth || 360;
    const height = container.clientHeight || 320;

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.set(0, 0.3, 6.2);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: !isMobile,
        alpha: true,
        powerPreference: 'high-performance',
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, isMobile ? 1.5 : 2));
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.3;
      container.appendChild(renderer.domElement);
    } catch {
      setWebglSupported(false);
      return;
    }

    // ── Lighting ──
    const ambient = new THREE.AmbientLight(0x050e1a, 2.5);
    scene.add(ambient);

    const cyanKey = new THREE.PointLight(0x00e5ff, 3.2, 25);
    cyanKey.position.set(4, 5, 5);
    scene.add(cyanKey);

    const tealFill = new THREE.PointLight(0x14b8a6, 2.0, 20);
    tealFill.position.set(-4, -3, 3);
    scene.add(tealFill);

    const topWhite = new THREE.PointLight(0xf8fafc, 1.2, 15);
    topWhite.position.set(0, 6, 2);
    scene.add(topWhite);

    // ── 3D Visual Hierarchy ──
    const rootGroup = new THREE.Group();
    scene.add(rootGroup);

    // 1. Volumetric Neural Core (Geodesic Wireframe Sphere)
    const nucleusGeo = new THREE.IcosahedronGeometry(1.25, 2);
    const nucleusMat = new THREE.MeshStandardMaterial({
      color: 0x00e5ff,
      wireframe: true,
      transparent: true,
      opacity: 0.65,
      roughness: 0.2,
      metalness: 0.8,
    });
    const nucleus = new THREE.Mesh(nucleusGeo, nucleusMat);
    rootGroup.add(nucleus);

    // Glowing Inner Nucleus Core
    const glowGeo = new THREE.SphereGeometry(0.75, 24, 24);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.2,
    });
    const glowSphere = new THREE.Mesh(glowGeo, glowMat);
    rootGroup.add(glowSphere);

    // 2. Multi-Axial Scan Rings (Layered Medical Tomography Disks)
    const ringsGroup = new THREE.Group();
    rootGroup.add(ringsGroup);

    const ringRadii = [1.8, 2.15, 2.45];
    const ringColors = [0x00e5ff, 0x14b8a6, 0x38bdf8];
    const ringMeshes: THREE.Mesh[] = [];

    ringRadii.forEach((r, i) => {
      const ringGeo = new THREE.TorusGeometry(r, 0.018, 16, 96);
      const ringMat = new THREE.MeshStandardMaterial({
        color: ringColors[i],
        emissive: ringColors[i],
        emissiveIntensity: 0.6,
        roughness: 0.3,
        metalness: 0.9,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = (Math.PI / 4) * (i + 1);
      ring.rotation.y = (Math.PI / 6) * i;
      ringsGroup.add(ring);
      ringMeshes.push(ring);
    });

    // 3. Synaptic Particle Lattice (ResNet50 Feature Points)
    const particleCount = isMobile ? 120 : 240;
    const posArray = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const radius = 1.35 + Math.random() * 1.2;
      posArray[i] = radius * Math.sin(phi) * Math.cos(theta);
      posArray[i + 1] = radius * Math.sin(phi) * Math.sin(theta);
      posArray[i + 2] = radius * Math.cos(phi);
    }
    const particlesGeo = new THREE.BufferGeometry();
    particlesGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    const particlesMat = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: isMobile ? 0.045 : 0.055,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });
    const particles = new THREE.Points(particlesGeo, particlesMat);
    rootGroup.add(particles);

    // 4. Volumetric Scanning Laser Beam Slice
    const sliceGeo = new THREE.RingGeometry(0.1, 2.3, 48);
    const sliceMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.12,
    });
    const slicePlane = new THREE.Mesh(sliceGeo, sliceMat);
    slicePlane.rotation.x = Math.PI / 2;
    rootGroup.add(slicePlane);

    // ── Animation Loop ──
    let animId = 0;
    let clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      if (!prefersReducedMotion) {
        // Base rotational dynamics
        const speed = isAnalyzing ? 1.4 : isComplete ? 0.35 : 0.8;
        rootGroup.rotation.y = elapsed * 0.3 * speed;
        nucleus.rotation.x = elapsed * 0.2 * speed;
        nucleus.rotation.z = elapsed * 0.15 * speed;

        // Counter-rotating rings
        ringMeshes.forEach((rm, idx) => {
          rm.rotation.z = elapsed * (0.2 + idx * 0.1) * (idx % 2 === 0 ? 1 : -1) * speed;
        });

        // Breathing nucleus scale
        const pulse = 1 + Math.sin(elapsed * (isAnalyzing ? 3.5 : 1.5)) * 0.04;
        nucleus.scale.set(pulse, pulse, pulse);

        // Sweeping axial scan plane
        slicePlane.position.y = Math.sin(elapsed * 2.2) * 1.4;
      }

      // Dynamic color shifts on failure or completion
      if (isFailed) {
        cyanKey.color.setHex(0xfb7185);
        nucleusMat.color.setHex(0xfb7185);
      } else if (isComplete) {
        cyanKey.color.setHex(0x34d399);
        nucleusMat.color.setHex(0x34d399);
      } else {
        cyanKey.color.setHex(0x00e5ff);
        nucleusMat.color.setHex(0x00e5ff);
      }

      renderer.render(scene, camera);
    };

    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup resources strictly
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);

      nucleusGeo.dispose();
      nucleusMat.dispose();
      glowGeo.dispose();
      glowMat.dispose();
      particlesGeo.dispose();
      particlesMat.dispose();
      sliceGeo.dispose();
      sliceMat.dispose();
      ringMeshes.forEach((rm) => {
        rm.geometry.dispose();
        (rm.material as THREE.Material).dispose();
      });

      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [isAnalyzing, isComplete, isFailed]);

  return (
    <div
      style={{
        width: '100%',
        maxWidth: '1360px',
        margin: '2rem auto',
        borderRadius: '24px',
        border: '1px solid #262626',
        background: '#040404',
        padding: '2rem 1.5rem',
        boxShadow: '0 20px 60px rgba(0, 0, 0, 0.85)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Background Radial Glow */}
      <div
        style={{
          position: 'absolute',
          top: '30%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '500px',
          height: '400px',
          background: isFailed
            ? 'radial-gradient(circle, rgba(251, 113, 133, 0.12) 0%, transparent 70%)'
            : isComplete
            ? 'radial-gradient(circle, rgba(52, 211, 153, 0.14) 0%, transparent 70%)'
            : 'radial-gradient(circle, rgba(0, 229, 255, 0.12) 0%, transparent 70%)',
          pointerEvents: 'none',
          transition: 'background 0.5s ease',
        }}
      />

      {/* Header Context Bar */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          borderBottom: '1px solid #1a1a1a',
          paddingBottom: '1.25rem',
          position: 'relative',
          zIndex: 2,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          {/* Scan thumbnail pill */}
          {imagePreviewUrl && (
            <div
              style={{
                width: '42px',
                height: '42px',
                borderRadius: '8px',
                border: '1px solid #333333',
                overflow: 'hidden',
                background: '#000000',
                flexShrink: 0,
              }}
            >
              <img
                src={imagePreviewUrl}
                alt="Active Scan"
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            </div>
          )}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span
                style={{
                  fontSize: '0.62rem',
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.16em',
                  textTransform: 'uppercase',
                  color: isFailed ? '#FB7185' : isComplete ? '#34D399' : '#00E5FF',
                  fontWeight: 600,
                }}
              >
                {selectedModality.toUpperCase()} RADIOLOGY INGESTION
              </span>
            </div>
            <h3
              style={{
                fontSize: '1.1rem',
                fontWeight: 700,
                color: '#F5F5F5',
                letterSpacing: '-0.02em',
                margin: '2px 0 0',
              }}
            >
              {isFailed
                ? 'Diagnostic Pipeline Execution Interrupted'
                : isComplete
                ? 'Inference & Explainability Verified'
                : 'Neural Computation & Multi-Modal Analysis'}
            </h3>
          </div>
        </div>

        {/* Live Status Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '20px',
              border: isFailed
                ? '1px solid rgba(251, 113, 133, 0.4)'
                : isComplete
                ? '1px solid rgba(52, 211, 153, 0.4)'
                : '1px solid rgba(0, 229, 255, 0.4)',
              background: isFailed
                ? 'rgba(251, 113, 133, 0.1)'
                : isComplete
                ? 'rgba(52, 211, 153, 0.1)'
                : 'rgba(0, 229, 255, 0.1)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.72rem',
              color: isFailed ? '#FB7185' : isComplete ? '#34D399' : '#00E5FF',
              fontWeight: 600,
            }}
          >
            {isAnalyzing && (
              <span
                style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: '#00E5FF',
                  boxShadow: '0 0 8px #00E5FF',
                }}
              />
            )}
            {isFailed
              ? 'PIPELINE ERROR'
              : isComplete
              ? '100% COMPLETE'
              : 'REAL-TIME EXECUTION'}
          </div>

          <span
            style={{
              fontSize: '0.72rem',
              fontFamily: 'var(--font-mono)',
              color: '#737373',
            }}
          >
            ⏱ {elapsedSec.toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Main 3D Core & Progress Center */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '2rem',
          alignItems: 'center',
          padding: '1.5rem 0',
          position: 'relative',
          zIndex: 2,
        }}
      >
        {/* 3D Visualizer Canvas */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: '340px',
            borderRadius: '20px',
            background: 'radial-gradient(circle at center, #070707 0%, #000000 100%)',
            border: '1px solid #1c1c1c',
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {webglSupported ? (
            <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
          ) : (
            // Graceful SVG Fallback for devices without WebGL
            <div
              style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#00E5FF',
              }}
            >
              <Activity className="size-16 animate-pulse" />
              <p
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.75rem',
                  color: '#A3A3A3',
                  marginTop: '1rem',
                }}
              >
                Volumetric Core (CSS Safe Fallback)
              </p>
            </div>
          )}

          {/* Spatial Corner Brackets */}
          <span
            style={{
              position: 'absolute',
              top: '12px',
              left: '12px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.6rem',
              color: 'rgba(0, 229, 255, 0.6)',
              letterSpacing: '0.1em',
            }}
          >
            + RESNET50.VOLUMETRIC.CORE
          </span>
          <span
            style={{
              position: 'absolute',
              bottom: '12px',
              right: '12px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.6rem',
              color: '#555555',
            }}
          >
            TENSOR 3D VIZ
          </span>
        </div>

        {/* Progress Display & Technical Feedback */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem' }}>
              <span
                style={{
                  fontSize: 'clamp(3.2rem, 6vw, 4.5rem)',
                  fontWeight: 800,
                  fontFamily: 'var(--font-mono)',
                  color: isFailed ? '#FB7185' : isComplete ? '#34D399' : '#00E5FF',
                  lineHeight: 1,
                  letterSpacing: '-0.04em',
                  textShadow: isComplete
                    ? '0 0 30px rgba(52, 211, 153, 0.4)'
                    : '0 0 30px rgba(0, 229, 255, 0.35)',
                }}
              >
                {progressPct}%
              </span>
              <span
                style={{
                  fontSize: '0.85rem',
                  fontFamily: 'var(--font-mono)',
                  color: '#A3A3A3',
                }}
              >
                {isComplete
                  ? 'Inference Verified'
                  : isFailed
                  ? 'Halted'
                  : 'Computed'}
              </span>
            </div>

            {/* Stage Title & Details */}
            <div style={{ marginTop: '0.75rem' }}>
              <p
                style={{
                  fontSize: '1rem',
                  fontWeight: 700,
                  color: '#F5F5F5',
                  letterSpacing: '-0.01em',
                  margin: 0,
                }}
              >
                {isComplete
                  ? 'All Diagnostic Stages Successfully Executed'
                  : currentStage
                  ? currentStage.label
                  : 'Initializing Neural Engine...'}
              </p>
              <p
                style={{
                  fontSize: '0.78rem',
                  fontFamily: 'var(--font-mono)',
                  color: isFailed ? '#FB7185' : '#737373',
                  marginTop: '0.35rem',
                  lineHeight: 1.5,
                }}
              >
                {executionError
                  ? executionError
                  : currentStage?.detail ||
                    'Preparing high-resolution tensors and executing gradient attribution passes.'}
              </p>
            </div>
          </div>

          {/* Clean Segmented Progress Bar */}
          <div
            style={{
              height: '8px',
              background: '#121212',
              borderRadius: '4px',
              overflow: 'hidden',
              border: '1px solid #262626',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${progressPct}%`,
                background: isFailed
                  ? 'linear-gradient(90deg, #FB7185, #E11D48)'
                  : isComplete
                  ? 'linear-gradient(90deg, #00E5FF, #34D399)'
                  : 'linear-gradient(90deg, #00E5FF, #14B8A6)',
                borderRadius: '4px',
                transition: 'width 0.4s ease-out',
                boxShadow: isComplete
                  ? '0 0 16px rgba(52, 211, 153, 0.6)'
                  : '0 0 16px rgba(0, 229, 255, 0.5)',
              }}
            />
          </div>

          {/* Action Row */}
          {isComplete ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}
            >
              <button
                type="button"
                onClick={onScrollToResults}
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  background: '#00E5FF',
                  color: '#000000',
                  border: 'none',
                  borderRadius: '12px',
                  padding: '0.85rem 1.25rem',
                  fontSize: '0.82rem',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'background 0.2s ease',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = '#33ebff')}
                onMouseLeave={(e) => (e.currentTarget.style.background = '#00E5FF')}
              >
                Inspect Clinical Results Below
                <ArrowDown className="size-4" />
              </button>
            </motion.div>
          ) : isFailed ? (
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <button
                type="button"
                onClick={onRetry}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  background: 'rgba(251, 113, 133, 0.15)',
                  border: '1px solid #FB7185',
                  color: '#FB7185',
                  borderRadius: '12px',
                  padding: '0.85rem 1.5rem',
                  fontSize: '0.8rem',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                <RefreshCw className="size-4" />
                Retry Diagnostic Analysis
              </button>
            </div>
          ) : null}
        </div>
      </div>

      {/* Illuminated Stage Track (Visual Pipeline Legibility) */}
      <div
        style={{
          borderTop: '1px solid #1a1a1a',
          paddingTop: '1.5rem',
          marginTop: '1rem',
          position: 'relative',
          zIndex: 2,
        }}
      >
        <span
          style={{
            fontSize: '0.62rem',
            fontFamily: 'var(--font-mono)',
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            color: '#737373',
            display: 'block',
            marginBottom: '0.75rem',
          }}
        >
          Pipeline Execution Sequence · Deterministic Trace
        </span>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            gap: '0.5rem',
          }}
        >
          {pipelineStages.map((stage, idx) => {
            const isDone = stage.status === 'done';
            const isRunning = stage.status === 'running';
            const isErr = stage.status === 'error' || stage.status === 'timeout';
            const isSkipped = stage.status === 'skipped';

            return (
              <div
                key={stage.id}
                style={{
                  borderRadius: '10px',
                  border: isRunning
                    ? '1px solid #00E5FF'
                    : isDone
                    ? '1px solid rgba(52, 211, 153, 0.3)'
                    : isErr
                    ? '1px solid rgba(251, 113, 133, 0.4)'
                    : '1px solid #1c1c1c',
                  background: isRunning
                    ? 'rgba(0, 229, 255, 0.08)'
                    : isDone
                    ? 'rgba(52, 211, 153, 0.05)'
                    : isErr
                    ? 'rgba(251, 113, 133, 0.08)'
                    : '#080808',
                  padding: '0.75rem',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  gap: '6px',
                  transition: 'all 0.2s ease',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <span
                    style={{
                      fontSize: '0.62rem',
                      fontFamily: 'var(--font-mono)',
                      color: '#555555',
                    }}
                  >
                    0{idx + 1}
                  </span>
                  {isDone ? (
                    <CheckCircle2 className="size-3.5 text-[#34D399]" />
                  ) : isRunning ? (
                    <span
                      style={{
                        width: '7px',
                        height: '7px',
                        borderRadius: '50%',
                        background: '#00E5FF',
                        boxShadow: '0 0 8px #00E5FF',
                      }}
                    />
                  ) : isErr ? (
                    <AlertTriangle className="size-3.5 text-[#FB7185]" />
                  ) : (
                    <Clock className="size-3.5 text-[#444444]" />
                  )}
                </div>

                <div>
                  <p
                    style={{
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      color: isDone ? '#F5F5F5' : isRunning ? '#00E5FF' : isErr ? '#FB7185' : '#737373',
                      lineHeight: 1.25,
                      margin: 0,
                    }}
                  >
                    {stage.label}
                  </p>
                  <span
                    style={{
                      display: 'block',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.62rem',
                      marginTop: '3px',
                      color: isDone
                        ? '#34D399'
                        : isRunning
                        ? '#00E5FF'
                        : isErr
                        ? '#FB7185'
                        : isSkipped
                        ? '#444444'
                        : '#555555',
                      textTransform: 'uppercase',
                    }}
                  >
                    {isRunning
                      ? 'COMPUTING'
                      : isDone
                      ? stage.durationMs
                        ? `${stage.durationMs}ms`
                        : 'VERIFIED'
                      : isErr
                      ? 'FAILED'
                      : isSkipped
                      ? 'SKIPPED'
                      : 'QUEUED'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default Diagnostic3DLoader;
