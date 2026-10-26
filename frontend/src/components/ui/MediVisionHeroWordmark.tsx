"use client";

import * as React from "react";
import { hwfHex, hwfRadius, hwfStep, hwfWarmth, hwfFitSize } from "./halftone-wordmark-footer";

export interface MediVisionHeroWordmarkProps {
  wordmark?: string;
  badge?: string;
  tagline?: string;
  primaryActionLabel?: string;
  secondaryActionLabel?: string;
  onPrimaryAction?: () => void;
  onSecondaryAction?: () => void;
  height?: string;
  className?: string;
}

const SANS_STACK = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

export function MediVisionHeroWordmark({
  wordmark = "MediVision",
  badge = "CLINICAL NEURAL INTELLIGENCE · MULTI-MODAL XAI",
  tagline = "Explainable medical AI diagnostic platform integrating ResNet50 neural vision, deep Q-learning symptom triage, and transparent attribution.",
  primaryActionLabel = "Launch Diagnostic Studio ↓",
  secondaryActionLabel = "Explore Intelligence Bento",
  onPrimaryAction,
  onSecondaryAction,
  height = "clamp(260px, 26vw, 380px)",
  className = "",
}: MediVisionHeroWordmarkProps) {
  const rootRef = React.useRef<HTMLDivElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  const colors = {
    background: "#000000",
    ink: "#E5E5E5",
    warm: "#00E5FF",
  };

  const dotGap = 6;
  const radius = 80;
  const grain = 0.15;
  const fontWeight = 800;

  React.useEffect(() => {
    const root = rootRef.current;
    const canvas = canvasRef.current;
    if (!root || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0;
    let h = 0;
    let dpr = 1;
    let px = -1e5;
    let py = -1e5;
    let raf = 0;
    let awake = true;
    let ring = { x: 0, y: 0, t: -1 };
    let dots = new Float32Array(0);
    let count = 0;
    let grainTile: HTMLCanvasElement | null = null;
    const ink = hwfHex(colors.ink) || [229, 229, 229];
    const warm = hwfHex(colors.warm) || [0, 229, 255];

    const layout = () => {
      const rect = root.getBoundingClientRect();
      w = Math.max(1, Math.round(rect.width));
      h = Math.max(1, Math.round(rect.height));
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);

      // Measure text to fit prominently and proportionally across width
      const mask = document.createElement("canvas");
      mask.width = w;
      mask.height = h;
      const m = mask.getContext("2d", { willReadFrequently: true });
      if (!m) return;

      const baseFont = `${fontWeight} 100px ${SANS_STACK}`;
      m.font = baseFont;
      const measured = m.measureText(wordmark).width;

      // Allow prominent proportion across desktop while maintaining edge breathing room
      const targetWidth = Math.min(w * 0.96, 1400);
      const fit = hwfFitSize(measured, targetWidth);
      m.font = `${fontWeight} ${fit}px ${SANS_STACK}`;
      const box = m.measureText(wordmark);
      const cap = box.actualBoundingBoxAscent || fit * 0.72;

      // Strictly centered horizontally and vertically
      const textX = (w - box.width) / 2;
      const textY = (h + cap) / 2;

      m.fillStyle = "#000";
      m.textBaseline = "alphabetic";
      m.fillText(wordmark, textX, textY);
      const img = m.getImageData(0, 0, w, h).data;

      const gap = dotGap;
      const cols = Math.ceil(w / gap);
      const rows = Math.ceil(h / gap);
      const tmp: number[] = [];

      for (let gy = 0; gy < rows; gy++) {
        for (let gx = 0; gx < cols; gx++) {
          let sum = 0;
          for (let sy = 0; sy < 3; sy++) {
            for (let sx = 0; sx < 3; sx++) {
              const x = Math.min(w - 1, Math.floor(gx * gap + ((sx + 0.5) * gap) / 3));
              const y = Math.min(h - 1, Math.floor(gy * gap + ((sy + 0.5) * gap) / 3));
              sum += img[(y * w + x) * 4 + 3];
            }
          }
          const cov = sum / (9 * 255);
          if (cov < 0.05) continue;
          const hx = gx * gap + gap / 2;
          const hy = gy * gap + gap / 2;
          tmp.push(hx, hy, 0, 0, hx, hy, hwfRadius(cov, gap));
        }
      }
      dots = new Float32Array(tmp);
      count = tmp.length / 7;

      // Subtle grain overlay
      grainTile = document.createElement("canvas");
      grainTile.width = 128;
      grainTile.height = 128;
      const g = grainTile.getContext("2d");
      if (g) {
        const id = g.createImageData(128, 128);
        for (let i = 0; i < id.data.length; i += 4) {
          const v = Math.random() * 255;
          id.data[i] = id.data[i + 1] = id.data[i + 2] = v;
          id.data[i + 3] = Math.random() < 0.35 ? 0 : 255;
        }
        g.putImageData(id, 0, 0);
      }
      awake = true;
    };

    const draw = (now: number) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = colors.background;
      ctx.fillRect(0, 0, w, h);

      const r = !reduced ? radius : 0;
      const ringAge = ring.t < 0 ? 1e9 : now - ring.t;
      const ringR = ringAge * 0.9;
      let moving = false;

      for (let i = 0; i < count; i++) {
        const j = i * 7;
        let x = dots[j];
        let y = dots[j + 1];
        let vx = dots[j + 2];
        let vy = dots[j + 3];
        const hx = dots[j + 4];
        const hy = dots[j + 5];

        if (!reduced) {
          if (ringAge < 800) {
            const dx = hx - ring.x;
            const dy = hy - ring.y;
            const d = Math.sqrt(dx * dx + dy * dy) + 0.001;
            const band = 1 - Math.min(1, Math.abs(d - ringR) / 26);
            if (band > 0) {
              const f = band * 3.2 * (1 - ringAge / 800);
              vx += (dx / d) * f;
              vy += (dy / d) * f;
            }
          }
          const s = hwfStep(x, y, vx, vy, hx, hy, px, py, r);
          x = s[0];
          y = s[1];
          vx = s[2];
          vy = s[3];
          if (Math.abs(vx) + Math.abs(vy) > 0.02 || Math.abs(x - hx) + Math.abs(y - hy) > 0.05) {
            moving = true;
          } else {
            x = hx;
            y = hy;
            vx = 0;
            vy = 0;
          }
          dots[j] = x;
          dots[j + 1] = y;
          dots[j + 2] = vx;
          dots[j + 3] = vy;
        }

        const k = hwfWarmth(x, y, hx, hy, 28);
        ctx.fillStyle =
          k > 0.02
            ? "rgb(" +
              Math.round(ink[0] + (warm[0] - ink[0]) * k) + "," +
              Math.round(ink[1] + (warm[1] - ink[1]) * k) + "," +
              Math.round(ink[2] + (warm[2] - ink[2]) * k) + ")"
            : colors.ink;

        ctx.beginPath();
        ctx.arc(x, y, dots[j + 6], 0, 6.2832);
        ctx.fill();
      }

      if (grainTile && grain > 0) {
        ctx.globalAlpha = grain * 0.06;
        ctx.globalCompositeOperation = "overlay";
        const pattern = ctx.createPattern(grainTile, "repeat");
        if (pattern) {
          ctx.fillStyle = pattern;
          ctx.fillRect(0, 0, w, h);
        }
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = "source-over";
      }

      awake = moving || ringAge < 800 || px > -1e4;
    };

    const loop = (now: number) => {
      raf = 0;
      draw(now);
      if (awake) raf = requestAnimationFrame(loop);
    };

    const wake = () => {
      awake = true;
      if (!raf) raf = requestAnimationFrame(loop);
    };

    const toLocal = (clientX: number, clientY: number) => {
      const b = canvas.getBoundingClientRect();
      return [clientX - b.left, clientY - b.top];
    };

    const onPointerMove = (e: PointerEvent) => {
      const [lx, ly] = toLocal(e.clientX, e.clientY);
      px = lx;
      py = ly;
      wake();
    };

    const onPointerLeave = () => {
      px = -1e5;
      py = -1e5;
      wake();
    };

    const onPointerDown = (e: PointerEvent) => {
      if (reduced) return;
      const [lx, ly] = toLocal(e.clientX, e.clientY);
      ring = { x: lx, y: ly, t: performance.now() };
      wake();
    };

    layout();
    draw(performance.now());
    wake();

    root.addEventListener("pointermove", onPointerMove);
    root.addEventListener("pointerleave", onPointerLeave);
    root.addEventListener("pointerdown", onPointerDown);

    const ro = new ResizeObserver(() => {
      layout();
      draw(performance.now());
      wake();
    });
    ro.observe(root);

    let alive = true;
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => {
        if (!alive) return;
        layout();
        draw(performance.now());
        wake();
      });
    }

    // Secondary timer to guarantee render after initial paint
    const timer = setTimeout(() => {
      if (!alive) return;
      layout();
      draw(performance.now());
    }, 100);

    return () => {
      alive = false;
      clearTimeout(timer);
      cancelAnimationFrame(raf);
      ro.disconnect();
      root.removeEventListener("pointermove", onPointerMove);
      root.removeEventListener("pointerleave", onPointerLeave);
      root.removeEventListener("pointerdown", onPointerDown);
    };
  }, [wordmark]);

  return (
    <section
      className={`hero-section ${className}`}
      style={{
        position: 'relative',
        width: '100%',
        backgroundColor: '#000000',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        paddingTop: '2.5rem',
        paddingBottom: '3.5rem',
        paddingLeft: '1.5rem',
        paddingRight: '1.5rem',
        boxSizing: 'border-box',
      }}
    >
      {/* ── Inner Content Wrapper (Strictly Centered) ── */}
      <div
        style={{
          width: '100%',
          maxWidth: '1440px',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          boxSizing: 'border-box',
        }}
      >
        {/* 1. Top Clinical Intelligence Badge */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            borderRadius: '9999px',
            border: '1px solid #262626',
            backgroundColor: '#080808',
            padding: '0.45rem 1.15rem',
            marginBottom: '1.75rem',
            boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
          }}
        >
          <span
            style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: '#00E5FF',
              boxShadow: '0 0 8px rgba(0,229,255,0.6)',
              display: 'inline-block',
            }}
          />
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              fontWeight: 600,
              letterSpacing: '0.12em',
              color: '#A3A3A3',
              textTransform: 'uppercase',
            }}
          >
            {badge}
          </span>
        </div>

        {/* 2. Halftone Canvas Wordmark (Prominent & Centered) */}
        <div
          ref={rootRef}
          style={{
            position: 'relative',
            width: '100%',
            maxWidth: '1400px',
            height: height,
            margin: '0 auto',
            cursor: 'pointer',
            touchAction: 'none',
            userSelect: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          aria-label={wordmark}
        >
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              display: 'block',
            }}
            aria-hidden="true"
          />
        </div>

        {/* 3. Subtitle & Tagline */}
        <p
          style={{
            maxWidth: '720px',
            margin: '1.5rem auto 0',
            padding: '0 1rem',
            fontSize: 'clamp(0.9rem, 1.25vw, 1.05rem)',
            color: '#A3A3A3',
            lineHeight: 1.7,
            fontWeight: 400,
            textAlign: 'center',
          }}
        >
          {tagline}
        </p>

        {/* 4. Action Buttons (Centered Row on Desktop) */}
        <div
          style={{
            marginTop: '2.25rem',
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1rem',
            padding: '0 1rem',
          }}
        >
          <button
            type="button"
            onClick={onPrimaryAction}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              borderRadius: '9999px',
              backgroundColor: '#00E5FF',
              padding: '0.85rem 2.25rem',
              fontSize: '0.85rem',
              fontWeight: 700,
              color: '#000000',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: '0 0 24px rgba(0,229,255,0.3)',
              fontFamily: 'var(--font-sans)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#33ebff';
              e.currentTarget.style.boxShadow = '0 0 32px rgba(0,229,255,0.5)';
              e.currentTarget.style.transform = 'translateY(-1px)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#00E5FF';
              e.currentTarget.style.boxShadow = '0 0 24px rgba(0,229,255,0.3)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            {primaryActionLabel}
          </button>

          {onSecondaryAction && (
            <button
              type="button"
              onClick={onSecondaryAction}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                borderRadius: '9999px',
                border: '1px solid #262626',
                backgroundColor: '#0D0D0D',
                padding: '0.85rem 2rem',
                fontSize: '0.85rem',
                fontWeight: 500,
                color: '#F5F5F5',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                fontFamily: 'var(--font-sans)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#383838';
                e.currentTarget.style.backgroundColor = '#141414';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#262626';
                e.currentTarget.style.backgroundColor = '#0D0D0D';
              }}
            >
              {secondaryActionLabel}
            </button>
          )}
        </div>

        {/* 5. Modality Performance Strip (Evenly Distributed & Centered) */}
        <div
          style={{
            marginTop: '2.75rem',
            borderTop: '1px solid #1a1a1a',
            paddingTop: '1.5rem',
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1.5rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            color: '#737373',
            maxWidth: '850px',
            width: '100%',
            margin: '2.75rem auto 0',
          }}
        >
          <span>
            Brain MRI: <strong style={{ color: '#34D399', fontWeight: 600 }}>95.44%</strong>
          </span>
          <span style={{ color: '#333333' }}>•</span>
          <span>
            Chest X-Ray: <strong style={{ color: '#34D399', fontWeight: 600 }}>92.95%</strong>
          </span>
          <span style={{ color: '#333333' }}>•</span>
          <span>
            Eye Fundus: <strong style={{ color: '#34D399', fontWeight: 600 }}>90.5%</strong>
          </span>
          <span style={{ color: '#333333' }}>•</span>
          <span>
            Skin Lesion: <strong style={{ color: '#34D399', fontWeight: 600 }}>89.0%</strong>
          </span>
        </div>
      </div>
    </section>
  );
}

export default MediVisionHeroWordmark;
