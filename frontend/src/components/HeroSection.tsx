import { useRef } from 'react';
import { MedicalScene3D } from './MedicalScene3D';
import type { HealthResponse, ModelsResponse } from '../api/types';

interface HeroSectionProps {
  health: HealthResponse | null;
  models: ModelsResponse | null;
  onExploreClick: () => void;
}

export function HeroSection({ health, models, onExploreClick }: HeroSectionProps) {
  const heroRef = useRef<HTMLDivElement>(null);

  const stat = (value: string, label: string, sub?: string) => (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontSize: 'clamp(1.25rem, 2.5vw, 1.75rem)',
        fontWeight: 700,
        letterSpacing: '-0.03em',
        color: 'var(--text-primary)',
        fontFamily: 'var(--font-sans)',
        lineHeight: 1,
        marginBottom: '0.25rem',
      }}>
        {value}
      </div>
      <div style={{
        fontSize: '0.55rem',
        fontFamily: 'var(--font-mono)',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        color: 'var(--text-muted)',
      }}>
        {label}
      </div>
      {sub && (
        <div style={{
          fontSize: '0.5rem',
          color: 'var(--text-dim)',
          fontFamily: 'var(--font-mono)',
          marginTop: '2px',
        }}>
          {sub}
        </div>
      )}
    </div>
  );

  const modelsLoaded = models?.cnns?.length ?? 0;
  const bestAcc = models?.cnn_test_accuracy
    ? Math.max(...Object.values(models.cnn_test_accuracy)) * 100
    : null;

  return (
    <section
      ref={heroRef}
      className="medical-grid"
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
        overflow: 'hidden',
        paddingTop: '60px',
      }}
    >
      {/* Ambient depth gradients */}
      <div style={{
        position: 'absolute',
        inset: 0,
        background: 'radial-gradient(ellipse 70% 50% at 65% 40%, rgba(0,242,254,0.04) 0%, transparent 65%)',
        pointerEvents: 'none',
      }} />
      <div style={{
        position: 'absolute',
        inset: 0,
        background: 'radial-gradient(ellipse 40% 60% at 20% 70%, rgba(0,212,170,0.025) 0%, transparent 60%)',
        pointerEvents: 'none',
      }} />

      <div className="hero-content-grid" style={{
        maxWidth: '1320px',
        margin: '0 auto',
        padding: '0 1.5rem',
        width: '100%',
        alignItems: 'center',
      }}>

        {/* Left: Text */}
        <div style={{ animation: 'fade-up 0.7s var(--ease-out) both' }}>
          {/* Eyebrow */}
          <p style={{
            fontSize: '0.58rem',
            fontFamily: 'var(--font-mono)',
            letterSpacing: '0.2em',
            textTransform: 'uppercase',
            color: 'var(--accent)',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}>
            <span style={{
              display: 'inline-block',
              width: '28px',
              height: '1px',
              background: 'linear-gradient(90deg, var(--accent), transparent)',
            }} />
            Explainable Medical AI
          </p>

          {/* Headline */}
          <h1 style={{
            fontSize: 'clamp(2.25rem, 5vw, 3.75rem)',
            fontWeight: 700,
            letterSpacing: '-0.04em',
            lineHeight: 1.05,
            color: 'var(--text-primary)',
            marginBottom: '1.5rem',
          }}>
            Medical Imaging,
            <br />
            <span className="gradient-text">Intelligently</span>
            <br />
            Explained.
          </h1>

          <p style={{
            fontSize: '0.88rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.7,
            maxWidth: '460px',
            marginBottom: '2.5rem',
          }}>
            ResNet50 CNN inference across four modalities — chest, brain, skin, eye.
            Every prediction accompanied by Grad-CAM, LIME, and SHAP explanations.
            Clinical synthesis via Gemma 4.
          </p>

          {/* CTA */}
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <button
              onClick={onExploreClick}
              style={{
                padding: '0.8rem 2rem',
                background: 'linear-gradient(135deg, rgba(0,242,254,0.12) 0%, rgba(0,212,170,0.08) 100%)',
                border: '1px solid var(--border-focus)',
                borderRadius: '10px',
                fontSize: '0.72rem',
                fontFamily: 'var(--font-mono)',
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                color: 'var(--accent)',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                fontWeight: 600,
                boxShadow: '0 0 20px rgba(0,242,254,0.08)',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = 'linear-gradient(135deg, rgba(0,242,254,0.2) 0%, rgba(0,212,170,0.14) 100%)';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)';
                (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 24px rgba(0,242,254,0.15)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = 'linear-gradient(135deg, rgba(0,242,254,0.12) 0%, rgba(0,212,170,0.08) 100%)';
                (e.currentTarget as HTMLElement).style.transform = '';
                (e.currentTarget as HTMLElement).style.boxShadow = '0 0 20px rgba(0,242,254,0.08)';
              }}
            >
              Start Analysis ↓
            </button>
          </div>

          {/* Stats */}
          <div style={{
            display: 'flex',
            gap: '2rem',
            marginTop: '3rem',
            paddingTop: '2rem',
            borderTop: '1px solid var(--border)',
            flexWrap: 'wrap',
          }}>
            {stat(
              `${modelsLoaded}`,
              'CNN Models',
              models?.cnns?.join(' · ') || '—',
            )}
            {stat(
              bestAcc ? `${bestAcc.toFixed(1)}%` : '—',
              'Best Test Acc.',
              'Brain MRI',
            )}
            {stat(
              health?.gpu?.cuda_available ? 'CUDA' : 'CPU',
              'Compute',
              health?.gpu?.gpu_name?.replace('NVIDIA GeForce ', '') || '—',
            )}
            {stat(
              health?.gpu?.cuda_available ? 'Active' : '—',
              'Gemma 4',
              'Remote API',
            )}
          </div>
        </div>

        {/* Right: 3D scene */}
        <div
          style={{
            height: '560px',
            position: 'relative',
          }}
        >
          <MedicalScene3D />

          {/* Decorative scan line overlay */}
          <div style={{
            position: 'absolute',
            inset: 0,
            pointerEvents: 'none',
            overflow: 'hidden',
            borderRadius: '12px',
          }}>
            <div style={{
              position: 'absolute',
              left: 0,
              right: 0,
              height: '2px',
              background: 'linear-gradient(90deg, transparent, rgba(0,242,254,0.3), transparent)',
              animation: 'scan-line 4s linear infinite',
              opacity: 0.4,
            }} />
          </div>
        </div>

      </div>

      {/* Bottom fade */}
      <div style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: '140px',
        background: 'linear-gradient(to bottom, transparent, var(--bg-void))',
        pointerEvents: 'none',
      }} />

      {/* Responsive layout styling */}
      <style>{`
        .hero-content-grid {
          display: grid;
          grid-template-columns: 1.05fr 0.95fr;
          gap: 3rem;
        }
        @media (max-width: 960px) {
          .hero-content-grid {
            grid-template-columns: 1fr !important;
            gap: 2.5rem !important;
            text-align: center;
          }
          .hero-content-grid > div:first-child p {
            justify-content: center;
          }
          .hero-content-grid > div:first-child > div:nth-of-type(1) {
            justify-content: center;
          }
          .hero-content-grid > div:first-child > div:nth-of-type(2) {
            justify-content: center;
          }
          .hero-content-grid > div:last-child {
            height: 380px !important;
          }
        }
      `}</style>
    </section>
  );
}
