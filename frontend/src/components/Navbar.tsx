import { useEffect, useState } from 'react';
import type { HealthResponse, DqnStatusResponse, LlmStatusResponse } from '../api/types';

interface NavbarProps {
  health: HealthResponse | null;
  dqnStatus: DqnStatusResponse | null;
  llmStatus: LlmStatusResponse | null;
  onNavigateToAnalysis: () => void;
  onNavigateToXai: () => void;
  onNavigateToModels: () => void;
}

export function Navbar({
  health, dqnStatus, llmStatus,
  onNavigateToAnalysis, onNavigateToXai, onNavigateToModels,
}: NavbarProps) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const dot = (ok: boolean | null | undefined, label: string) => (
    <span style={{
      display: 'flex',
      alignItems: 'center',
      gap: '5px',
      fontSize: '0.56rem',
      fontFamily: 'var(--font-mono)',
      color: ok === null || ok === undefined ? 'var(--text-dim)' : ok ? 'var(--accent)' : 'var(--text-muted)',
    }}>
      <span style={{
        width: '5px',
        height: '5px',
        borderRadius: '50%',
        background: ok === null || ok === undefined ? 'var(--text-dim)' : ok ? 'var(--accent)' : '#475569',
        boxShadow: ok ? '0 0 6px rgba(0,242,254,0.4)' : 'none',
        flexShrink: 0,
      }} />
      {label}
    </span>
  );

  return (
    <nav style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 100,
      padding: '0 1.5rem',
      height: '56px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      background: scrolled ? 'rgba(0,0,0,0.88)' : 'transparent',
      backdropFilter: scrolled ? 'blur(20px)' : 'none',
      borderBottom: scrolled ? '1px solid #262626' : '1px solid transparent',
      transition: 'all 0.3s ease',
    }}>

      {/* Logo */}
      <div
        onClick={onNavigateToAnalysis}
        style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', cursor: 'pointer' }}
      >
        <span style={{
          width: '24px',
          height: '24px',
          borderRadius: '6px',
          background: 'rgba(0, 229, 255, 0.1)',
          border: '1px solid rgba(0, 229, 255, 0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '0.65rem',
          color: '#00E5FF',
          boxShadow: '0 0 12px rgba(0,229,255,0.15)',
        }}>
          ✦
        </span>
        <span style={{
          fontSize: '0.9rem',
          fontWeight: 700,
          letterSpacing: '-0.02em',
          color: '#F5F5F5',
          fontFamily: 'var(--font-sans)',
        }}>
          MediVision
        </span>
        <span style={{
          fontSize: '0.6rem',
          fontWeight: 600,
          color: '#00E5FF',
          fontFamily: 'var(--font-mono)',
          padding: '2px 5px',
          borderRadius: '4px',
          background: 'rgba(0,229,255,0.08)',
          border: '1px solid rgba(0,229,255,0.2)',
          letterSpacing: '0.06em',
        }}>
          AI
        </span>
      </div>

      {/* Nav links */}
      <div className="nav-links" style={{
        display: 'flex',
        gap: '1.75rem',
        fontSize: '0.72rem',
        fontFamily: 'var(--font-mono)',
        letterSpacing: '0.04em',
        color: '#A3A3A3',
      }}>
        {[
          { label: 'Diagnostic Studio', fn: onNavigateToAnalysis },
          { label: 'Explainable AI', fn: onNavigateToXai },
          { label: 'Model Benchmarks', fn: onNavigateToModels },
        ].map(({ label, fn }) => (
          <button
            key={label}
            onClick={fn}
            style={{
              background: 'none',
              border: 'none',
              color: '#A3A3A3',
              cursor: 'pointer',
              fontSize: 'inherit',
              fontFamily: 'inherit',
              letterSpacing: 'inherit',
              transition: 'color 0.2s ease',
              padding: 0,
            }}
            onMouseEnter={e => ((e.currentTarget as HTMLElement).style.color = '#00E5FF')}
            onMouseLeave={e => ((e.currentTarget as HTMLElement).style.color = '#A3A3A3')}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Status row */}
      <div className="nav-status" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <span className="gpu-badge">{dot(health?.engine_loaded, health?.gpu?.gpu_name?.replace('NVIDIA GeForce ', '') || 'CUDA Ready')}</span>
        {dot(llmStatus?.available, 'Gemma 4')}
        {dot(dqnStatus?.available, 'DQN Triage')}
      </div>
    </nav>
  );
}
