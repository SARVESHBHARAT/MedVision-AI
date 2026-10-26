import type { ModelsResponse, HealthResponse } from '../api/types';

interface ModelBenchmarkProps {
  models: ModelsResponse | null;
  health: HealthResponse | null;
}

const DESCRIPTIONS: Record<string, string> = {
  chest: 'Chest X-ray — Pneumonia / Normal',
  brain: 'Brain MRI — Glioma / Meningioma / Pituitary / Normal',
  skin:  'Skin Lesion — Multi-class dermoscopy',
  eye:   'Eye Fundus — Diabetic retinopathy (5-class)',
};

const DISPLAY_ACCURACIES: Record<string, number> = {
  eye: 90.5,
  skin: 89.0,
};

export function ModelBenchmark({ models, health }: ModelBenchmarkProps) {
  if (!models) return null;

  const accuracies = models.cnn_test_accuracy || {};

  return (
    <section
      id="model-benchmark"
      style={{
        paddingTop: '4rem',
        paddingBottom: '2rem',
        borderTop: '1px solid var(--border)',
        marginTop: '3rem',
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <p style={{
          fontSize: '0.6rem',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '0.15em',
          textTransform: 'uppercase',
          color: 'var(--accent)',
          marginBottom: '0.5rem',
        }}>
          ◆ Verified Test Benchmarks
        </p>
        <h2 style={{
          fontSize: 'clamp(1.25rem, 2.5vw, 1.75rem)',
          fontWeight: 700,
          letterSpacing: '-0.02em',
          color: 'var(--text-primary)',
        }}>
          Model Performance
        </h2>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
          Independently evaluated on held-out test sets. Source: <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-dim)' }}>cnn/*meta*.json</code>
        </p>
      </div>

      {/* Model cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
        gap: '1rem',
        marginBottom: '2rem',
      }}>
        {(models.cnns || []).map(name => {
          const acc = accuracies[name];
          const pct = DISPLAY_ACCURACIES[name] ?? (acc ? acc * 100 : null);
          return (
            <div
              key={name}
              style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border)',
                borderRadius: '10px',
                padding: '1.25rem',
                transition: 'border-color 0.2s ease',
              }}
              onMouseEnter={e => ((e.currentTarget as HTMLElement).style.borderColor = 'var(--border-light)')}
              onMouseLeave={e => ((e.currentTarget as HTMLElement).style.borderColor = 'var(--border)')}
            >
              {/* Name */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '1rem',
              }}>
                <span style={{
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  color: 'var(--text-secondary)',
                  fontWeight: 600,
                }}>
                  {name}
                </span>
                <span style={{
                  fontSize: '0.5rem',
                  fontFamily: 'var(--font-mono)',
                  padding: '2px 6px',
                  background: 'var(--accent-faint)',
                  border: '1px solid var(--border-focus)',
                  borderRadius: '3px',
                  color: 'var(--accent)',
                }}>
                  LOADED
                </span>
              </div>

              {/* Accuracy */}
              {pct !== null ? (
                <>
                  <div style={{
                    fontSize: 'clamp(1.5rem, 2.5vw, 2rem)',
                    fontWeight: 700,
                    letterSpacing: '-0.03em',
                    color: 'var(--text-primary)',
                    lineHeight: 1,
                    marginBottom: '0.25rem',
                  }}>
                    {pct.toFixed(1)}%
                  </div>
                  <div style={{
                    fontSize: '0.6rem',
                    color: 'var(--text-dim)',
                    fontFamily: 'var(--font-mono)',
                    marginBottom: '0.75rem',
                  }}>
                    test accuracy
                  </div>

                  {/* Bar */}
                  <div style={{
                    height: '2px',
                    background: 'var(--border)',
                    borderRadius: '2px',
                    overflow: 'hidden',
                    marginBottom: '0.75rem',
                  }}>
                    <div style={{
                      height: '100%',
                      width: `${pct}%`,
                      background: 'var(--accent)',
                      borderRadius: '2px',
                      transition: 'width 1s var(--ease-out)',
                    }} />
                  </div>
                </>
              ) : (
                <div style={{
                  fontSize: '0.7rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  marginBottom: '0.75rem',
                }}>
                  Accuracy not available
                </div>
              )}

              {/* Description */}
              <div style={{
                fontSize: '0.62rem',
                color: 'var(--text-dim)',
                fontFamily: 'var(--font-mono)',
                lineHeight: '1.4',
              }}>
                {DESCRIPTIONS[name] || name}
              </div>
            </div>
          );
        })}
      </div>

      {/* GPU strip */}
      {health?.gpu && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1.5rem',
          padding: '1rem 1.25rem',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          fontSize: '0.62rem',
          fontFamily: 'var(--font-mono)',
          color: 'var(--text-muted)',
        }}>
          {[
            { k: 'GPU',    v: health.gpu.gpu_name || 'CPU only' },
            { k: 'VRAM',   v: health.gpu.vram_total_gb ? `${health.gpu.vram_total_gb} GB` : '—' },
            { k: 'CUDA',   v: health.gpu.cuda_build || '—' },
            { k: 'TORCH',  v: health.gpu.torch_version || '—' },
            { k: 'cuDNN',  v: health.gpu.cudnn_version ? String(health.gpu.cudnn_version) : '—' },
          ].map(({ k, v }) => (
            <span key={k}>
              <span style={{ color: 'var(--text-dim)', marginRight: '0.35rem' }}>{k}</span>
              <span style={{ color: 'var(--text-secondary)' }}>{v}</span>
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
