import type { PipelineStage } from '../App';
import type { ModelsResponse, HealthResponse, FullAnalysisResult } from '../api/types';
import { XaiViewer } from './XaiViewer';

interface AnalysisDashboardProps {
  analysis: FullAnalysisResult;
  pipelineStages: PipelineStage[];
  modelsMeta: ModelsResponse | null;
  health: HealthResponse | null;
  onRetryXai: (method: 'gradcam' | 'lime' | 'shap') => void;
  onRetryReport?: () => void;
  isRetryingReport?: boolean;
}


function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        marginBottom: '1rem',
        paddingBottom: '0.5rem',
        borderBottom: '1px solid var(--border)',
      }}>
        <span style={{
          fontSize: '0.58rem',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
        }}>
          {title}
        </span>
        <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
      </div>
      {children}
    </div>
  );
}

function ConfidenceBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100);
  return (
    <div style={{ marginBottom: '0.5rem' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginBottom: '4px',
      }}>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
          {label}
        </span>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
          {pct}%
        </span>
      </div>
      <div style={{ height: '2px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: pct > 80 ? 'var(--accent)' : pct > 50 ? 'var(--warning)' : '#94a3b8',
          borderRadius: '2px',
          transition: 'width 0.8s var(--ease-out)',
          boxShadow: pct > 80 ? 'var(--accent-glow)' : 'none',
        }} />
      </div>
    </div>
  );
}


export function AnalysisDashboard({
  analysis,
  pipelineStages,
  modelsMeta,
  health,
  onRetryXai,
  onRetryReport,
  isRetryingReport = false,
}: AnalysisDashboardProps) {
  const { predictionResponse, xaiResults, llmStatus, symptoms } = analysis;
  const cnn = predictionResponse?.cnn_result;
  const dqn = predictionResponse?.dqn_result;
  const report = predictionResponse?.report;

  const getStageStatus = (id: string) =>
    pipelineStages.find(s => s.id === id)?.status;

  const cardStyle: React.CSSProperties = {
    background: 'var(--bg-surface)',
    border: '1px solid var(--border)',
    borderRadius: '12px',
    padding: '1.5rem',
    animation: 'fade-up 0.5s var(--ease-out) both',
  };

  return (
    <div style={{ paddingTop: '3rem', animation: 'fade-up 0.5s var(--ease-out) both' }}>

      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <p style={{
          fontSize: '0.6rem',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          color: 'var(--accent)',
          marginBottom: '0.5rem',
        }}>
          ◆ Analysis Complete — {new Date(analysis.timestamp).toLocaleTimeString()}
        </p>
        <h2 style={{
          fontSize: 'clamp(1.5rem, 3vw, 2rem)',
          fontWeight: 700,
          letterSpacing: '-0.02em',
          color: 'var(--text-primary)',
        }}>
          {cnn?.prediction || 'Result Unavailable'}
        </h2>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
          {analysis.scanType.toUpperCase()} · {cnn?.model?.toUpperCase() || 'ResNet50'}
        </p>
      </div>

      {/* ── Top Row: Prediction + Image ── */}
      <div className="dashboard-top-grid" style={{
        gap: '1.5rem',
        marginBottom: '1.5rem',
      }}>

        {/* Prediction card */}
        <div style={cardStyle}>
          <Section title="Prediction">
            {cnn ? (
              <>
                <div style={{
                  display: 'flex',
                  alignItems: 'baseline',
                  gap: '1rem',
                  marginBottom: '1.5rem',
                }}>
                  <span style={{
                    fontSize: 'clamp(1.75rem, 3vw, 2.5rem)',
                    fontWeight: 700,
                    letterSpacing: '-0.03em',
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-sans)',
                  }}>
                    {(cnn.confidence * 100).toFixed(1)}%
                  </span>
                  <span style={{
                    fontSize: '0.7rem',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-mono)',
                  }}>
                    confidence
                  </span>
                </div>

                <div style={{ marginBottom: '1rem' }}>
                  {Object.entries(cnn.all_scores)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .map(([cls, score]) => (
                      <ConfidenceBar key={cls} label={cls} value={score as number} />
                    ))}
                </div>

                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '0.75rem',
                  borderTop: '1px solid var(--border)',
                  paddingTop: '0.75rem',
                }}>
                  <div>
                    <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.1em', marginBottom: '2px' }}>MODEL</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>ResNet50</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.1em', marginBottom: '2px' }}>MODALITY</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{analysis.scanType.toUpperCase()}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.1em', marginBottom: '2px' }}>EMERGENCY</div>
                    <div style={{
                      fontSize: '0.7rem',
                      fontFamily: 'var(--font-mono)',
                      color: predictionResponse?.emergency ? 'var(--error)' : 'var(--success)',
                    }}>
                      {predictionResponse?.emergency ? 'YES — Call 108' : 'No'}
                    </div>
                  </div>
                  {modelsMeta?.cnn_test_accuracy?.[analysis.scanType] && (
                    <div>
                      <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.1em', marginBottom: '2px' }}>TEST ACC.</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                        {(modelsMeta.cnn_test_accuracy[analysis.scanType] * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>CNN result unavailable</p>
            )}
          </Section>
        </div>

        {/* Image viewer */}
        <div style={cardStyle}>
          <Section title="Medical Image">
            <div style={{
              background: '#000',
              borderRadius: '8px',
              overflow: 'hidden',
              height: '280px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <img
                src={analysis.imageBlobUrl}
                alt="Medical scan"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                }}
              />
            </div>
          </Section>
        </div>
      </div>

      {/* ── Explainable AI (XAI) Suite ── */}
      <div style={{ marginBottom: '1.5rem' }}>
        {/* Retries banner if any stage errored or timed out */}
        {(['gradcam', 'lime', 'shap'] as const).some(
          m => getStageStatus(m) === 'error' || getStageStatus(m) === 'timeout'
        ) && (
          <div style={{
            marginBottom: '1rem',
            padding: '0.85rem 1.25rem',
            borderRadius: '12px',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            background: 'rgba(245, 158, 11, 0.08)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '0.75rem',
          }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--warning)', fontFamily: 'var(--font-mono)' }}>
              ⏱ One or more XAI attribution stages timed out or encountered an issue.
            </span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {(['gradcam', 'lime', 'shap'] as const)
                .filter(m => getStageStatus(m) === 'error' || getStageStatus(m) === 'timeout')
                .map(m => (
                  <button
                    key={m}
                    onClick={() => onRetryXai(m)}
                    style={{
                      padding: '0.35rem 0.75rem',
                      background: 'var(--bg-panel)',
                      border: '1px solid var(--border-focus)',
                      borderRadius: '6px',
                      fontSize: '0.62rem',
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--accent)',
                      cursor: 'pointer',
                      textTransform: 'uppercase',
                      fontWeight: 600,
                    }}
                  >
                    ↻ Retry {m.toUpperCase()}
                  </button>
                ))}
            </div>
          </div>
        )}

        <XaiViewer
          originalImageUrl={analysis.imageBlobUrl}
          gradcamResult={xaiResults.gradcam || null}
          limeResult={xaiResults.lime || null}
          shapResult={xaiResults.shap || null}
          predictedClass={cnn?.prediction || 'Diagnosis Pending'}
        />
      </div>

      {/* ── Bottom Row: Gemma Report + DQN + System ── */}
      <div className="dashboard-bottom-grid" style={{
        gap: '1.5rem',
        marginBottom: '1.5rem',
      }}>

        {/* Gemma report */}
        <div style={cardStyle}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '1rem',
            borderBottom: '1px solid var(--border)',
            paddingBottom: '0.5rem',
          }}>
            <span style={{
              fontSize: '0.58rem',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
            }}>
              AI Clinical Report — Gemma 4
            </span>
            {onRetryReport && (
              <button
                onClick={onRetryReport}
                disabled={isRetryingReport || getStageStatus('gemma') === 'running'}
                style={{
                  padding: '0.25rem 0.65rem',
                  fontSize: '0.58rem',
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  borderRadius: '4px',
                  border: '1px solid var(--border-focus)',
                  background: 'var(--bg-panel)',
                  color: 'var(--accent)',
                  cursor: (isRetryingReport || getStageStatus('gemma') === 'running') ? 'not-allowed' : 'pointer',
                  opacity: (isRetryingReport || getStageStatus('gemma') === 'running') ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  transition: 'all 0.2s ease',
                }}
              >
                <span>↻</span>
                <span>{isRetryingReport || getStageStatus('gemma') === 'running' ? 'Generating...' : 'Regenerate'}</span>
              </button>
            )}
          </div>
          {(() => {
            const isError = !report || report.startsWith('Gemma report unavailable:') || report.startsWith('Gemma 4 API not configured');
            const isLoading = isRetryingReport || getStageStatus('gemma') === 'running';

            if (isLoading) {
              return (
                <div style={{
                  padding: '1.25rem',
                  background: 'var(--bg-base)',
                  borderRadius: '8px',
                  border: '1px solid var(--border)',
                  fontSize: '0.72rem',
                  fontFamily: 'var(--font-mono)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem',
                }}>
                  <div style={{ color: 'var(--accent)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ animation: 'pulse-ring 1.2s ease-out infinite' }}>●</span>
                    GENERATING CLINICAL REPORT (Gemma 4)...
                  </div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', lineHeight: '1.5' }}>
                    Synthesizing CNN diagnostic findings, confidence probabilities, and presenting symptoms into physician summary...
                  </div>
                </div>
              );
            }

            if (!isError && report) {
              return (
                <div style={{
                  fontSize: '0.78rem',
                  lineHeight: '1.8',
                  color: 'var(--text-secondary)',
                  whiteSpace: 'pre-wrap',
                }}>
                  {report}
                </div>
              );
            }

            return (
              <div style={{
                padding: '1.25rem',
                background: 'var(--bg-base)',
                borderRadius: '8px',
                border: '1px solid rgba(239, 68, 68, 0.25)',
                fontSize: '0.72rem',
                fontFamily: 'var(--font-mono)',
              }}>
                <div style={{
                  marginBottom: '0.5rem',
                  fontWeight: 600,
                  color: 'var(--error)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                }}>
                  <span>⚠</span>
                  <span>{getStageStatus('gemma') === 'timeout' ? 'REPORT GENERATION TIMED OUT' : 'AI CLINICAL REPORT UNAVAILABLE'}</span>
                </div>
                <div style={{ color: 'var(--text-muted)', lineHeight: '1.6', marginBottom: '1rem' }}>
                  {report || pipelineStages.find(s => s.id === 'gemma')?.detail || (llmStatus?.available === false ? (llmStatus?.reason || 'API key not configured in .env') : 'Transient provider error encountered.')}
                </div>
                {onRetryReport && (
                  <button
                    onClick={onRetryReport}
                    style={{
                      padding: '0.45rem 1rem',
                      fontSize: '0.65rem',
                      fontFamily: 'var(--font-mono)',
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                      fontWeight: 600,
                      borderRadius: '6px',
                      border: '1px solid var(--border-focus)',
                      background: 'var(--bg-surface)',
                      color: 'var(--accent)',
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    <span>↻</span>
                    <span>Retry Report Generation</span>
                  </button>
                )}
              </div>
            );
          })()}

          {/* Symptoms */}
          {symptoms.length > 0 && (
            <Section title="Reported Symptoms">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {symptoms.map((s: string) => (
                  <span key={s} style={{
                    padding: '2px 8px',
                    fontSize: '0.62rem',
                    fontFamily: 'var(--font-mono)',
                    background: 'var(--bg-base)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                    color: 'var(--text-secondary)',
                  }}>
                    {s}
                  </span>
                ))}
              </div>
            </Section>
          )}
        </div>

        {/* DQN + System info column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

          {/* DQN Triage */}
          <div style={{ ...cardStyle, padding: '1.25rem' }}>
            <div style={{
              fontSize: '0.55rem',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
              marginBottom: '0.75rem',
            }}>
              DQN Triage
            </div>

            {dqn && !dqn.error ? (
              <div>
                <div style={{
                  fontSize: '1.1rem',
                  fontWeight: 700,
                  color: dqn.priority === 1 ? 'var(--error)' : dqn.priority === 2 ? 'var(--warning)' : 'var(--success)',
                  marginBottom: '0.25rem',
                }}>
                  {dqn.priority_label}
                </div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  Emergency: {dqn.emergency ? 'Yes' : 'No'}
                </div>
              </div>
            ) : (
              <div>
                <div style={{
                  fontSize: '0.72rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  marginBottom: '0.25rem',
                }}>
                  Status: Not configured
                </div>
                <div style={{
                  fontSize: '0.6rem',
                  color: 'var(--text-dim)',
                  fontFamily: 'var(--font-mono)',
                  lineHeight: '1.5',
                }}>
                  Reason: No verified DQN checkpoint available.
                </div>
              </div>
            )}
          </div>

          {/* System info */}
          <div style={{ ...cardStyle, padding: '1.25rem' }}>
            <div style={{
              fontSize: '0.55rem',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: 'var(--text-muted)',
              marginBottom: '0.75rem',
            }}>
              System
            </div>

            {[
              { k: 'GPU',    v: health?.gpu?.gpu_name || 'Unknown' },
              { k: 'VRAM',   v: health?.gpu?.vram_total_gb ? `${health.gpu.vram_total_gb} GB` : '—' },
              { k: 'CUDA',   v: health?.gpu?.cuda_build || '—' },
              { k: 'DEVICE', v: health?.gpu?.device?.toUpperCase() || '—' },
              { k: 'LLM',    v: llmStatus?.model || '—' },
            ].map(({ k, v }) => (
              <div key={k} style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '0.35rem 0',
                borderBottom: '1px solid var(--border)',
                fontSize: '0.65rem',
                fontFamily: 'var(--font-mono)',
              }}>
                <span style={{ color: 'var(--text-dim)' }}>{k}</span>
                <span style={{ color: 'var(--text-secondary)', textAlign: 'right', maxWidth: '60%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v}</span>
              </div>
            ))}
          </div>

          {/* Disclaimer */}
          <div style={{
            padding: '1rem',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: '10px',
            fontSize: '0.6rem',
            color: 'var(--text-dim)',
            fontFamily: 'var(--font-mono)',
            lineHeight: '1.6',
          }}>
            ⚠ AI-assisted analysis only. Not a substitute for clinical diagnosis. All results must be
            reviewed and confirmed by a qualified physician before clinical use.
          </div>
        </div>
      </div>

      <style>{`
        .dashboard-top-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
        }
        .dashboard-bottom-grid {
          display: grid;
          grid-template-columns: 2fr 1fr;
        }
        @media (max-width: 860px) {
          .dashboard-top-grid, .dashboard-bottom-grid {
            grid-template-columns: 1fr !important;
            gap: 1.25rem !important;
          }
        }
      `}</style>

    </div>
  );
}
