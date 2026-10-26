import React, { useMemo } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Clock,
  ArrowDown,
  RefreshCw,
} from 'lucide-react';
import type { PipelineStage } from '../App';

interface HorizontalProgress3DProps {
  isAnalyzing: boolean;
  pipelineStages: PipelineStage[];
  selectedModality: string;
  imagePreviewUrl?: string | null;
  executionError?: string | null;
  onRetry?: () => void;
  onScrollToResults?: () => void;
  generateReport?: boolean;
}

export const HorizontalProgress3D: React.FC<HorizontalProgress3DProps> = ({
  isAnalyzing,
  pipelineStages,
  selectedModality,
  imagePreviewUrl,
  executionError,
  onRetry,
  onScrollToResults,
  generateReport = true,
}) => {
  // Compute progress strictly from verified stage states (Part 2.C requirements)
  const { progressPct, currentStage, failedStage, isComplete, activeStages } = useMemo(() => {
    // 6 stages if report enabled; 5 stages if report disabled
    const active = pipelineStages.filter(
      (s) => (s.id !== 'gemma' || generateReport) && s.status !== 'skipped'
    );
    const stageCount = active.length || 6;

    let completedCount = 0;
    let running: PipelineStage | null = null;
    let failed: PipelineStage | null = null;

    active.forEach((s) => {
      if (s.status === 'done') {
        completedCount += 1;
      } else if (s.status === 'running') {
        running = s;
      } else if (s.status === 'error' || s.status === 'timeout') {
        failed = s;
      }
    });

    // Stage thresholds as requested:
    // 6-stage: 16%, 33%, 50%, 66%, 83%, 100%
    // 5-stage: 20%, 40%, 60%, 80%, 100%
    const stepPct = 100 / stageCount;
    let basePct = Math.round(completedCount * stepPct);

    // If currently running, advance to halfway within this stage's bracket
    if (running && !failed && !executionError) {
      basePct = Math.min(99, Math.round(basePct + stepPct * 0.45));
    }

    const cnnDone = active.find((s) => s.id === 'cnn')?.status === 'done';
    const coreFailed = Boolean(executionError || active.some((s) => s.id !== 'gemma' && (s.status === 'error' || s.status === 'timeout')));
    const allStagesSettled = active.every((s) => s.status === 'done' || s.status === 'skipped' || s.status === 'error' || s.status === 'timeout');

    const complete = !isAnalyzing && !coreFailed && cnnDone && (allStagesSettled || completedCount === stageCount);
    const failedState = !isAnalyzing && coreFailed;

    const finalPct = complete
      ? 100
      : failedState
      ? Math.min(99, basePct)
      : Math.min(98, Math.max(isAnalyzing ? 5 : 0, basePct));

    return {
      progressPct: finalPct,
      currentStage: running || (coreFailed ? failed : null) || active.find((s) => s.status === 'pending') || null,
      failedStage: coreFailed ? failed : null,
      isComplete: complete,
      activeStages: active,
    };
  }, [pipelineStages, isAnalyzing, executionError, generateReport]);

  const isFailed = Boolean(failedStage);

  // Threshold ticks for visual reference across the 3D track
  const tickThresholds = useMemo(() => {
    const count = activeStages.length || 6;
    const step = 100 / count;
    const ticks: number[] = [];
    for (let i = 1; i < count; i++) {
      ticks.push(Math.round(i * step));
    }
    return ticks;
  }, [activeStages.length]);

  return (
    <div
      id="horizontal-progress-3d"
      style={{
        width: '100%',
        maxWidth: '1360px',
        margin: '2rem auto 2.5rem',
        borderRadius: '20px',
        border: isFailed
          ? '1px solid rgba(251, 113, 133, 0.4)'
          : isComplete
          ? '1px solid rgba(52, 211, 153, 0.4)'
          : '1px solid #262626',
        background: '#040404',
        padding: '2rem',
        boxShadow: isFailed
          ? '0 10px 40px rgba(251, 113, 133, 0.12)'
          : isComplete
          ? '0 10px 40px rgba(52, 211, 153, 0.12)'
          : '0 10px 40px rgba(0, 0, 0, 0.85)',
        position: 'relative',
        overflow: 'hidden',
        transition: 'border-color 0.4s ease, box-shadow 0.4s ease',
      }}
    >
      {/* Subtle Ambient Glow Behind Track */}
      <div
        style={{
          position: 'absolute',
          top: '40%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '70%',
          height: '140px',
          background: isFailed
            ? 'radial-gradient(ellipse, rgba(251, 113, 133, 0.08) 0%, transparent 70%)'
            : isComplete
            ? 'radial-gradient(ellipse, rgba(52, 211, 153, 0.1) 0%, transparent 70%)'
            : 'radial-gradient(ellipse, rgba(0, 229, 255, 0.09) 0%, transparent 70%)',
          pointerEvents: 'none',
          transition: 'background 0.5s ease',
        }}
      />

      {/* Top Meta Area: Scan Context + Large Percentage */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          gap: '1.5rem',
          marginBottom: '1.75rem',
          position: 'relative',
          zIndex: 2,
        }}
      >
        {/* Left Side: Scan Context */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {imagePreviewUrl && (
            <div
              style={{
                width: '46px',
                height: '46px',
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
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.14em',
                  textTransform: 'uppercase',
                  color: isFailed ? '#FB7185' : isComplete ? '#34D399' : '#00E5FF',
                  fontWeight: 600,
                }}
              >
                ◆ DIAGNOSTIC PIPELINE · {selectedModality.toUpperCase()}
              </span>
            </div>
            <h3
              style={{
                fontSize: '1.25rem',
                fontWeight: 700,
                color: '#F5F5F5',
                letterSpacing: '-0.02em',
                margin: '3px 0 0',
              }}
            >
              {isFailed
                ? 'Analysis Interrupted'
                : isComplete
                ? 'Full Diagnostic Analysis Verified'
                : 'Multi-Modal Neural Inference'}
            </h3>
          </div>
        </div>

        {/* Right Side: Large 3D Monospace Percentage */}
        <div style={{ textAlign: 'right' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', justifyContent: 'flex-end' }}>
            <span
              id="progress-percentage-display"
              style={{
                fontSize: 'clamp(2.75rem, 5vw, 3.75rem)',
                fontWeight: 800,
                fontFamily: 'var(--font-mono)',
                color: isFailed ? '#FB7185' : isComplete ? '#34D399' : '#00E5FF',
                lineHeight: 1,
                letterSpacing: '-0.04em',
                textShadow: isFailed
                  ? '0 0 24px rgba(251, 113, 133, 0.4)'
                  : isComplete
                  ? '0 0 28px rgba(52, 211, 153, 0.5)'
                  : '0 0 24px rgba(0, 229, 255, 0.4)',
                transition: 'color 0.3s ease, text-shadow 0.3s ease',
              }}
            >
              {progressPct}%
            </span>
            <span
              style={{
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                color: '#737373',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
              }}
            >
              {isComplete ? 'Complete' : isFailed ? 'Halted' : 'Progress'}
            </span>
          </div>
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────
          HORIZONTAL 3D PROGRESS LINE
          A thin, dark, recessed track with a bright cyan/teal progress beam,
          layered 3D extrusion, and luminous leading edge.
          ────────────────────────────────────────────────────────── */}
      <div
        style={{
          position: 'relative',
          width: '100%',
          height: '18px',
          borderRadius: '9px',
          background: '#09090b',
          border: '1px solid #1f1f23',
          boxShadow: 'inset 0 3px 6px rgba(0, 0, 0, 0.95), inset 0 1px 2px rgba(0, 0, 0, 0.8), 0 1px 0 rgba(255, 255, 255, 0.05)',
          overflow: 'visible',
          margin: '0 0 1.25rem',
          zIndex: 2,
        }}
      >
        {/* Subtle Stage Boundary Ticks on Track */}
        {tickThresholds.map((threshold) => (
          <div
            key={threshold}
            style={{
              position: 'absolute',
              top: '2px',
              bottom: '2px',
              left: `${threshold}%`,
              width: '1px',
              background: 'rgba(255, 255, 255, 0.08)',
              zIndex: 3,
            }}
          />
        ))}

        {/* 3D Extruded Progressive Beam */}
        <div
          style={{
            position: 'absolute',
            top: '2px',
            bottom: '2px',
            left: '2px',
            width: `calc(${progressPct}% - 4px)`,
            minWidth: progressPct > 0 ? '12px' : '0px',
            borderRadius: '6px',
            background: isFailed
              ? 'linear-gradient(180deg, #ffa8ba 0%, #fb7185 30%, #e11d48 70%, #9f1239 100%)'
              : isComplete
              ? 'linear-gradient(180deg, #a7f3d0 0%, #34d399 30%, #10b981 70%, #047857 100%)'
              : 'linear-gradient(180deg, #7df9ff 0%, #00e5ff 30%, #00b4d8 70%, #007799 100%)',
            boxShadow: isFailed
              ? 'inset 0 1px 1px rgba(255, 255, 255, 0.8), 0 0 18px rgba(251, 113, 133, 0.5)'
              : isComplete
              ? 'inset 0 1px 1px rgba(255, 255, 255, 0.85), 0 0 20px rgba(52, 211, 153, 0.6)'
              : 'inset 0 1px 1px rgba(255, 255, 255, 0.85), 0 0 20px rgba(0, 229, 255, 0.5)',
            transition: 'width 0.4s cubic-bezier(0.2, 0.8, 0.3, 1), background 0.3s ease',
            zIndex: 4,
          }}
        >
          {/* Internal Specular Bevel Line */}
          <div
            style={{
              position: 'absolute',
              top: '0',
              left: '0',
              right: '0',
              height: '35%',
              background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.4) 0%, transparent 100%)',
              borderRadius: '5px 5px 0 0',
            }}
          />

          {/* Luminous 3D Leading Edge */}
          {progressPct > 0 && (
            <div
              style={{
                position: 'absolute',
                top: '-3px',
                bottom: '-3px',
                right: '-4px',
                width: '8px',
                borderRadius: '4px',
                background: '#ffffff',
                boxShadow: isFailed
                  ? '0 0 12px #fb7185, 0 0 24px rgba(251, 113, 133, 0.8)'
                  : isComplete
                  ? '0 0 14px #34d399, 0 0 28px rgba(52, 211, 153, 0.8)'
                  : '0 0 14px #00e5ff, 0 0 28px rgba(0, 229, 255, 0.9)',
                transition: 'box-shadow 0.3s ease',
                zIndex: 5,
              }}
            />
          )}
        </div>
      </div>

      {/* Current Stage Status Text & Feedback */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          marginBottom: '1.75rem',
          position: 'relative',
          zIndex: 2,
        }}
      >
        <div>
          <span
            style={{
              fontSize: '0.7rem',
              fontFamily: 'var(--font-mono)',
              color: isFailed ? '#FB7185' : isComplete ? '#34D399' : '#00E5FF',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              display: 'block',
            }}
          >
            {isFailed
              ? 'FAILED OPERATION'
              : isComplete
              ? 'STAGE VERIFICATION COMPLETE'
              : 'CURRENT STAGE'}
          </span>
          <p
            id="current-stage-label"
            style={{
              fontSize: '0.92rem',
              fontWeight: 700,
              color: '#F5F5F5',
              margin: '2px 0 0',
            }}
          >
            {isComplete
              ? activeStages.length === 5
                ? 'All 5 Diagnostic Stages Finished With Verified Results'
                : 'All 6 Diagnostic Stages Finished With Verified Results'
              : currentStage?.label || 'Initializing Neural Engine...'}
          </p>
          <p
            id="current-stage-detail"
            style={{
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              color: isFailed ? '#FB7185' : '#888888',
              margin: '3px 0 0',
            }}
          >
            {executionError
              ? executionError
              : currentStage?.detail ||
                'Executing real backend model passes with transparent attribution.'}
          </p>
        </div>

        {/* Action button on complete or error */}
        {isComplete && onScrollToResults && (
          <button
            id="view-results-btn"
            type="button"
            onClick={onScrollToResults}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              background: '#00E5FF',
              color: '#000000',
              border: 'none',
              borderRadius: '10px',
              padding: '0.65rem 1.1rem',
              fontSize: '0.78rem',
              fontFamily: 'var(--font-mono)',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'background 0.2s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#33ebff')}
            onMouseLeave={(e) => (e.currentTarget.style.background = '#00E5FF')}
          >
            View Diagnostic Results
            <ArrowDown className="size-4" />
          </button>
        )}

        {isFailed && onRetry && (
          <button
            id="retry-analysis-btn"
            type="button"
            onClick={onRetry}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              background: 'rgba(251, 113, 133, 0.12)',
              border: '1px solid #FB7185',
              color: '#FB7185',
              borderRadius: '10px',
              padding: '0.65rem 1.1rem',
              fontSize: '0.78rem',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            <RefreshCw className="size-3.5" />
            Retry Pipeline
          </button>
        )}
      </div>

      {/* ──────────────────────────────────────────────────────────
          ALL SIX STAGES DISPLAYED IN EXACT ORDER:
          IMAGE NORMALIZATION → RESNET50 → GRAD-CAM → LIME → SHAP → GEMMA 4
          ────────────────────────────────────────────────────────── */}
      <div
        style={{
          borderTop: '1px solid #1a1a1a',
          paddingTop: '1.25rem',
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
          Workflow Stages · Sequence & Verification
        </span>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
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
                    ? '1px solid rgba(52, 211, 153, 0.35)'
                    : isErr
                    ? '1px solid rgba(251, 113, 133, 0.45)'
                    : '1px solid #1a1a1a',
                  background: isRunning
                    ? 'rgba(0, 229, 255, 0.06)'
                    : isDone
                    ? 'rgba(52, 211, 153, 0.04)'
                    : isErr
                    ? 'rgba(251, 113, 133, 0.08)'
                    : '#080808',
                  padding: '0.75rem 0.85rem',
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
                      color: isRunning ? '#00E5FF' : isDone ? '#34D399' : '#555555',
                      fontWeight: 600,
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
                        animation: isAnalyzing ? 'pulse 1.5s infinite' : 'none',
                      }}
                    />
                  ) : isErr ? (
                    <AlertTriangle className="size-3.5 text-[#FB7185]" />
                  ) : isSkipped ? (
                    <span style={{ fontSize: '0.75rem', color: '#555555', fontFamily: 'var(--font-mono)' }}>—</span>
                  ) : (
                    <Clock className="size-3.5 text-[#444444]" />
                  )}
                </div>

                <div>
                  <p
                    style={{
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      color: isDone ? '#F5F5F5' : isRunning ? '#00E5FF' : isErr ? '#FB7185' : '#737373',
                      lineHeight: 1.3,
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
                        ? '#555555'
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

export default HorizontalProgress3D;
