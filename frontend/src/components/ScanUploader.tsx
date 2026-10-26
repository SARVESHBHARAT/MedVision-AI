import { useState, useCallback, useRef } from 'react';
import type { PipelineStage } from '../App';

interface ScanUploaderProps {
  selectedModality: string;
  onModalityChange: (m: string) => void;
  selectedImage: File | Blob | null;
  imagePreviewUrl: string | null;
  onImageSelected: (file: File | Blob, preview: string) => void;
  onClearImage: () => void;
  symptomsList: string[];
  selectedSymptoms: string[];
  onToggleSymptom: (s: string) => void;
  onAddCustomSymptom: (s: string) => void;
  generateReport: boolean;
  onToggleGenerateReport: (v: boolean) => void;
  onRunAnalysis: () => void;
  isAnalyzing: boolean;
  pipelineStages: PipelineStage[];
}

const MODALITIES = [
  {
    id: 'chest',
    label: 'Chest X-Ray',
    tag: 'PNEUMONIA · RESNET50',
    accuracy: '92.95%',
    sub: 'Binary pulmonary radiographic classification for pneumonia and normal thoracic structures.',
    icon: '🫁',
  },
  {
    id: 'brain',
    label: 'Brain MRI',
    tag: 'NEURO-ONCOLOGY · 4-CLASS',
    accuracy: '95.44%',
    sub: 'T1/T2 axial MRI multi-class pathology classification (Glioma, Meningioma, Pituitary, Normal).',
    icon: '🧠',
  },
  {
    id: 'eye',
    label: 'Eye Fundus',
    tag: 'RETINOPATHY · 5-CLASS',
    accuracy: '90.5%',
    sub: 'High-resolution retinal fundus photography graded from stage 0 (No DR) to stage 4 (Proliferative DR).',
    icon: '👁',
  },
  {
    id: 'skin',
    label: 'Skin Lesion',
    tag: 'DERMOSCOPY · MULTI-CLASS',
    accuracy: '89.0%',
    sub: 'Dermoscopic lesion analysis evaluating melanocytic, basal cell, and benign keratotic pathologies.',
    icon: '🔬',
  },
];

const STAGE_ICON: Record<string, string> = {
  pending: '○',
  running: '●',
  done: '✓',
  error: '⚠',
  timeout: '⏱',
  skipped: '—',
};

const STAGE_COLOR: Record<string, string> = {
  pending: '#737373',
  running: '#00E5FF',
  done: '#34D399',
  error: '#FB7185',
  timeout: '#FBBF24',
  skipped: '#525252',
};

export function ScanUploader({
  selectedModality,
  onModalityChange,
  selectedImage,
  imagePreviewUrl,
  onImageSelected,
  onClearImage,
  symptomsList,
  selectedSymptoms,
  onToggleSymptom,
  onAddCustomSymptom,
  generateReport,
  onToggleGenerateReport,
  onRunAnalysis,
  isAnalyzing,
  pipelineStages,
}: ScanUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [customSymptom, setCustomSymptom] = useState('');
  const [showSymptoms, setShowSymptoms] = useState(true);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndProcessFile = useCallback(
    (file: File) => {
      setValidationError(null);
      const validExtensions = /\.(jpe?g|png|webp|tiff?|bmp|dcm)$/i;
      const isImage = file.type.startsWith('image/') || validExtensions.test(file.name);
      if (!isImage) {
        setValidationError('Unsupported format. Please upload JPEG, PNG, WEBP, or TIFF medical scans.');
        return;
      }
      const MAX_SIZE_MB = 25;
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        setValidationError(`File exceeds maximum size limit of ${MAX_SIZE_MB}MB.`);
        return;
      }
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    },
    [onImageSelected]
  );

  // Drag & Drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragging(false), []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) {
        validateAndProcessFile(file);
      }
    },
    [validateAndProcessFile]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        validateAndProcessFile(file);
      }
    },
    [validateAndProcessFile]
  );

  const handleAddCustom = () => {
    if (customSymptom.trim()) {
      onAddCustomSymptom(customSymptom);
      setCustomSymptom('');
    }
  };

  // Pipeline progress calculation
  const doneCount = pipelineStages.filter((s) => s.status === 'done' || s.status === 'skipped').length;
  const totalStages = pipelineStages.length;
  const progressPct = isAnalyzing
    ? Math.round((doneCount / totalStages) * 90) + 5
    : doneCount === totalStages
    ? 100
    : 0;

  return (
    <div className="workspace-container" style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>

      {/* ──────────────────────────────────────────────────────────
          SECTION 1: MODALITY SELECTION (Spacious, Large Cards)
          ────────────────────────────────────────────────────────── */}
      <div
        style={{
          background: '#080808',
          border: '1px solid #262626',
          borderRadius: '16px',
          padding: '2rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <span style={{
              fontSize: '0.65rem',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.16em',
              color: '#00E5FF',
              textTransform: 'uppercase',
              display: 'block',
              marginBottom: '0.35rem',
            }}>
              Section 01 · Diagnostic Target
            </span>
            <h3 style={{
              fontSize: '1.25rem',
              fontWeight: 700,
              color: '#F5F5F5',
              letterSpacing: '-0.02em',
              margin: 0,
            }}>
              Select Medical Imaging Modality
            </h3>
          </div>
          <span style={{
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
            color: '#A3A3A3',
            background: '#121212',
            padding: '4px 12px',
            borderRadius: '20px',
            border: '1px solid #262626',
          }}>
            4 Active CNN Engines
          </span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: '1rem',
        }}>
          {MODALITIES.map((m) => {
            const isSelected = selectedModality === m.id;
            return (
              <button
                key={m.id}
                onClick={() => onModalityChange(m.id)}
                disabled={isAnalyzing}
                style={{
                  background: isSelected ? '#10171B' : '#0D0D0D',
                  border: isSelected ? '1px solid #00E5FF' : '1px solid #262626',
                  borderRadius: '12px',
                  padding: '1.5rem',
                  textAlign: 'left',
                  cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
                  position: 'relative',
                  overflow: 'hidden',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.65rem',
                  boxShadow: isSelected ? '0 0 24px rgba(0,229,255,0.08)' : 'none',
                  opacity: isAnalyzing ? 0.6 : 1,
                }}
              >
                {isSelected && (
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '2px',
                    background: '#00E5FF',
                  }} />
                )}

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '1.75rem', lineHeight: 1 }}>{m.icon}</span>
                  <span style={{
                    fontSize: '0.65rem',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    color: isSelected ? '#00E5FF' : '#34D399',
                    background: isSelected ? 'rgba(0,229,255,0.1)' : 'rgba(52,211,153,0.08)',
                    padding: '3px 8px',
                    borderRadius: '4px',
                    border: `1px solid ${isSelected ? 'rgba(0,229,255,0.3)' : 'rgba(52,211,153,0.2)'}`,
                  }}>
                    {m.accuracy} Acc
                  </span>
                </div>

                <div>
                  <h4 style={{
                    fontSize: '1rem',
                    fontWeight: 700,
                    color: isSelected ? '#F5F5F5' : '#D4D4D4',
                    margin: '0 0 0.2rem',
                    letterSpacing: '-0.01em',
                  }}>
                    {m.label}
                  </h4>
                  <span style={{
                    fontSize: '0.62rem',
                    fontFamily: 'var(--font-mono)',
                    color: isSelected ? '#00E5FF' : '#737373',
                    letterSpacing: '0.04em',
                    textTransform: 'uppercase',
                  }}>
                    {m.tag}
                  </span>
                </div>

                <p style={{
                  fontSize: '0.75rem',
                  color: '#A3A3A3',
                  lineHeight: 1.5,
                  margin: 0,
                }}>
                  {m.sub}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────
          SECTION 2 & 3 & 4: DUAL-COLUMN SPACIOUS WORKSPACE
          Left: Medical Image Upload + Symptoms
          Right: Analysis Pipeline + Full Execution Action
          ────────────────────────────────────────────────────────── */}
      <div className="workspace-grid" style={{
        display: 'grid',
        gridTemplateColumns: '1.2fr 1fr',
        gap: '2rem',
        alignItems: 'start',
      }}>

        {/* LEFT COLUMN: Large Upload & Clinical Context */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

          {/* Section 2: Medical Image Upload */}
          <div style={{
            background: '#080808',
            border: '1px solid #262626',
            borderRadius: '16px',
            padding: '2rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div>
                <span style={{
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.16em',
                  color: '#00E5FF',
                  textTransform: 'uppercase',
                  display: 'block',
                  marginBottom: '0.35rem',
                }}>
                  Section 02 · Scan Acquisition
                </span>
                <h3 style={{
                  fontSize: '1.25rem',
                  fontWeight: 700,
                  color: '#F5F5F5',
                  letterSpacing: '-0.02em',
                  margin: 0,
                }}>
                  Medical Image Upload
                </h3>
              </div>

              {imagePreviewUrl && (
                <button
                  type="button"
                  onClick={() => {
                    setValidationError(null);
                    onClearImage();
                  }}
                  disabled={isAnalyzing}
                  style={{
                    padding: '6px 14px',
                    borderRadius: '6px',
                    border: '1px solid #383838',
                    background: '#121212',
                    color: '#FB7185',
                    fontSize: '0.72rem',
                    fontFamily: 'var(--font-mono)',
                    cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  Clear Scan ✕
                </button>
              )}
            </div>

            {imagePreviewUrl ? (
              <div style={{
                position: 'relative',
                width: '100%',
                minHeight: '380px',
                maxHeight: '520px',
                background: '#000000',
                borderRadius: '12px',
                border: '1px solid #262626',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                padding: '1rem',
              }}>
                <img
                  src={imagePreviewUrl}
                  alt="Medical scan preview"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '480px',
                    width: 'auto',
                    height: 'auto',
                    objectFit: 'contain',
                    borderRadius: '8px',
                    display: 'block',
                  }}
                />

                <div style={{
                  position: 'absolute',
                  bottom: '12px',
                  left: '12px',
                  background: 'rgba(0,0,0,0.8)',
                  backdropFilter: 'blur(8px)',
                  border: '1px solid #262626',
                  borderRadius: '6px',
                  padding: '4px 10px',
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  color: '#00E5FF',
                }}>
                  Scan Loaded · Ready for CNN Inference
                </div>

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isAnalyzing}
                  style={{
                    position: 'absolute',
                    top: '12px',
                    right: '12px',
                    background: 'rgba(0,0,0,0.8)',
                    backdropFilter: 'blur(8px)',
                    border: '1px solid #383838',
                    borderRadius: '6px',
                    padding: '6px 12px',
                    fontSize: '0.68rem',
                    fontFamily: 'var(--font-mono)',
                    color: '#F5F5F5',
                    cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                  }}
                >
                  Replace File ↺
                </button>
              </div>
            ) : (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  minHeight: '360px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: `2px dashed ${isDragging ? '#00E5FF' : '#262626'}`,
                  borderRadius: '12px',
                  background: isDragging ? 'rgba(0,229,255,0.04)' : '#040404',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  padding: '2.5rem 1.5rem',
                  textAlign: 'center',
                  gap: '1rem',
                }}
              >
                <div style={{
                  width: '64px',
                  height: '64px',
                  borderRadius: '16px',
                  background: '#0D0D0D',
                  border: '1px solid #262626',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.75rem',
                  color: isDragging ? '#00E5FF' : '#A3A3A3',
                  boxShadow: isDragging ? '0 0 20px rgba(0,229,255,0.2)' : 'none',
                }}>
                  ↑
                </div>

                <div>
                  <h4 style={{
                    fontSize: '1rem',
                    fontWeight: 600,
                    color: isDragging ? '#00E5FF' : '#F5F5F5',
                    margin: '0 0 0.35rem',
                  }}>
                    Drag and drop high-resolution medical scan
                  </h4>
                  <p style={{
                    fontSize: '0.8rem',
                    color: '#737373',
                    margin: 0,
                  }}>
                    or click to browse local files (DICOM, PNG, JPEG, TIFF, WEBP up to 25MB)
                  </p>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  color: '#A3A3A3',
                  background: '#0D0D0D',
                  border: '1px solid #262626',
                  padding: '6px 14px',
                  borderRadius: '20px',
                }}>
                  <span>✓ 224×224 Normalization</span>
                  <span>•</span>
                  <span>✓ RGB Channel Conversion</span>
                  <span>•</span>
                  <span>✓ Zero-loss Ingest</span>
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.dcm"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            {validationError && (
              <div style={{
                marginTop: '1.25rem',
                padding: '0.85rem 1rem',
                borderRadius: '8px',
                background: 'rgba(251, 113, 133, 0.08)',
                border: '1px solid rgba(251, 113, 133, 0.3)',
                color: '#FB7185',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                lineHeight: 1.5,
              }}>
                ⚠ Validation Alert: {validationError}
              </div>
            )}
          </div>

          {/* Section 3: Symptoms and Clinical Context */}
          <div style={{
            background: '#080808',
            border: '1px solid #262626',
            borderRadius: '16px',
            padding: '2rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
              <div>
                <span style={{
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.16em',
                  color: '#A78BFA',
                  textTransform: 'uppercase',
                  display: 'block',
                  marginBottom: '0.35rem',
                }}>
                  Section 03 · Clinical Context (Optional)
                </span>
                <h3 style={{
                  fontSize: '1.25rem',
                  fontWeight: 700,
                  color: '#F5F5F5',
                  letterSpacing: '-0.02em',
                  margin: 0,
                }}>
                  Patient Symptoms & DQN Triage
                </h3>
              </div>

              <button
                type="button"
                onClick={() => setShowSymptoms((v) => !v)}
                style={{
                  fontSize: '0.7rem',
                  fontFamily: 'var(--font-mono)',
                  color: '#A3A3A3',
                  background: '#121212',
                  border: '1px solid #262626',
                  borderRadius: '6px',
                  padding: '6px 12px',
                  cursor: 'pointer',
                }}
              >
                {showSymptoms ? 'Collapse ▲' : 'Expand ▼'}
              </button>
            </div>

            <p style={{
              fontSize: '0.8rem',
              color: '#A3A3A3',
              lineHeight: 1.6,
              marginBottom: '1.25rem',
            }}>
              Supplying reported symptoms conditions the Deep Q-Network triage agent, calculating
              evidence alignment and priority escalation vectors alongside visual CNN features.
            </p>

            {/* Selected symptoms pills */}
            {selectedSymptoms.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '1.25rem' }}>
                {selectedSymptoms.map((s) => (
                  <span
                    key={s}
                    onClick={() => onToggleSymptom(s)}
                    style={{
                      padding: '5px 12px',
                      fontSize: '0.72rem',
                      fontFamily: 'var(--font-mono)',
                      background: 'rgba(167, 139, 250, 0.1)',
                      border: '1px solid rgba(167, 139, 250, 0.4)',
                      borderRadius: '6px',
                      color: '#A78BFA',
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                    }}
                  >
                    {s} <span style={{ opacity: 0.7 }}>✕</span>
                  </span>
                ))}
              </div>
            )}

            {showSymptoms && (
              <>
                <div style={{
                  maxHeight: '160px',
                  overflowY: 'auto',
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '6px',
                  marginBottom: '1.25rem',
                  padding: '0.75rem',
                  background: '#040404',
                  borderRadius: '8px',
                  border: '1px solid #1f1f1f',
                }}>
                  {symptomsList.map((s) => {
                    const isChecked = selectedSymptoms.includes(s);
                    return (
                      <button
                        key={s}
                        type="button"
                        onClick={() => onToggleSymptom(s)}
                        style={{
                          padding: '5px 11px',
                          fontSize: '0.7rem',
                          fontFamily: 'var(--font-mono)',
                          background: isChecked ? 'rgba(167, 139, 250, 0.15)' : '#0D0D0D',
                          border: `1px solid ${isChecked ? '#A78BFA' : '#262626'}`,
                          borderRadius: '6px',
                          color: isChecked ? '#A78BFA' : '#A3A3A3',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        {isChecked ? `✓ ${s}` : `+ ${s}`}
                      </button>
                    );
                  })}
                </div>

                {/* Custom symptom input */}
                <div style={{ display: 'flex', gap: '8px', marginBottom: '1.25rem' }}>
                  <input
                    value={customSymptom}
                    onChange={(e) => setCustomSymptom(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddCustom()}
                    placeholder="Enter custom clinical observation or symptom..."
                    style={{
                      flex: 1,
                      padding: '0.75rem 1rem',
                      background: '#040404',
                      border: '1px solid #262626',
                      borderRadius: '8px',
                      fontSize: '0.8rem',
                      fontFamily: 'var(--font-mono)',
                      color: '#F5F5F5',
                      outline: 'none',
                    }}
                  />
                  <button
                    type="button"
                    onClick={handleAddCustom}
                    style={{
                      padding: '0.75rem 1.25rem',
                      background: '#121212',
                      border: '1px solid #383838',
                      borderRadius: '8px',
                      fontSize: '0.75rem',
                      fontFamily: 'var(--font-mono)',
                      color: '#F5F5F5',
                      cursor: 'pointer',
                      fontWeight: 600,
                    }}
                  >
                    Add Symptom
                  </button>
                </div>
              </>
            )}

            {/* Gemma Report Toggle */}
            <label style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              fontSize: '0.8rem',
              color: '#F5F5F5',
              cursor: 'pointer',
              userSelect: 'none',
              padding: '0.75rem 1rem',
              background: '#0D0D0D',
              border: '1px solid #262626',
              borderRadius: '8px',
            }}>
              <input
                id="gemma-report-checkbox"
                type="checkbox"
                checked={generateReport}
                onChange={(e) => onToggleGenerateReport(e.target.checked)}
                style={{ accentColor: '#00E5FF', width: '16px', height: '16px' }}
              />
              <span>Generate structured Gemma 4 AI clinical synthesis & findings report (Optional)</span>
            </label>
          </div>
        </div>

        {/* RIGHT COLUMN: Section 4 (Analysis Pipeline) + Section 5 (Run Action) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

          {/* Section 4: Analysis Pipeline */}
          <div style={{
            background: '#080808',
            border: '1px solid #262626',
            borderRadius: '16px',
            padding: '2rem',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '1.25rem',
            }}>
              <div>
                <span style={{
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.16em',
                  color: '#00E5FF',
                  textTransform: 'uppercase',
                  display: 'block',
                  marginBottom: '0.35rem',
                }}>
                  Section 04 · Neural Execution
                </span>
                <h3 style={{
                  fontSize: '1.25rem',
                  fontWeight: 700,
                  color: '#F5F5F5',
                  letterSpacing: '-0.02em',
                  margin: 0,
                }}>
                  Multi-Modal Analysis Pipeline
                </h3>
              </div>

              {isAnalyzing && (
                <span style={{
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  color: '#00E5FF',
                  letterSpacing: '0.1em',
                  background: 'rgba(0,229,255,0.1)',
                  padding: '4px 10px',
                  borderRadius: '12px',
                  border: '1px solid rgba(0,229,255,0.3)',
                  animation: 'pulse-ring 1.5s infinite',
                }}>
                  PROCESSING PIPELINE
                </span>
              )}
            </div>

            <p style={{
              fontSize: '0.8rem',
              color: '#A3A3A3',
              lineHeight: 1.6,
              marginBottom: '1.5rem',
            }}>
              Deterministic pipeline sequence chaining tensor normalization, PyTorch ResNet50 inference,
              feature attribution maps, and LLM clinical interpretation.
            </p>

            {/* Progress Bar */}
            <div style={{
              height: '6px',
              background: '#121212',
              borderRadius: '3px',
              overflow: 'hidden',
              marginBottom: '2rem',
              border: '1px solid #262626',
            }}>
              <div
                style={{
                  height: '100%',
                  width: `${progressPct}%`,
                  background: 'linear-gradient(90deg, #00E5FF, #34D399)',
                  transition: 'width 0.4s ease',
                }}
              />
            </div>

            {/* Pipeline Stage Rows */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {pipelineStages.map((stage, i) => {
                const isRunning = stage.status === 'running';
                const isDone = stage.status === 'done';
                return (
                  <div
                    key={stage.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '1rem',
                      padding: '1rem 1.25rem',
                      borderRadius: '8px',
                      background: isRunning ? 'rgba(0,229,255,0.04)' : '#0D0D0D',
                      border: isRunning ? '1px solid rgba(0,229,255,0.4)' : '1px solid #1c1c1c',
                      transition: 'all 0.2s ease',
                    }}
                  >
                    {/* Stage index */}
                    <span style={{
                      fontSize: '0.65rem',
                      fontFamily: 'var(--font-mono)',
                      color: '#737373',
                      width: '20px',
                    }}>
                      {String(i + 1).padStart(2, '0')}
                    </span>

                    {/* Status icon */}
                    <span style={{
                      fontSize: '0.85rem',
                      color: STAGE_COLOR[stage.status] || '#737373',
                      width: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}>
                      {STAGE_ICON[stage.status] || '○'}
                    </span>

                    {/* Label & Description */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: '0.82rem',
                        fontWeight: 600,
                        color: stage.status === 'pending' ? '#737373' : '#F5F5F5',
                        letterSpacing: '-0.01em',
                      }}>
                        {stage.label}
                      </div>
                      {stage.detail && (
                        <div style={{
                          fontSize: '0.7rem',
                          fontFamily: 'var(--font-mono)',
                          color: STAGE_COLOR[stage.status] || '#A3A3A3',
                          marginTop: '2px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}>
                          {stage.detail}
                        </div>
                      )}
                    </div>

                    {/* Duration badge */}
                    {stage.durationMs && isDone && (
                      <span style={{
                        fontSize: '0.65rem',
                        fontFamily: 'var(--font-mono)',
                        color: '#34D399',
                        background: 'rgba(52,211,153,0.1)',
                        padding: '3px 8px',
                        borderRadius: '4px',
                        flexShrink: 0,
                      }}>
                        {(stage.durationMs / 1000).toFixed(1)}s
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Section 5: Primary Analysis Action (Large, prominent) */}
          <div style={{
            background: '#080808',
            border: '1px solid #262626',
            borderRadius: '16px',
            padding: '2rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
          }}>
            <div>
              <span style={{
                fontSize: '0.65rem',
                fontFamily: 'var(--font-mono)',
                letterSpacing: '0.16em',
                color: '#00E5FF',
                textTransform: 'uppercase',
                display: 'block',
                marginBottom: '0.35rem',
              }}>
                Section 05 · Primary Execution
              </span>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: 700,
                color: '#F5F5F5',
                letterSpacing: '-0.02em',
                margin: 0,
              }}>
                Execute Diagnostic Pipeline
              </h3>
            </div>

            <button
              id="run-analysis-btn"
              type="button"
              onClick={onRunAnalysis}
              disabled={!selectedImage || isAnalyzing}
              style={{
                width: '100%',
                padding: '1.25rem',
                borderRadius: '12px',
                border: !selectedImage || isAnalyzing ? '1px solid #262626' : '1px solid #00E5FF',
                background: !selectedImage || isAnalyzing
                  ? '#121212'
                  : 'linear-gradient(135deg, #00E5FF 0%, #00B4D8 100%)',
                color: !selectedImage || isAnalyzing ? '#737373' : '#000000',
                fontSize: '0.88rem',
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                cursor: !selectedImage || isAnalyzing ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: !selectedImage || isAnalyzing ? 'none' : '0 0 32px rgba(0,229,255,0.3)',
              }}
            >
              {isAnalyzing ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
                  <span style={{
                    display: 'inline-block',
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: '#00E5FF',
                    animation: 'pulse-ring 1s infinite',
                  }} />
                  Analyzing Multi-Modal Matrix...
                </span>
              ) : !selectedImage ? (
                'Select or Upload Medical Scan to Begin'
              ) : (
                '◆ Run Full Diagnostic Analysis'
              )}
            </button>

            <div style={{
              fontSize: '0.72rem',
              color: '#737373',
              fontFamily: 'var(--font-mono)',
              lineHeight: 1.6,
              background: '#040404',
              padding: '0.85rem 1rem',
              borderRadius: '8px',
              border: '1px solid #1a1a1a',
            }}>
              <strong>Clinical Protocol:</strong> This platform performs AI-assisted secondary triage and transparent attribution. Not intended as an autonomous clinical diagnosis. Must be reviewed by board-certified radiologists or attending clinicians.
            </div>
          </div>

        </div>

      </div>

      <style>{`
        @media (max-width: 980px) {
          .workspace-grid {
            grid-template-columns: 1fr !important;
            gap: 1.5rem !important;
          }
        }
      `}</style>
    </div>
  );
}
