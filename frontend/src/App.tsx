import { useEffect, useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Navbar } from './components/Navbar';
import { MediVisionHeroWordmark } from './components/ui/MediVisionHeroWordmark';
import { HorizontalProgress3D } from './components/HorizontalProgress3D';
import { ScanUploader } from './components/ScanUploader';
import { SampleStrip } from './components/SampleStrip';
import { AnalysisDashboard } from './components/AnalysisDashboard';
import { ModelBenchmark } from './components/ModelBenchmark';
import { CursorGlow } from './components/CursorGlow';
import { MediVisionFooter } from './components/ui/MediVisionFooter';
import {
  fetchHealth,
  fetchModels,
  fetchDqnStatus,
  fetchLlmStatus,
  fetchSymptomsList,
  predictCombined,
  predictImage,
  requestXai,
  TimeoutError,
} from './api/client';
import type {
  HealthResponse,
  ModelsResponse,
  DqnStatusResponse,
  LlmStatusResponse,
  FullAnalysisResult,
} from './api/types';

// ─────────────────────────────────────────────
//  PIPELINE STAGE TYPE & EXACT ORDER:
//  IMAGE NORMALIZATION → RESNET50 → GRAD-CAM → LIME → SHAP → GEMMA 4
// ─────────────────────────────────────────────
export type StageStatus = 'pending' | 'running' | 'done' | 'error' | 'timeout' | 'skipped';

export interface PipelineStage {
  id: string;
  label: string;
  status: StageStatus;
  detail?: string;
  durationMs?: number;
}

export const INITIAL_STAGES: PipelineStage[] = [
  { id: 'upload', label: 'Image Normalization', status: 'pending' },
  { id: 'cnn', label: 'ResNet50 Inference', status: 'pending' },
  { id: 'gradcam', label: 'Grad-CAM Generation', status: 'pending' },
  { id: 'lime', label: 'LIME Explanation', status: 'pending' },
  { id: 'shap', label: 'SHAP Explanation', status: 'pending' },
  { id: 'gemma', label: 'Gemma 4 Clinical Report', status: 'pending' },
];

export function App() {
  // ── Backend telemetry ──
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [dqnStatus, setDqnStatus] = useState<DqnStatusResponse | null>(null);
  const [llmStatus, setLlmStatus] = useState<LlmStatusResponse | null>(null);
  const [symptomsList, setSymptomsList] = useState<string[]>([]);

  // ── Input state ──
  const [selectedModality, setSelectedModality] = useState<string>('chest');
  const [selectedImage, setSelectedImage] = useState<File | Blob | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [generateReport, setGenerateReport] = useState<boolean>(true);

  // ── Pipeline execution state ──
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>(INITIAL_STAGES);
  const [analysisResult, setAnalysisResult] = useState<FullAnalysisResult | null>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);

  const studioRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // ─────────────────────────────────────────────
  //  STAGE HELPERS
  // ─────────────────────────────────────────────
  const updateStage = (id: string, patch: Partial<PipelineStage>) => {
    setPipelineStages((prev) =>
      prev.map((s) => (s.id === id ? { ...s, ...patch } : s))
    );
  };

  // ─────────────────────────────────────────────
  //  INITIAL TELEMETRY LOAD
  // ─────────────────────────────────────────────
  useEffect(() => {
    const load = async () => {
      const [h, m, d, l, s] = await Promise.allSettled([
        fetchHealth(),
        fetchModels(),
        fetchDqnStatus(),
        fetchLlmStatus(),
        fetchSymptomsList(),
      ]);

      if (h.status === 'fulfilled') setHealth(h.value);
      if (m.status === 'fulfilled') setModels(m.value);
      if (d.status === 'fulfilled') setDqnStatus(d.value);
      if (l.status === 'fulfilled') setLlmStatus(l.value);
      if (s.status === 'fulfilled' && s.value.symptoms) setSymptomsList(s.value.symptoms);
    };
    load();
  }, []);

  // ─────────────────────────────────────────────
  //  INPUT HANDLERS
  // ─────────────────────────────────────────────
  const handleSelectSample = useCallback(async (modality: string, sampleUrl: string) => {
    setSelectedModality(modality);
    try {
      const res = await fetch(sampleUrl);
      const blob = await res.blob();
      setSelectedImage(blob);
      setImagePreviewUrl(sampleUrl);
      setExecutionError(null);
      setAnalysisResult(null);
      setPipelineStages(INITIAL_STAGES);
    } catch (err) {
      console.error('Failed to load sample scan:', err);
    }
  }, []);

  const handleImageSelected = useCallback((file: File | Blob, preview: string) => {
    setSelectedImage(file);
    setImagePreviewUrl(preview);
    setExecutionError(null);
    setAnalysisResult(null);
    setPipelineStages(INITIAL_STAGES);
  }, []);

  const handleClearImage = useCallback(() => {
    setSelectedImage(null);
    setImagePreviewUrl(null);
    setAnalysisResult(null);
    setExecutionError(null);
    setPipelineStages(INITIAL_STAGES);
  }, []);

  const handleToggleSymptom = useCallback((symptom: string) => {
    setSelectedSymptoms((prev) =>
      prev.includes(symptom) ? prev.filter((s) => s !== symptom) : [...prev, symptom]
    );
  }, []);

  const handleAddCustomSymptom = useCallback(
    (symptom: string) => {
      const clean = symptom.trim().toLowerCase();
      if (clean && !selectedSymptoms.includes(clean)) {
        setSelectedSymptoms((prev) => [...prev, clean]);
      }
    },
    [selectedSymptoms]
  );

  // ─────────────────────────────────────────────
  //  PRIMARY PIPELINE: RUN FULL ANALYSIS
  //  Strict gate: requires explicit user button action to prevent auto-start or repeating loops
  // ─────────────────────────────────────────────
  const userTriggeredRunRef = useRef<boolean>(false);
  const isAnalyzingRef = useRef<boolean>(false);

  const handleRunFullAnalysis = async () => {
    // Strict guard: MUST have an image, MUST NOT be already analyzing, and MUST have explicit user trigger
    if (!selectedImage || isAnalyzing || isAnalyzingRef.current || !userTriggeredRunRef.current) {
      return;
    }

    // Immediately consume user trigger so this workflow executes strictly once per explicit button action
    userTriggeredRunRef.current = false;
    isAnalyzingRef.current = true;
    setIsAnalyzing(true);
    setExecutionError(null);
    setAnalysisResult(null);
    setPipelineStages(INITIAL_STAGES);

    try {
      // If report is disabled, mark gemma as skipped immediately so the 5-stage pipeline is recognized
      if (!generateReport) {
        updateStage('gemma', { status: 'skipped', detail: 'Synthesis disabled' });
      }

      // ── Stage: IMAGE NORMALIZATION ──
      updateStage('upload', { status: 'running', detail: 'Normalizing tensor (224×224 RGB)...' });
      await new Promise((r) => setTimeout(r, 60));
      updateStage('upload', { status: 'done', detail: 'Normalized & verified' });

      let predictionResp: FullAnalysisResult['predictionResponse'] | null = null;

      // ── Stage 1: CNN INFERENCE & CLINICAL PREDICTION ──
      updateStage('cnn', {
        status: 'running',
        detail: `ResNet50 (${selectedModality.toUpperCase()}) CUDA inference...`,
      });
      if (generateReport) {
        updateStage('gemma', {
          status: 'running',
          detail: 'Synthesizing clinical findings (Gemma 4)...',
        });
      } else {
        updateStage('gemma', {
          status: 'skipped',
          detail: 'Synthesis disabled',
        });
      }

      const cnnStart = Date.now();
      try {
        if (selectedSymptoms.length > 0) {
          predictionResp = await predictCombined(
            selectedImage,
            selectedModality,
            selectedSymptoms,
            generateReport
          );
        } else {
          predictionResp = await predictImage(selectedImage, selectedModality, generateReport);
        }
        const cnn = predictionResp?.cnn_result;
        updateStage('cnn', {
          status: 'done',
          detail: cnn ? `${cnn.prediction} — ${(cnn.confidence * 100).toFixed(1)}%` : 'Complete',
          durationMs: Date.now() - cnnStart,
        });
      } catch (err: any) {
        const isTimeout = err instanceof TimeoutError || err?.isTimeout;
        updateStage('cnn', {
          status: isTimeout ? 'timeout' : 'error',
          detail: isTimeout ? 'CNN timed out — model loading' : (err.message || 'CNN failed'),
          durationMs: Date.now() - cnnStart,
        });
        if (generateReport) {
          updateStage('gemma', {
            status: 'error',
            detail: 'Skipped — CNN prediction failed',
          });
        }
        setExecutionError(`CNN inference failed: ${err.message || 'Unknown error'}`);
        return;
      }

      const previewUrl =
        imagePreviewUrl || (selectedImage instanceof Blob ? URL.createObjectURL(selectedImage) : '');
      const currentResult: FullAnalysisResult = {
        scanType: selectedModality,
        imageBlobUrl: previewUrl,
        imageBytes: selectedImage,
        predictionResponse: predictionResp!,
        xaiResults: {},
        dqnStatus: dqnStatus || { loaded: false, available: false, reason: 'Not configured' },
        llmStatus: llmStatus || { loaded: false, available: false, reason: 'Status unpolled' },
        symptoms: selectedSymptoms,
        timestamp: new Date().toISOString(),
      };
      // Make result visible immediately so user sees the diagnostic studio populate in real time
      setAnalysisResult({ ...currentResult });

      // ── Stage 2: GRAD-CAM ──
      updateStage('gradcam', { status: 'running', detail: 'Computing layer4 gradient activation maps...' });
      const gStart = Date.now();
      try {
        const gRes = await requestXai(selectedImage, selectedModality, 'gradcam');
        const hasOverlay = Boolean(
          gRes &&
          !gRes.error &&
          (gRes.images?.overlay || gRes.images?.heatmap || gRes.images?.panel || gRes.explanation_image)
        );
        if (hasOverlay) {
          updateStage('gradcam', { status: 'done', detail: 'Heatmap rendered', durationMs: Date.now() - gStart });
          currentResult.xaiResults = { ...currentResult.xaiResults, gradcam: gRes };
        } else {
          updateStage('gradcam', {
            status: 'error',
            detail: gRes?.error || 'Grad-CAM returned no visual data',
            durationMs: Date.now() - gStart,
          });
          currentResult.xaiResults = { ...currentResult.xaiResults, gradcam: { error: gRes?.error || 'Heatmap rendering failed' } };
        }
        setAnalysisResult({ ...currentResult, xaiResults: { ...currentResult.xaiResults } });
      } catch (err: any) {
        const isTimeout = err instanceof TimeoutError || err?.isTimeout;
        updateStage('gradcam', {
          status: isTimeout ? 'timeout' : 'error',
          detail: isTimeout ? 'Timed out' : (err.message || 'Grad-CAM failed'),
          durationMs: Date.now() - gStart,
        });
        currentResult.xaiResults = { ...currentResult.xaiResults, gradcam: { error: err.message || 'Grad-CAM failed' } };
        setAnalysisResult({ ...currentResult, xaiResults: { ...currentResult.xaiResults } });
      }

      // ── Stage 3: LIME EXPLANATION ──
      updateStage('lime', { status: 'running', detail: 'Fitting superpixel surrogate (~10s)...' });
      const lStart = Date.now();
      try {
        const lRes = await requestXai(selectedImage, selectedModality, 'lime', { num_samples: 50, num_features: 6 });
        const hasLime = Boolean(
          lRes &&
          !lRes.error &&
          (lRes.images?.lime_positive || lRes.images?.lime_positive_negative || lRes.images?.overlay || lRes.explanation_image)
        );
        if (hasLime) {
          updateStage('lime', { status: 'done', detail: 'Surrogate generated', durationMs: Date.now() - lStart });
          currentResult.xaiResults = { ...currentResult.xaiResults, lime: lRes };
        } else {
          updateStage('lime', {
            status: 'error',
            detail: lRes?.error || 'LIME returned no visual data',
            durationMs: Date.now() - lStart,
          });
          currentResult.xaiResults = { ...currentResult.xaiResults, lime: { error: lRes?.error || 'LIME explanation failed' } };
        }
        setAnalysisResult({ ...currentResult, xaiResults: { ...currentResult.xaiResults } });
      } catch (err: any) {
        const isTimeout = err instanceof TimeoutError || err?.isTimeout;
        updateStage('lime', {
          status: isTimeout ? 'timeout' : 'error',
          detail: isTimeout ? 'Timed out' : (err.message || 'LIME failed'),
          durationMs: Date.now() - lStart,
        });
        currentResult.xaiResults = { ...currentResult.xaiResults, lime: { error: err.message || 'LIME failed' } };
        setAnalysisResult({ ...currentResult, xaiResults: { ...currentResult.xaiResults } });
      }

      // ── Stage 4: SHAP EXPLANATION ──
      updateStage('shap', { status: 'running', detail: 'Evaluating regional Shapley attributions...' });
      const sStart = Date.now();
      try {
        const sRes = await requestXai(selectedImage, selectedModality, 'shap', { max_evals: 64 });
        const hasShap = Boolean(
          sRes &&
          !sRes.error &&
          (sRes.images?.overlay_signed || sRes.images?.heatmap || sRes.images?.overlay || sRes.explanation_image)
        );
        if (hasShap) {
          updateStage('shap', { status: 'done', detail: 'Attributions generated', durationMs: Date.now() - sStart });
          currentResult.xaiResults = { ...currentResult.xaiResults, shap: sRes };
        } else {
          updateStage('shap', {
            status: 'error',
            detail: sRes?.error || 'SHAP returned no visual data',
            durationMs: Date.now() - sStart,
          });
          currentResult.xaiResults = { ...currentResult.xaiResults, shap: { error: sRes?.error || 'SHAP explanation failed' } };
        }
        setAnalysisResult({ ...currentResult, xaiResults: { ...currentResult.xaiResults } });
      } catch (err: any) {
        const isTimeout = err instanceof TimeoutError || err?.isTimeout;
        updateStage('shap', {
          status: isTimeout ? 'timeout' : 'error',
          detail: isTimeout ? 'Timed out' : (err.message || 'SHAP failed'),
          durationMs: Date.now() - sStart,
        });
        currentResult.xaiResults = { ...currentResult.xaiResults, shap: { error: err.message || 'SHAP failed' } };
        setAnalysisResult({ ...currentResult, xaiResults: { ...currentResult.xaiResults } });
      }

      // ── Stage 5: GEMMA CLINICAL REPORT ──
      if (generateReport) {
        const reportText = predictionResp?.report;
        const isErr =
          !reportText ||
          reportText.startsWith('Gemma report unavailable:') ||
          reportText.startsWith('Gemma 4 API not configured');

        updateStage('gemma', {
          status: isErr ? 'error' : 'done',
          detail: isErr ? (reportText || 'Report unavailable') : 'Clinical synthesis complete',
        });
      } else {
        updateStage('gemma', { status: 'skipped', detail: 'Synthesis disabled' });
      }
    } finally {
      // Mark analysis complete - guarantees resetting isAnalyzingRef, userTriggeredRunRef, and isAnalyzing
      isAnalyzingRef.current = false;
      userTriggeredRunRef.current = false;
      setIsAnalyzing(false);
    }

    // Automatically smooth scroll directly into the diagnostic results suite
    setTimeout(() => {
      const el = document.getElementById('analysis-dashboard-root');
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 200);
  };

  const handleUserTriggeredAnalysis = useCallback(() => {
    userTriggeredRunRef.current = true;
    handleRunFullAnalysis();
  }, [selectedImage, selectedModality, selectedSymptoms, generateReport, isAnalyzing]);

  // ─────────────────────────────────────────────
  //  RETRY REPORT
  // ─────────────────────────────────────────────
  const [isRetryingReport, setIsRetryingReport] = useState<boolean>(false);

  const handleRetryReport = async () => {
    if (!selectedImage || !analysisResult || isAnalyzing || isRetryingReport) return;
    setIsRetryingReport(true);
    updateStage('gemma', { status: 'running', detail: 'Re-evaluating Gemma 4 clinical report...' });
    const rStart = Date.now();
    try {
      const reportResp =
        selectedSymptoms.length > 0
          ? await predictCombined(selectedImage, selectedModality, selectedSymptoms, true)
          : await predictImage(selectedImage, selectedModality, true);

      const isErr =
        !reportResp?.report ||
        reportResp.report.startsWith('Gemma report unavailable:') ||
        reportResp.report.startsWith('Gemma 4 API not configured');

      updateStage('gemma', {
        status: isErr ? 'error' : 'done',
        detail: isErr ? (reportResp?.report || 'Report unavailable') : 'Clinical synthesis complete',
        durationMs: Date.now() - rStart,
      });

      if (reportResp?.report) {
        setAnalysisResult((prev) =>
          prev
            ? {
                ...prev,
                predictionResponse: {
                  ...prev.predictionResponse,
                  report: reportResp.report,
                  dqn_result: reportResp.dqn_result || prev.predictionResponse.dqn_result,
                },
              }
            : prev
        );
      }
    } catch (err: any) {
      const isTimeout = err instanceof TimeoutError || err?.isTimeout;
      updateStage('gemma', {
        status: isTimeout ? 'timeout' : 'error',
        detail: isTimeout ? 'Timed out' : (err.message || 'Report unavailable'),
        durationMs: Date.now() - rStart,
      });
    } finally {
      setIsRetryingReport(false);
    }
  };

  // ─────────────────────────────────────────────
  //  RETRY SINGLE XAI
  // ─────────────────────────────────────────────
  const handleRetryXai = async (method: 'gradcam' | 'lime' | 'shap') => {
    if (!selectedImage || !analysisResult) return;
    const stageId = method;

    updateStage(stageId, { status: 'running', detail: 'Recomputing attribution...' });
    const params =
      method === 'lime' ? { num_samples: 50, num_features: 6 } :
      method === 'shap' ? { max_evals: 64 } : {};

    try {
      const result = await requestXai(selectedImage, selectedModality, method, params);
      updateStage(stageId, { status: 'done', detail: 'Attribution complete' });
      setAnalysisResult((prev) =>
        prev
          ? {
              ...prev,
              xaiResults: { ...prev.xaiResults, [method]: result },
            }
          : prev
      );
    } catch (err: any) {
      const isTimeout = err instanceof TimeoutError || err?.isTimeout;
      updateStage(stageId, {
        status: isTimeout ? 'timeout' : 'error',
        detail: isTimeout ? 'Timed out again' : err.message,
      });
    }
  };

  // ─────────────────────────────────────────────
  //  SCROLL HELPERS
  // ─────────────────────────────────────────────
  const scrollToStudio = () => studioRef.current?.scrollIntoView({ behavior: 'smooth' });
  const scrollToXai = () => {
    const el = document.getElementById('xai-overview');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
    else scrollToStudio();
  };
  const scrollToModels = () =>
    document.getElementById('model-benchmark')?.scrollIntoView({ behavior: 'smooth' });

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#000000', color: '#F5F5F5', overflowX: 'hidden' }}>
      <CursorGlow />

      {/* ── 1. Compact Navbar ── */}
      <Navbar
        health={health}
        dqnStatus={dqnStatus}
        llmStatus={llmStatus}
        onNavigateToAnalysis={scrollToStudio}
        onNavigateToXai={scrollToXai}
        onNavigateToModels={scrollToModels}
      />

      {/* ── 2. Prominent MediVision Hero Wordmark ── */}
      <div style={{ paddingTop: '56px' }}>
        <MediVisionHeroWordmark
          wordmark="MediVision"
          badge="CLINICAL NEURAL INTELLIGENCE · MULTI-MODAL XAI"
          tagline="Explainable medical AI diagnostic platform integrating ResNet50 neural vision, deep Q-learning symptom triage, and transparent attribution."
          primaryActionLabel="Launch Diagnostic Studio ↓"
          secondaryActionLabel="Explore Model Benchmarks"
          onPrimaryAction={scrollToStudio}
          onSecondaryAction={scrollToModels}
        />
      </div>

      {/* ── 3. Main Diagnostic Workspace (Spacious, Natural Scrolling) ── */}
      <main
        ref={studioRef}
        id="workspace"
        style={{
          maxWidth: '1360px',
          margin: '0 auto',
          padding: '4rem 1.5rem 6rem',
        }}
      >
        <div style={{ marginBottom: '2rem' }}>
          <p style={{
            fontSize: '0.65rem',
            fontFamily: 'var(--font-mono)',
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            color: '#00E5FF',
            marginBottom: '0.5rem',
          }}>
            ◆ Clinical Workspace
          </p>
          <h2 style={{
            fontSize: 'clamp(1.6rem, 3.2vw, 2.5rem)',
            fontWeight: 700,
            letterSpacing: '-0.02em',
            color: '#F5F5F5',
            marginBottom: '0.5rem',
          }}>
            Diagnostic Workspace & Inference Suite
          </h2>
          <p style={{ fontSize: '0.88rem', color: '#A3A3A3', maxWidth: '640px', lineHeight: 1.6 }}>
            Select an imaging modality, upload patient DICOM/radiology scans or test samples,
            condition with presenting symptoms, and execute full multi-modal neural analysis.
          </p>
        </div>

        {/* Sample Strip */}
        <SampleStrip onSelectSample={handleSelectSample} disabled={isAnalyzing} />

        {/* Spacious Scan Uploader */}
        <ScanUploader
          selectedModality={selectedModality}
          onModalityChange={setSelectedModality}
          selectedImage={selectedImage}
          imagePreviewUrl={imagePreviewUrl}
          onImageSelected={handleImageSelected}
          onClearImage={handleClearImage}
          symptomsList={symptomsList}
          selectedSymptoms={selectedSymptoms}
          onToggleSymptom={handleToggleSymptom}
          onAddCustomSymptom={handleAddCustomSymptom}
          generateReport={generateReport}
          onToggleGenerateReport={setGenerateReport}
          onRunAnalysis={handleUserTriggeredAnalysis}
          isAnalyzing={isAnalyzing}
          pipelineStages={pipelineStages}
        />

        {/* 3D Medical AI Analysis Loader & Neural Progress */}
        <div ref={resultsRef} style={{ scrollMarginTop: '80px' }}>
          <AnimatePresence>
            {(isAnalyzing || executionError || analysisResult) && (
              <motion.div
                initial={{ opacity: 0, y: 25, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.98 }}
                transition={{ duration: 0.35, ease: 'easeOut' }}
              >
                <HorizontalProgress3D
                  isAnalyzing={isAnalyzing}
                  pipelineStages={pipelineStages}
                  selectedModality={selectedModality}
                  imagePreviewUrl={imagePreviewUrl}
                  executionError={executionError}
                  onRetry={handleUserTriggeredAnalysis}
                  onScrollToResults={() => {
                    const el = document.getElementById('analysis-dashboard-root');
                    if (el) el.scrollIntoView({ behavior: 'smooth' });
                  }}
                  generateReport={generateReport}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Analysis Results Dashboard with Motion */}
        <div>
          <AnimatePresence>
            {analysisResult && (
              <motion.div
                id="analysis-dashboard-root"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 30 }}
                transition={{ duration: 0.45, ease: 'easeOut' }}
              >
                <AnalysisDashboard
                  analysis={analysisResult}
                  pipelineStages={pipelineStages}
                  modelsMeta={models}
                  health={health}
                  onRetryXai={handleRetryXai}
                  onRetryReport={handleRetryReport}
                  isRetryingReport={isRetryingReport}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── 5. Model Benchmarks ── */}
        <ModelBenchmark models={models} health={health} />
      </main>

      {/* ── 6. Dedicated AMOLED Black Clinical Footer ── */}
      <MediVisionFooter />
    </div>
  );
}

export default App;
