import type {
  HealthResponse,
  ModelsResponse,
  DqnStatusResponse,
  LlmStatusResponse,
  XaiMethodsResponse,
  PredictionResponse,
  XaiResultResponse,
} from './types';

/**
 * In development, all requests go through the Vite dev-server proxy
 * (/api → http://localhost:8000) which avoids cross-origin issues.
 * In production, set VITE_API_URL to the backend's actual origin.
 */
export const API_BASE = import.meta.env.VITE_API_URL || '/api';

// ─────────────────────────────────────────────
//  TIMEOUT CONFIGURATION (milliseconds)
// ─────────────────────────────────────────────
export const TIMEOUTS = {
  status:    8_000,   // GET health / models / status endpoints
  cnn:      90_000,   // CNN inference (first call warms up GPU — up to 90s)
  gradcam:  60_000,   // Grad-CAM: usually fast after warmup
  lime:    180_000,   // LIME: slowest XAI method
  shap:    120_000,   // SHAP: moderate
  gemma:    60_000,   // Gemma API (remote call)
};

// ─────────────────────────────────────────────
//  FETCH WITH ABORT/TIMEOUT
// ─────────────────────────────────────────────
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } catch (err: any) {
    if (err?.name === 'AbortError') {
      throw new TimeoutError(`Request timed out after ${timeoutMs / 1000}s`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export class TimeoutError extends Error {
  readonly isTimeout = true;
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
  }
}

// ─────────────────────────────────────────────
//  STATUS / TELEMETRY ENDPOINTS
// ─────────────────────────────────────────────
export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/health`, {}, TIMEOUTS.status);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/models`, {}, TIMEOUTS.status);
  if (!res.ok) throw new Error(`Models fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchGpuStatus(): Promise<HealthResponse['gpu']> {
  const res = await fetchWithTimeout(`${API_BASE}/system/gpu`, {}, TIMEOUTS.status);
  if (!res.ok) throw new Error(`GPU status fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDqnStatus(): Promise<DqnStatusResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/dqn/status`, {}, TIMEOUTS.status);
  if (!res.ok) throw new Error(`DQN status fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchLlmStatus(): Promise<LlmStatusResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/llm/status`, {}, TIMEOUTS.status);
  if (!res.ok) throw new Error(`LLM status fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSymptomsList(): Promise<{ symptoms: string[]; count: number }> {
  const res = await fetchWithTimeout(`${API_BASE}/symptoms/list`, {}, TIMEOUTS.status);
  if (!res.ok) throw new Error(`Symptoms fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchXaiMethods(): Promise<XaiMethodsResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/xai/methods`, {}, TIMEOUTS.status);
  if (!res.ok) throw new Error(`XAI methods fetch failed: ${res.statusText}`);
  return res.json();
}

// ─────────────────────────────────────────────
//  PREDICTION ENDPOINTS
// ─────────────────────────────────────────────
export async function predictImage(
  imageBlob: Blob,
  scanType: string,
  generateReport: boolean = true
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'scan.png');

  const url = `${API_BASE}/predict/image/${scanType}?generate_report=${generateReport}`;
  const timeoutMs = generateReport ? TIMEOUTS.cnn + TIMEOUTS.gemma : TIMEOUTS.cnn;
  const res = await fetchWithTimeout(url, { method: 'POST', body: formData }, timeoutMs);

  if (!res.ok) {
    const errText = await res.text().catch(() => res.statusText);
    throw new Error(`Prediction error (${res.status}): ${errText}`);
  }
  return res.json();
}

export async function predictSymptoms(
  symptoms: string[],
  generateReport: boolean = true
): Promise<PredictionResponse> {
  const res = await fetchWithTimeout(
    `${API_BASE}/predict/symptoms`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms, generate_report: generateReport }),
    },
    TIMEOUTS.cnn
  );

  if (!res.ok) {
    const errText = await res.text().catch(() => res.statusText);
    throw new Error(`Symptoms triage error (${res.status}): ${errText}`);
  }
  return res.json();
}

export async function predictCombined(
  imageBlob: Blob,
  scanType: string,
  symptoms: string[],
  generateReport: boolean = true
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'scan.png');

  const url = `${API_BASE}/predict/combined/${scanType}?symptoms=${encodeURIComponent(symptoms.join(','))}&generate_report=${generateReport}`;
  const timeoutMs = generateReport ? TIMEOUTS.cnn + TIMEOUTS.gemma : TIMEOUTS.cnn;
  const res = await fetchWithTimeout(url, { method: 'POST', body: formData }, timeoutMs);

  if (!res.ok) {
    const errText = await res.text().catch(() => res.statusText);
    throw new Error(`Combined analysis error (${res.status}): ${errText}`);
  }
  return res.json();
}

// ─────────────────────────────────────────────
//  XAI ENDPOINTS (individual with per-method timeouts)
// ─────────────────────────────────────────────
export async function requestXai(
  imageBlob: Blob,
  scanType: string,
  method: 'gradcam' | 'lime' | 'shap',
  params: Record<string, any> = {}
): Promise<XaiResultResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'scan.png');

  const qs = Object.entries(params)
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`)
    .join('&');
  const url = `${API_BASE}/xai/${method}/${scanType}${qs ? `?${qs}` : ''}`;

  const methodTimeout =
    method === 'lime' ? TIMEOUTS.lime :
    method === 'shap' ? TIMEOUTS.shap :
    TIMEOUTS.gradcam;

  const res = await fetchWithTimeout(
    url,
    { method: 'POST', body: formData },
    methodTimeout
  );

  if (!res.ok) {
    const errText = await res.text().catch(() => res.statusText);
    throw new Error(`${method.toUpperCase()} XAI failed (${res.status}): ${errText}`);
  }
  return res.json();
}
