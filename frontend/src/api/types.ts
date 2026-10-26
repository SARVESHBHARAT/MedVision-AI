export interface HealthResponse {
  status: string;
  engine_loaded: boolean;
  xai?: Record<string, {
    method: string;
    label: string;
    description: string;
    available: boolean;
    reason: string | null;
  }>;
  gpu: {
    cuda_available: boolean;
    torch_version?: string;
    cuda_build?: string;
    device: string;
    gpu_name?: string | null;
    vram_total_gb?: number | null;
    cudnn_version?: number | null;
    error?: string;
  };
}

export interface ModelsResponse {
  cnns: string[];
  dqn: boolean;
  gemma4: boolean;
  device: string;
  cnn_test_accuracy: Record<string, number>;
  cnn_accuracy_source: string;
}

export interface DqnStatusResponse {
  loaded: boolean;
  available: boolean;
  checkpoint_expected?: string;
  checkpoint_found?: boolean;
  symptom_count?: number;
  reason?: string | null;
  note?: string;
}

export interface LlmStatusResponse {
  loaded: boolean;
  available: boolean;
  provider?: string;
  mode?: string;
  model?: string;
  sdk_installed?: boolean;
  api_key_set?: boolean;
  reason?: string | null;
  note?: string;
}

export interface XaiMethodItem {
  method: string;
  label: string;
  description: string;
  available: boolean;
  reason?: string | null;
}

export interface XaiMethodsResponse {
  methods: Record<string, XaiMethodItem>;
  defaults?: Record<string, any>;
  tunable?: Record<string, any>;
  error?: string;
}

export interface CnnResult {
  model: string;
  prediction: string;
  confidence: number; // 0.0 to 1.0 (float)
  all_scores: Record<string, number>;
}

export interface DqnResult {
  model?: string;
  matched_symptoms?: string[];
  action?: number;
  priority?: number; // 1: Critical, 2: Moderate, 3: Non-urgent
  priority_label?: string;
  emergency?: boolean;
  error?: string;
}

export interface PredictionResponse {
  scan_type?: string;
  symptoms?: string[];
  cnn_result?: CnnResult;
  dqn_result?: DqnResult;
  emergency: boolean;
  call_108: boolean;
  report?: string | null;
  error?: string;
}

export interface XaiResultResponse {
  method: string;
  method_label?: string;
  scan_type: string;
  predicted_class: string;
  confidence: number;
  target_class: string;
  images: {
    original?: string;
    heatmap?: string;
    overlay?: string;
    overlay_signed?: string;
    lime_positive?: string;
    lime_positive_negative?: string;
    panel?: string;
    [key: string]: string | undefined;
  };
  explanation_image?: string;
  parameters?: Record<string, any>;
  metadata?: Record<string, any>;
  warnings?: string[];
  error?: string;
}

export interface FullAnalysisResult {
  scanType: string;
  imageBlobUrl: string;
  imageBytes: Blob;
  predictionResponse: PredictionResponse;
  xaiResults: {
    gradcam?: XaiResultResponse | { error: string };
    lime?: XaiResultResponse | { error: string };
    shap?: XaiResultResponse | { error: string };
  };
  dqnStatus: DqnStatusResponse;
  llmStatus: LlmStatusResponse;
  symptoms: string[];
  timestamp: string;
}
