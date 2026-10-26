"""
MedVision AI — FastAPI Backend
Production-grade medical AI API
Author: Sarvesh Bharat
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional
from functools import partial
import uvicorn
import io
import sys
from pathlib import Path
from PIL import Image
from gradcam import generate_gradcam_image, create_gradcam_figure   # noqa: F401  (legacy shim)
import io as io_module

# ─────────────────────────────────────────────
#  EXPLAINABLE AI (unified XAI layer)
# ─────────────────────────────────────────────
try:
    from xai import DEFAULT_PARAMS as XAI_DEFAULT_PARAMS
    from xai import TUNABLE_PARAMS as XAI_TUNABLE_PARAMS
    from xai import available_methods as xai_available_methods
except ImportError as _xai_import_error:     # optionals missing -> XAI degrades, prediction keeps working
    XAI_DEFAULT_PARAMS, XAI_TUNABLE_PARAMS, xai_available_methods = {}, {}, None
    _XAI_IMPORT_ERROR = str(_xai_import_error)
else:
    _XAI_IMPORT_ERROR = None

#: Where explanations requested with `?save=true` are written (research artefacts).
XAI_OUTPUT_DIR = Path(__file__).resolve().parent / "xai_outputs"

#: Hard bounds so an API caller cannot make the server do unbounded work.
XAI_LIMITS = {
    "num_samples": (10, 2000),
    "num_features": (1, 50),
    "batch_size": (1, 128),
    "max_evals": (10, 1024),
    "random_seed": (0, 2 ** 31 - 1),
}

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from medvision_inference import MedVisionEngine

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="MedVision AI",
    description="Production-grade multi-modal medical AI — CNN + DQN + Groq LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load engine once at startup
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    print("[MedVision API] Loading inference engine...")
    engine = MedVisionEngine()
    print("[MedVision API] Ready!")


# ─────────────────────────────────────────────
#  REQUEST MODELS
# ─────────────────────────────────────────────
class SymptomsRequest(BaseModel):
    symptoms: list[str]
    patient_info: Optional[dict] = None
    generate_report: Optional[bool] = True

class CombinedRequest(BaseModel):
    symptoms: list[str]
    scan_type: str
    patient_info: Optional[dict] = None
    generate_report: Optional[bool] = True


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "MedVision AI is running!",
        "version": "1.0.0",
        "author": "Sarvesh Bharat",
        "endpoints": [
            "GET  /health",
            "GET  /models",
            "GET  /dqn/status",
            "POST /dqn/predict",
            "POST /predict/symptoms",
            "POST /predict/image/{scan_type}",
            "POST /predict/combined/{scan_type}",
            "GET  /symptoms/list",
            "GET  /xai/methods",
            "POST /xai/{method}/{scan_type}   # method = gradcam | lime | shap",
            "POST /gradcam/{scan_type}        # legacy GRAD-CAM overlay",
        ]
    }


# ─────────────────────────────────────────────
#  SYSTEM / COMPONENT STATUS (truthful availability, never fabricated)
# ─────────────────────────────────────────────
def _gpu_status() -> dict:
    try:
        import torch as _t
        cuda = bool(_t.cuda.is_available())
        info = {
            "cuda_available": cuda,
            "torch_version": _t.__version__,
            "cuda_build": _t.version.cuda,
            "device": "cuda" if cuda else "cpu",
            "gpu_name": None,
            "vram_total_gb": None,
            "cudnn_version": None,
        }
        if cuda:
            try:
                info["gpu_name"] = _t.cuda.get_device_name(0)
                info["vram_total_gb"] = round(_t.cuda.get_device_properties(0).total_memory / 1e9, 2)
                info["cudnn_version"] = _t.backends.cudnn.version()
            except Exception:
                pass
        return info
    except Exception as exc:
        return {"cuda_available": False, "device": "cpu", "error": str(exc)}


@app.get("/health")
async def health():
    status = {"status": "healthy", "engine_loaded": engine is not None}
    if engine is not None:
        try:
            status["xai"] = engine.xai_methods().get("methods", {})
        except Exception:
            status["xai"] = {}
    status["gpu"] = _gpu_status()
    return status


@app.get("/system/gpu")
async def system_gpu():
    """Truthful CUDA/GPU report: torch build, device name, VRAM. No fabrication."""
    return _gpu_status()


@app.get("/dqn/status")
async def dqn_status():
    """Honest DQN triage state: loaded or exactly which artifact is missing."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    try:
        return engine.dqn_status()
    except AttributeError:
        loaded = bool(getattr(engine.dqn, "model", None) is not None)
        return {"loaded": loaded,
                "reason": None if loaded else "dqn_model.pth not found in project root"}


@app.post("/dqn/predict")
async def dqn_predict(request: SymptomsRequest):
    """
    Direct DQN prediction endpoint.
    Returns priority level, Q-values, policy probabilities, and research disclaimer.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")
    return engine.dqn.predict(request.symptoms)


@app.get("/llm/status")
async def llm_status():
    """Honest Gemma 4 LLM state: SDK present? API key set? Never a fake response."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    try:
        return engine.llm_status()
    except AttributeError:
        ok = bool(getattr(getattr(engine, "reporter", None), "client", None) is not None)
        return {"loaded": ok, "provider": "gemini",
                "reason": None if ok else "GEMMA_API_KEY not set or google-genai unavailable"}


# Verified test accuracies (read from cnn/*meta*.json + evaluation files — do not edit by hand).
VERIFIED_TEST_ACCURACY = {
    "chest": 0.9295,   # cnn/chest_xray_meta.json test_acc
    "brain": 0.9544,   # cnn/brain_test_results_v2.json accuracy
    "skin": 0.5932,    # cnn/skin_meta.json test_acc
    "eye": 0.8333,     # cnn/eye_meta_v3.json test_metrics.accuracy
}


@app.get("/models")
async def get_models():
    """Returns list of loaded models and their status."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    status = engine.available_models()
    if isinstance(status, dict):
        status = dict(status)
        status["cnn_test_accuracy"] = dict(VERIFIED_TEST_ACCURACY)
        status["cnn_accuracy_source"] = "cnn/*meta*.json (independent evaluation files)"
    return status


@app.post("/predict/symptoms")
async def predict_symptoms(request: SymptomsRequest):
    """
    Symptom-based triage using DQN.
    Returns priority level (1/2/3) and emergency flag.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")

    result = engine.predict_symptoms(
        symptoms=request.symptoms,
        generate_report=request.generate_report,
        patient_info=request.patient_info
    )
    return result


@app.post("/predict/image/{scan_type}")
async def predict_image(
    scan_type: str,
    file: UploadFile = File(...),
    generate_report: bool = True
):
    """
    Image-based diagnosis using CNN.
    scan_type: chest | brain | skin | eye
    Runs in a thread pool so the async event loop is never blocked.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    valid_types = ["chest", "brain", "skin", "eye"]
    if scan_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scan_type. Must be one of {valid_types}"
        )

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).copy()  # copy so it's safe to pass to threadpool
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    result = await run_in_threadpool(
        partial(engine.predict_image,
                image=image, scan_type=scan_type, generate_report=generate_report)
    )
    return result


@app.post("/predict/combined/{scan_type}")
async def predict_combined(
    scan_type: str,
    symptoms: str = Query(..., description="Comma-separated symptom list"),
    file: UploadFile = File(...),
    generate_report: bool = True
):
    """
    Full pipeline: image + symptoms → CNN + DQN + Groq LLM report.
    symptoms: comma-separated string e.g. "fever,cough,chest pain"
    scan_type: chest | brain | skin | eye
    Runs in a thread pool so the async event loop is never blocked.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    # Parse symptoms
    symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]

    # Read image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).copy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    result = await run_in_threadpool(
        partial(engine.predict_combined,
                image=image, scan_type=scan_type,
                symptoms=symptom_list, generate_report=generate_report)
    )
    return result


@app.get("/symptoms/list")
async def get_symptoms_list():
    """Returns the full list of supported symptoms for DQN triage."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if engine.dqn.model is None:
        return {"symptoms": [], "count": 0}
    return {
        "symptoms": engine.dqn.symptoms,
        "count": len(engine.dqn.symptoms)
    }

# ─────────────────────────────────────────────
#  EXPLAINABILITY ENDPOINTS
# ─────────────────────────────────────────────
def _clamp(name: str, value):
    """Clamp an optional numeric explanation setting to its safe range."""
    if value is None:
        return None
    low, high = XAI_LIMITS[name]
    return int(max(low, min(high, int(value))))


def _xai_settings_payload() -> dict:
    """Method availability + tunable settings consumed by the Streamlit UI."""
    if xai_available_methods is None:
        return {"methods": {}, "defaults": {}, "tunable": {},
                "error": f"XAI layer unavailable: {_XAI_IMPORT_ERROR}"}
    return {
        "methods": xai_available_methods(probe=True),
        "defaults": XAI_DEFAULT_PARAMS,
        "tunable": XAI_TUNABLE_PARAMS,
    }


def _run_xai(image: Image.Image, scan_type: str, method: str, target_class=None,
             save: bool = False, **params) -> dict:
    """Shared validation + execution path for the XAI endpoints."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if scan_type not in engine.cnns:
        raise HTTPException(status_code=400, detail=f"{scan_type} CNN not loaded")
    if xai_available_methods is None:
        raise HTTPException(status_code=503, detail=f"XAI layer unavailable: {_XAI_IMPORT_ERROR}")

    available = xai_available_methods(probe=True)
    if method not in available:
        raise HTTPException(status_code=400,
                            detail=f"unknown XAI method '{method}'. Use one of {sorted(available)}")
    if not available[method].get("available", True):
        raise HTTPException(status_code=503,
                            detail=f"{method} unavailable: {available[method].get('reason')}")

    cleaned = {key: value for key, value in params.items() if value is not None}
    try:
        payload = engine.explain_image(
            image=image, scan_type=scan_type, method=method, target_class=target_class,
            save_dir=str(XAI_OUTPUT_DIR) if save else None, **cleaned)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                     # never leak a stack trace to the UI
        raise HTTPException(status_code=500, detail=f"{method} explanation failed: {exc}")

    if isinstance(payload, dict) and payload.get("error"):
        raise HTTPException(status_code=400, detail=payload["error"])
    return payload


@app.get("/xai/methods")
async def xai_methods():
    """XAI methods available in this build + their tunable settings."""
    return _xai_settings_payload()


@app.post("/xai/{method}/{scan_type}")
async def xai_explain(
    method: str,
    scan_type: str,
    file: UploadFile = File(...),
    target_class: Optional[str] = None,
    num_samples: Optional[int] = None,
    num_features: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_evals: Optional[int] = None,
    mask_value: Optional[str] = None,
    reduce: Optional[str] = None,
    prefer: Optional[str] = None,
    segmentation: Optional[str] = None,
    alpha: Optional[float] = None,
    target_layer_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    make_panel: bool = True,
    save: bool = False,
):
    """
    Explain a scan with the selected method (gradcam | lime | shap).
    Runs in a thread pool so heavy XAI compute never blocks the event loop.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).copy()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    return await run_in_threadpool(
        partial(_run_xai,
                image, scan_type, method,
                target_class=target_class,
                save=save,
                num_samples=_clamp("num_samples", num_samples),
                num_features=_clamp("num_features", num_features),
                batch_size=_clamp("batch_size", batch_size),
                max_evals=_clamp("max_evals", max_evals),
                random_seed=_clamp("random_seed", random_seed),
                mask_value=mask_value,
                reduce=reduce,
                prefer=prefer,
                segmentation=segmentation,
                alpha=alpha,
                target_layer_path=target_layer_path,
                make_panel=make_panel)
    )


@app.post("/gradcam/{scan_type}")
async def get_gradcam(
    scan_type: str,
    file: UploadFile = File(...)
):
    """
    GRAD-CAM heatmap overlay as a PNG (kept for backwards compatibility).

    Backed by the unified XAI layer (`xai.gradcam`), so this endpoint and the
    Grad-CAM button in the UI always show the same explanation.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    if scan_type not in engine.cnns:
        raise HTTPException(status_code=400, detail=f"{scan_type} CNN not loaded")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    cnn_result = engine.cnns[scan_type].predict(image)
    result = engine.explain_image_result(image, scan_type, method="gradcam", make_panel=False)
    if isinstance(result, dict):                 # error dict rather than an XAIResult
        raise HTTPException(status_code=400, detail=result.get("error", "gradcam failed"))
    overlay = result.explanation_image or result.images.get("overlay")

    buf = io_module.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png",
                             headers={"X-Prediction": str(cnn_result["prediction"]),
                                      "X-Confidence": str(cnn_result["confidence"]),
                                      "X-Target-Layer": str(result.metadata.get("target_layer", ""))})


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )