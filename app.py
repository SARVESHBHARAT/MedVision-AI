"""
MedVision AI — Streamlit Frontend
Production-grade medical AI interface
Author: Spandan Das
"""

import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
import hashlib

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
API_URL = "http://localhost:8000"
st.set_page_config(
    page_title="MedVision AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    * { font-family: 'DM Sans', sans-serif; }
    
    .main { background-color: #0a0f1e; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a0f1e 100%);
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #0099ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.1rem;
        color: #8899aa;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .emergency-banner {
        background: linear-gradient(135deg, #ff0040, #ff4060);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        animation: pulse 1.5s infinite;
        margin: 1rem 0;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 64, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(255, 0, 64, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 64, 0); }
    }

    .emergency-text {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }

    .result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
    }

    .priority-1 {
        border-left: 4px solid #ff0040;
        background: rgba(255, 0, 64, 0.05);
    }
    .priority-2 {
        border-left: 4px solid #ff8800;
        background: rgba(255, 136, 0, 0.05);
    }
    .priority-3 {
        border-left: 4px solid #00ff88;
        background: rgba(0, 255, 136, 0.05);
    }

    .metric-card {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099ff) !important;
        color: #0a0f1e !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3) !important;
    }

    .report-box {
        background: rgba(0, 212, 255, 0.03);
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        color: #ccd9e8;
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    .scan-badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .sidebar .sidebar-content {
        background: rgba(0,0,0,0.3) !important;
    }

    div[data-testid="stSidebarContent"] {
        background: rgba(10, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.1) !important;
    }

    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #e8f0fe !important;
    }

    .stRadio > div {
        gap: 0.5rem;
    }

    p, li { color: #aabbcc; }
    
    .divider {
        border: none;
        border-top: 1px solid rgba(0, 212, 255, 0.1);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def get_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=3)
        return r.json()
    except:
        return {}

def predict_symptoms(symptoms, generate_report=True):
    r = requests.post(
        f"{API_URL}/predict/symptoms",
        json={"symptoms": symptoms, "generate_report": generate_report},
        timeout=60
    )
    return r.json()

def predict_image(image_bytes, scan_type, generate_report=True):
    r = requests.post(
        f"{API_URL}/predict/image/{scan_type}",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"generate_report": generate_report},
        timeout=60
    )
    return r.json()

def priority_color(p):
    return {1: "#ff0040", 2: "#ffaa00", 3: "#00ff88"}.get(p, "#8899aa")

def priority_label(p):
    return {
        1: "🔴 PRIORITY 1 — EMERGENCY / CRITICAL",
        2: "🟡 PRIORITY 2 — URGENT / MODERATE",
        3: "🟢 PRIORITY 3 — ROUTINE / NON-URGENT",
    }.get(p, f"Priority {p}")

def priority_explanation(p):
    return {
        1: "Priority 1 (Emergency): Critical presentation requiring immediate clinical intervention, stabilization, or emergency transfer.",
        2: "Priority 2 (Urgent): Potentially serious presentation requiring timely clinical evaluation within hours and active monitoring.",
        3: "Priority 3 (Routine): Stable presentation suitable for standard outpatient consultation, supportive care, or scheduled follow-up.",
    }.get(p, "Standard clinical triage evaluation.")

def get_gradcam(image_bytes, scan_type):
    try:
        r = requests.post(
            f"{API_URL}/gradcam/{scan_type}",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=60
        )
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
    except:
        pass
    return None


# ─────────────────────────────────────────────
#  EXPLAINABLE AI HELPERS
# ─────────────────────────────────────────────
XAI_TIMEOUT = {"gradcam": 90, "lime": 300, "shap": 300}
XAI_ICONS = {"gradcam": "🔥", "lime": "🧩", "shap": "📊"}
XAI_ORDER = ["gradcam", "lime", "shap"]


def get_xai_methods():
    """Method availability + tunable settings advertised by the backend."""
    try:
        r = requests.get(f"{API_URL}/xai/methods", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


@st.cache_data(show_spinner=False, max_entries=8)
def request_xai(image_bytes: bytes, scan_type: str, method: str, params_json: str) -> dict:
    """
    Run ONE explanation method (never all three) through the FastAPI backend.

    Cached per (image, scan type, method, settings) so re-rendering the page
    does not recompute an explanation.  Only JSON/base64 data is cached — the
    models live in the API process, so no model or GPU state is held here.
    """
    params = json.loads(params_json) if params_json else {}
    try:
        r = requests.post(
            f"{API_URL}/xai/{method}/{scan_type}",
            files={"file": ("scan.png", image_bytes, "image/png")},
            params=params,
            timeout=XAI_TIMEOUT.get(method, 300),
        )
    except Exception as exc:
        return {"error": f"XAI request failed: {exc}"}

    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return {"error": f"{method.upper()} explanation failed: {detail}"}
    return r.json()


def b64_to_image(data_url: str):
    """base64 (optionally data-URL) PNG -> PIL image."""
    if not data_url:
        return None
    payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def xai_settings_ui(method: str, tunable: dict, key_prefix: str) -> dict:
    """Settings widgets for the selected method (kept small and explicit)."""
    params = {}
    spec = (tunable or {}).get(method, {})

    if method == "gradcam":
        alpha = spec.get("alpha", {})
        params["alpha"] = st.slider("Overlay opacity", float(alpha.get("min", 0.1)),
                                    float(alpha.get("max", 0.9)),
                                    float(alpha.get("default", 0.45)), 0.05,
                                    key=f"{key_prefix}_alpha")
    elif method == "lime":
        samples = spec.get("num_samples", {})
        params["num_samples"] = st.slider(
            "LIME samples (model evaluations)", int(samples.get("min", 50)),
            int(samples.get("max", 1000)), int(samples.get("default", 200)),
            int(samples.get("step", 50)), key=f"{key_prefix}_lime_samples",
            help="Higher = smoother attribution but slower. The default is deliberately modest.")
        features = spec.get("num_features", {})
        params["num_features"] = st.slider(
            "Highlighted superpixels", int(features.get("min", 3)), int(features.get("max", 30)),
            int(features.get("default", 8)), key=f"{key_prefix}_lime_features")
        segmentation = spec.get("segmentation", {})
        params["segmentation"] = st.selectbox(
            "Segmentation", segmentation.get("options", ["quickshift", "slic", "grid"]),
            index=0, key=f"{key_prefix}_lime_seg")
        params["random_seed"] = st.number_input(
            "Random seed", min_value=0, max_value=100000, value=42, step=1,
            key=f"{key_prefix}_lime_seed")
    elif method == "shap":
        evals = spec.get("max_evals", {})
        params["max_evals"] = st.slider(
            "SHAP evaluations (max_evals)", int(evals.get("min", 32)), int(evals.get("max", 512)),
            int(evals.get("default", 128)), int(evals.get("step", 32)),
            key=f"{key_prefix}_shap_evals",
            help="Perturbation budget. Higher = finer attribution but slower.")
        mask = spec.get("mask_value", {})
        params["mask_value"] = st.selectbox(
            "Masked-region fill", mask.get("options", ["blur(16,16)", "inpaint_telea"]),
            index=0, key=f"{key_prefix}_shap_mask")
        reduce_spec = spec.get("reduce", {})
        params["reduce"] = st.selectbox(
            "Channel reduction", reduce_spec.get("options", ["mean", "sum", "max_abs"]),
            index=0, key=f"{key_prefix}_shap_reduce")
        prefer = spec.get("prefer", {})
        params["prefer"] = st.selectbox(
            "Explainer", prefer.get("options", ["partition", "gradient"]), index=0,
            key=f"{key_prefix}_shap_prefer",
            help="partition = PartitionExplainer (perturbation). gradient = GradientExplainer fallback.")
        params["random_seed"] = st.number_input(
            "Random seed", min_value=0, max_value=100000, value=42, step=1,
            key=f"{key_prefix}_shap_seed")

    return {key: value for key, value in params.items()
            if value is not None and (not isinstance(value, str) or value != "")}


def render_xai_section(image_bytes: bytes, scan_type: str, prediction: str,
                       confidence: float, class_names: list, key_prefix: str = "xai"):
    """
    Explainable AI panel: pick ONE method, run it on demand, show the result.

    Nothing is computed until the user presses the button, and LIME/SHAP results
    are cached per (image, method, settings) so the page can be re-rendered
    without repeating the work.
    """
    st.markdown("### 🔍 Explainable AI")
    st.caption("Explanations show which image regions contributed to the model's prediction. "
               "They describe model behaviour, not clinical causality, and are not evidence of "
               "where a disease is located.")

    info = get_xai_methods()
    methods = info.get("methods") or {}
    tunable = info.get("tunable") or {}
    if not methods:
        st.info("Explainable AI is unavailable: " + str(info.get("error", "backend did not report XAI support")))
        return

    available = [m for m in XAI_ORDER if methods.get(m, {}).get("available", True)]
    if not available:
        st.info("No XAI method is available in this build.")
        return

    labels = {m: f"{XAI_ICONS.get(m, '')} {methods.get(m, {}).get('label', m)}" for m in available}
    choice = st.radio("Explanation method", [labels[m] for m in available], horizontal=True,
                      key=f"{key_prefix}_method", label_visibility="collapsed")
    method = available[[labels[m] for m in available].index(choice)]
    st.caption(methods.get(method, {}).get("description", ""))

    target_options = ["Predicted class (default)"] + list(class_names or [])
    target_pick = st.selectbox("Class to explain", target_options, index=0,
                               key=f"{key_prefix}_target")

    with st.expander("Explanation settings", expanded=False):
        params = xai_settings_ui(method, tunable, key_prefix)
    if target_pick != target_options[0]:
        params["target_class"] = target_pick

    params_json = json.dumps(params, sort_keys=True)
    cache_key = f"{scan_type}|{method}|{params_json}|{hashlib.md5(image_bytes).hexdigest()[:10]}"

    if st.button(f"Generate {methods.get(method, {}).get('label', method)} explanation",
                 key=f"{key_prefix}_run_{method}"):
        with st.spinner(f"Running {methods.get(method, {}).get('label', method)}... "
                        f"({'this can take a while' if method in ('lime', 'shap') else 'fast'})"):
            payload = request_xai(image_bytes, scan_type, method, params_json)
        st.session_state[f"{key_prefix}_result"] = {"key": cache_key, "payload": payload}

    state = st.session_state.get(f"{key_prefix}_result")
    if not state or state.get("key") != cache_key:
        st.caption("Press the button above to generate an explanation for this image and method.")
        return

    payload = state.get("payload") or {}
    if payload.get("error"):
        st.error(payload["error"])
        return

    images = payload.get("images") or {}
    panel = b64_to_image(images.get("panel"))
    primary = b64_to_image(images.get(method_primary_key(method))) or b64_to_image(payload.get("explanation_image"))
    if panel is not None:
        st.image(panel, use_container_width=True)
    elif primary is not None:
        st.image(primary, use_container_width=True)

    st.markdown(
        f"**Method:** {payload.get('method_label', method)}  |  "
        f"**Prediction:** `{payload.get('predicted_class', prediction)}`  |  "
        f"**Confidence:** `{float(payload.get('confidence', confidence / 100.0)) * 100:.1f}%`  |  "
        f"**Explained class:** `{payload.get('target_class', '')}`")

    gallery = [k for k in ("original", "heatmap", "overlay", "overlay_signed",
                           "lime_positive", "lime_positive_negative") if images.get(k)]
    if gallery:
        columns = st.columns(min(3, len(gallery)))
        for index, key in enumerate(gallery):
            image = b64_to_image(images[key])
            if image is None:
                continue
            with columns[index % len(columns)]:
                st.image(image, caption=XAI_IMAGE_CAPTIONS.get(key, key), use_container_width=True)

    with st.expander("Explanation details (settings + metadata)"):
        st.json({"parameters": payload.get("parameters", {}),
                 "metadata": payload.get("metadata", {}),
                 "settings_fingerprint": payload.get("settings_fingerprint", ""),
                 "warnings": payload.get("warnings", [])})

    download = primary if primary is not None else panel
    if download is not None:
        buffer = io.BytesIO()
        download.save(buffer, format="PNG")
        st.download_button(
            "⬇️ Download explanation (PNG)", buffer.getvalue(),
            file_name=f"{scan_type}_{method}_explanation.png", mime="image/png",
            key=f"{key_prefix}_dl_{method}")


XAI_IMAGE_CAPTIONS = {
    "original": "Original scan (model input resolution)",
    "heatmap": "Attribution heatmap",
    "overlay": "Attribution overlay",
    "overlay_signed": "Signed contribution (warm = toward, cool = away from the explained class)",
    "lime_positive": "LIME regions that supported this prediction",
    "lime_positive_negative": "LIME regions that supported / opposed this prediction",
    "panel": "Full explanation figure",
}


def method_primary_key(method: str) -> str:
    """Which returned view should be shown as the main figure for a method."""
    return {"gradcam": "overlay", "lime": "lime_positive", "shap": "overlay_signed"}.get(method, "overlay")


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.5rem; font-weight: 800; 
                    background: linear-gradient(135deg, #00d4ff, #00ff88);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🏥 MedVision AI
        </div>
        <div style='color: #556677; font-size: 0.75rem; letter-spacing: 2px; 
                    text-transform: uppercase; margin-top: 0.3rem;'>
            Medical Intelligence
        </div>
    </div>
    <hr style='border-color: rgba(0,212,255,0.1);'>
    """, unsafe_allow_html=True)

    # API Status
    api_ok = check_api()
    status_color = "#00ff88" if api_ok else "#ff0040"
    status_text = "ONLINE" if api_ok else "OFFLINE"
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem;'>
        <div style='width:8px; height:8px; border-radius:50%; background:{status_color}; 
                    box-shadow: 0 0 6px {status_color};'></div>
        <span style='color:{status_color}; font-size:0.75rem; font-weight:600; 
                     letter-spacing:1px;'>API {status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    if api_ok:
        models = get_models()
        loaded = models.get("cnns", [])

        # Check DQN and LLM status
        _dqn_ok = bool(models.get('dqn'))
        _gemma_ok = bool(models.get('gemma4', models.get('groq')))
        _dqn_reason = "checkpoint missing"
        _gemma_reason = "GEMMA_API_KEY not set"
        try:
            _d = requests.get(f"{API_URL}/dqn/status", timeout=3).json()
            _dqn_ok = bool(_d.get("loaded", _d.get("available", _dqn_ok)))
            if _d.get("reason"):
                _dqn_reason = str(_d["reason"])
            elif _dqn_ok:
                _dqn_reason = f"ready ({int(_d.get('symptom_count', 0))} symptoms)"
        except Exception:
            pass
        try:
            _g = requests.get(f"{API_URL}/llm/status", timeout=3).json()
            _gemma_ok = bool(_g.get("loaded", _g.get("available", _gemma_ok)))
            if _g.get("reason"):
                _gemma_reason = str(_g["reason"])
            elif _gemma_ok:
                _gemma_reason = f"ready ({_g.get('model', 'gemma')})"
        except Exception:
            pass

        st.markdown("**Loaded Models:**")
        model_icons = {"chest": "🫁", "brain": "🧠", "skin": "🔬", "eye": "👁️"}
        verified_acc = {"chest": "92.95% test", "brain": "95.44% test",
                        "skin": "59.32% test", "eye": "83.33% test"}
        for m in ["chest", "brain", "skin", "eye"]:
            icon = model_icons[m]
            ok = m in loaded
            color = "#00ff88" if ok else "#334455"
            acc = verified_acc[m] if ok else "not loaded"
            st.markdown(f"<span style='color:{color};'>{'✅' if ok else '⬜'} {icon} {m.upper()} — {acc}</span>",
                       unsafe_allow_html=True)

        # DQN Triage Model
        dqn_color = "#00ff88" if _dqn_ok else "#ff0040"
        dqn_badge = "READY / LOADED (50 symptoms)" if _dqn_ok else "NOT LOADED"
        st.markdown(f"<span style='color:{dqn_color};'>{'✅' if _dqn_ok else '⬜'} 🤖 DQN TRIAGE — {dqn_badge}</span>",
                   unsafe_allow_html=True)

        # Gemma 4 LLM
        gemma_color = "#00ff88" if _gemma_ok else "#ff0040"
        gemma_badge = "READY / CONNECTED" if _gemma_ok else "NOT CONFIGURED"
        st.markdown(f"<span style='color:{gemma_color};'>{'✅' if _gemma_ok else '⬜'} 💬 GEMMA 4 LLM — {gemma_badge}</span>",
                   unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(0,212,255,0.1);'>", unsafe_allow_html=True)

        _dev = str(models.get('device', '') or '').upper()
        st.markdown(f"<span style='color:#8899aa; font-size:0.75rem;'>🖥️ Device: {_dev or 'unknown'}</span>",
                   unsafe_allow_html=True)
        try:
            _gpu = requests.get(f"{API_URL}/system/gpu", timeout=3).json()
            if _gpu.get("cuda_available"):
                _gpu_line = (f"CUDA: ✅ {_gpu.get('gpu_name', '')} "
                             f"({_gpu.get('vram_total_gb', '?')} GB VRAM, "
                             f"torch {_gpu.get('torch_version', '')}, "
                             f"cu{str(_gpu.get('cuda_build', ''))})")
            else:
                _gpu_line = "CUDA: ❌ CPU-only (torch has no usable GPU here)"
        except Exception:
            _gpu_line = "CUDA: unknown"
        st.markdown(f"<span style='color:#8899aa; font-size:0.75rem;'>{_gpu_line}</span>",
                   unsafe_allow_html=True)
        st.markdown(f"<span style='color:#8899aa; font-size:0.75rem;'>DQN: {'✅ ' + _dqn_reason if _dqn_ok else '❌ ' + _dqn_reason} | "
                    f"Gemma 4: {'✅ ' + _gemma_reason if _gemma_ok else '❌ ' + _gemma_reason}</span>",
                   unsafe_allow_html=True)
        if not _dqn_ok or not _gemma_ok:
            st.caption("CNN + XAI demo unaffected. Triage/reports need dqn_model.pth / GEMMA_API_KEY.")
    else:
        st.error("Start the API: `py -3.12 main.py`")

    st.markdown("<hr style='border-color: rgba(0,212,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#556677; font-size:0.75rem; line-height:1.6;'>
    ⚠️ <b style='color:#8899aa;'>Disclaimer</b><br>
    MedVision AI is an AI-assisted tool. All diagnoses must be confirmed by a licensed physician.
    </div>
    """, unsafe_allow_html=True)


# Scan types available for imaging — only offer models the backend actually loaded
AVAILABLE_SCAN_TYPES = ["chest", "brain", "skin", "eye"]
loaded_models = models.get("cnns", []) if api_ok else []
available_scans = [m for m in AVAILABLE_SCAN_TYPES if m in loaded_models] or list(AVAILABLE_SCAN_TYPES)
# ─────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">MedVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Multi-Modal Medical Intelligence System</div>', unsafe_allow_html=True)

if not api_ok:
    st.error("⚠️ API is offline. Please start the backend: `py -3.12 main.py`")
    st.stop()

# Mode selector
mode = st.radio(
    "Select Analysis Mode",
    ["🖼️ Medical Imaging", "🩺 Symptom Triage", "🔬 Combined Analysis"],
    horizontal=True
)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODE 1: MEDICAL IMAGING
# ─────────────────────────────────────────────
if mode == "🖼️ Medical Imaging":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Upload Medical Scan")
        
        scan_type = st.selectbox(
            "Scan Type",
            options=available_scans,
            format_func=lambda x: {
                "chest": "🫁 Chest X-Ray (Pneumonia/Normal)",
                "brain": "🧠 Brain MRI (Tumor Detection)",
                "skin":  "🔬 Skin Lesion (Cancer Detection)",
                "eye":   "👁️ Retinal Scan (Diabetic Retinopathy)"
            }.get(x, x.upper()),
        )

        uploaded = st.file_uploader(
            "Upload image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        generate_report = st.checkbox("Generate AI Doctor Report (Gemma 4)", value=True)

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption=f"{scan_type.upper()} Scan", use_container_width=True)

        if uploaded is not None and st.button("🔍 ANALYZE SCAN"):
            with st.spinner("Analyzing medical image..."):
                img_bytes = uploaded.getvalue()
                result = predict_image(img_bytes, scan_type, generate_report)

            # Keep the prediction across Streamlit reruns so the Explainable AI
            # buttons below can be used without losing the result.
            if result.get("error"):
                st.session_state.pop("imaging_result", None)
            else:
                st.session_state["imaging_result"] = {
                    "scan_type": scan_type,
                    "image": img_bytes,
                    "result": result,
                }

        cached_run = st.session_state.get("imaging_result")
        if cached_run and cached_run.get("scan_type") == scan_type and uploaded is not None:
            img_bytes = cached_run["image"]
            result = cached_run["result"]

            with col2:
                st.markdown("### Analysis Results")

                if "error" in result:
                    st.error(result["error"])
                else:
                    cnn = result.get("cnn_result", {})
                    emergency = result.get("emergency", False)

                    # Emergency banner
                    if emergency:
                        st.markdown("""
                        <div class="emergency-banner">
                            <p class="emergency-text">🚨 EMERGENCY DETECTED — CALL 108 NOW!</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Main prediction
                    conf = cnn.get("confidence", 0) * 100
                    pred = cnn.get("prediction", "Unknown")

                    st.markdown(f"""
                    <div class="result-card {'priority-1' if emergency else 'priority-3'}">
                        <div class="scan-badge">{scan_type.upper()} SCAN</div>
                        <div style='margin-top:1rem;'>
                            <div style='font-family: Syne, sans-serif; font-size: 1.8rem; 
                                        font-weight: 700; color: {"#ff0040" if emergency else "#00ff88"};'>
                                {pred}
                            </div>
                            <div style='color: #8899aa; font-size: 0.9rem; margin-top:0.3rem;'>
                                Confidence: <b style='color:#00d4ff;'>{conf:.1f}%</b>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # All scores
                    st.markdown("**Confidence Scores:**")
                    scores = cnn.get("all_scores", {})
                    for cls, score in sorted(scores.items(), key=lambda x: -x[1]):
                        pct = score * 100
                        color = "#ff0040" if cls == pred and emergency else "#00d4ff" if cls == pred else "#334455"
                        st.markdown(f"""
                        <div style='display:flex; align-items:center; gap:0.8rem; margin:0.3rem 0;'>
                            <div style='width:120px; color:#8899aa; font-size:0.85rem;'>{cls}</div>
                            <div style='flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:8px;'>
                                <div style='width:{pct}%; background:{color}; height:8px; border-radius:4px;'></div>
                            </div>
                            <div style='width:50px; text-align:right; color:{color}; font-size:0.85rem; font-weight:600;'>{pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Explainable AI panel is rendered full-width below (see below)

                    # LLM Report
                    if generate_report and result.get("report"):
                        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                        st.markdown("### 📋 AI Doctor Report")
                        st.markdown(f'<div class="report-box">{result["report"]}</div>', 
                                   unsafe_allow_html=True)

    # ── Explainable AI (full width, below the result columns) ──────────
    cached_run = st.session_state.get("imaging_result")
    if cached_run and cached_run.get("scan_type") == scan_type and uploaded is not None \
            and not cached_run["result"].get("error"):
        cnn_result = cached_run["result"].get("cnn_result", {}) or {}
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_xai_section(
            image_bytes=cached_run["image"],
            scan_type=scan_type,
            prediction=cnn_result.get("prediction", "Unknown"),
            confidence=float(cnn_result.get("confidence", 0)) * 100,
            class_names=list((cnn_result.get("all_scores") or {}).keys()),
            key_prefix=f"imaging_{scan_type}",
        )

    if not uploaded:
        with col2:
            st.markdown("""
            <div style='height:300px; display:flex; align-items:center; justify-content:center;
                        border: 2px dashed rgba(0,212,255,0.15); border-radius:16px;
                        color: #334455; font-size:1rem; text-align:center;'>
                Upload a medical image<br>to see analysis results
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODE 2: SYMPTOM TRIAGE
# ─────────────────────────────────────────────
elif mode == "🩺 Symptom Triage":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Enter Patient Symptoms")

        # Common symptoms for easy selection
        common_symptoms = [
            "fever", "cough", "chest pain", "shortness of breath",
            "headache", "fatigue", "nausea", "vomiting", "dizziness",
            "abdominal pain", "back pain", "joint pain", "skin rash",
            "blurred vision", "loss of appetite", "weight loss",
            "sweating", "chills", "sore throat", "runny nose"
        ]

        selected = st.multiselect(
            "Select symptoms (or type custom):",
            options=common_symptoms,
            placeholder="Choose symptoms..."
        )

        custom = st.text_input(
            "Additional symptoms (comma-separated):",
            placeholder="e.g. chest tightness, dry cough"
        )

        if custom:
            extra = [s.strip() for s in custom.split(",") if s.strip()]
            selected = list(set(selected + extra))

        generate_report = st.checkbox("Generate AI Doctor Report", value=True)

        if selected:
            st.markdown(f"**Selected:** {', '.join(selected)}")

        if selected and st.button("🔍 ANALYZE SYMPTOMS"):
            with st.spinner("Running DQN triage analysis..."):
                result = predict_symptoms(selected, generate_report)
                st.session_state["symptom_triage_result"] = {
                    "symptoms": sorted(list(selected)),
                    "result": result,
                }

        cached_sym = st.session_state.get("symptom_triage_result")
        has_current_cache = bool(cached_sym and cached_sym.get("symptoms") == sorted(list(selected)))

        if has_current_cache:
            result = cached_sym["result"]
            with col2:
                st.markdown("### Triage Results")

                dqn = result.get("dqn_result", {})
                emergency = result.get("emergency", False)

                if dqn.get("error"):
                    st.warning(f"⚠️ {dqn.get('error')} — Symptom triage is unavailable for this build.")
                    st.info("See the sidebar for loaded-model status. Medical Imaging mode works normally.")
                else:
                    priority = dqn.get("priority", 2)
                    p_color = priority_color(priority)
                    p_label = priority_label(priority)
                    p_exp = priority_explanation(priority)

                    # Emergency banner
                    if emergency:
                        st.markdown("""
                        <div class="emergency-banner">
                            <p class="emergency-text">🚨 EMERGENCY — CALL 108 IMMEDIATELY!</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Priority card with explanation
                    st.markdown(f"""
                    <div class="result-card priority-{priority}">
                        <div class="scan-badge">DQN SYMPTOM TRIAGE</div>
                        <div style='font-family: Syne, sans-serif; font-size: 1.8rem; 
                                    font-weight: 800; color: {p_color}; margin-top:0.5rem;'>
                            {p_label}
                        </div>
                        <div style='color:#00d4ff; font-weight:600; margin-top:0.4rem; font-size:0.92rem; line-height:1.4;'>
                            {p_exp}
                        </div>
                        <div style='margin-top:0.8rem; color:#ccd9e8; font-size:0.9rem;'>
                            <b>Matched symptoms:</b> {', '.join(dqn.get('matched_symptoms', [])) or 'None matched'}
                        </div>
                        {f"<div style='margin-top:0.3rem; color:#8899aa; font-size:0.82rem;'>Unmatched: {', '.join(dqn.get('unmatched_symptoms', []))}</div>" if dqn.get('unmatched_symptoms') else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    # Q-values breakdown
                    q_vals = dqn.get("q_values") or {}
                    if q_vals:
                        st.markdown("**Q-Values & Policy Confidence:**")
                        probs = dqn.get("policy_probabilities") or {}
                        for p_name, q_val in q_vals.items():
                            prob_pct = (probs.get(p_name, 0.0)) * 100
                            is_selected = (p_name == f"Priority {priority}")
                            bar_color = p_color if is_selected else "#334455"
                            st.markdown(f"""
                            <div style='display:flex; align-items:center; gap:0.8rem; margin:0.3rem 0;'>
                                <div style='width:90px; color:#8899aa; font-size:0.82rem;'>{p_name}</div>
                                <div style='flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:8px;'>
                                    <div style='width:{max(5, min(100, prob_pct))}%; background:{bar_color}; height:8px; border-radius:4px;'></div>
                                </div>
                                <div style='width:65px; text-align:right; color:{bar_color}; font-size:0.82rem; font-weight:600;'>Q={q_val:+.2f}</div>
                                <div style='width:50px; text-align:right; color:#8899aa; font-size:0.80rem;'>({prob_pct:.1f}%)</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Disclaimer
                    st.markdown("""
                    <div style='margin-top:1rem; border:1px solid rgba(0,212,255,0.2); background:rgba(0,212,255,0.04); 
                                border-radius:8px; padding:0.6rem 0.8rem; color:#88aacc; font-size:0.78rem;'>
                        ℹ️ <b>DQN Architecture:</b> Deep Q-Network trained on medical symptom taxonomy (symptoms2.csv). Discrete Q-action optimization for clinical triage prioritization.
                    </div>
                    """, unsafe_allow_html=True)

                    # LLM Report
                    if generate_report and result.get("report"):
                        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                        st.markdown("### 📋 AI Doctor Report")
                        st.markdown(f'<div class="report-box">{result["report"]}</div>',
                                   unsafe_allow_html=True)
        elif selected:
            with col2:
                st.markdown("""
                <div style='height:300px; display:flex; align-items:center; justify-content:center;
                            border: 2px dashed rgba(0,212,255,0.15); border-radius:16px;
                            color: #8899aa; font-size:1rem; text-align:center;'>
                    Click <b>🔍 ANALYZE SYMPTOMS</b><br>to run DQN triage
                </div>
                """, unsafe_allow_html=True)
        else:
            with col2:
                st.markdown("""
                <div style='height:300px; display:flex; align-items:center; justify-content:center;
                            border: 2px dashed rgba(0,212,255,0.15); border-radius:16px;
                            color: #334455; font-size:1rem; text-align:center;'>
                    Select symptoms<br>to see triage results
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODE 3: COMBINED — primary full-demo mode (one button → full pipeline)
# ─────────────────────────────────────────────
elif mode == "🔬 Combined Analysis":
    st.markdown("### 🔬 Combined Medical Analysis")
    st.caption("One button runs the complete available pipeline on the SAME uploaded image: "
               "CNN → Grad-CAM → LIME → SHAP → DQN status → Gemma 4 report → emergency check.")

    SCAN_LABELS = {"chest": "🫁 Chest X-Ray", "brain": "🧠 Brain MRI",
                   "skin": "🔬 Skin Lesion", "eye": "👁️ Retinal Scan"}
    # Verified test accuracies (cnn/*meta*.json) — displayed only, never edited here.
    VERIFIED_ACC = {"chest": "92.95%", "brain": "95.44%", "skin": "59.32%", "eye": "83.33%"}
    XAI_BLURB = {
        "gradcam": "Shows regions receiving strong class-discriminative activation.",
        "lime": "Shows local superpixel regions that influenced the prediction.",
        "shap": "Shows positive/negative regional contribution to the model output.",
    }

    # ── SECTION 1 — INPUT ─────────────────────────────────────────────
    in1, in2, in3 = st.columns([1.2, 1.4, 1], gap="large")

    with in1:
        st.markdown("**1. Upload Medical Image**")
        scan_type = st.selectbox("Modality", available_scans,
                                 format_func=lambda x: SCAN_LABELS.get(x, x.upper()))
        uploaded = st.file_uploader("Image (JPG / PNG)", type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
        if uploaded:
            st.image(Image.open(uploaded),
                     caption=f"{SCAN_LABELS.get(scan_type, scan_type)} — uploaded scan",
                     use_container_width=True)

    with in2:
        st.markdown("**2. Add Symptoms**")
        common_symptoms = ["fever", "cough", "chest pain", "shortness of breath", "headache",
                           "fatigue", "nausea", "dizziness", "abdominal pain", "blurred vision"]
        selected = st.multiselect("Predefined symptoms", common_symptoms, placeholder="Select...")
        custom = st.text_input("Custom symptoms (comma-separated)",
                               placeholder="e.g. chest tightness, dry cough")
        if custom:
            selected = list(dict.fromkeys(selected + [s.strip() for s in custom.split(",") if s.strip()]))
        if selected:
            st.markdown(f"**Selected:** {', '.join(selected)}")

    with in3:
        st.markdown("**3. Run Analysis**")
        generate_report = st.checkbox("AI Doctor Report (Gemma 4)", value=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("🔬 RUN FULL ANALYSIS")

    if run_clicked and uploaded is None:
        st.warning("Upload a medical image first, then press RUN FULL ANALYSIS.")

    # ── ONE-BUTTON PIPELINE (single run; kept across Streamlit re-renders) ──
    if run_clicked and uploaded is not None:
        img_bytes = uploaded.getvalue()
        display_key = f"{scan_type}|{hashlib.md5(img_bytes).hexdigest()[:16]}"
        run = {"key": display_key, "scan_type": scan_type, "image": img_bytes,
               "symptoms": list(selected), "generate_report": bool(generate_report),
               "status": "RUNNING", "report_status": "NOT_STARTED",
               "errors": {}, "xai": {}}
        # Persist the run identity before any slow work starts.  A regular
        # Streamlit rerun can therefore render this same run, but only a fresh
        # button click can start a new analysis or a new Gemma request.
        st.session_state["combined_result"] = run

        progress = st.progress(0.0)
        status_box = st.empty()

        def _step(frac, text):
            progress.progress(float(frac))
            status_box.markdown(f"**⏳ {text}**")

        # 1) CNN prediction — these exact bytes are reused for every XAI method below
        _step(0.05, "Running CNN prediction...")
        try:
            img_result = predict_image(img_bytes, scan_type, False)
            if img_result.get("error"):
                raise RuntimeError(img_result["error"])
        except Exception as exc:
            img_result = {"error": str(exc)}
            run["errors"]["cnn"] = str(exc)
        run["img_result"] = img_result

        # 2–4) Grad-CAM / LIME / SHAP — same image bytes, validated backend defaults
        xai_steps = [("gradcam", 0.20, "Generating Grad-CAM..."),
                     ("lime", 0.45, "Generating LIME explanation..."),
                     ("shap", 0.70, "Generating SHAP explanation...")]
        for method, frac, label in xai_steps:
            _step(frac, label)
            if run["errors"].get("cnn"):
                run["xai"][method] = {"error": "skipped — CNN prediction failed"}
                continue
            payload = request_xai(img_bytes, scan_type, method, "{}")
            run["xai"][method] = payload
            if payload.get("error"):
                run["errors"][method] = payload["error"]

        # 5) DQN triage — real state only (checkpoint genuinely missing in this build)
        _step(0.80, "Checking symptom triage...")
        try:
            dqn_status = requests.get(f"{API_URL}/dqn/status", timeout=5).json()
        except Exception as exc:
            dqn_status = {"loaded": False, "reason": f"DQN status unreachable: {exc}"}
        sym_result = None
        if selected:
            try:
                sym_result = predict_symptoms(list(selected), False)
            except Exception as exc:
                run["errors"]["dqn"] = str(exc)
        run["dqn_status"] = dqn_status
        run["sym_result"] = sym_result

        # 6) Gemma 4 report — through the existing combined backend path (real Gemini API)
        run["report"], run["report_error"] = None, None
        if generate_report:
            run["report_status"] = "RUNNING"
            _step(0.90, "Generating Gemma 4 report...")
            try:
                r = requests.post(
                    f"{API_URL}/predict/combined/{scan_type}",
                    files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                    params={"symptoms": ",".join(selected), "generate_report": "true"},
                    timeout=240)
                if r.status_code == 200:
                    combined_payload = r.json()
                    run["combined"] = combined_payload
                    run["report"] = combined_payload.get("report")
                    if not run["report"]:
                        run["report_error"] = "backend returned no report text"
                else:
                    try:
                        run["report_error"] = str(r.json().get("detail", r.text))
                    except Exception:
                        run["report_error"] = str(r.text[:300])
            except Exception as exc:
                run["report_error"] = f"{type(exc).__name__}: {exc}"
            report_text = str(run.get("report") or "")
            run["report_status"] = (
                "SUCCESS" if report_text and not report_text.startswith(
                    ("Gemma 4 API not configured", "Report generation failed", "Gemma report unavailable")
                ) else "ERROR"
            )

        # 7) Emergency assessment — existing rule set only, nothing invented
        emergency = bool(img_result.get("emergency"))
        if sym_result:
            emergency = emergency or bool(sym_result.get("emergency"))
        if run.get("combined"):
            emergency = emergency or bool(run["combined"].get("emergency"))
        run["emergency"] = emergency

        _step(0.97, "Building final analysis...")
        progress.progress(1.0)
        status_box.markdown("**✅ Analysis complete — results below.**")
        run["status"] = "SUCCESS"
        st.session_state["combined_result"] = run

    # ── RENDER — only for the CURRENT image (a new upload clears old results) ──
    cur_key = f"{scan_type}|{hashlib.md5(uploaded.getvalue()).hexdigest()[:16]}" if uploaded else None
    run = st.session_state.get("combined_result")
    if run and (cur_key is None or run.get("key") != cur_key):
        st.session_state.pop("combined_result", None)   # never show a stale image's analysis
        run = None
    if uploaded and run is None:
        st.info("Press **🔬 RUN FULL ANALYSIS** to run the full pipeline on this image.")

    if run:
        cnn = (run.get("img_result") or {}).get("cnn_result") or {}
        pred = cnn.get("prediction", "Unknown")
        conf = float(cnn.get("confidence", 0)) * 100
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── SECTION 2 — CNN PREDICTION & SCAN ─────────────────────────
        st.markdown("## 🧠 Medical Scan & CNN Prediction")
        if run["errors"].get("cnn"):
            st.error(f"CNN prediction failed: {run['errors']['cnn']}")
        else:
            col_img, col_pred = st.columns([1, 1.3], gap="large")
            with col_img:
                st.image(Image.open(io.BytesIO(run["image"])),
                         caption=f"{SCAN_LABELS.get(scan_type, scan_type)} — Analyzed Input Scan",
                         use_container_width=True)
            with col_pred:
                acc_map = models.get("cnn_test_accuracy") or {}
                _acc = acc_map.get(scan_type)
                test_acc_txt = (f"{_acc * 100:.2f}%" if isinstance(_acc, (int, float))
                                else VERIFIED_ACC.get(scan_type, "n/a"))
                _border = "priority-1" if run.get("emergency") else "priority-3"
                st.markdown(f"""
                <div class="result-card {_border}">
                    <div class="scan-badge">CNN PREDICTION</div>
                    <div style='font-family:Syne,sans-serif; font-size:1.8rem; font-weight:700;
                                color:{"#ff0040" if run.get("emergency") else "#00ff88"}; margin-top:0.6rem;'>{pred}</div>
                    <div style='color:#8899aa; margin-top:0.4rem;'>
                        Confidence: <b style='color:#00d4ff;'>{conf:.1f}%</b> &nbsp;·&nbsp;
                        Modality: <b style='color:#ccd9e8;'>{SCAN_LABELS.get(scan_type, scan_type)}</b> &nbsp;·&nbsp;
                        Model: <b style='color:#ccd9e8;'>ResNet50</b>
                    </div>
                    <div style='color:#556677; font-size:0.78rem; margin-top:0.5rem;'>
                        Verified test accuracy: {test_acc_txt}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                scores = cnn.get("all_scores") or {}
                if scores:
                    st.markdown("**Confidence scores:**")
                    for cls, score in sorted(scores.items(), key=lambda x: -x[1]):
                        pct = score * 100
                        color = "#00ff88" if cls == pred else "#334455"
                        st.markdown(f"""
                        <div style='display:flex; align-items:center; gap:0.8rem; margin:0.25rem 0;'>
                            <div style='width:150px; color:#8899aa; font-size:0.82rem;'>{cls}</div>
                            <div style='flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:8px;'>
                                <div style='width:{pct}%; background:{color}; height:8px; border-radius:4px;'></div>
                            </div>
                            <div style='width:55px; text-align:right; color:{color}; font-size:0.82rem; font-weight:600;'>{pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── SECTION 3 — EXPLAINABLE AI (same image, same predicted class) ──
        st.markdown("## 🔍 Explainable AI")
        st.caption("All three methods run on the exact same uploaded image used for the prediction "
                   "above and explain the predicted class.")
        xai_labels = (get_xai_methods().get("methods") or {})
        xcols = st.columns(3)
        for _i, method in enumerate(XAI_ORDER):
            _label = xai_labels.get(method, {}).get("label", method.upper())
            with xcols[_i]:
                st.markdown(f"**{XAI_ICONS.get(method, '')} {_label}**")
                payload = (run.get("xai") or {}).get(method) or {}
                if payload.get("error"):
                    st.error(f"{_label} unavailable: {payload['error']}")
                    continue
                images = payload.get("images") or {}
                figure = (b64_to_image(images.get(method_primary_key(method)))
                          or b64_to_image(images.get("panel"))
                          or b64_to_image(payload.get("explanation_image")))
                if figure is None:
                    st.error(f"{_label}: no visualization returned by the backend")
                else:
                    st.image(figure, use_container_width=True)
                _explained = payload.get("target_class") or payload.get("predicted_class") or pred
                st.caption(f"Explained class: `{_explained}`")
                if payload.get("predicted_class") and payload["predicted_class"] != pred:
                    st.warning("Explained class differs from the displayed prediction — "
                               "press RUN FULL ANALYSIS again.")
                st.caption(XAI_BLURB.get(method, ""))

        # ── SECTION 4 — XAI INTERPRETATION + DISCLAIMER ───────────────
        st.markdown("#### How to interpret these explanations")
        st.markdown(
            "- **Grad-CAM:** Shows regions receiving strong class-discriminative activation.\n"
            "- **LIME:** Shows local superpixel regions that influenced the prediction.\n"
            "- **SHAP:** Shows positive/negative regional contribution to the model output.")
        st.markdown("""
        <div style='border:1px solid rgba(255,136,0,0.45); background:rgba(255,136,0,0.07);
                    border-radius:10px; padding:0.9rem 1.1rem; color:#ffd9b3; font-size:0.92rem;'>
        ⚠️ <b>These explanations describe model behaviour, not clinical causality, and are not
        evidence of where disease is located.</b>
        </div>
        """, unsafe_allow_html=True)

        # ── SECTION 5 — DQN TRIAGE ────────────────────────────────────
        st.markdown("## 🤖 DQN Symptom Triage")
        dqn_status = run.get("dqn_status") or {}
        sym_result = run.get("sym_result")
        dqn_result = (sym_result or {}).get("dqn_result") or None
        if dqn_result and "priority" in dqn_result:
            _p = dqn_result["priority"]
            _qvals = dqn_result.get("q_values") or {}
            _probs = dqn_result.get("policy_probabilities") or {}
            _exp = priority_explanation(_p)
            st.markdown(f"""
            <div class="result-card priority-{_p}">
                <div class="scan-badge">DQN SYMPTOM TRIAGE</div>
                <div style='font-family:Syne,sans-serif; font-size:1.6rem; font-weight:700;
                            color:{priority_color(_p)}; margin-top:0.5rem;'>{priority_label(_p)}</div>
                <div style='color:#00d4ff; font-weight:600; margin-top:0.3rem; font-size:0.92rem; line-height:1.4;'>
                    {_exp}
                </div>
                <div style='color:#ccd9e8; margin-top:0.6rem; font-size:0.9rem;'>
                    <b>Matched symptoms:</b> {', '.join(dqn_result.get('matched_symptoms', [])) or 'None matched'}
                </div>
                {f"<div style='margin-top:0.3rem; color:#8899aa; font-size:0.82rem;'>Unmatched: {', '.join(dqn_result.get('unmatched_symptoms', []))}</div>" if dqn_result.get('unmatched_symptoms') else ""}
            </div>""", unsafe_allow_html=True)

            if _qvals:
                st.markdown("**Q-Values & Policy Confidence:**")
                for p_name, q_val in _qvals.items():
                    prob_pct = (_probs.get(p_name, 0.0)) * 100
                    is_selected = (p_name == f"Priority {_p}")
                    bar_color = priority_color(_p) if is_selected else "#334455"
                    st.markdown(f"""
                    <div style='display:flex; align-items:center; gap:0.8rem; margin:0.25rem 0;'>
                        <div style='width:90px; color:#8899aa; font-size:0.82rem;'>{p_name}</div>
                        <div style='flex:1; background:rgba(255,255,255,0.05); border-radius:4px; height:8px;'>
                            <div style='width:{max(5, min(100, prob_pct))}%; background:{bar_color}; height:8px; border-radius:4px;'></div>
                        </div>
                        <div style='width:65px; text-align:right; color:{bar_color}; font-size:0.82rem; font-weight:600;'>Q={q_val:+.2f}</div>
                        <div style='width:50px; text-align:right; color:#8899aa; font-size:0.80rem;'>({prob_pct:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("""
            <div style='margin-top:0.6rem; border:1px solid rgba(0,212,255,0.2); background:rgba(0,212,255,0.04); 
                        border-radius:8px; padding:0.5rem 0.8rem; color:#88aacc; font-size:0.78rem;'>
                ℹ️ <b>DQN Architecture:</b> Deep Q-Network trained on medical symptom taxonomy (symptoms2.csv). Discrete Q-action optimization for clinical triage prioritization.
            </div>
            """, unsafe_allow_html=True)
        elif dqn_status.get("loaded"):
            st.info("DQN checkpoint is loaded (50 symptoms ready) but no symptoms were entered for this run. "
                    "Add symptoms and press RUN FULL ANALYSIS to see triage.")
        else:
            _reason = dqn_status.get("reason") or "dqn_model.pth not provided"
            _expected = dqn_status.get("checkpoint_expected", "dqn_model.pth (project root)")
            st.markdown(f"""
            <div class="result-card">
                <div class="scan-badge">DQN SYMPTOM TRIAGE</div>
                <div style='color:#ffcc80; margin-top:0.6rem; font-weight:600;'>
                    Status: Not configured
                </div>
                <div style='color:#8899aa; margin-top:0.4rem;'>Reason: {_reason}</div>
                <div style='color:#8899aa; margin-top:0.5rem; font-size:0.85rem; line-height:1.6;'>
                    Expected checkpoint: <code style='color:#00d4ff;'>{_expected}</code><br>
                    Symptom triage cannot be calculated until the trained DQN checkpoint is provided.
                </div>
            </div>""", unsafe_allow_html=True)

        # ── SECTION 6 — GEMMA 4 REPORT (real Gemini API response) ─────
        st.markdown("## 🩺 AI Doctor Report — Gemma 4")
        _report = run.get("report")
        _report_error = run.get("report_error")
        if not run.get("generate_report"):
            st.info("Report generation was disabled for this run (checkbox above).")
        elif _report and not str(_report).startswith(
                ("Gemma 4 API not configured", "Report generation failed", "Gemma report unavailable")):
            st.markdown(f'<div class="report-box">{_report}</div>', unsafe_allow_html=True)
            st.caption("AI-generated research/educational report — not a medical diagnosis.")
        else:
            _detail = _report if _report else (_report_error or "unknown error")
            st.warning(f"Gemma 4 report unavailable — {_detail}")

        # ── SECTION 7 — EMERGENCY ASSESSMENT ──────────────────────────
        st.markdown("## 🚨 Emergency Assessment")
        if run.get("emergency"):
            st.markdown("""
            <div class="emergency-banner">
                <p class="emergency-text">🚨 EMERGENCY FLAG — CALL 108!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='border:1px solid rgba(0,255,136,0.35); background:rgba(0,255,136,0.06);
                        border-radius:10px; padding:0.9rem 1.1rem; color:#b8ffd9; font-size:0.95rem;'>
                ✅ Emergency assessment: No emergency flag detected by the configured rule set.
            </div>
            """, unsafe_allow_html=True)
        st.caption("Automated rule check over the CNN prediction and (when available) DQN triage "
                   "— not a medical diagnosis.")

        # ── SECTION 8 — FINAL UNIFIED SUMMARY ─────────────────────────
        st.markdown("## 📋 Final Analysis Summary")

        def _xai_cell(method):
            payload = (run.get("xai") or {}).get(method) or {}
            if payload.get("error"):
                return (f"<span style='color:#ff8800;'>⚠ unavailable — "
                        f"{str(payload['error'])[:90]}</span>")
            _cls = payload.get("target_class") or payload.get("predicted_class") or pred
            return (f"<span style='color:#00ff88;'>✅ generated</span> "
                    f"<span style='color:#8899aa;'>(explained class: {_cls})</span>")

        if run.get("generate_report"):
            if _report and not str(_report).startswith(
                    ("Gemma 4 API not configured", "Report generation failed", "Gemma report unavailable")):
                _gemma_cell = "<span style='color:#00ff88;'>✅ Report generated</span>"
            else:
                _gemma_cell = (f"<span style='color:#ff8800;'>⚠ unavailable — "
                               f"{str(_report or _report_error)[:90]}</span>")
        else:
            _gemma_cell = "<span style='color:#8899aa;'>➖ disabled for this run</span>"

        if "priority" in (dqn_result or {}):
            _p = dqn_result["priority"]
            _probs = dqn_result.get("policy_probabilities") or {}
            _pct = _probs.get(f"Priority {_p}", 0.0) * 100
            _dqn_cell = (f"<span style='color:{priority_color(_p)}; font-weight:600;'>✅ "
                         f"{priority_label(_p)}</span> "
                         f"<span style='color:#8899aa;'>({_pct:.1f}% confidence)</span>")
        elif dqn_status.get("loaded"):
            _dqn_cell = "<span style='color:#00ff88;'>✅ Ready (no symptoms entered for this run)</span>"
        else:
            _dqn_cell = "<span style='color:#ff8800;'>⚠ Not configured</span>"

        _emg_cell = ("<span style='color:#ff0040;'>🚨 Emergency flag detected</span>"
                     if run.get("emergency") else
                     "<span style='color:#00ff88;'>No emergency flag detected by the "
                     "configured rule set</span>")

        summary_rows = [
            ("Modality", SCAN_LABELS.get(scan_type, scan_type)),
            ("Analyzed Image", "<span style='color:#00ff88;'>✅ Image uploaded & verified</span>"),
            ("CNN Prediction", f"<b style='color:#00d4ff;'>{pred}</b>"),
            ("Confidence", f"{conf:.1f}%"),
            ("Patient Symptoms", ", ".join(run.get("symptoms") or []) or "None entered"),
            ("Grad-CAM", _xai_cell("gradcam")),
            ("LIME", _xai_cell("lime")),
            ("SHAP", _xai_cell("shap")),
            ("DQN Triage", _dqn_cell),
            ("Gemma 4 Report", _gemma_cell),
            ("Emergency Assessment", _emg_cell),
        ]
        st.markdown("<div style='margin-top:0.5rem;'>" + "".join(
            f"<div style='display:flex; gap:1rem; padding:0.5rem 0.2rem; "
            f"border-bottom:1px solid rgba(255,255,255,0.06);'>"
            f"<div style='width:210px; color:#8899aa; font-size:0.78rem; "
            f"text-transform:uppercase; letter-spacing:1px; padding-top:2px;'>{k}</div>"
            f"<div style='color:#ccd9e8; font-size:0.95rem;'>{v}</div></div>"
            for k, v in summary_rows) + "</div>", unsafe_allow_html=True)
