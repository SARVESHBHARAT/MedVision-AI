"""
MedVision AI — Unified Inference Engine
Loads all 4 CNNs + DQN and exposes a single predict() interface
Author: Sarvesh Bharat
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:   # optional LLM SDK — the basic CNN demo must work even without it
    genai = None
    genai_types = None
from dotenv import load_dotenv
import os
import time

load_dotenv()

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CNN_DIR  = BASE_DIR / "cnn"

MODEL_PATHS = {
    "chest": CNN_DIR / "chest_xray_resnet50.pth",
    "brain": CNN_DIR / "brain_resnet50_v2.pth",
    "skin":  CNN_DIR / "skin_resnet50.pth",
    # ACTIVE eye model: five-class Eye V3 (eye_meta_v3.json). Do NOT point this
    # back at the V1 binary/V2 files or at V4/V5.
    "eye":   CNN_DIR / "eye_resnet50_v3.pth",
}
META_PATHS = {
    "chest": CNN_DIR / "chest_xray_meta.json",
    "brain": CNN_DIR / "brain_meta_v2.json",
    "skin":  CNN_DIR / "skin_meta.json",
    "eye":   CNN_DIR / "eye_meta_v3.json",
}
DQN_PATH      = BASE_DIR / "dqn_model.pth"
DISEASE_PATH  = BASE_DIR / "dia_3.csv"
SYMPTOMS_PATH = BASE_DIR / "symptoms2.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MedVision Inference] Device: {DEVICE}")

# ─────────────────────────────────────────────
#  IMAGE TRANSFORMS
# ─────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ACTIVE Eye V3 is trained on 384x384 fundus-cache inputs (eye_meta_v3.json,
# img_size=384). The live prediction path and the XAI layer MUST use this
# transform for eye so explanations match the displayed prediction. All other
# modalities stay on the shared 224px transform.
eye_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

MODALITY_TRANSFORMS = {
    "eye": eye_transform,
}

# ─────────────────────────────────────────────
#  BUILD RESNET50 ARCHITECTURE
# ─────────────────────────────────────────────
def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


# ─────────────────────────────────────────────
#  CNN LOADER
# ─────────────────────────────────────────────
class CNNModel:
    def __init__(self, name: str):
        self.name = name
        meta_path  = META_PATHS[name]
        model_path = MODEL_PATHS[name]

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train {name} CNN first!")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.class_names = self.meta["class_names"]
        self.num_classes = self.meta["num_classes"]

        self.model = build_resnet50(self.num_classes)
        ckpt = torch.load(model_path, map_location=DEVICE)
        # Accept both checkpoint formats: a dict wrapper ({"model_state_dict": ...})
        # or a raw state_dict (skin_trainer.py saves the raw state_dict; the others save the wrapper}.
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()

        # ── XAI plumbing (read-only: no weight, no architecture change) ──
        # The explainability layer re-uses these exact objects, which guarantees
        # that Grad-CAM / LIME / SHAP explain the same input the prediction saw.
        # Eye V3 was trained at 384px (see eye_meta_v3.json img_size=384);
        # all other active models use the shared 224px transform.
        self.transform  = MODALITY_TRANSFORMS.get(name, img_transform)
        self.model_path = model_path
        self.meta_path  = meta_path
        self.device     = DEVICE

        print(f"  [OK] {name.upper()} CNN loaded | Classes: {self.class_names}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Returns top prediction + confidence scores for all classes."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        # NOTE: self.transform is modality-specific (Eye V3 = 384px, others = 224px).
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx   = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        top_conf  = float(probs[top_idx])

        all_scores = {cls: round(float(p), 4) for cls, p in zip(self.class_names, probs)}

        return {
            "model":      self.name,
            "prediction": top_class,
            "confidence": round(top_conf, 4),
            "all_scores": all_scores,
        }


# ─────────────────────────────────────────────
#  DQN TRIAGE MODEL
# ─────────────────────────────────────────────
class DQNTriage:
    PRIORITY_MAP = {0: 1, 1: 2, 2: 3}  # 3 actions → priority 1/2/3

    def __init__(self):
        try:
            from dqn import DQNTriageInference
            self.inference = DQNTriageInference(
                model_path=DQN_PATH,
                meta_path=BASE_DIR / "dqn_meta.json",
                device=DEVICE,
            )
            self.model = self.inference.model
            self.symptoms = self.inference.symptoms
            self.num_symptoms = self.inference.num_symptoms
        except Exception as exc:
            print(f"  [WARN] Failed to initialize DQN module: {exc}")
            self.inference = None
            self.model = None
            self.symptoms = []
            self.num_symptoms = 0

    @torch.no_grad()
    def predict(self, symptom_list: list[str]) -> dict:
        """symptom_list: list of symptom strings the patient has."""
        if self.inference is None or not self.inference.loaded:
            return {
                "error": "DQN model not loaded",
                "loaded": False,
                "reason": "No verified DQN checkpoint available.",
                "disclaimer": "Research prototype — not clinically validated.",
            }
        return self.inference.predict(symptom_list)

    def status(self) -> dict:
        """Returns truthful status of DQN component."""
        if self.inference is not None:
            return self.inference.status()
        return {
            "loaded": False,
            "available": False,
            "checkpoint_expected": str(DQN_PATH),
            "checkpoint_found": bool(DQN_PATH.exists()),
            "symptom_count": 0,
            "reason": "No verified DQN checkpoint available.",
            "disclaimer": "Research prototype — not clinically validated.",
        }


# ─────────────────────────────────────────────
#  GEMMA 4 LLM REPORT (remote Gemini API — no local weights)
# ─────────────────────────────────────────────
#: Model used for doctor reports. Overridable via GEMMA_MODEL env var.
#: (verified live 2026-10-05 via ListModels + generate_content: valid IDs include
#: "gemma-4-31b-it" and "gemma-4-26b-a4b-it"; "gemma-3-27b-it" returns 404.)
GEMMA_MODEL_NAME = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
# A report is one user-initiated operation.  Keep its transport bounded at the
# SDK layer so a timed-out request is not left running in a detached thread.
GEMMA_REQUEST_TIMEOUT_MS = 60_000


class GroqReporter:  # keeping same class name so rest of code doesn't break
    def __init__(self):
        # Accept GEMMA_API_KEY (documented name) with GEMINI_API_KEY as alias.
        api_key = os.getenv("GEMMA_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMMA_MODEL", GEMMA_MODEL_NAME)
        if genai is None:
            print("  [WARN] google-genai SDK not available - LLM reports disabled")
            self.client = None
        elif not api_key:
            print("  [WARN] GEMMA_API_KEY not found in .env - LLM reports disabled")
            self.client = None
        else:
            # Disable SDK retries here.  The Streamlit click maps to exactly
            # one Gemma request; retrying in the client after a timeout made
            # request counts and completion state ambiguous.
            self.client = genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(
                    timeout=GEMMA_REQUEST_TIMEOUT_MS,
                    retry_options=genai_types.HttpRetryOptions(attempts=1),
                ),
            )
            print(f"  [OK] Gemma 4 LLM ready ({self.model_name})")

    def generate_report(self, cnn_result: dict = None,
                        dqn_result: dict = None,
                        patient_info: dict = None) -> str:
        if self.client is None:
            return "Gemma 4 API not configured. Add GEMMA_API_KEY to .env"

        # Build context string
        context_parts = []

        if cnn_result:
            context_parts.append(
                f"Medical imaging analysis ({cnn_result['model'].upper()} scan): "
                f"Detected '{cnn_result['prediction']}' with {cnn_result['confidence']*100:.1f}% confidence. "
                f"All findings: {cnn_result['all_scores']}"
            )

        if dqn_result and "priority" in dqn_result:
            context_parts.append(
                f"Symptom triage: Patient reported symptoms {dqn_result.get('matched_symptoms', [])}. "
                f"AI triage assigned {dqn_result['priority_label']}. "
                f"Emergency: {dqn_result['emergency']}"
            )

        if patient_info:
            context_parts.append(f"Patient info: {patient_info}")

        context = "\n".join(context_parts)

        prompt = f"""You are MedVision AI, an advanced medical assistant powered by Gemma 4. Based on the following AI analysis results, provide a clear, professional medical report for the attending physician.

{context}

Please provide:
1. Summary of findings
2. Possible diagnosis or differential diagnosis
3. Recommended next steps
4. Urgency level and whether emergency care is needed

Keep the report concise, professional, and actionable. Always remind that this is AI-assisted analysis and final diagnosis must be confirmed by a licensed physician.
Do NOT use markdown formatting, asterisks, bold, or bullet points. Write in plain text only."""

        print("[Gemma] request started")
        print(f"[Gemma] model = {self.model_name}")

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            print(f"[Gemma] request sent (attempt {attempt} of {max_attempts})")
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Model returned an empty response")
                report = text.strip()
                print("[Gemma] request finished | status=200 OK")
                print(f"[Gemma] response length = {len(report)}")
                return report
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                is_transient = any(code in message for code in ["500", "503", "504", "429", "INTERNAL", "ResourceExhausted", "DeadlineExceeded", "DEADLINE_EXCEEDED", "timeout", "timed out"])
                if attempt < max_attempts and is_transient:
                    print(f"[Gemma] Transient error encountered ({type(exc).__name__}). Retrying in 1.5s...")
                    time.sleep(1.5)
                    continue
                return f"Gemma report unavailable: {message}"


# ─────────────────────────────────────────────
#  EMERGENCY DETECTION
# ─────────────────────────────────────────────
EMERGENCY_CONDITIONS = {
    "chest": ["PNEUMONIA"],
    "brain": ["glioma", "meningioma", "pituitary"],
    "skin":  ["Melanoma", "Basal Cell Carcinoma", "melanoma", "basal cell carcinoma"],
    "eye":   ["Proliferate_DR", "Severe"],
}

def check_emergency(cnn_result: dict, dqn_result: dict = None) -> bool:
    model  = cnn_result.get("model", "")
    pred   = cnn_result.get("prediction", "")
    conf   = cnn_result.get("confidence", 0)

    cnn_emergency = (
        model in EMERGENCY_CONDITIONS and
        pred in EMERGENCY_CONDITIONS[model] and
        conf > 0.7
    )
    dqn_emergency = dqn_result.get("emergency", False) if dqn_result else False
    return cnn_emergency or dqn_emergency


# ─────────────────────────────────────────────
#  MEDVISION ENGINE  (main interface)
# ─────────────────────────────────────────────
class MedVisionEngine:
    def __init__(self, load_models: list[str] = None):
        """
        load_models: list of CNN names to load e.g. ["chest", "brain"]
                     Default: loads all available trained models
        """
        print("\n[MedVision] Initializing inference engine...")
        self.cnns     = {}
        self.dqn      = DQNTriage()
        self.reporter = GroqReporter()

        models_to_load = load_models or list(MODEL_PATHS.keys())
        for name in models_to_load:
            if MODEL_PATHS[name].exists():
                try:
                    self.cnns[name] = CNNModel(name)
                except Exception as e:
                    print(f"  [WARN] Could not load {name}: {e}")
            else:
                print(f"  [INFO] {name.upper()} model not trained yet - skipping")

        print(f"\n[MedVision] Ready! Loaded CNNs: {list(self.cnns.keys())}\n")

    def predict_image(self, image: Image.Image, scan_type: str,
                      generate_report: bool = True,
                      patient_info: dict = None) -> dict:
        """
        scan_type: "chest" | "brain" | "skin" | "eye"
        Returns full prediction dict with optional LLM report.
        """
        if scan_type not in self.cnns:
            return {"error": f"{scan_type} CNN not loaded. Train it first!"}

        cnn_result = self.cnns[scan_type].predict(image)
        emergency  = check_emergency(cnn_result)

        result = {
            "scan_type":  scan_type,
            "cnn_result": cnn_result,
            "emergency":  emergency,
            "call_108":   emergency,
            "report":     None,
        }

        if generate_report:
            result["report"] = self.reporter.generate_report(
                cnn_result=cnn_result,
                patient_info=patient_info
            )

        return result

    def predict_symptoms(self, symptoms: list[str],
                         generate_report: bool = True,
                         patient_info: dict = None) -> dict:
        """Symptom-based triage using DQN."""
        dqn_result = self.dqn.predict(symptoms)
        emergency  = dqn_result.get("emergency", False)

        result = {
            "symptoms":   symptoms,
            "dqn_result": dqn_result,
            "emergency":  emergency,
            "call_108":   emergency,
            "report":     None,
        }

        if generate_report:
            result["report"] = self.reporter.generate_report(
                dqn_result=dqn_result,
                patient_info=patient_info
            )

        return result

    def predict_combined(self, image: Image.Image, scan_type: str,
                         symptoms: list[str],
                         generate_report: bool = True,
                         patient_info: dict = None) -> dict:
        """Full pipeline: image + symptoms → combined triage + LLM report."""
        cnn_result = self.cnns[scan_type].predict(image) if scan_type in self.cnns else None
        dqn_result = self.dqn.predict(symptoms) if symptoms else None
        emergency  = check_emergency(cnn_result, dqn_result) if cnn_result else \
                     (dqn_result.get("emergency", False) if dqn_result else False)

        result = {
            "scan_type":  scan_type,
            "cnn_result": cnn_result,
            "dqn_result": dqn_result,
            "emergency":  emergency,
            "call_108":   emergency,
            "report":     None,
        }

        if generate_report:
            result["report"] = self.reporter.generate_report(
                cnn_result=cnn_result,
                dqn_result=dqn_result,
                patient_info=patient_info
            )

        return result

    def available_models(self) -> dict:
        return {
            "cnns":   list(self.cnns.keys()),
            "dqn":    self.dqn.model is not None,
            "gemma4": self.reporter.client is not None,
            "device": str(DEVICE),
        }

    def dqn_status(self) -> dict:
        """Truthful DQN triage availability."""
        if hasattr(self.dqn, "status"):
            return self.dqn.status()
        loaded = bool(getattr(self.dqn, "model", None) is not None)
        return {
            "loaded": loaded,
            "available": loaded,
            "checkpoint_expected": str(DQN_PATH),
            "checkpoint_found": bool(DQN_PATH.exists()),
            "symptom_count": int(getattr(self.dqn, "num_symptoms", 0) or 0) if loaded else 0,
            "reason": None if loaded else "No verified DQN checkpoint available.",
            "disclaimer": "Research prototype — not clinically validated.",
        }

    def llm_status(self) -> dict:
        """Truthful Gemma 4 LLM availability. Local/API nature + missing dependency."""
        import os as _os
        try:
            from google import genai as _genai_check
            sdk = True
        except ImportError:
            sdk, _genai_check = False, None
        key_set = bool(_os.getenv("GEMMA_API_KEY") or _os.getenv("GEMINI_API_KEY"))
        ok = bool(getattr(getattr(self, "reporter", None), "client", None) is not None)
        model = getattr(getattr(self, "reporter", None), "model_name", "gemma-4-31b-it")
        status = {"loaded": ok, "available": ok, "provider": "google-genai (Gemini API)",
                  "mode": "remote API (no local weights)", "model": model,
                  "sdk_installed": sdk, "api_key_set": key_set}
        if ok:
            status["reason"] = None
        else:
            if not sdk:
                status["reason"] = "google-genai package not installed (pip install google-genai)"
            elif not key_set:
                status["reason"] = "GEMMA_API_KEY not set in environment/.env"
            else:
                status["reason"] = "LLM client failed to initialise"
            status["note"] = ("Without a key the app fails gracefully: reports return "
                              "'Gemma 4 API not configured'. CNN + XAI demo unaffected; "
                              "nothing is downloaded automatically.")
        return status

    # ─────────────────────────────────────────────
    #  EXPLAINABLE AI (XAI) — read-only helpers
    # ─────────────────────────────────────────────
    @staticmethod
    def xai_methods() -> dict:
        """
        Methods available in this environment plus the wording shown in the UI.

        Returns a JSON-safe mapping; if the optional `xai` dependencies
        (lime / shap) are missing the affected method is marked unavailable
        instead of breaking the API.
        """
        try:
            from xai import available_methods
            return {"methods": available_methods()}
        except Exception as exc:                       # pragma: no cover - env issue
            return {"methods": {}, "error": f"XAI layer unavailable: {exc}"}

    def explain_image(self, image: Image.Image, scan_type: str,
                      method: str = "gradcam",
                      target_class=None,
                      save_dir: str = None,
                      **params) -> dict:
        """
        Explain an image with Grad-CAM | LIME | SHAP using the loaded modality model.

        Returns the standardized JSON-safe payload (predicted class, confidence,
        base64 PNG visualizations, settings, metadata).  For research work call
        `explain_image_result` instead, which returns the full `XAIResult`
        including the raw attribution arrays.
        """
        result = self.explain_image_result(image, scan_type, method=method,
                                           target_class=target_class, **params)
        if isinstance(result, dict):        # error dict (e.g. model not loaded)
            return result

        payload = result.to_payload()
        if save_dir:
            try:
                payload["saved_files"] = result.save(save_dir)
            except Exception as exc:        # saving is optional, never fatal
                payload.setdefault("warnings", []).append(f"could not save artefacts: {exc}")
        return payload

    def explain_image_result(self, image: Image.Image, scan_type: str,
                             method: str = "gradcam",
                             target_class=None,
                             **params):
        """
        Same as `explain_image` but returns the `XAIResult` object (PIL images +
        numpy attribution maps).  Used by scripts/notebooks and the smoke test.
        """
        if scan_type not in self.cnns:
            return {"error": f"{scan_type} CNN not loaded. Train it first!"}
        if image is None:
            return {"error": "no image supplied"}

        from xai import explain_image as xai_explain_image     # lazy: optional deps
        return xai_explain_image(self.cnns[scan_type], image,
                                 method=method, target_class=target_class, **params)


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    engine = MedVisionEngine()
    print("\n[Test] Available models:", engine.available_models())

    # Test symptom triage
    test_symptoms = ["fever", "cough", "chest pain", "shortness of breath"]
    print(f"\n[Test] Symptom triage for: {test_symptoms}")
    result = engine.predict_symptoms(test_symptoms, generate_report=False)
    print(f"  Priority: {result['dqn_result'].get('priority_label')}")
    print(f"  Emergency: {result['emergency']}")
    print(f"  Call 108: {result['call_108']}")

    # Test Gemma 4 report generation
    print("\n[Test] Gemma 4 report generation...")
    report = engine.reporter.generate_report(
        cnn_result={"model": "chest", "prediction": "PNEUMONIA", "confidence": 0.92, "all_scores": {}},
        patient_info={"age": 45, "gender": "Male"}
    )
    print(f"  Report: {report[:200]}...")
