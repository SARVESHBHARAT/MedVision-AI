"""
DQN Inference Engine for Medical Symptom Triage
Loads dqn_model.pth and dqn_meta.json once, processes free-text or structured symptoms,
and outputs deterministic triage assessments with Q-values and clinical research disclaimers.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from .config import (
    STATE_DIM,
    CANONICAL_SYMPTOMS,
    SYMPTOM_ALIASES,
    ACTION_PRIORITY_MAP,
    ACTION_NAMES,
    DQN_MODEL_PATH,
    DQN_META_PATH,
)
from .model import DQNNetwork


class DQNTriageInference:
    """
    Production-grade DQN Inference Wrapper for FastAPI & Streamlit.
    """
    PRIORITY_MAP = ACTION_PRIORITY_MAP  # {0: 1, 1: 2, 2: 3}

    def __init__(
        self,
        model_path: Path = DQN_MODEL_PATH,
        meta_path: Path = DQN_META_PATH,
        device: Optional[torch.device] = None,
    ):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[DQNNetwork] = None
        self.meta: Dict[str, Any] = {}
        self.symptoms: List[str] = CANONICAL_SYMPTOMS
        self.num_symptoms: int = len(CANONICAL_SYMPTOMS)
        self.loaded: bool = False
        self.load_error: Optional[str] = None

        self._load()

    def _load(self):
        """Loads checkpoint and metadata once at startup."""
        if not self.model_path.exists():
            self.load_error = f"Checkpoint not found at {self.model_path}"
            print(f"[DQN Inference] [INFO] {self.load_error}. Symptom triage disabled.")
            return

        try:
            # Load metadata if exists
            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
                    if "canonical_symptoms" in self.meta:
                        self.symptoms = self.meta["canonical_symptoms"]
                        self.num_symptoms = len(self.symptoms)

            ckpt = torch.load(self.model_path, map_location=self.device)
            state_dim = ckpt.get("state_dim", self.num_symptoms)
            action_dim = ckpt.get("action_dim", 3)
            hidden_dims = ckpt.get("hidden_dims", [128, 128, 64])

            dropout = ckpt.get("dropout", 0.2)

            self.model = DQNNetwork(
                input_dim=state_dim,
                output_dim=action_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            ).to(self.device)

            if isinstance(ckpt, dict) and "policy_net" in ckpt:
                state_dict = ckpt["policy_net"]
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.loaded = True
            self.load_error = None
            print(f"[DQN Inference] [OK] DQN Triage loaded | {self.num_symptoms} canonical symptoms | Device: {self.device}")

        except Exception as exc:
            self.loaded = False
            self.load_error = str(exc)
            self.model = None
            print(f"[DQN Inference] [ERROR] Failed to load DQN: {exc}")

    def encode_symptoms(self, symptom_list: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Encodes a list of symptom strings into the 50-dimensional feature vector.
        Returns: (feature_vector, matched_symptoms, unmatched_symptoms)
        """
        vec = np.zeros(self.num_symptoms, dtype=np.float32)
        matched = []
        unmatched = []

        for item in symptom_list:
            clean = item.strip().lower()
            if not clean:
                continue

            matched_idx = None
            # Direct match in canonical list
            if clean in self.symptoms:
                matched_idx = self.symptoms.index(clean)
            # Match in aliases
            elif clean in SYMPTOM_ALIASES:
                matched_idx = SYMPTOM_ALIASES[clean]
            else:
                # Substring match fallback
                for canonical, idx in SYMPTOM_ALIASES.items():
                    if clean in canonical or canonical in clean:
                        matched_idx = idx
                        break

            if matched_idx is not None and matched_idx < self.num_symptoms:
                vec[matched_idx] = 1.0
                matched.append(clean)
            else:
                unmatched.append(clean)

        return vec, matched, unmatched

    @torch.no_grad()
    def predict(self, symptoms: Union[List[str], str]) -> Dict[str, Any]:
        """
        Runs DQN triage inference on the given symptoms.
        Never crashes; returns honest error dict if model is unconfigured.
        """
        if not self.loaded or self.model is None:
            return {
                "loaded": False,
                "error": "No verified DQN checkpoint available.",
                "reason": self.load_error or "dqn_model.pth not found",
                "disclaimer": "Research prototype — not clinically validated.",
            }

        if isinstance(symptoms, str):
            symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]
        else:
            symptom_list = list(symptoms)

        vec, matched, unmatched = self.encode_symptoms(symptom_list)

        tensor = torch.FloatTensor(vec).unsqueeze(0).to(self.device)
        q_values_raw = self.model(tensor).cpu().numpy()[0]
        action = int(np.argmax(q_values_raw))
        priority = self.PRIORITY_MAP.get(action, 2)

        # Compute softmax probabilities over Q-values as policy confidence
        exp_q = np.exp(q_values_raw - np.max(q_values_raw))
        probs = exp_q / np.sum(exp_q)

        q_dict = {
            "Priority 1": round(float(q_values_raw[0]), 4),
            "Priority 2": round(float(q_values_raw[1]), 4),
            "Priority 3": round(float(q_values_raw[2]), 4),
        }
        policy_probs = {
            "Priority 1": round(float(probs[0]), 4),
            "Priority 2": round(float(probs[1]), 4),
            "Priority 3": round(float(probs[2]), 4),
        }

        return {
            "model": "dqn_triage",
            "loaded": True,
            "matched_symptoms": matched,
            "unmatched_symptoms": unmatched,
            "active_feature_count": int(np.sum(vec)),
            "action": action,
            "priority": priority,
            "priority_label": f"Priority {priority}",
            "action_name": ACTION_NAMES.get(action, f"Priority {priority}"),
            "emergency": (priority == 1),
            "q_values": q_dict,
            "policy_probabilities": policy_probs,
            "is_synthetic": True,
            "status": "Loaded",
            "disclaimer": "Research prototype — simulated clinical triage MDP. Not clinically validated for hospital deployment.",
        }

    def status(self) -> Dict[str, Any]:
        """Truthful DQN status dictionary for API endpoint."""
        return {
            "loaded": self.loaded,
            "available": self.loaded,
            "checkpoint_expected": str(self.model_path),
            "checkpoint_found": bool(self.model_path.exists()),
            "symptom_count": self.num_symptoms if self.loaded else 0,
            "state_dim": self.num_symptoms if self.loaded else 0,
            "action_count": 3,
            "actions": ACTION_NAMES,
            "is_synthetic": True,
            "environment_type": "simulated_clinical_triage_mdp",
            "reason": None if self.loaded else (self.load_error or "No verified DQN checkpoint available."),
            "disclaimer": "Research prototype — simulated triage environment. Not clinically validated.",
        }
