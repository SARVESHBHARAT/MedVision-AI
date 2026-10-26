"""
DQN Configuration & Clinical Symptom Ontology
Defines state dimensions, action space, clinical urgency rules, reward matrix,
and symptom mappings grounded on symptoms2.csv taxonomy.

RESEARCH PROTOTYPE DISCLAIMER:
This environment and policy represent a simulated research demonstration of
Reinforcement Learning for clinical symptom triage and are NOT clinically validated.
"""

from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
DQN_MODEL_PATH = BASE_DIR / "dqn_model.pth"
DQN_META_PATH = BASE_DIR / "dqn_meta.json"
DQN_OUTPUT_DIR = BASE_DIR / "dqn_outputs"
SYMPTOMS_CSV_PATH = BASE_DIR / "symptoms2.csv"

# ─────────────────────────────────────────────
#  CANONICAL SYMPTOM ONTOLOGY (50 Dimensions)
# ─────────────────────────────────────────────
# Curated from symptoms2.csv chief complaints & clinical triage categories:
# [0-14]  High-risk / Acute / Life-threatening signs (Priority 1 candidates)
# [15-31] Urgent / Significant symptoms (Priority 2 candidates)
# [32-49] Non-urgent / Mild outpatient symptoms (Priority 3 candidates)
CANONICAL_SYMPTOMS = [
    # Tier 1: Red flags / Acute life-threatening (0..14)
    "chest pain",                   # 0
    "chest tightness",              # 1
    "chest pressure",               # 2
    "shortness of breath",          # 3
    "difficulty breathing",         # 4
    "altered mental status",        # 5
    "fainting / syncope",           # 6
    "vomiting blood",               # 7
    "bloody diarrhea",              # 8
    "spontaneous bleeding",         # 9
    "sudden severe headache",       # 10
    "seizures / convulsions",       # 11
    "sudden vision loss",           # 12
    "cyanosis (turning blue)",      # 13
    "anaphylaxis / throat swelling",# 14

    # Tier 2: Urgent / High-risk symptoms (15..31)
    "high fever (>38.5C)",          # 15
    "fever with chills / rigors",   # 16
    "severe abdominal pain",        # 17
    "persistent vomiting",          # 18
    "calf pain / leg swelling",     # 19
    "severe back / spinal pain",    # 20
    "eye pain / severe irritation", # 21
    "rapid heart rate / palpitations", # 22
    "black / dark stools (melena)", # 23
    "pain or soreness of breast",   # 24
    "abscess / severe boil",        # 25
    "acute dizziness / vertigo",    # 26
    "severe joint swelling",        # 27
    "difficulty swallowing",        # 28
    "alcohol abuse / withdrawal",   # 29
    "unexplained weight loss",      # 30
    "delusions / hallucinations",   # 31

    # Tier 3: Non-urgent / Mild outpatient symptoms (32..49)
    "cough",                        # 32
    "dry cough",                    # 33
    "fever",                        # 34
    "mild headache",                # 35
    "fatigue / tiredness",          # 36
    "runny nose / congestion",      # 37
    "sore throat",                  # 38
    "skin rash / itching",          # 39
    "constipation",                 # 40
    "mild diarrhea",                # 41
    "minor joint pain",             # 42
    "arm / muscle ache",            # 43
    "loss of appetite",             # 44
    "ear pressure / ear pain",      # 45
    "anxiety / nervousness",        # 46
    "mild nausea",                  # 47
    "sweating",                     # 48
    "blurred vision",               # 49
]

STATE_DIM = len(CANONICAL_SYMPTOMS)  # 50

# ─────────────────────────────────────────────
#  ACTION SPACE (3 Discrete Actions)
# ─────────────────────────────────────────────
ACTION_PRIORITY_MAP = {
    0: 1,  # Action 0 -> Priority 1 (Immediate Emergency / Resuscitation / Call 108)
    1: 2,  # Action 1 -> Priority 2 (Urgent / Clinical Evaluation within 24 Hours)
    2: 3,  # Action 2 -> Priority 3 (Non-Urgent / Routine Outpatient Appointment)
}

PRIORITY_ACTION_MAP = {v: k for k, v in ACTION_PRIORITY_MAP.items()}

ACTION_NAMES = {
    0: "Priority 1 (Immediate Emergency Care)",
    1: "Priority 2 (Urgent Care within 24 Hours)",
    2: "Priority 3 (Non-Urgent / Routine Appointment)",
}

ACTION_DESCRIPTIONS = {
    0: "Life-threatening acute presentation requiring emergency intervention (Call 108).",
    1: "Urgent medical condition requiring same-day or 24-hour medical assessment.",
    2: "Mild, stable condition appropriate for standard scheduled clinical visit.",
}

NUM_ACTIONS = len(ACTION_PRIORITY_MAP)  # 3

# ─────────────────────────────────────────────
#  CLINICAL REWARD MATRIX (Asymmetric Safety Penalties)
# ─────────────────────────────────────────────
# Reward structure: [True Urgency (1,2,3)][Chosen Priority (1,2,3)]
# Prioritizes patient safety by penalizing under-triage severely.
CLINICAL_REWARD_MATRIX = {
    1: {  # True condition: Priority 1 (Emergency)
        1: +10.0,  # Correct triage
        2: -15.0,  # Under-triage (Dangerous delay)
        3: -30.0,  # Severe under-triage (Critical safety hazard)
    },
    2: {  # True condition: Priority 2 (Urgent)
        1: -3.0,   # Moderate over-triage (Precautionary, mild resource inefficiency)
        2: +10.0,  # Correct triage
        3: -10.0,  # Under-triage (Unacceptable clinical delay)
    },
    3: {  # True condition: Priority 3 (Non-urgent)
        1: -6.0,   # Over-triage (Emergency department crowding)
        2: -2.0,   # Minor over-triage
        3: +10.0,  # Correct triage
    },
}

# ─────────────────────────────────────────────
#  SYNONYM & ALIAS MAPPING
# ─────────────────────────────────────────────
# Maps common inputs and symptoms2.csv terms to canonical symptom indices
SYMPTOM_ALIASES = {
    # Chest / Cardiac / Respiratory
    "chest pain": 0,
    "chest tightness": 1,
    "chest tightness, heaviness or pressure": 1,
    "chest pressure": 2,
    "pressure in chest": 2,
    "shortness of breath": 3,
    "short of breath": 3,
    "out of breath": 3,
    "difficulty breathing": 4,
    "breathlessness": 4,
    "dyspnea": 4,
    "wheezing": 4,
    "cyanosis": 13,
    "turning blue": 13,
    "throat swelling": 14,
    "anaphylaxis": 14,
    "severe allergic reaction": 14,

    # Neurological / Altered Status
    "altered mental status": 5,
    "change in behavior": 5,
    "confusion": 5,
    "disorientation": 5,
    "fainting": 6,
    "fainting spells": 6,
    "passing out": 6,
    "syncope": 6,
    "loss of consciousness": 6,
    "vomiting blood": 7,
    "blood in vomit": 7,
    "hematemesis": 7,
    "bloody diarrhea": 8,
    "blood in stool": 8,
    "spontaneous bleeding": 9,
    "bleeding tendency": 9,
    "easy bruising": 9,
    "sudden severe headache": 10,
    "thunderclap headache": 10,
    "worst headache": 10,
    "seizures": 11,
    "seizure": 11,
    "convulsions": 11,
    "sudden vision loss": 12,
    "loss of vision": 12,

    # Urgent / Tier 2
    "high fever": 15,
    "elevated temperature": 15,
    "fever with chills": 16,
    "chills and fever": 16,
    "rigors": 16,
    "severe abdominal pain": 17,
    "upper abdominal pain": 17,
    "upper belly pain": 17,
    "lower abdominal pain": 17,
    "lower belly pain": 17,
    "persistent vomiting": 18,
    "vomiting": 18,
    "calf pain": 19,
    "lower leg pain": 19,
    "leg swelling": 19,
    "severe back pain": 20,
    "back ache or pain": 20,
    "spinal pain": 20,
    "back pain": 20,
    "eye pain": 21,
    "eye pain (irritation)": 21,
    "palpitations": 22,
    "rapid heart rate": 22,
    "racing heart": 22,
    "dark stools": 23,
    "black stools": 23,
    "melena": 23,
    "breast pain": 24,
    "pain or soreness of breast": 24,
    "abscess": 25,
    "abscess (collection of pus)": 25,
    "boil": 25,
    "boil (furuncle)": 25,
    "dizziness": 26,
    "vertigo": 26,
    "lightheaded": 26,
    "joint swelling": 27,
    "difficulty swallowing": 28,
    "dysphagia": 28,
    "alcohol abuse": 29,
    "excessive drinking": 29,
    "weight loss": 30,
    "unexplained weight loss": 30,
    "delusions": 31,
    "hallucinations": 31,
    "delusions or hallucinations": 31,
    "behavioral disturbances": 31,

    # Non-Urgent / Tier 3
    "cough": 32,
    "dry cough": 33,
    "wet cough": 32,
    "fever": 34,
    "low grade fever": 34,
    "mild fever": 34,
    "headache": 35,
    "mild headache": 35,
    "head pain": 35,
    "fatigue": 36,
    "tiredness": 36,
    "lethargy": 36,
    "runny nose": 37,
    "nasal congestion": 37,
    "cold": 37,
    "stuffy nose": 37,
    "sore throat": 38,
    "throat irritation": 38,
    "skin rash": 39,
    "rash": 39,
    "itching": 39,
    "constipation": 40,
    "decreased stools": 40,
    "diarrhea": 41,
    "mild diarrhea": 41,
    "joint pain": 42,
    "elbow ache or pain": 42,
    "arm ache or pain": 43,
    "muscle ache": 43,
    "loss of appetite": 44,
    "decreased appetite": 44,
    "ear pressure": 45,
    "pain in the ear": 45,
    "anxiety": 46,
    "nervousness": 46,
    "anxiety (nervousness)": 46,
    "panic attack": 46,
    "depressed": 46,
    "nausea": 47,
    "mild nausea": 47,
    "sweating": 48,
    "night sweats": 48,
    "blurred vision": 49,
    "double vision (diplopia)": 49,
    "double vision": 49,
}

# ─────────────────────────────────────────────
#  DEFAULT TRAINING CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_TRAIN_CONFIG = {
    "state_dim": STATE_DIM,
    "action_dim": NUM_ACTIONS,
    "hidden_dims": [128, 128, 64],
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "gamma": 0.95,
    "batch_size": 64,
    "buffer_size": 50000,
    "min_buffer_size": 500,
    "target_update_freq": 10,
    "episodes": 2500,
    "eval_episodes": 500,
    "epsilon_start": 1.0,
    "epsilon_end": 0.02,
    "epsilon_decay": 0.996,
    "random_seed": 42,
    "environment_version": "v1.0-simulated-clinical-triage",
    "is_synthetic": True,
    "disclaimer": "Research prototype — simulated triage environment. Not clinically validated."
}
