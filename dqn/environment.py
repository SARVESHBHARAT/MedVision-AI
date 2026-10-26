"""
Clinical Triage Reinforcement Learning Environment
Simulated Markov Decision Process (MDP) for medical symptom triage.

RESEARCH PROTOTYPE / SCIENTIFIC INTEGRITY STATEMENT:
This environment is a simulated research prototype designed to model clinical
urgency decision-making under asymmetric safety penalties (penalizing under-triage
of life-threatening conditions much more severely than over-triage).
It is grounded on the symptoms2.csv taxonomy and standard Emergency Severity Index (ESI)
principles, but it is NOT a clinically validated hospital deployment tool.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from .config import (
    STATE_DIM,
    CANONICAL_SYMPTOMS,
    ACTION_PRIORITY_MAP,
    CLINICAL_REWARD_MATRIX,
)


class ClinicalTriageEnv:
    """
    Simulated Clinical Symptom Triage MDP.

    State:
        s in {0, 1}^D binary feature vector representing presence of patient symptoms.
        D = 50 canonical symptom dimensions.

    Action:
        a in {0, 1, 2}
        a = 0 -> Priority 1 (Immediate Emergency / Resuscitation)
        a = 1 -> Priority 2 (Urgent / 24-Hour Evaluation)
        a = 2 -> Priority 3 (Non-Urgent / Routine Appointment)

    Reward:
        R(s, a) = CLINICAL_REWARD_MATRIX[true_urgency][chosen_priority]
        Asymmetric clinical loss matrix prioritizing safety.
    """

    def __init__(self, state_dim: int = STATE_DIM, seed: Optional[int] = None):
        self.state_dim = state_dim
        self.rng = np.random.RandomState(seed)
        self.current_state: Optional[np.ndarray] = None
        self.current_urgency: Optional[int] = None  # 1, 2, or 3
        self.current_syndrome_name: Optional[str] = None
        self.done: bool = True

        # Define clinical syndrome templates grounded on symptoms2.csv
        self._init_clinical_syndromes()

    def _init_clinical_syndromes(self):
        """Clinical syndrome clusters defining co-occurring symptoms by urgency level."""
        # Tier 1: Life-threatening Emergency Syndromes (True Priority 1)
        self.p1_syndromes = [
            {
                "name": "Acute Coronary / Cardiac Emergency",
                "core": [0, 1, 2],  # chest pain, chest tightness, chest pressure
                "optional": [3, 4, 6, 22, 48],  # shortness of breath, difficulty breathing, fainting, palpitations, sweating
                "urgency": 1,
            },
            {
                "name": "Acute Respiratory Failure",
                "core": [3, 4, 13],  # shortness of breath, difficulty breathing, cyanosis
                "optional": [0, 15, 22, 48],  # chest pain, high fever, palpitations, sweating
                "urgency": 1,
            },
            {
                "name": "Acute Stroke / Intracranial Emergency",
                "core": [10, 5],  # sudden severe headache, altered mental status
                "optional": [6, 11, 12, 49],  # fainting, seizures, vision loss, blurred vision
                "urgency": 1,
            },
            {
                "name": "Severe Upper GI Bleed / Acute Shock",
                "core": [7, 17],  # vomiting blood, severe abdominal pain
                "optional": [6, 22, 23, 26],  # fainting, palpitations, dark stools, dizziness
                "urgency": 1,
            },
            {
                "name": "Severe Anaphylaxis / Airway Compromise",
                "core": [14, 4],  # anaphylaxis / throat swelling, difficulty breathing
                "optional": [13, 22, 39],  # cyanosis, palpitations, skin rash
                "urgency": 1,
            },
            {
                "name": "Generalized Convulsive Status / Seizure",
                "core": [11, 5],  # seizures, altered mental status
                "optional": [6, 13, 48],  # fainting, cyanosis, sweating
                "urgency": 1,
            },
            {
                "name": "Severe Lower GI Hemorrhage",
                "core": [8, 9],  # bloody diarrhea, spontaneous bleeding
                "optional": [6, 17, 26],  # fainting, severe abdominal pain, dizziness
                "urgency": 1,
            },
        ]

        # Tier 2: Urgent High-Risk Syndromes (True Priority 2)
        self.p2_syndromes = [
            {
                "name": "High-Grade Febrile / Sepsis-Risk Presentation",
                "core": [15, 16],  # high fever, fever with chills/rigors
                "optional": [17, 36, 44, 48],  # abdominal pain, fatigue, loss of appetite, sweating
                "urgency": 2,
            },
            {
                "name": "Acute Surgical Abdomen / Appendicitis / Cholecystitis",
                "core": [17, 18],  # severe abdominal pain, persistent vomiting
                "optional": [15, 26, 40, 44],  # high fever, dizziness, constipation, loss of appetite
                "urgency": 2,
            },
            {
                "name": "Deep Vein Thrombosis (DVT) Risk",
                "core": [19, 27],  # calf pain/leg swelling, joint swelling
                "optional": [20, 36],  # severe back pain, fatigue
                "urgency": 2,
            },
            {
                "name": "Severe Ophthalmic / Acute Eye Inflammation",
                "core": [21, 49],  # eye pain/irritation, blurred vision
                "optional": [35],  # mild headache
                "urgency": 2,
            },
            {
                "name": "Severe Deep Tissue Abscess / Cellulitis",
                "core": [25, 15],  # abscess/boil, high fever
                "optional": [16, 39],  # chills, skin rash
                "urgency": 2,
            },
            {
                "name": "Acute Dysphagia / Esophageal Obstruction",
                "core": [28, 24],  # difficulty swallowing, breast/chest soreness
                "optional": [18, 44],  # vomiting, loss of appetite
                "urgency": 2,
            },
            {
                "name": "Acute Substance Withdrawal / Agitation",
                "core": [29, 31],  # alcohol abuse/withdrawal, delusions/hallucinations
                "optional": [22, 46, 48],  # palpitations, anxiety, sweating
                "urgency": 2,
            },
            {
                "name": "Subacute Upper GI Melena",
                "core": [23, 17],  # dark/black stools, abdominal pain
                "optional": [26, 30, 36],  # dizziness, weight loss, fatigue
                "urgency": 2,
            },
        ]

        # Tier 3: Non-Urgent Outpatient Syndromes (True Priority 3)
        self.p3_syndromes = [
            {
                "name": "Upper Respiratory Tract Infection (Common Cold / URI)",
                "core": [32, 37, 38],  # cough, runny nose, sore throat
                "optional": [34, 35, 36],  # low grade fever, mild headache, fatigue
                "urgency": 3,
            },
            {
                "name": "Mild Viral Gastroenteritis",
                "core": [41, 47],  # mild diarrhea, mild nausea
                "optional": [34, 36, 44],  # mild fever, fatigue, loss of appetite
                "urgency": 3,
            },
            {
                "name": "Mild Tension Headache / Fatigue",
                "core": [35, 36],  # mild headache, fatigue
                "optional": [46, 43],  # anxiety, muscle ache
                "urgency": 3,
            },
            {
                "name": "Mild Allergic Dermatitis / Rash",
                "core": [39],  # skin rash / itching
                "optional": [37, 36],  # runny nose, fatigue
                "urgency": 3,
            },
            {
                "name": "Mechanical Low Back Pain / Musculoskeletal",
                "core": [42, 43],  # minor joint pain, arm/muscle ache
                "optional": [36],  # fatigue
                "urgency": 3,
            },
            {
                "name": "Functional Constipation / Bloating",
                "core": [40, 44],  # constipation, loss of appetite
                "optional": [47],  # mild nausea
                "urgency": 3,
            },
            {
                "name": "Mild Anxiety / Stress Reaction",
                "core": [46],  # anxiety / nervousness
                "optional": [35, 36, 48],  # mild headache, fatigue, sweating
                "urgency": 3,
            },
            {
                "name": "Mild Otitis Externa / Ear Discomfort",
                "core": [45],  # ear pressure / ear pain
                "optional": [38, 37],  # sore throat, runny nose
                "urgency": 3,
            },
        ]

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def reset(self) -> np.ndarray:
        """
        Samples a new simulated patient presentation.
        Returns the initial symptom state vector.
        """
        # Sample urgency tier with balanced representation
        tier_choice = self.rng.choice([1, 2, 3], p=[0.33, 0.34, 0.33])
        if tier_choice == 1:
            syndrome = self.p1_syndromes[self.rng.randint(0, len(self.p1_syndromes))]
        elif tier_choice == 2:
            syndrome = self.p2_syndromes[self.rng.randint(0, len(self.p2_syndromes))]
        else:
            syndrome = self.p3_syndromes[self.rng.randint(0, len(self.p3_syndromes))]

        state = np.zeros(self.state_dim, dtype=np.float32)

        # Activate core symptoms
        for idx in syndrome["core"]:
            if idx < self.state_dim:
                state[idx] = 1.0

        # Probabilistically activate secondary symptoms
        for idx in syndrome.get("optional", []):
            if idx < self.state_dim and self.rng.rand() < 0.60:
                state[idx] = 1.0

        # Add small background noise / non-specific symptom (e.g. fatigue or mild headache)
        if self.rng.rand() < 0.25:
            noise_idx = self.rng.choice([36, 44, 46])  # fatigue, appetite loss, anxiety
            if noise_idx < self.state_dim:
                state[noise_idx] = 1.0

        self.current_state = state
        self.current_urgency = syndrome["urgency"]
        self.current_syndrome_name = syndrome["name"]
        self.done = False

        return self.current_state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes triage action.
        Returns (next_state, reward, done, info).
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        chosen_priority = ACTION_PRIORITY_MAP.get(action, 2)
        true_urgency = self.current_urgency

        # Compute clinically aligned reward
        reward = float(CLINICAL_REWARD_MATRIX[true_urgency][chosen_priority])

        # Diagnostic flags
        correct = (chosen_priority == true_urgency)
        under_triaged = (chosen_priority > true_urgency)
        critical_under_triage = (true_urgency == 1 and chosen_priority == 3)
        over_triaged = (chosen_priority < true_urgency)

        self.done = True
        next_state = self.current_state.copy()

        info = {
            "true_urgency": true_urgency,
            "chosen_priority": chosen_priority,
            "syndrome": self.current_syndrome_name,
            "correct": correct,
            "under_triaged": under_triaged,
            "critical_under_triage": critical_under_triage,
            "over_triaged": over_triaged,
            "reward": reward,
        }

        return next_state, reward, self.done, info

    def get_optimal_action(self, state: np.ndarray) -> int:
        """
        Returns the Bayes-optimal action under the clinical safety matrix for a given state.
        Red flags (indices 0..14) -> Priority 1 (Action 0)
        Urgent signs (indices 15..31) -> Priority 2 (Action 1)
        Mild signs (indices 32..49) -> Priority 3 (Action 2)
        """
        # Red flags present?
        if np.any(state[:15] > 0.5):
            return 0  # Action 0 -> Priority 1

        # Urgent signs present?
        if np.any(state[15:32] > 0.5):
            return 1  # Action 1 -> Priority 2

        # Non-urgent signs
        return 2  # Action 2 -> Priority 3
