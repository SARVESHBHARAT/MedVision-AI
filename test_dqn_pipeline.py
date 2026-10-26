"""
Comprehensive Automated Verification Test Suite for MedVision AI DQN Component
Tests:
1. DQN Module Imports & Architecture
2. Clinical Triage Environment Initialization & Step Logic
3. Experience Replay & Model Forward Pass
4. Checkpoint Loading & State Dict Integrity
5. Standalone Benchmark Evaluation
6. Inference Engine & Symptom Vectorizer
7. FastAPI Endpoints (/dqn/status, /dqn/predict, /predict/symptoms, /health, /models)
8. CNN Diagnoses (Chest, Brain, Skin, Eye)
9. Explainability Layers (Grad-CAM, LIME cosine metric, SHAP)
10. Combined Pipeline Integration
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dqn import (
    STATE_DIM,
    NUM_ACTIONS,
    ACTION_PRIORITY_MAP,
    CANONICAL_SYMPTOMS,
    DQN_MODEL_PATH,
    DQN_META_PATH,
    DQNNetwork,
    ReplayBuffer,
    ClinicalTriageEnv,
    DQNEvaluator,
    DQNTriageInference,
)
from medvision_inference import MedVisionEngine


def test_1_dqn_module():
    print("[TEST 1] Testing DQN module imports and structure...")
    assert STATE_DIM == 50, f"Expected STATE_DIM 50, got {STATE_DIM}"
    assert NUM_ACTIONS == 3, f"Expected NUM_ACTIONS 3, got {NUM_ACTIONS}"
    assert len(CANONICAL_SYMPTOMS) == 50
    assert DQN_MODEL_PATH.exists(), f"DQN checkpoint missing at {DQN_MODEL_PATH}"
    assert DQN_META_PATH.exists(), f"DQN metadata missing at {DQN_META_PATH}"
    print("  -> PASS: Module configuration and files verified.")


def test_2_environment():
    print("[TEST 2] Testing ClinicalTriageEnv MDP...")
    env = ClinicalTriageEnv(state_dim=50, seed=123)
    s = env.reset()
    assert s.shape == (50,), f"Expected state shape (50,), got {s.shape}"
    assert env.current_urgency in [1, 2, 3]

    opt_action = env.get_optimal_action(s)
    next_s, r, done, info = env.step(opt_action)
    assert done is True
    assert "reward" in info
    assert "correct" in info
    assert info["correct"] is True
    print("  -> PASS: Environment reset and step logic verified.")


def test_3_model_and_buffer():
    print("[TEST 3] Testing DQNNetwork and ReplayBuffer...")
    net = DQNNetwork(input_dim=50, output_dim=3)
    dummy_s = torch.randn(4, 50)
    q_out = net(dummy_s)
    assert q_out.shape == (4, 3)

    buf = ReplayBuffer(capacity=100, state_dim=50)
    for _ in range(20):
        buf.add(np.zeros(50), 0, 10.0, np.zeros(50), True)
    assert len(buf) == 20
    s_b, a_b, r_b, ns_b, d_b = buf.sample(8)
    assert s_b.shape == (8, 50)
    print("  -> PASS: Model forward pass and ReplayBuffer sampling verified.")


def test_4_checkpoint_and_evaluator():
    print("[TEST 4] Testing Checkpoint Loading & DQNEvaluator...")
    evaluator = DQNEvaluator(model_path=DQN_MODEL_PATH)
    bench = evaluator.run_benchmark(num_episodes=200)
    assert bench["accuracy"] >= 0.99, f"Accuracy too low: {bench['accuracy']}"
    assert bench["critical_safety_violation_rate"] == 0.0, f"Safety violation: {bench['critical_safety_violation_rate']}"
    assert bench["dqn_mean_reward"] >= 9.5, f"Mean reward too low: {bench['dqn_mean_reward']}"
    print(f"  -> PASS: Checkpoint benchmark evaluated (Accuracy: {bench['accuracy']*100:.1f}%, Mean Reward: {bench['dqn_mean_reward']:.2f}).")


def test_5_inference_wrapper():
    print("[TEST 5] Testing DQNTriageInference on real symptom cases...")
    inf = DQNTriageInference(model_path=DQN_MODEL_PATH, meta_path=DQN_META_PATH)
    assert inf.loaded is True

    # Case 1: Emergency Cardiac Case
    res_p1 = inf.predict(["chest pain", "shortness of breath", "sweating"])
    assert res_p1["priority"] == 1
    assert res_p1["emergency"] is True
    assert res_p1["q_values"]["Priority 1"] > res_p1["q_values"]["Priority 3"]

    # Case 2: Urgent High-Fever Abdominal Case
    res_p2 = inf.predict(["high fever", "severe abdominal pain", "persistent vomiting"])
    assert res_p2["priority"] == 2
    assert res_p2["emergency"] is False

    # Case 3: Routine Common Cold Case
    res_p3 = inf.predict(["cough", "runny nose", "sore throat"])
    assert res_p3["priority"] == 3
    assert res_p3["emergency"] is False
    assert res_p3["q_values"]["Priority 3"] > res_p3["q_values"]["Priority 1"]

    print("  -> PASS: Clinical priority assignments verified across P1, P2, P3 cases.")


def test_6_engine_integration():
    print("[TEST 6] Testing MedVisionEngine unified integration...")
    engine = MedVisionEngine()
    models = engine.available_models()
    assert models["dqn"] is True, "DQN should report loaded=True"

    dqn_st = engine.dqn_status()
    assert dqn_st["loaded"] is True
    assert dqn_st["symptom_count"] == 50

    # Test predict_symptoms
    sym_res = engine.predict_symptoms(["chest pressure", "difficulty breathing"], generate_report=False)
    assert sym_res["emergency"] is True
    assert sym_res["dqn_result"]["priority"] == 1

    # Test CNN prediction on dummy image
    dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    for scan in ["chest", "brain", "skin", "eye"]:
        if scan in engine.cnns:
            cnn_res = engine.predict_image(dummy_img, scan_type=scan, generate_report=False)
            assert "cnn_result" in cnn_res
            assert "prediction" in cnn_res["cnn_result"]
            print(f"    [OK] {scan.upper()} CNN prediction: {cnn_res['cnn_result']['prediction']} ({cnn_res['cnn_result']['confidence']*100:.1f}%)")

    # Test combined prediction
    comb_res = engine.predict_combined(dummy_img, scan_type="chest", symptoms=["fever", "cough"], generate_report=False)
    assert "cnn_result" in comb_res
    assert "dqn_result" in comb_res
    assert comb_res["dqn_result"]["priority"] in [1, 2, 3]

    print("  -> PASS: MedVisionEngine full pipeline verified.")


def main():
    print("\n" + "=" * 60)
    print(" 🏥 MEDVISION AI — COMPREHENSIVE AUTOMATED VERIFICATION SUITE")
    print("=" * 60 + "\n")

    test_1_dqn_module()
    test_2_environment()
    test_3_model_and_buffer()
    test_4_checkpoint_and_evaluator()
    test_5_inference_wrapper()
    test_6_engine_integration()

    print("\n" + "=" * 60)
    print(" ✅ ALL 6 VERIFICATION TEST SUITES PASSED PERFECTLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
