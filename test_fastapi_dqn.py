"""
FastAPI TestClient End-to-End Suite for MedVision AI
Tests all API endpoints including DQN status, prediction, CNN, and Combined Analysis.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path
import io
from PIL import Image
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from main import app, startup_event
import asyncio

# Run startup event to initialize engine
asyncio.run(startup_event())
client = TestClient(app)


def test_api():
    print("\n" + "=" * 60)
    print(" [MedVision API] FASTAPI TESTCLIENT VERIFICATION")
    print("=" * 60)

    # 1. Health & Root
    r_root = client.get("/")
    assert r_root.status_code == 200
    print("  [OK] GET / -> 200")

    r_health = client.get("/health")
    assert r_health.status_code == 200
    print("  [OK] GET /health -> 200")

    # 2. Models
    r_models = client.get("/models")
    assert r_models.status_code == 200
    models_data = r_models.json()
    assert models_data["dqn"] is True
    print(f"  [OK] GET /models -> 200 | DQN: {models_data['dqn']} | CNNs: {models_data['cnns']}")

    # 3. DQN Status
    r_dqn_st = client.get("/dqn/status")
    assert r_dqn_st.status_code == 200
    dqn_st = r_dqn_st.json()
    assert dqn_st["loaded"] is True
    assert dqn_st["symptom_count"] == 50
    print(f"  [OK] GET /dqn/status -> 200 | Loaded: {dqn_st['loaded']} | Symptoms: {dqn_st['symptom_count']}")

    # 4. DQN Direct Prediction (POST /dqn/predict)
    r_dqn_p1 = client.post("/dqn/predict", json={"symptoms": ["chest pain", "shortness of breath"]})
    assert r_dqn_p1.status_code == 200
    res_p1 = r_dqn_p1.json()
    assert res_p1["priority"] == 1
    assert res_p1["emergency"] is True
    print(f"  [OK] POST /dqn/predict (P1) -> Priority {res_p1['priority']} | Q-values: {res_p1['q_values']}")

    r_dqn_p3 = client.post("/dqn/predict", json={"symptoms": ["cough", "runny nose"]})
    assert r_dqn_p3.status_code == 200
    res_p3 = r_dqn_p3.json()
    assert res_p3["priority"] == 3
    assert res_p3["emergency"] is False
    print(f"  [OK] POST /dqn/predict (P3) -> Priority {res_p3['priority']} | Q-values: {res_p3['q_values']}")

    # 5. Symptom Triage (POST /predict/symptoms)
    r_sym = client.post("/predict/symptoms", json={"symptoms": ["fever", "chills"], "generate_report": False})
    assert r_sym.status_code == 200
    print(f"  [OK] POST /predict/symptoms -> 200 | Priority: {r_sym.json()['dqn_result']['priority_label']}")

    # 6. CNN Image Prediction (POST /predict/image/chest)
    dummy_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG")
    buf.seek(0)
    files = {"file": ("chest.jpg", buf, "image/jpeg")}
    r_img = client.post("/predict/image/chest?generate_report=false", files=files)
    assert r_img.status_code == 200
    print(f"  [OK] POST /predict/image/chest -> 200 | Prediction: {r_img.json()['cnn_result']['prediction']}")

    # 7. Symptoms List
    r_sym_list = client.get("/symptoms/list")
    assert r_sym_list.status_code == 200
    assert r_sym_list.json()["count"] == 50
    print(f"  [OK] GET /symptoms/list -> 200 | Count: {r_sym_list.json()['count']}")

    print("\n" + "=" * 60)
    print(" [OK] ALL FASTAPI ENDPOINTS VERIFIED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_api()
