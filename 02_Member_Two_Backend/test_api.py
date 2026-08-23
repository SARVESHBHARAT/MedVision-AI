from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_image_to_completed_analysis():
    image = client.post('/api/images', json={'filename': 'scan.png', 'content_type': 'image/png', 'size_bytes': 2048})
    assert image.status_code == 201
    job = client.post(f"/api/images/{image.json()['id']}/analyses")
    assert job.status_code == 202
    completed = client.post(f"/api/analyses/{job.json()['id']}/complete")
    assert completed.json()['status'] == 'completed'
    assert completed.json()['result']['model_version'] == 'baseline-0.1'

def test_rejects_unsupported_image_type():
    response = client.post('/api/images', json={'filename': 'scan.gif', 'content_type': 'image/gif', 'size_bytes': 400})
    assert response.status_code == 415
