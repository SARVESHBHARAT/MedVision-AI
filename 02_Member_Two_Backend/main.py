from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(title="MedVision AI API", version="0.1.0")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}

class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class ImageCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str
    size_bytes: int = Field(gt=0)

class ImageRecord(ImageCreate):
    id: UUID
    created_at: datetime

class AnalysisJob(BaseModel):
    id: UUID
    image_id: UUID
    status: JobStatus
    created_at: datetime
    result: dict | None = None

images: dict[UUID, ImageRecord] = {}
jobs: dict[UUID, AnalysisJob] = {}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "medvision-api"}

@app.post("/api/images", response_model=ImageRecord, status_code=status.HTTP_201_CREATED)
def create_image(payload: ImageCreate):
    if payload.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Only PNG and JPEG images are supported")
    if payload.size_bytes > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB limit")
    image = ImageRecord(id=uuid4(), created_at=datetime.now(timezone.utc), **payload.model_dump())
    images[image.id] = image
    return image

@app.post("/api/images/{image_id}/analyses", response_model=AnalysisJob, status_code=status.HTTP_202_ACCEPTED)
def create_analysis(image_id: UUID):
    if image_id not in images:
        raise HTTPException(404, "Image not found")
    job = AnalysisJob(id=uuid4(), image_id=image_id, status=JobStatus.queued, created_at=datetime.now(timezone.utc))
    jobs[job.id] = job
    return job

@app.post("/api/analyses/{job_id}/complete", response_model=AnalysisJob)
def complete_analysis(job_id: UUID):
    """Temporary worker callback used until the inference worker is connected."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Analysis job not found")
    if job.status == JobStatus.completed:
        return job
    job.status = JobStatus.completed
    job.result = {
        "finding": "No urgent pattern detected",
        "confidence": 0.82,
        "model_version": "baseline-0.1",
        "disclaimer": "Experimental output — not for diagnostic use.",
    }
    return job

@app.get("/api/analyses/{job_id}", response_model=AnalysisJob)
def get_analysis(job_id: UUID):
    if job_id not in jobs:
        raise HTTPException(404, "Analysis job not found")
    return jobs[job_id]
