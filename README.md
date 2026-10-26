# 🏥 MedVision AI — Multi-Modal Clinical Intelligence Platform

> **Production-grade multi-modal diagnostic system combining PyTorch ResNet50 CNNs, Explainable AI (Grad-CAM, LIME, SHAP), Deep Q-Network (DQN) reinforcement learning symptom triage, and Google AI Studio Gemma 4 clinical report synthesis.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8.0-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Google GenAI](https://img.shields.io/badge/Google%20GenAI-Gemma%204-4285F4?style=flat&logo=google&logoColor=white)](https://aistudio.google.com/)

---

## 🔬 System Overview & Problem Formulation

In modern clinical triage and multi-modal diagnostics, medical AI systems encounter two distinct data streams:
1. **Unstructured Imaging Data:** Chest Radiographs, Brain Magnetic Resonance Imaging (MRI), Dermoscopic Skin Lesions, and Diabetic Retinopathy Fundus Scans.
2. **Tabular & Free-Text Chief Complaints:** Acute red flags (chest tightness, syncope, sudden hemiparesis), urgent symptoms (high fever with rigors, persistent vomiting), and routine outpatient complaints.

Standard supervised machine learning treats all misclassifications symmetrically through standard cross-entropy loss. However, in real-world clinical triage:
- **Under-triaging a life-threatening acute case** (e.g., categorizing an evolving stroke or tension pneumothorax as routine) represents a catastrophic, irreversible clinical error.
- **Over-triaging a mild condition** (e.g., escalating mild viral pharyngitis) causes emergency department resource congestion, but does not risk patient mortality.

MedVision AI addresses this asymmetry by pairing fine-tuned deep convolutional networks with an asymmetric-loss **Deep Q-Network (DQN) Reinforcement Learning agent**, verifiable attribution layers (**Grad-CAM**, **LIME**, **SHAP**), and structured **Gemma 4** doctor-level report generation.

---

## 🧠 End-to-End Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Patient / Clinician Input                │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
    [Medical Image Upload]                                          [Chief Complaint Triage]
  Chest / Brain / Eye / Skin                                        389 Symptoms (symptoms2.csv)
               │                                                               │
               ▼                                                               ▼
   ┌───────────────────────┐                                       ┌───────────────────────┐
   │ PyTorch ResNet50 CNNs │                                       │  Deep Q-Network (DQN) │
   │ Modality Classification│                                      │  Asymmetric Loss MDP  │
   └───────────┬───────────┘                                       └───────────┬───────────┘
               │ Class & Confidence                                            │ Q-Values & Triage
               ▼                                                               ▼
   ┌───────────────────────┐                                       ┌───────────────────────┐
   │ Explainable AI (XAI)  │                                       │ Urgency Tier Decision │
   │  Grad-CAM | LIME | SHAP│                                      │ Priority 1: Emergency │
   └───────────┬───────────┘                                       │ Priority 2: Urgent    │
               │                                                   │ Priority 3: Routine   │
               │ Visual Attributions                               └───────────┬───────────┘
               └───────────────────────────────┬───────────────────────────────┘
                                               │ Combined Diagnostic Vector
                                               ▼
                               ┌───────────────────────────────┐
                               │ Google AI Studio Gemma 4 LLM  │
                               │  Clinical Synthesis Engine    │
                               └───────────────┬───────────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │  Structured Physician Report    │
                              │  • Diagnostic Summary           │
                              │  • Emergency Detection (108)    │
                              │  • Differential Diagnoses       │
                              │  • Recommended Interventions    │
                              └─────────────────────────────────┘
```

---

## 📊 Verified Model Benchmarks

All model benchmarks reflect real test evaluation metrics verified in accompanying evaluation metadata:

| Modality / Component | Architecture / Base | Task & Classes | Verified Test Accuracy | Verification Artifact |
| :--- | :--- | :--- | :---: | :--- |
| **🫁 Chest X-Ray** | ResNet50 (Transfer Learning) | Pneumonia vs. Normal | **92.95%** | `cnn/chest_xray_meta.json` |
| **🧠 Brain MRI** | ResNet50 (Transfer Learning) | Glioma, Meningioma, No Tumor, Pituitary | **95.44%** | `cnn/brain_test_results_v2.json` |
| **👁️ Eye Fundus** | ResNet50 (Transfer Learning) | Diabetic Retinopathy (5 stages: Normal to Severe) | **90.50%** | `cnn/eye_meta_v3.json` |
| **🔬 Skin Lesion** | ResNet50 (ISIC 2019) | Multi-class Dermoscopy Classification | **89.00%** | `cnn/skin_meta.json` |
| **🤖 DQN Triage** | Double Q-Network (DQN) | 50-dim state vector $\rightarrow$ Priority 1 / 2 / 3 | **100.0%** (Policy) | `dqn_meta.json` (0.00% safety violations) |

---

## 🤖 Deep Q-Network (DQN) Clinical Formulation

The symptom triage module formulates patient presentation as a **Markov Decision Process (MDP)**:
- **State Space ($\mathcal{S}$):** A 50-dimensional feature space derived from the `symptoms2.csv` taxonomy (389 clinical complaints). Symptoms map into:
  - **Tier 1 (Indices 0–14):** Acute Life-Threatening Red Flags (*chest pain, dyspnea, altered sensorium, sudden hemiplegia, syncope*).
  - **Tier 2 (Indices 15–31):** Urgent Presentations (*high fever with chills, acute abdominal pain, melena, intractable emesis*).
  - **Tier 3 (Indices 32–49):** Stable Ambulatory Complaints (*mild rhinitis, low-grade pyrexia, minor arthralgia*).
- **Action Space ($\mathcal{A}$):**
  - **$a = 0$ $\rightarrow$ Priority 1 (Emergency / Resuscitation / Call 108):** Immediate intervention within minutes.
  - **$a = 1$ $\rightarrow$ Priority 2 (Urgent / 24-Hour Evaluation):** Same-day emergency/urgent care assessment.
  - **$a = 2$ $\rightarrow$ Priority 3 (Non-Urgent / Routine Appointment):** Standard outpatient clinic visit.
- **Asymmetric Safety Reward Matrix ($\mathcal{R}$):** Under-triaging a Priority 1 patient is assigned an extreme penalty ($\mathcal{R} = -100.0$), ensuring the learned policy guarantees zero critical under-triage errors.

---

## 👁️ Explainable AI (XAI) Suite

To satisfy clinical verification standards, every visual prediction can be interrogated using three independent explainability modalities:
1. **Grad-CAM:** Computes gradients flowing into the final convolutional layer of ResNet50 to visualize class-discriminative activation heatmaps.
2. **LIME:** Segments images into superpixels and generates local surrogate linear models under random masking to identify positive and negative support regions.
3. **SHAP (PartitionExplainer):** Applies cooperative game-theoretic Shapley value attribution across image partitions to quantify exact feature attributions.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.12+
- Node.js 18+ and npm
- (Optional) NVIDIA CUDA-capable GPU

### 2. Backend Installation

```bash
# Clone the repository
git clone https://github.com/SARVESHBHARAT/MedVision-AI.git
cd MedVision-AI

# Create and activate Python virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Create local .env from example template
cp .env.example .env

# Edit .env and insert your free Google AI Studio key:
# GEMMA_API_KEY=your_key_here
# GEMMA_MODEL=gemma-4-31b-it (or gemma-4-26b-a4b-it)
```
> *Note: If no API key is provided, the application will still perform imaging diagnostics, XAI attributions, and DQN symptom triage seamlessly; report text will indicate that Gemma API is not configured.*

### 4. Model Weights Setup

Check model status using the built-in helper:
```bash
python scripts/download_weights.py
```
Place your trained ResNet50 checkpoints in `cnn/`:
- `cnn/brain_resnet50_v2.pth` (94.5 MB)
- `cnn/chest_xray_resnet50.pth` (94.5 MB)
- `cnn/eye_resnet50_v3.pth` (94.5 MB)
- `cnn/skin_resnet50.pth` (94.5 MB)

*(Pre-trained weights are available via the [GitHub Releases](https://github.com/SARVESHBHARAT/MedVision-AI/releases) section or can be trained from scratch using the included `cnn/*_trainer*.py` scripts).*

### 5. Launch the Application

**Terminal 1 — FastAPI Backend:**
```bash
python main.py
# API runs at http://127.0.0.1:8000 (Swagger docs at http://127.0.0.1:8000/docs)
```

**Terminal 2 — React 19 Frontend:**
```bash
cd frontend
npm install
npm run dev
# Dashboard launches at http://localhost:5173
```

---

## 🔌 API Reference

| HTTP Method | Route | Description |
| :--- | :--- | :--- |
| `POST` | `/predict/combined/{scan_type}?generate_report=true` | Full multi-modal pipeline (Image + Symptoms + DQN + XAI + Gemma 4 report) |
| `POST` | `/predict/image/{scan_type}` | Image-only classification with optional report |
| `POST` | `/predict/symptoms` | Chief complaint symptom triage via DQN agent |
| `POST` | `/xai/gradcam/{scan_type}` | Computes Grad-CAM activation heatmap overlay |
| `POST` | `/xai/lime/{scan_type}` | Computes LIME superpixel attribution explanation |
| `POST` | `/xai/shap/{scan_type}` | Computes SHAP partition feature attribution map |
| `GET` | `/dqn/status` | Reports DQN agent status, loaded weights, and symptom count |
| `GET` | `/system/gpu` | Real-time CUDA device name, VRAM allocation, and PyTorch status |
| `GET` | `/health` | Backend health check and subsystem status |

---

## 📁 Repository Structure

```
MedVision-AI/
├── cnn/                           # Reproducible CNN training scripts & verified metadata
│   ├── brain_meta_v2.json         # Brain classes & validation metrics
│   ├── brain_trainer_v2.py        # Brain MRI ResNet50 trainer
│   ├── chest_xray_meta.json       # Chest X-Ray classes & metrics
│   ├── chest_xray_trainer.py      # Chest X-Ray ResNet50 trainer
│   ├── eye_meta_v3.json           # Diabetic Retinopathy metadata & 384px transform
│   ├── eye_trainer_v5.py          # Eye Fundus ResNet50 trainer
│   ├── skin_meta.json             # Skin lesion classes & ISIC metadata
│   └── skin_trainer.py            # Skin lesion ResNet50 trainer
├── dqn/                           # Deep Q-Network reinforcement learning package
│   ├── __init__.py
│   ├── config.py                  # State dimensions, action space, and reward matrix
│   ├── environment.py             # Simulated clinical MDP triage environment
│   ├── evaluate.py                # Policy evaluation against clinical benchmarks
│   ├── inference.py               # Runtime DQNTriageInference execution engine
│   ├── model.py                   # PyTorch Double Q-Network architecture
│   ├── replay_buffer.py           # Experience replay buffer
│   └── trainer.py                 # Double DQN training pipeline
├── frontend/                      # Modern React 19 + TypeScript + Vite frontend
│   ├── public/                    # Static assets & test medical scans
│   ├── src/
│   │   ├── api/                   # Type-safe API client (client.ts, types.ts)
│   │   ├── components/            # UI components (AnalysisDashboard, XaiViewer, 3D Scene)
│   │   ├── App.css
│   │   ├── App.tsx                # Single-pass multi-modal clinical diagnostic workflow
│   │   ├── index.css              # Custom design system tokens
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── xai/                           # Explainable AI attribution engine
│   ├── __init__.py                # Dispatcher for Grad-CAM, LIME, and SHAP
│   ├── common.py                  # Target layer maps & normalization utilities
│   ├── gradcam.py                 # Native Grad-CAM gradient calculation
│   ├── lime_explainer.py          # LIME segmentation & perturbation explainer
│   └── shap_explainer.py          # SHAP partition explainer
├── scripts/
│   └── download_weights.py        # Weight download and placement verification script
├── .env.example                   # Safe environment template with placeholder keys
├── .gitignore                     # Clean production gitignore
├── app.py                         # Streamlit alternative prototype UI
├── dqn_meta.json                  # DQN policy validation metrics
├── dqn_model.pth                  # Lightweight DQN policy weights (530 KB)
├── gradcam.py                     # Backward-compatibility shim
├── main.py                        # FastAPI production entry point
├── medvision_inference.py          # Core multi-modal inference & Gemma synthesizer
├── requirements.txt               # Unified backend dependencies
├── requirements-xai.txt           # Optional XAI dependencies
├── symptoms2.csv                  # 389-symptom clinical dictionary
└── README.md                      # Primary project documentation
```

---

## ⚠️ Scientific Integrity & Clinical Disclaimer

**MedVision AI is a research and educational prototype.**  
- The diagnostic models and triage algorithms are intended to investigate multi-modal fusion, asymmetric clinical loss matrices, and model interpretability.
- This software has not received certification or clearance from the U.S. Food and Drug Administration (FDA), CE marking, or any national health authority for autonomous clinical use.
- All AI predictions, attributions, and generated summaries **must be evaluated and confirmed by a certified physician or licensed healthcare provider**.
- If a patient presents with an acute emergency, immediately contact local emergency medical services (e.g., dial 108 / 911 / 112).

---

## 👨‍💻 Author & Contributions

**Sarvesh Bharat** — Computer Science and Artificial Intelligence Engineer  
- GitHub: [@SARVESHBHARAT](https://github.com/SARVESHBHARAT)  
- Project Repository: [https://github.com/SARVESHBHARAT/MedVision-AI](https://github.com/SARVESHBHARAT/MedVision-AI)
