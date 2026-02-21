# 🫀 Liver Disease Prediction — End-to-End MLOps on AWS SageMaker

An end-to-end machine learning pipeline for predicting liver disease from patient blood panel data, built on AWS SageMaker with full MLOps capabilities including feature engineering, model comparison, CI/CD, real-time and batch inference, and automated monitoring.

---

## 📋 Table of Contents

- [Business Use Case](#business-use-case)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [Inference](#inference)
- [Monitoring](#monitoring)
- [CI/CD](#cicd)
- [Results](#results)
- [Tech Stack](#tech-stack)

---

## Business Use Case

Liver disease affects millions of patients globally. Early detection through routine blood tests significantly improves patient outcomes. This system automates the classification of patient records as **liver disease** or **no disease** from 10 blood panel markers, providing a scalable screening tool to assist clinical decision-making.

> ⚠️ This system is a **decision-support tool**, not a diagnostic replacement. All predictions should be reviewed by a qualified clinician.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│   S3 (raw CSV) → SKLearn Processor → Feature Store (online+offline) │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
│                                                                  │
│   LiverDiseasePreprocessing                                      │
│          │                                                       │
│          ├──► TrainXGBoost (built-in algorithm)                  │
│          └──► TrainRandomForest (SKLearn estimator)              │
│                     │                                            │
│                     └──► EvaluateModels ──► CheckAccuracy        │
│                                                │                 │
│                                     (accuracy ≥ 0.75)            │
│                                                │                 │
│                                     RegisterLiverDiseaseModel    │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                  DEPLOYMENT & INFERENCE                          │
│                                                                  │
│   Model Registry (approve) ──► Real-Time Endpoint               │
│                          └──► Batch Transform Job               │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│               MONITORING & OBSERVABILITY                         │
│                                                                  │
│   Data Quality Monitor (hourly) + Model Quality Monitor         │
│   CloudWatch Alarms (latency / errors / CPU)                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Liver Patient Dataset (LPD)](https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset) |
| Records | 30,691 patient records |
| Raw features | 10 (Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase, SGPT, SGOT, Total Proteins, Albumin, A/G Ratio) |
| Engineered features | 3 (Bilirubin Ratio, Enzyme Ratio, Age Group) |
| Total features | 13 |
| Target | Binary — 1 = liver disease, 0 = no disease |
| Class split | ~71% disease / ~29% no disease |
| Encoding | ISO-8859-1 (Latin-1) |

### Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `bilirubin_ratio` | Direct / (Total + ε) | Conjugation efficiency indicator |
| `enzyme_ratio` | SGPT / (SGOT + ε) | Liver vs cardiac damage pattern |
| `age_group` | Binned 0–3 | Non-linear age risk encoding |

---

## Project Structure

```
liver-disease-sagemaker/
│
├── liver_disease_sagemaker_pipeline.py   # Main pipeline — all components
├── streamlit_app.py                      # Real-time inference UI
├── requirements.txt                      # Python dependencies
├── README.md
│
├── scripts/                              # Auto-generated by pipeline at runtime
│   ├── preprocessing.py                  # SKLearn processing step
│   ├── rf_train.py                       # Random Forest training
│   ├── evaluate.py                       # Model comparison
│   └── inference.py                      # Endpoint handler
│
└── cicd/                                 # Auto-generated by pipeline at runtime
    ├── buildspec.yml                     # CodeBuild spec
    ├── cicd_trigger.py                   # Trigger pipeline execution
    └── cicd_check_status.py             # Poll execution status
```

---

## Pipeline Components

### 1. Feature Store
- Feature group: `liver-disease-features`
- Online store for real-time retrieval + offline store (S3 + Athena) for training
- Idempotent creation — skips if group already exists

### 2. SageMaker Pipeline (5 steps)

| Step | Type | Description |
|---|---|---|
| `LiverDiseasePreprocessing` | ProcessingStep | Clean, encode, engineer features, split 80/20 |
| `TrainXGBoost` | TrainingStep | Built-in XGBoost algorithm (no custom script) |
| `TrainRandomForest` | TrainingStep | SKLearn Random Forest via custom script |
| `EvaluateModels` | ProcessingStep | Compare both models, output `evaluation.json` |
| `CheckAccuracy` | ConditionStep | Register only if accuracy >= 0.75 |

### 3. Model Registry
- Group: `LiverDiseaseModelGroup`
- Workflow: `PendingManualApproval` → `Approved` → deployed
- Auto-approval wired into pipeline on success

### 4. Deployment
- Real-time endpoint: `liver-disease-endpoint` (ml.m5.large)
- 100% data capture enabled for Model Monitor
- Batch Transform: features-only CSV (label column auto-stripped)

### 5. Monitoring
- **Data Quality Monitor** — hourly, detects feature drift vs training baseline
- **Model Quality Monitor** — hourly, tracks AUC / accuracy / F1
- **CloudWatch Alarms** — latency > 500ms, any 5xx error, CPU > 80%

### 6. CI/CD
- CodePipeline triggers on GitHub push to `main`
- CodeBuild runs unit tests → triggers SageMaker Pipeline
- Live DAG with per-step status, failure reports, and remediation hints

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### AWS Setup

1. **SageMaker execution role** — ensure it has:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
   - `AmazonSageMakerFeatureStoreAccess`

2. **Configure constants** at the top of `liver_disease_sagemaker_pipeline.py`:

```python
BUCKET = "your-sagemaker-bucket"
PREFIX = "liver-disease"
REGION = "us-east-1"
```

3. **Download the dataset** from Kaggle:

```bash
kaggle datasets download -d abhi8923shriv/liver-disease-patient-dataset
unzip liver-disease-patient-dataset.zip
# File needed: Liver_Patient_Dataset__LPD__train.csv
```

---

## Running the Pipeline

### Full end-to-end run

```bash
python liver_disease_sagemaker_pipeline.py
```

This automatically runs all 10 steps:

1. Uploads data to S3
2. Ingests into Feature Store
3. Deletes and recreates pipeline (clears stale cache)
4. Runs pipeline with live DAG output
5. Extracts all ARNs and S3 URIs automatically
6. Approves model and deploys endpoint
7. Creates CloudWatch alarms
8. Runs batch transform (label column auto-stripped)
9. Sets up monitoring schedules
10. Generates monitoring report

### Run individual components

```python
# Extract artifacts from a previous execution
artifacts = extract_pipeline_artifacts("arn:aws:sagemaker:...:execution/...")

# Deploy from existing artifacts
predictor = approve_and_deploy(artifacts, endpoint_name="liver-disease-endpoint")

# Generate monitoring report
generate_monitoring_report("liver-disease-endpoint")

# Cleanup (avoid charges)
cleanup("liver-disease-endpoint")
```

---

## Inference

### Streamlit App

```bash
pip install streamlit
streamlit run streamlit_app.py
```

Enter patient values → click Run Prediction → get probability score, confidence bar, and prediction history.

### Direct endpoint call

```python
import boto3

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

# 13 features — NO label column
# Order: Age, Gender(0/1), TotalBili, DirectBili, AlkPhos,
#        SGPT, SGOT, TotProt, Albumin, AGRatio,
#        BiliRatio, EnzymeRatio, AgeGroup
payload = "65,1,3.5,1.2,450,120,95,5.5,2.8,0.7,0.34,1.26,2"

response = runtime.invoke_endpoint(
    EndpointName="liver-disease-endpoint",
    ContentType="text/csv",
    Body=payload
)

score = float(response["Body"].read())
print(f"Score: {score:.4f}  →  {'liver_disease' if score >= 0.5 else 'no_disease'}")
```

> **Important:** Send 13 features with **no label column**. The model returns a raw probability (0–1). Apply threshold (default 0.5) for binary prediction.

---

## Monitoring

| Monitor | Tracks | Alert Condition |
|---|---|---|
| Data Quality | Feature distributions vs training baseline | Feature drifts beyond constraint |
| Model Quality | Accuracy, AUC, F1, Precision, Recall | Any metric below threshold |

```python
# Generate full report anytime
generate_monitoring_report("liver-disease-endpoint")
```

---

## CI/CD

DAG output on successful run:

```
✅  LiverDiseasePreprocessing     [Succeeded]
    └─► ✅  TrainXGBoost          [Succeeded]
    └─► ✅  TrainRandomForest     [Succeeded]
    └─► ✅  EvaluateModels        [Succeeded]
    └─► ✅  CheckAccuracy         [Succeeded]
         ↳  Condition outcome: TRUE
```

On failure, the DAG prints the exact step, reason, training job ARN, and remediation hints.

---

## Results

| Model | Accuracy | AUC | F1 |
|---|---|---|---|
| **XGBoost** ✅ | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9991 | 1.0000 | 0.9994 |

**Winner: XGBoost** — registered in Model Registry and deployed to endpoint.

> ⚠️ AUC = 1.0 on both models warrants investigation for data leakage before production deployment. A strict patient-level train/test split is recommended.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud | AWS SageMaker, S3, CloudWatch, CodePipeline, CodeBuild |
| Pipeline orchestration | SageMaker Pipelines |
| Feature management | SageMaker Feature Store |
| Model governance | SageMaker Model Registry |
| Training — XGBoost | SageMaker built-in XGBoost 1.5-1 |
| Training — Random Forest | SKLearn 1.0-1 (SageMaker estimator) |
| Monitoring | SageMaker Model Monitor |
| Inference UI | Streamlit |
| Language | Python 3.8+ |

---

## Known Issues

- **CloudWatch dashboard** requires `cloudwatch:PutDashboard` IAM permission — manual creation instructions are printed if unavailable
- **SNS notifications** require `sns:CreateTopic` — silently skipped if unavailable
- Pipeline **deletes and recreates** on every run to clear SageMaker's definition cache — this is intentional
- Batch transform automatically **strips the label column** before scoring
