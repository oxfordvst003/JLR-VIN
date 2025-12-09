# JLR VIN Detection - Comprehensive Model Testing Plan

**Project:** JLR Vehicle Identification Number (VIN) Character Detection  
**Version:** 1.0  
**Date:** December 9, 2025  
**Repository:** [oxfordvst003/JLR-VIN](https://dagshub.com/oxfordvst003/JLR-VIN)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Models Overview](#2-models-overview)
3. [Data Versioning Strategy](#3-data-versioning-strategy)
4. [Testing Phases](#4-testing-phases)
5. [Metrics & Evaluation Framework](#5-metrics--evaluation-framework)
6. [Hyperparameter Tuning Strategy](#6-hyperparameter-tuning-strategy)
7. [Explainability Framework](#7-explainability-framework)
8. [Observability & Monitoring](#8-observability--monitoring)
9. [Model Governance](#9-model-governance)
10. [Execution Schedule](#10-execution-schedule)
11. [Appendices](#11-appendices)

---

## 1. Executive Summary

This document outlines the comprehensive testing plan for two machine learning models in the JLR VIN detection pipeline:

| Model | Purpose | Architecture | Status |
|-------|---------|--------------|--------|
| **YOLOv8s** | Character Detection | Ultralytics YOLOv8 Small | Trained |
| **TrOCR** | Character Recognition | Vision Encoder-Decoder (Microsoft) | Trained |

### Pipeline Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input Image   │ ──► │  YOLO Detection │ ──► │  TrOCR OCR      │ ──► VIN String
│   (VIN Plate)   │     │  (33 Classes)   │     │  (Seq2Seq)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Testing Objectives
- ✅ Validate model performance after data cleaning (deduplication, 60:20:20 split)
- ✅ Establish baseline metrics with full MLOps tracking
- ✅ Compare fixed-seed vs Optuna-tuned YOLO models
- ✅ Ensure reproducibility and compliance for production deployment

---

## 2. Models Overview

### 2.1 YOLOv8s - Character Detection Model

**Location:** `YOLO_trained_fixed_seed/vin_yolov8s_best_fixed_seed.pt`  
**Alternative:** `YOLO_trained_optuna/vin_yolov8s_best_autotuned.pt`

| Parameter | Fixed Seed Model | Optuna Model |
|-----------|-----------------|--------------|
| Base Model | yolov8s.pt | yolov8s.pt |
| Image Size | 1024×1024 | 1024×1024 |
| Epochs | 50 | 100 (after 15 HPO trials) |
| Batch Size | 16 | Variable (8-16) |
| Optimizer | AdamW | AdamW |
| Seed | 12345 | N/A |
| Classes | 33 (0-9, A-Z excl. I,O,Q, *) | 33 |

**Class Distribution:**
```python
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z', '*'
]
```

### 2.2 TrOCR - Character Recognition Model

**Location:** `best_trocr_v1/best_trocr_v1/`

| Parameter | Value |
|-----------|-------|
| Base Model | microsoft/trocr-base-printed |
| Architecture | VisionEncoderDecoderModel |
| Task | Sequence-to-Sequence OCR |
| Output Format | `*XXXXXXXXXXXXXXXXX*` (17 chars + asterisks) |
| Training Data | 2,385 train / 417 val images |

---

## 3. Data Versioning Strategy

### 3.1 Current Data State

| Split | Images | Labels | Ratio |
|-------|--------|--------|-------|
| Train | 1,525 | 1,525 | 60% |
| Val | 508 | 508 | 20% |
| Test | 510 | 510 | 20% |
| **Total** | **2,543** | **2,543** | **100%** |

### 3.2 DVC Tracking

```yaml
# Data tracked via DVC
Data/
├── train/images/  # 1,525 images
├── train/labels/  # 1,525 YOLO format labels
├── val/images/    # 508 images
├── val/labels/    # 508 labels
├── test/images/   # 510 images
└── test/labels/   # 510 labels
```

**DVC Commands:**
```bash
# Track data version
dvc add Data/
git add Data.dvc .gitignore
git commit -m "Data v1.0: Deduplicated, 60:20:20 split"

# Push to DagsHub storage
dvc push -r dagshub

# Tag version
git tag -a data-v1.0 -m "Clean data: 2543 images, no duplicates"
```

### 3.3 Data Versioning Checkpoints

| Version | Description | Data Hash | Date |
|---------|-------------|-----------|------|
| v0.1 | Raw data (with duplicates) | `abc123...` | Initial |
| v1.0 | Deduplicated, 60:20:20 split | `{current_hash}` | 2025-12-09 |
| v1.1 | Post-testing adjustments | TBD | Planned |

---

## 4. Testing Phases

### Phase 1: Baseline Validation (YOLO)

**Duration:** 2-4 hours  
**Objective:** Validate YOLO models on cleaned data

```bash
# Test Fixed Seed Model
python -m mlops.pipeline evaluate \
    --model YOLO_trained_fixed_seed/vin_yolov8s_best_fixed_seed.pt \
    --split test \
    --conf 0.25 \
    --iou 0.5

# Test Optuna Model
python -m mlops.pipeline evaluate \
    --model YOLO_trained_optuna/vin_yolov8s_best_autotuned.pt \
    --split test \
    --conf 0.25 \
    --iou 0.5
```

**Expected Outputs:**
- mAP50, mAP50-95, Precision, Recall per model
- Confusion matrix
- Per-class performance breakdown

### Phase 2: TrOCR Evaluation

**Duration:** 1-2 hours  
**Objective:** Validate TrOCR OCR accuracy

```python
# TrOCR Evaluation Script
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load model
processor = TrOCRProcessor.from_pretrained("best_trocr_v1/best_trocr_v1")
model = VisionEncoderDecoderModel.from_pretrained("best_trocr_v1/best_trocr_v1")
model.eval()

# Metrics to compute
# - Character Error Rate (CER)
# - Word Error Rate (WER) - full VIN accuracy
# - Per-character accuracy
```

### Phase 3: End-to-End Pipeline Testing

**Duration:** 3-4 hours  
**Objective:** Test full YOLO → TrOCR pipeline

```
Input Image → YOLO Detection → Crop Characters → TrOCR → VIN String
```

**Test Cases:**
| Test Case | Description | Pass Criteria |
|-----------|-------------|---------------|
| TC-001 | Clean VIN plate, good lighting | 100% accuracy |
| TC-002 | Partially obscured VIN | ≥95% accuracy |
| TC-003 | Angled capture | ≥90% accuracy |
| TC-004 | Low contrast | ≥85% accuracy |
| TC-005 | Multiple VINs in frame | All detected |

### Phase 4: Retraining with MLOps (Optional)

**Duration:** 8-12 hours  
**Objective:** Full retraining with new data splits

```bash
# Full MLOps training pipeline
python -m mlops.pipeline train \
    --model yolov8s.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 1024 \
    --lr0 0.001 \
    --seed 12345 \
    --description "Retraining on deduplicated data v1.0"
```

---

## 5. Metrics & Evaluation Framework

### 5.1 YOLO Detection Metrics

| Metric | Formula | Target | Weight |
|--------|---------|--------|--------|
| **mAP50** | Mean AP @ IoU=0.5 | ≥0.90 | 30% |
| **mAP50-95** | Mean AP @ IoU=0.5:0.95 | ≥0.75 | 25% |
| **Precision** | TP/(TP+FP) | ≥0.85 | 20% |
| **Recall** | TP/(TP+FN) | ≥0.85 | 20% |
| **F1 Score** | 2×P×R/(P+R) | ≥0.85 | 5% |

### 5.2 TrOCR Recognition Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **CER** | Character Error Rate | ≤2% |
| **WER** | Word Error Rate (full VIN) | ≤5% |
| **Exact Match** | Complete VIN correct | ≥95% |
| **Per-Char Accuracy** | Individual character accuracy | ≥99% |

### 5.3 Combined Pipeline Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **End-to-End Accuracy** | Full pipeline correct VIN | ≥92% |
| **Inference Time** | Per-image processing | ≤500ms |
| **Throughput** | Images per second | ≥2 img/s |

### 5.4 MLflow Metric Logging

```python
# Automatic logging via MLOps pipeline
mlflow_tracker.log_metrics({
    "mAP50": 0.923,
    "mAP50_95": 0.781,
    "precision": 0.889,
    "recall": 0.876,
    "f1_score": 0.882,
    "inference_time_ms": 234,
})
```

---

## 6. Hyperparameter Tuning Strategy

### 6.1 YOLO Hyperparameter Space

| Parameter | Search Space | Current Best |
|-----------|--------------|--------------|
| `lr0` | LogUniform[1e-4, 1e-2] | 0.001 |
| `lrf` | Uniform[0.01, 0.2] | 0.01 |
| `momentum` | Uniform[0.85, 0.98] | 0.937 |
| `weight_decay` | LogUniform[1e-4, 1e-2] | 0.0005 |
| `warmup_epochs` | Integer[1, 5] | 3 |
| `hsv_h` | Uniform[0.0, 0.1] | 0.015 |
| `hsv_s` | Uniform[0.0, 0.9] | 0.7 |
| `hsv_v` | Uniform[0.0, 0.9] | 0.4 |
| `scale` | Uniform[0.0, 0.9] | 0.5 |
| `translate` | Uniform[0.0, 0.2] | 0.1 |
| `mosaic` | Uniform[0.0, 1.0] | 1.0 |

### 6.2 Optuna Configuration

```python
# params.yaml configuration
optuna:
  enabled: true
  n_trials: 15
  study_name: vin_yolo_hpo
  pruning: true
  direction: maximize
  metric: mAP50
```

### 6.3 TrOCR Hyperparameters (Future)

| Parameter | Default | Search Space |
|-----------|---------|--------------|
| `learning_rate` | 5e-5 | [1e-5, 1e-4] |
| `batch_size` | 8 | [4, 8, 16] |
| `num_epochs` | 3 | [3, 5, 10] |
| `warmup_steps` | 500 | [100, 1000] |
| `weight_decay` | 0.01 | [0.0, 0.1] |

---

## 7. Explainability Framework

### 7.1 YOLO Explainability

#### 7.1.1 Attention Heatmaps
```bash
python -m mlops.pipeline explain --model path/to/model.pt --samples 20
```

**Outputs:**
- Grad-CAM style attention visualizations
- Detection confidence overlays
- Bounding box attribution maps

#### 7.1.2 Error Analysis

```python
from mlops import error_analyzer

error_analyzer.load_model("YOLO_trained_fixed_seed/vin_yolov8s_best_fixed_seed.pt")
analysis = error_analyzer.analyze_predictions(
    images_dir="Data/test/images",
    labels_dir="Data/test/labels",
    output_dir="mlops/outputs/error_analysis"
)

# Outputs:
# - Confusion matrix (33×33 classes)
# - False positive analysis
# - False negative analysis  
# - Per-class precision/recall/F1
```

#### 7.1.3 Confidence Distribution

```python
from mlops import confidence_analyzer

confidence_analyzer.load_model("path/to/model.pt")
analysis = confidence_analyzer.analyze_confidence_distribution(
    images_dir="Data/test/images",
    output_dir="mlops/outputs/confidence"
)

# Outputs:
# - Confidence histogram
# - Per-class confidence distributions
# - Low-confidence sample identification
```

### 7.2 TrOCR Explainability

#### 7.2.1 Attention Visualization
```python
# Encoder-decoder attention weights
outputs = model.generate(
    pixel_values,
    output_attentions=True,
    return_dict_in_generate=True
)
# Visualize cross-attention between image patches and text tokens
```

#### 7.2.2 Character-Level Analysis
- Per-character confidence scores
- Confusion pairs (e.g., O↔0, I↔1, B↔8)
- Sequence position accuracy

### 7.3 Model Cards

Automatically generated documentation:

```json
{
  "model_details": {
    "name": "VIN Character Detection Model",
    "architecture": "YOLOv8s",
    "task": "Object Detection (Character Recognition)"
  },
  "intended_use": {
    "primary_use": "VIN character detection on vehicle plates",
    "out_of_scope": ["General OCR", "Non-VIN text"]
  },
  "training_data": {
    "source": "JLR/OVST VIN images",
    "size": "2,543 images",
    "split": "60:20:20"
  },
  "evaluation_results": {
    "mAP50": 0.923,
    "precision": 0.889,
    "recall": 0.876
  },
  "ethical_considerations": [
    "Contains vehicle identification data",
    "Data protection policies apply"
  ]
}
```

---

## 8. Observability & Monitoring

### 8.1 Real-Time Monitoring

#### Resource Metrics (Collected every 5 seconds)
```
┌─────────────────────────────────────────────────────────────┐
│  System Metrics                                             │
├─────────────────────────────────────────────────────────────┤
│  CPU Usage:        ████████░░░░  67%                        │
│  Memory:           ██████░░░░░░  52% (8.3/16 GB)            │
│  GPU Memory:       ████████████  95% (3.8/4 GB)             │
│  GPU Utilization:  ██████████░░  83%                        │
│  Disk I/O:         ███░░░░░░░░░  28%                        │
└─────────────────────────────────────────────────────────────┘
```

#### Training Progress
```python
from mlops import resource_monitor, metrics_collector

# Start monitoring
resource_monitor.start()

# During training, metrics are automatically collected:
# - Epoch loss (box, cls, dfl)
# - mAP50, mAP50-95
# - Learning rate
# - GPU temperature and power
```

### 8.2 Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| `HIGH_GPU_MEMORY` | GPU memory > 95% | WARNING |
| `HIGH_CPU_USAGE` | CPU > 95% for 5min | WARNING |
| `LOSS_SPIKE` | Loss increases 5× | WARNING |
| `NAN_LOSS` | Loss = NaN | CRITICAL |
| `TRAINING_STALL` | No progress 30min | WARNING |

### 8.3 Logging Configuration

```yaml
# mlops/mlops_config.py
observability:
  log_level: INFO
  log_format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
  metrics_interval: 10  # seconds
  enable_gpu_monitoring: true
  enable_alerts: true
```

### 8.4 Dashboards

**MLflow UI:**
```bash
mlflow ui --port 5000
# Access at http://localhost:5000
```

**DagsHub Dashboard:**
- Experiments: https://dagshub.com/oxfordvst003/JLR-VIN/experiments
- Data: https://dagshub.com/oxfordvst003/JLR-VIN/data

---

## 9. Model Governance

### 9.1 Model Lifecycle Stages

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ DEVELOPMENT  │ ─► │   STAGING    │ ─► │  PRODUCTION  │ ─► │   ARCHIVED   │
│              │    │              │    │              │    │              │
│ • Training   │    │ • Validation │    │ • Deployed   │    │ • Historical │
│ • Tuning     │    │ • Review     │    │ • Monitored  │    │ • Reference  │
│ • Testing    │    │ • Approval   │    │ • Active     │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 9.2 Model Registry

```python
from mlops import model_registry, ModelStage

# Register model
model = model_registry.register_model(
    model_id="vin_character_detector",
    model_path="path/to/model.pt",
    metrics={"mAP50": 0.923, "precision": 0.889},
    parameters={"epochs": 50, "batch": 16},
    data_hash="abc123...",
    description="YOLO v8s trained on deduplicated data v1.0",
    created_by="ml_engineer"
)

# List versions
versions = model_registry.list_models("vin_character_detector")
# [v1, v2, v3, ...]

# Get production model
prod_model = model_registry.get_production_model("vin_character_detector")
```

### 9.3 Approval Workflow

```python
from mlops import approval_workflow, ModelStage

# Request promotion to production
request = approval_workflow.request_approval(
    model_id="vin_character_detector",
    version="v3",
    target_stage=ModelStage.PRODUCTION,
    requested_by="ml_engineer",
    justification="Model meets all quality thresholds. mAP50=0.923"
)

# Reviewer approves
approval_workflow.review_request(
    request_id=request.request_id,
    approved=True,
    reviewer="tech_lead",
    comments="Approved. Metrics exceed requirements."
)
```

### 9.4 Compliance Checks

| Check | Requirement | Auto-Validated |
|-------|-------------|----------------|
| Minimum mAP50 | ≥ 0.85 | ✅ |
| Minimum Precision | ≥ 0.80 | ✅ |
| Minimum Recall | ≥ 0.80 | ✅ |
| Data Lineage | Hash recorded | ✅ |
| Documentation | Model card exists | ✅ |
| Approval | Approved by reviewer | ✅ |

```python
from mlops import compliance_checker

result = compliance_checker.check_model(model)
# {
#   "passed": true,
#   "checks": [
#     {"name": "minimum_metrics", "passed": true},
#     {"name": "data_lineage", "passed": true},
#     {"name": "documentation", "passed": true},
#     {"name": "approval_status", "passed": true}
#   ]
# }
```

### 9.5 Model Lineage

```python
from mlops import lineage_tracker

# Record training lineage
lineage_tracker.record_training_lineage(
    model_id="vin_character_detector",
    version="v3",
    data_source={
        "hash": "abc123",
        "path": "Data/",
        "splits": {"train": 1525, "val": 508, "test": 510}
    },
    code_version="commit_sha",
    parent_model="v2",
    training_config={"epochs": 50, "lr0": 0.001}
)

# Query lineage
lineage = lineage_tracker.get_lineage("vin_character_detector", "v3")
```

---

## 10. Execution Schedule

### 10.1 Testing Timeline

| Phase | Task | Duration | Owner |
|-------|------|----------|-------|
| **Day 1** | Environment setup, dependency verification | 2 hrs | ML Engineer |
| **Day 1** | YOLO baseline evaluation (both models) | 3 hrs | ML Engineer |
| **Day 1** | TrOCR evaluation | 2 hrs | ML Engineer |
| **Day 2** | End-to-end pipeline testing | 4 hrs | ML Engineer |
| **Day 2** | Explainability analysis | 3 hrs | ML Engineer |
| **Day 3** | Results documentation | 2 hrs | ML Engineer |
| **Day 3** | Governance review | 2 hrs | Tech Lead |

### 10.2 Quick Start Commands

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Verify setup
python setup_mlops.py

# 3. Check current status
python -m mlops.pipeline status

# 4. Run YOLO evaluation
python -m mlops.pipeline evaluate --split test

# 5. Run explainability
python -m mlops.pipeline explain --samples 20

# 6. View results in MLflow
mlflow ui --port 5000
```

### 10.3 Deliverables Checklist

- [ ] YOLO Fixed Seed Model Evaluation Report
- [ ] YOLO Optuna Model Evaluation Report
- [ ] TrOCR Evaluation Report
- [ ] End-to-End Pipeline Performance Report
- [ ] Explainability Analysis (Heatmaps, Error Analysis)
- [ ] Model Cards (JSON + Markdown)
- [ ] Governance Compliance Report
- [ ] Final Recommendations Document

---

## 11. Appendices

### Appendix A: File Locations

| Asset | Path |
|-------|------|
| YOLO Fixed Seed Model | `YOLO_trained_fixed_seed/vin_yolov8s_best_fixed_seed.pt` |
| YOLO Optuna Model | `YOLO_trained_optuna/vin_yolov8s_best_autotuned.pt` |
| TrOCR Model | `best_trocr_v1/best_trocr_v1/` |
| Data | `Data/` (DVC tracked) |
| MLOps Pipeline | `mlops/pipeline.py` |
| Config | `config.py`, `params.yaml` |
| Training Notebooks | `Yolov8s_OCR_Training_*.ipynb`, `TCOCR_Training_test.ipynb` |

### Appendix B: Environment Requirements

```bash
# Key packages (from requirements.txt)
ultralytics==8.3.221
torch==2.5.0
transformers==4.48.0
dagshub==0.6.3
dvc==3.64.2
mlflow  # (via dagshub)
```

### Appendix C: DagsHub Integration

```bash
# Remote configuration
export DAGSHUB_USER=oxfordvst003
export DAGSHUB_TOKEN=<your_token>

# MLflow tracking
export MLFLOW_TRACKING_URI=https://dagshub.com/oxfordvst003/JLR-VIN.mlflow

# DVC remote
dvc remote add dagshub https://dagshub.com/oxfordvst003/JLR-VIN.dvc
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user oxfordvst003
dvc remote modify dagshub --local password <token>
```

### Appendix D: Contact Information

| Role | Name | Responsibility |
|------|------|----------------|
| ML Engineer | TBD | Training, evaluation, MLOps |
| Tech Lead | TBD | Governance approval |
| Data Owner | TBD | Data quality, versioning |

---

**Document Version History:**

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-09 | Initial creation | ML Team |

---

*This document is part of the JLR VIN Detection MLOps framework. For updates, see the [mlops/README.md](mlops/README.md).*
