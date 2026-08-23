# FinShield AI — Model 1: KYC Document Forgery Detector

> **An end-to-end deep learning pipeline for detecting fake KYC documents (Aadhaar & PAN cards) in digital banking systems.**

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Architecture](#project-architecture)
- [Model Design](#model-design)
- [Dataset Generation Pipelines](#dataset-generation-pipelines)
- [Training Strategy](#training-strategy)
- [Evaluation and Explainability](#evaluation-and-explainability)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)

---

## Overview

FinShield AI Model 1 is a research-grade fraud detection system built to identify forged Indian KYC documents — specifically Aadhaar and PAN cards — as they appear in digital onboarding flows.

The system combines:
1. **A synthetic dataset generation engine** that produces realistic real and forged document images using DeepFace-powered face analysis, procedural composition, and multi-category tampering simulation.
2. **A dual-branch neural network** that simultaneously analyses the visual (spatial) and frequency-domain (DCT) features of each document to classify it as `real (0)` or `fake (1)`.
3. **Explainability via Grad-CAM** to visualize exactly which document regions the model focuses on during inference.

---

## Key Results

Evaluated on a held-out test set of **144 samples** (identity-aware split, zero face leakage):

| Metric     | Score  |
|------------|--------|
| **Accuracy**   | 86.81% |
| **Precision**  | 82.09% |
| **Recall**     | 88.71% |
| **F1 Score**   | 85.27% |
| **AUC-ROC**    | **92.19%** |

> The model achieves a **0.922 AUC-ROC**, meaning it reliably separates real from forged documents across all decision thresholds.

---

## Project Architecture

```
fin_modal1/
+-- src/                        # Dataset generation pipelines
|   +-- pipeline.py             # Aadhaar card orchestrator
|   +-- pan_pipeline.py         # PAN card orchestrator
|   +-- card_composer.py        # Aadhaar card image compositor
|   +-- pan_card_composer.py    # PAN card image compositor
|   +-- augmentor.py            # Post-render forgery augmentations
|   +-- face_analyzer.py        # DeepFace gender+age analysis
|   +-- template_extractor.py   # Clean template extraction
|   +-- data_utils.py           # Name/DOB/Aadhaar number generators
|   +-- config.py               # Pipeline configuration
+-- training/                   # Model training and evaluation
|   +-- model.py                # DualBranchForgeryDetector architecture
|   +-- train.py                # Two-phase training loop
|   +-- evaluate.py             # Metrics + plots
|   +-- dataset.py              # ForgeryDataset + identity-aware splitting
|   +-- grad_cam.py             # Grad-CAM explainability
|   +-- dct_utils.py            # DCT frequency feature computation
|   +-- config.py               # Training hyperparameters
+-- output/                     # Generated Aadhaar dataset + CSV
+-- output_pan/                 # Generated PAN dataset + CSV
+-- checkpoints/                # Saved model checkpoints
+-- results/                    # Metrics, plots, confusion matrix
+-- requirements.txt
```

---

## Model Design

### Dual-Branch Forgery Detector

The core model (`DualBranchForgeryDetector`) is a custom dual-stream architecture that processes every document image through **two parallel branches** simultaneously, then fuses their representations:

```
Input Image (224x224 RGB)
        |
        +-----------------------+
        |                       |
        v                       v
 [Spatial Branch]        [Frequency Branch]
 EfficientNet-B0         DCT-CNN (3-layer Conv2D)
 ImageNet pretrained     Trained from scratch
        |                       |
        v                       v
  1280-d features          128-d features
        |                       |
        +-----------+-----------+
                    |
               concat -> 1408-d
                    |
            [Fusion Head]
     Linear(1408->256) + BN + ReLU + Dropout(0.4)
     Linear(256->64)   + BN + ReLU + Dropout(0.4)
     Linear(64->2)
                    |
               Logits [Real, Fake]
```

**Why two branches?**
- The **spatial branch** learns visual forgery artifacts: inconsistent fonts, misaligned face regions, unnatural textures.
- The **frequency branch** learns high-frequency DCT anomalies: JPEG re-compression artifacts, copy-paste traces, and GAN synthesis fingerprints that are invisible to the naked eye but detectable as DCT coefficient discontinuities.

---

### Spatial Branch (EfficientNet-B0)

- **Backbone**: `timm` EfficientNet-B0, pretrained on ImageNet.
- **Classifier head removed** (`num_classes=0`, `global_pool="avg"`).
- **Output**: 1280-dimensional feature vector per image.
- Weights **frozen in Phase 1**, **fine-tuned at LR=1e-5** in Phase 2 to preserve ImageNet representations.

---

### Frequency Branch (DCT-CNN)

A lightweight, custom 3-layer convolutional network trained from scratch on **Discrete Cosine Transform (DCT) coefficient maps**:

```
Input: (B, 3, 224, 224) — per-channel DCT coefficients (log-scaled, normalized)
  -> Conv2d(3->32, k=3) + BN + ReLU + MaxPool2d   -> 112x112
  -> Conv2d(32->64, k=3) + BN + ReLU + MaxPool2d  -> 56x56
  -> Conv2d(64->128, k=3) + BN + ReLU + AdaptiveAvgPool -> 1x1
  -> Linear(128 -> 128)
Output: (B, 128) — frequency feature vector
```

DCT features expose compression artifacts and high-frequency inconsistencies that GAN-generated faces or digitally edited text fields leave behind, even after downsampling.

---

### Fusion Head

```
in_dim = 1280 (spatial) + 128 (freq) = 1408

Linear(1408 -> 256) -> BatchNorm1d -> ReLU -> Dropout(0.4)
Linear(256  ->  64) -> BatchNorm1d -> ReLU -> Dropout(0.4)
Linear(64   ->   2)   # logits: [real, fake]
```

---

### Focal Loss

**Focal Loss** (alpha=0.25, gamma=2.0) combined with **inverse-frequency class weights** computed from the training split:

```
FL(pt) = alpha * (1 - pt)^gamma * CE(logits, targets)
```

Focal Loss down-weights easy, well-classified examples and forces the model to focus training signal on hard-to-classify boundary cases.

---

## Dataset Generation Pipelines

### Aadhaar Card Pipeline (`src/pipeline.py`)

A 6-step procedural generation process:

| Step | Description |
|------|-------------|
| **1** | Copy original real Aadhaar card images -> `output/real/` (label=0) |
| **2** | Extract clean card templates (mask text/face regions with background colour) |
| **3** | Analyse TPDNE face images with **DeepFace** (gender + age estimation) -> cached |
| **4** | Generate real synthetic cards (label=0) — semantically consistent face/name/DOB |
| **5** | Generate fake/forged cards (label=1) — intentional mismatches + visual augmentations |
| **6** | Write labeled `output/dataset.csv` |

### PAN Card Pipeline (`src/pan_pipeline.py`)

Mirrors the Aadhaar pipeline for PAN cards with:

- PAN numbers in **official Indian government format**: `AAAPL1234C`
  - Pos 1-3: Random uppercase letters | Pos 4: Entity type (`P` = person)
  - Pos 5: First letter of surname | Pos 6-9: Sequential digits (0001-9999) | Pos 10: Check letter
- Fake PAN numbers use wrong digit count, wrong length, or mixed-case attacks.
- Father and applicant share the **same surname** for realistic family consistency.

---

### Forgery Categories

Each fake card is assigned **2-3 tampering categories** drawn from a weighted probability distribution:

**Semantic-level (card composer)**

| Category | Attack |
|----------|--------|
| `semantic` | Gender swap, age mismatch (+/-25 years), invalid document number |
| `partial_editing` | Single-field text replacement with subtle font/position shift artifacts |
| `face_tampering` | Face brightness alteration (spliced face simulation) |
| `text_tampering` | Font inconsistency + character spacing anomalies on 1-2 fields |

**Image-level (post-render augmentor)**

| Category | Attacks Applied |
|----------|-----------------|
| `image_quality` | JPEG recompression (quality 15-45), Gaussian blur, noise, HSV colour jitter |
| `structural` | Affine warp (+/-5 degree rotation + shear) |
| `border_crop` | Copy-paste border artifact rectangles, edge crop + resize |
| `text_tampering` | Text region blur (selective smudge/erasure simulation) |

---

### Anti-Leakage Splitting

**Identity-aware group splitting** using `sklearn.model_selection.GroupShuffleSplit`.

All documents sharing the same `face_file` (TPDNE image) are assigned to the **same split group**, preventing the model from memorizing face identities instead of learning forgery patterns.

```
Split ratios: 70% / 15% / 15%  (train / val / test)
Verified: 0 face leakage across any split boundary
```

---

## Training Strategy

### Phase 1 — Backbone Frozen (Warm-Up)

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 16 |
| Learning Rate | 1e-3 (AdamW) |
| Scheduler | CosineAnnealingWarmRestarts (T0=2, T_mult=2) |
| Trainable | Frequency branch + Fusion head only |
| Early Stop | Disabled (fixed warm-up) |

The EfficientNet-B0 backbone is completely frozen. Only the frequency branch and fusion head are trained, adapting to the document domain without corrupting ImageNet representations.

---

### Phase 2 — Full Fine-Tune (Differential LR)

| Parameter | Value |
|-----------|-------|
| Epochs (max) | 30 |
| Batch Size | 16 |
| LR — Backbone | **1e-5** (very low, protects pretrained weights) |
| LR — Freq Branch | 5e-4 |
| LR — Fusion Head | 1e-3 |
| Scheduler | CosineAnnealingWarmRestarts (T0=5, T_mult=2) |
| Weight Decay | 1e-4 |
| Gradient Clipping | 1.0 |
| Early Stop Patience | 7 epochs |

Differential learning rates ensure the pretrained backbone adapts slowly while newly initialized layers learn more aggressively. Best checkpoint is saved only when `val_loss` strictly improves (exact float comparison, never overwritten by a worse result).

**Checkpoint management:**
- `checkpoints/best_model.pt` — globally best validation loss checkpoint
- `checkpoints/best_phase1_*.pt` / `best_phase2_*.pt` — per-phase bests
- `checkpoints/final_model.pt` — model state at end of training
- `results/training_history.json` — written after **every epoch** to prevent loss on crash

---

## Evaluation and Explainability

### Evaluation (`training/evaluate.py`)

```bash
python -m training.evaluate
```

Produces:
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap -> `results/confusion_matrix.png`
- ROC curve + AUC -> `results/roc_curve.png`
- Training history curves (loss + accuracy, phase boundary marked) -> `results/training_curves.png`
- JSON metrics -> `results/test_metrics.json`

### Grad-CAM Explainability (`training/grad_cam.py`)

Grad-CAM hooks the **last convolutional layer of EfficientNet-B0** (`conv_head`) to produce spatial attention heatmaps:

```bash
python -m training.grad_cam --image path/to/document.jpg
```

Outputs a 3-panel visualization:
1. Original document image
2. Raw Grad-CAM heatmap (jet colormap)
3. Overlay with prediction label and confidence score

This allows inspection of whether the model is attending to meaningful forgery regions (face splice boundaries, tampered text fields) rather than spurious background features.

---

## Project Structure

```
fin_modal1/
|
+-- src/
|   +-- __init__.py
|   +-- augmentor.py            # PIL-based fake augmentations (8 attack types)
|   +-- card_composer.py        # Aadhaar card composition + tampering
|   +-- config.py               # Pipeline paths, targets, augmentation probs
|   +-- data_utils.py           # Indian name/DOB/Aadhaar generators
|   +-- download_faces.py       # TPDNE face image downloader
|   +-- face_analyzer.py        # DeepFace wrapper (gender + age, cached JSON)
|   +-- pan_card_composer.py    # PAN card composition + tampering
|   +-- pan_config.py           # PAN pipeline configuration
|   +-- pan_pipeline.py         # PAN card dataset generation orchestrator
|   +-- pipeline.py             # Aadhaar dataset generation orchestrator
|   +-- template_extractor.py   # Template masking + metadata extraction
|
+-- training/
|   +-- __init__.py
|   +-- config.py               # All training hyperparameters (centralized)
|   +-- dataset.py              # ForgeryDataset + identity-aware GroupShuffleSplit
|   +-- dct_utils.py            # Per-channel DCT computation + normalization
|   +-- evaluate.py             # Metrics, confusion matrix, ROC, history plots
|   +-- grad_cam.py             # Grad-CAM with EfficientNet-B0 forward/backward hooks
|   +-- model.py                # DualBranchForgeryDetector + FocalLoss
|   +-- train.py                # Two-phase training loop with checkpointing
|
+-- output/                     # Aadhaar dataset images + dataset.csv
+-- output_pan/                 # PAN dataset images + pan_dataset.csv
+-- checkpoints/                # .pt checkpoint files
+-- results/                    # Evaluation outputs (PNG plots + metrics JSON)
+-- data/                       # Raw real Aadhaar images + TPDNE faces + templates
|
+-- audit.py                    # Dataset audit utility
+-- calibrate.py                # Aadhaar template calibration
+-- calibrate_pan.py            # PAN template calibration
+-- verify_checkpoint.py        # Checkpoint integrity checker
+-- requirements.txt
```

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note**: `NUM_WORKERS=0` is set by default in `training/config.py` for PyTorch multiprocessing compatibility on Windows.

### Data Setup

Place original real Aadhaar card images in `data/real/` and TPDNE face images in `data/faces/`. The PAN sample card is already at `src/sample-pan-card.jpg`.

---

## Usage

### 1. Generate Datasets

```bash
# Aadhaar dataset (full run)
python -m src.pipeline

# PAN dataset
python -m src.pan_pipeline --n-real 300 --n-fake 300

# Dry run (face analysis only, no card images written)
python -m src.pipeline --dry-run

# Skip template extraction (reuse existing templates)
python -m src.pipeline --skip-extract --verbose
```

### 2. Train the Model

```bash
# Default configuration
python -m training.train

# Custom epochs/batch size
python -m training.train --phase1-epochs 5 --phase2-epochs 30 --phase1-batch 16
```

### 3. Evaluate

```bash
# Evaluate best checkpoint
python -m training.evaluate

# Evaluate a specific checkpoint
python -m training.evaluate --checkpoint checkpoints/best_model.pt
```

### 4. Grad-CAM Visualization

```bash
python -m training.grad_cam --image path/to/document.jpg
python -m training.grad_cam --image path/to/document.jpg --checkpoint checkpoints/best_model.pt
```

### 5. Verify Checkpoint

```bash
python verify_checkpoint.py
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch >= 2.0 |
| Backbone | EfficientNet-B0 via `timm` >= 0.9 |
| Face Analysis | DeepFace >= 0.0.79 (gender + age estimation) |
| Image Processing | OpenCV >= 4.8, Pillow >= 10.0 |
| Data Handling | NumPy, pandas |
| ML Utilities | scikit-learn (GroupShuffleSplit, metrics) |
| Visualization | matplotlib, seaborn |
| Scheduler | CosineAnnealingWarmRestarts |
| Optimizer | AdamW with differential learning rates |
| Loss | Custom Focal Loss (alpha=0.25, gamma=2.0) |

---

## Key Design Decisions

- **EfficientNet-B0 over ViT**: More parameter-efficient for this dataset size; stronger inductive bias for local texture patterns critical in document forensics. ViTs need much larger datasets to converge.
- **DCT frequency branch**: JPEG compression, GAN generation, and image editing all leave artifacts in the frequency domain that spatial CNNs may miss. The DCT branch is purpose-built to detect these invisible signatures.
- **Focal Loss**: The difficulty distribution across fakes is heavily skewed (some are very subtle). Focal Loss emphasizes hard examples during training.
- **Identity-aware splitting**: Without group splitting, inflated test accuracy from face identity memorization would render the evaluation meaningless.

---

*FinShield AI Model 1 — Built for robust KYC fraud detection in digital banking.*
