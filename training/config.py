"""
config.py — Training hyperparameters and paths for Document Forgery Detection.

Centralises every tunable knob so experiments are reproducible.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AADHAAR_CSV     = os.path.join(BASE_DIR, "output", "dataset.csv")
PAN_CSV         = os.path.join(BASE_DIR, "output_pan", "pan_dataset.csv")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

# ─── Dataset ──────────────────────────────────────────────────────────────────
IMG_SIZE        = 224          # EfficientNet-B0 native input size
SPLIT_RATIOS    = (0.70, 0.15, 0.15)   # train / val / test
RANDOM_SEED     = 42

# ImageNet normalisation stats
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# ─── Model ────────────────────────────────────────────────────────────────────
NUM_CLASSES     = 2            # real vs fake
SPATIAL_DIM     = 1280         # EfficientNet-B0 feature dim
FREQ_DIM        = 128          # Frequency branch output dim
FUSION_HIDDEN   = [256, 64]    # FC layers after concat
DROPOUT         = 0.4

# ─── Training — Phase 1 (backbone frozen) ─────────────────────────────────────
PHASE1_EPOCHS   = 5
PHASE1_LR       = 1e-3
PHASE1_BATCH    = 16

# ─── Training — Phase 2 (full fine-tune) ──────────────────────────────────────
PHASE2_EPOCHS   = 30
PHASE2_LR_BACKBONE  = 1e-5     # low LR for pretrained weights
PHASE2_LR_FREQ      = 5e-4
PHASE2_LR_FUSION    = 1e-3
PHASE2_BATCH    = 16

# ─── Common Training ─────────────────────────────────────────────────────────
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
EARLY_STOP_PATIENCE = 7

# ─── Focal Loss ───────────────────────────────────────────────────────────────
FOCAL_ALPHA     = 0.25
FOCAL_GAMMA     = 2.0

# ─── Misc ─────────────────────────────────────────────────────────────────────
NUM_WORKERS     = 0            # Windows-safe (0 = main process)
PIN_MEMORY      = False        # CPU training
