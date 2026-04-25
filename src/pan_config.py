"""
pan_config.py — Central configuration for PAN card dataset pipeline.

All coordinate regions are expressed as relative fractions (0.0 – 1.0) of
(image_width, image_height) so they work regardless of scan resolution.

Calibrated from reference PAN card (615×380 px):
  ┌─────────────────────────────────────────────────────────────────┐
  │ आयकर विभाग     [Emblem]       भारत सरकार                      │
  │ INCOME TAX DEPT               GOVT. OF INDIA       (header)    │
  ├─────────────────────────────────────────────────────────────────┤
  │    स्थायी लेखा संख्या कार्ड / Permanent Account Number Card    │
  │    <PAN NUMBER>  (e.g. ABCDE1234F)                             │
  ├───────────────────────────────────────────┬─────────────────────┤
  │ [Photo]  नाम / Name:                     │                     │
  │          <NAME VALUE>                     │   [QR Code]         │
  │          पिता का नाम / Father's Name:     │                     │
  │          <FATHER NAME VALUE>              │                     │
  │          जन्म की तारीख / Date of Birth:   │                     │
  │ <DOB>                                     │                     │
  └───────────────────────────────────────────┴─────────────────────┘

MASKING STRATEGY (only VALUE areas are masked — labels stay intact):
  - Face:       erase photo box, paste new face
  - PAN Num:    erase PAN number text, write new one
  - Name:       erase applicant name value
  - Father:     erase father's name value
  - DOB:        erase date value
  - QR:         left untouched (ignored)
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR            = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAN_SAMPLE          = os.path.join(os.path.dirname(__file__), "clean_pan_sample.png")
FACES_DIR           = os.path.join(BASE_DIR, "data", "faces")
PAN_TEMPLATES_DIR   = os.path.join(BASE_DIR, "data", "pan_templates")
PAN_OUTPUT_DIR      = os.path.join(BASE_DIR, "output_pan")
PAN_OUTPUT_REAL     = os.path.join(PAN_OUTPUT_DIR, "real")
PAN_OUTPUT_FAKE     = os.path.join(PAN_OUTPUT_DIR, "fake")
FONT_DIR            = os.path.join(BASE_DIR, "data", "fonts")
PAN_TEMPLATE_META   = os.path.join(PAN_TEMPLATES_DIR, "metadata.json")
PAN_DATASET_CSV     = os.path.join(PAN_OUTPUT_DIR, "pan_dataset.csv")

# ─── PAN Card Region Map (relative fractions) ─────────────────────────────────
# Format: (x_min, y_min, x_max, y_max)  — all in [0.0, 1.0]
#
# Calibrated by user via calibrate_pan.py
PAN_REGIONS = {
    "face":         (0.020, 0.240, 0.210, 0.540),   # Photo area (left side)
    "pan_num":      (0.280, 0.380, 0.650, 0.460),   # PAN number (ABCDE1234F)
    "name":         (0.020, 0.590, 0.345, 0.640),   # Applicant name value
    "father_name":  (0.020, 0.700, 0.485, 0.740),   # Father's name value
    "dob":          (0.015, 0.830, 0.240, 0.880),   # Date of birth value
    "qr":           (0.670, 0.250, 0.970, 0.760),   # qr area
}

# ─── Text Rendering ───────────────────────────────────────────────────────────
# Font sizes as fraction of card HEIGHT
PAN_FONT_SIZE_NAME       = 0.038   # Name value (reduced by ~2 font sizes)
PAN_FONT_SIZE_FATHER     = 0.038   # Father's name
PAN_FONT_SIZE_PAN_NUM    = 0.058   # PAN number (bold, larger)
PAN_FONT_SIZE_DOB        = 0.042   # DOB value

# Text colours
PAN_TEXT_COLOR      = (0, 0, 0)    # Black text
PAN_TEXT_COLOR_NUM  = (0, 0, 0)    # Black bold PAN number

# ─── PAN Number Format ────────────────────────────────────────────────────────
# Govt. format: AAAPL1234C
#   Pos 1-3: Random uppercase letters (A-Z)
#   Pos 4  : Entity type (P=Person, C=Company, H=HUF, F=Firm, etc.)
#   Pos 5  : First letter of surname
#   Pos 6-9: 4 sequential digits (0001-9999)
#   Pos 10 : Alphabetic check letter (A-Z)
PAN_ENTITY_TYPES = ["P"]   # Only persons for our dataset

# ─── Dataset Generation Targets ───────────────────────────────────────────────
PAN_TARGET_REAL = 299
PAN_TARGET_FAKE = 299

# ─── Face Detection ───────────────────────────────────────────────────────────
HAAR_SCALE_FACTOR  = 1.1
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (60, 60)

# Min face image size to accept
MIN_FACE_IMG_SIZE_KB = 50

# ─── Fake Category Probabilities ──────────────────────────────────────────────
PAN_FAKE_CATEGORIES_PROBS = {
    "semantic":         0.50,   # Gender/age mismatch
    "partial_editing":  0.40,   # Edit one field with bad font/shift
    "face_tampering":   0.35,   # Face brightness mismatch
    "text_tampering":   0.45,   # Font change, char spacing, text blur
    "image_quality":    0.60,   # JPEG, blur, noise, color jitter
    "structural":       0.25,   # Affine warp rotation
    "border_crop":      0.20,   # Edge crop, copy-paste border artifacts
}

# ─── Augmentation Settings ────────────────────────────────────────────────────
PAN_AUG_SETTINGS = {
    "jpeg_compress":    (15, 45),
    "gaussian_blur":    (0.6, 1.8),
    "gaussian_noise":   (5, 20),
    "affine_warp":      (-3, 3),
    "face_shift":       (5, 15),
}

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
