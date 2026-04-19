"""
config.py — Central configuration for the KYC Aadhaar dataset pipeline.

All coordinate regions are expressed as relative fractions (0.0 – 1.0) of
(image_width, image_height) so they work regardless of scan resolution.

Calibrated from reference card (approx 1024×680 px):
  ┌─────────────────────────────────────────────────────────────────┐
  │  [Emblem]   भरत सरकार / GOVERNMENT OF INDIYA    (header 0–27%) │
  ├─────────────────────────────────────────────────────────────────┤
  │[Photo ]  नाम / Name:                     [QR code]            │
  │ 6-27%    <name value>           28-78%     76-97%              │
  │ 28-78%   जन्म तारीख / DOB: <date value>  38-80%              │
  │          <gender value>                                        │
  ├──────────────────────────────────────────────────────────────--─┤
  │         XXXX  XXXX  XXXX  (Aadhaar number 80–89%)              │
  │         आधार - आदमी का अधिकार         (88–98%)                │
  └─────────────────────────────────────────────────────────────────┘

MASKING STRATEGY:
  - Face:       erase entire photo box, paste new face
  - Name:       erase only the name VALUE row (label "नाम / Name:" stays)
  - DOB:        erase only the DATE VALUE at end of that line (label stays)
  - Gender:     erase only the gender word ("Male"/"Female")
  - Aadhaar:    erase entire number, write new one centred
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_DIR       = os.path.join(BASE_DIR, "data", "aadhaar", "real")
FACES_DIR      = os.path.join(BASE_DIR, "data", "faces")
TEMPLATES_DIR  = os.path.join(BASE_DIR, "data", "templates")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
OUTPUT_REAL    = os.path.join(OUTPUT_DIR, "real")
OUTPUT_FAKE    = os.path.join(OUTPUT_DIR, "fake")
FONT_DIR       = os.path.join(BASE_DIR, "data", "fonts")
CACHE_FILE     = os.path.join(BASE_DIR, "data", "face_analysis_cache.json")
TEMPLATE_META  = os.path.join(TEMPLATES_DIR, "metadata.json")
DATASET_CSV    = os.path.join(OUTPUT_DIR, "dataset.csv")

# ─── Aadhaar Card Region Map (relative fractions) ─────────────────────────────
# Format: (x_min, y_min, x_max, y_max)  — all in [0.0, 1.0]
#
# REGIONS are used for TWO purposes:
#   1. template_extractor  → fills these areas with background colour
#   2. card_composer       → places new content in exactly these areas
#
# Only VALUE areas are masked — static labels stay in the template intact.
REGIONS = {
    "face":        (0.030, 0.330, 0.250, 0.720),
    "name":        (0.280, 0.380, 0.720, 0.460),
  "dob": (0.560, 0.470, 0.760, 0.580),
    "gender":      (0.280, 0.540, 0.430, 0.620),
   "aadhaar_num": (0.170, 0.740, 0.720, 0.840)
}

# ─── Text Rendering ───────────────────────────────────────────────────────────
# Font sizes as fraction of card HEIGHT — tuned to match real card typography
FONT_SIZE_NAME     = 0.050   # Name value  (≈34 px on 680 h card)
FONT_SIZE_VALUE    = 0.042   # DOB / Gender values  (≈29 px)
FONT_SIZE_AADHAAR  = 0.072   # Large Aadhaar number (≈49 px)

# Text colours — pure black to match the real card ink
TEXT_COLOR_DARK  = (0,   0,   0)     # Black for name / DOB / gender
TEXT_COLOR_NUM   = (0,   0,   0)     # Black bold for Aadhaar number

# ─── Dataset Generation Targets ───────────────────────────────────────────────
TARGET_REAL_SYNTHETIC = 200   # Generated real (label=0) samples
TARGET_FAKE           = 200   # Generated fake (label=1) samples

# Min face image size to accept — rejects tiny CelebA thumbnails (~3 KB)
MIN_FACE_IMG_SIZE_KB = 50

# ─── Face Detection ───────────────────────────────────────────────────────────
HAAR_SCALE_FACTOR  = 1.1
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (60, 60)

# ─── High-Fidelity Fake Configuration ─────────────────────────────────────────

# Probabilities for picking EACH category when generating a fake sample.
# We aim to pick exactly 2-3 categories total to keep the FAKE realistic.
FAKE_CATEGORIES_PROBS = {
    "semantic":         0.50, # Wrong gender, wrong age, malformed aadhaar
    "partial_editing":  0.40, # Editing only one specific text field
    "face_tampering":   0.35, # Face scale, shift, brightness
    "text_tampering":   0.45, # Blur, font variation, character shift in text
    "image_quality":    0.60, # JPEG, Blur, Noise, Color
    "structural":       0.25, # Affine wrap, Aadhaar global shift
    "border_crop":      0.20, # Cropped edges, copy-paste border artifacts
}

# ─── Individual Augmentation Settings ─────────────────────────────────────────
AUG_SETTINGS = {
    "jpeg_compress":    (15, 45),    # Quality range
    "gaussian_blur":    (0.6, 1.8),  # Radius range
    "gaussian_noise":   (5, 20),     # Sigma range
    "affine_warp":      (-3, 3),     # Rotation angle range
    "face_shift":       (5, 15),     # Pixel shift range
}

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
