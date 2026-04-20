"""
calibrate_pan.py — Visual calibration tool for PAN card regions.

Draws colored bounding boxes for all PAN_REGIONS on the sample PAN card image.
Similar to calibrate.py (Aadhaar) but specific to PAN card layout.

PAN Card layout (approx 615×380 px):
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
  │ <DOB>             <Signature>             │                     │
  └───────────────────────────────────────────┴─────────────────────┘
"""

import os, sys
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
PAN_IMAGE = os.path.join(BASE, "src", "sample-pan-card.jpg")

if not os.path.exists(PAN_IMAGE):
    sys.exit(f"PAN card image not found: {PAN_IMAGE}")

print(f"Using: {PAN_IMAGE}")

card = Image.open(PAN_IMAGE).convert("RGB")
w, h = card.size
print(f"Card dimensions: {w} x {h} px")

# ─── PAN Card Region Map (relative fractions) ────────────────────────────────
# Format: (x_min, y_min, x_max, y_max) — all in [0.0, 1.0]
#
# These define the VALUE areas that will be masked (blanked) and replaced
# with new synthetic data during card generation.

PAN_REGIONS = {
    "face":         (0.018, 0.260, 0.220, 0.560),   # Photo area (left side)
    "pan_num":      (0.280, 0.380, 0.650, 0.460),  # PAN number (ABCDE1234F)
    "name":         (0.035, 0.605, 0.360, 0.670),   # Applicant name value
    "father_name":  (0.035, 0.740, 0.540, 0.800),   # Father's name value
    "dob":         (0.015, 0.920, 0.270, 0.990),   # Date of birth value
    "qr": (0.670, 0.250, 0.970, 0.760),# qr area
}

COLORS = {
    "face":         (255, 80,  80),    # Red
    "pan_num":      (80,  200, 80),    # Green
    "name":         (80,  80,  255),   # Blue
    "father_name":  (255, 165, 0),     # Orange
    "dob":          (200, 0,   200),   # Purple
     "qr":          (200, 0,   200),   # Purple
   
}

LABELS = {
    "face":         "FACE",
    "pan_num":      "PAN NUMBER",
    "name":         "NAME",
    "father_name":  "FATHER NAME",
    "dob":          "DOB",
    "qr":           "QR CODE",
   
}

# ─── Draw overlay boxes ──────────────────────────────────────────────────────
for region_name, (rx0, ry0, rx1, ry1) in PAN_REGIONS.items():
    x0, y0 = int(rx0 * w), int(ry0 * h)
    x1, y1 = int(rx1 * w), int(ry1 * h)
    col = COLORS[region_name]

    # Semi-transparent filled rectangle with solid border
    overlay = Image.new("RGBA", card.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([x0, y0, x1, y1], fill=col + (60,), outline=col + (255,), width=3)

    # Add label text inside the box
    try:
        label_font = ImageFont.truetype("C:/Windows/Fonts/Arial.ttf", 12)
    except:
        label_font = ImageFont.load_default()
    label = LABELS[region_name]
    ov_draw.text((x0 + 4, y0 + 2), label, fill=(255, 255, 255, 220), font=label_font)

    card = card.convert("RGBA")
    card = Image.alpha_composite(card, overlay).convert("RGB")

# ─── Save calibration image ──────────────────────────────────────────────────
out = os.path.join(BASE, "debug_calibration_pan.jpg")
card.save(out, quality=95)
print(f"\nSaved -> {out}")

# ─── Print region details ────────────────────────────────────────────────────
print("\nPAN Card Regions (pixel coordinates):")
print("=" * 60)
for region_name, (rx0, ry0, rx1, ry1) in PAN_REGIONS.items():
    x0, y0, x1, y1 = int(rx0 * w), int(ry0 * h), int(rx1 * w), int(ry1 * h)
    print(f"  {region_name:14s}: px ({x0:3d},{y0:3d}) -> ({x1:3d},{y1:3d})   "
          f"size {x1-x0:3d}x{y1-y0:3d}")
