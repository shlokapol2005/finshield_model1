"""
calibrate.py — Visual calibration tool.

Draws colored bounding boxes for all REGIONS on the FIRST real card image.
"""

import os, sys
from PIL import Image, ImageDraw, ImageFont

BASE    = os.path.dirname(os.path.abspath(__file__))
REAL    = os.path.join(BASE, "data", "aadhaar", "real")
imgs    = sorted(f for f in os.listdir(REAL)
                 if f.lower().endswith((".jpg",".jpeg",".png",".bmp")))
if not imgs:
    sys.exit("No images in data/aadhaar/real/")

card_path = os.path.join(REAL, imgs[0])
print(f"Using: {card_path}")

card = Image.open(card_path).convert("RGB")
w, h = card.size
print(f"Card dimensions: {w} x {h} px")

REGIONS = {
    "face":        (0.030, 0.330, 0.250, 0.720),
    "name":        (0.280, 0.380, 0.720, 0.460),
   "dob": (0.560, 0.470, 0.760, 0.580),
    "gender":      (0.280, 0.540, 0.430, 0.620),
  "aadhaar_num": (0.170, 0.740, 0.720, 0.840)
}

COLORS = {
    "face":        (255, 80,  80),   # red
    "name":        (80,  200, 80),   # green
    "dob":         (80,  80,  255),  # blue
    "gender":      (255, 165, 0),    # orange
    "aadhaar_num": (200, 0,   200),  # purple
}

for name, (rx0, ry0, rx1, ry1) in REGIONS.items():
    x0, y0 = int(rx0*w), int(ry0*h)
    x1, y1 = int(rx1*w), int(ry1*h)
    col = COLORS[name]
    overlay = Image.new("RGBA", card.size, (0,0,0,0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([x0, y0, x1, y1], fill=col+(60,), outline=col+(255,), width=3)
    card = card.convert("RGBA")
    card = Image.alpha_composite(card, overlay).convert("RGB")

out = os.path.join(BASE, "debug_calibration.jpg")
card.save(out, quality=95)
print(f"\nSaved -> {out}")
for name, (rx0, ry0, rx1, ry1) in REGIONS.items():
    x0,y0,x1,y1 = int(rx0*w),int(ry0*h),int(rx1*w),int(ry1*h)
    print(f"  {name:12s}: px ({x0},{y0}) -> ({x1},{y1})   size {x1-x0}x{y1-y0}")
