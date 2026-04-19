"""
template_extractor.py — Extract clean Aadhaar card templates from real images.

Strategy (since all 90 cards share the same layout/font/format):
  1. For each real card image:
     a. Detect the face region using OpenCV Haarcascade (fallback: fixed coords)
     b. Mask face + text zones with the card's own background colour — this
        preserves layout, emblem colours, fonts and lines exactly.
     c. Save the cleaned template PNG + record face bounding box.
  2. Write data/templates/metadata.json with per-template info.

The metadata JSON is later read by card_composer.py to know
exactly where to paste new faces and render text.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .config import (
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    HAAR_SCALE_FACTOR,
    REAL_DIR,
    REGIONS,
    TEMPLATE_META,
    TEMPLATES_DIR,
)

logger = logging.getLogger(__name__)


# ─── Haarcascade Loader ───────────────────────────────────────────────────────

def _load_face_cascade() -> cv2.CascadeClassifier:
    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(xml)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haarcascade from {xml}")
    return cascade


# ─── Background Colour Sampler ────────────────────────────────────────────────

def _sample_background_color(img: np.ndarray) -> tuple:
    """
    Sample the average background colour from four corner patches of the card.
    Returns (R, G, B) as a tuple of ints.
    """
    h, w = img.shape[:2]
    patch_size = max(8, min(h, w) // 20)

    patches = [
        img[:patch_size, :patch_size],
        img[:patch_size, w - patch_size:],
        img[h - patch_size:, :patch_size],
        img[h - patch_size:, w - patch_size:],
    ]
    combined = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    median = np.median(combined, axis=0).astype(int)
    return (int(median[2]), int(median[1]), int(median[0]))   # BGR → RGB


# ─── Region Pixel Converter ───────────────────────────────────────────────────

def _rel_to_px(region: tuple, w: int, h: int) -> tuple:
    """Convert relative (xmin, ymin, xmax, ymax) → absolute pixel coords."""
    x1 = int(region[0] * w)
    y1 = int(region[1] * h)
    x2 = int(region[2] * w)
    y2 = int(region[3] * h)
    return x1, y1, x2, y2


# ─── Face Detection ───────────────────────────────────────────────────────────

def _detect_face_region(gray: np.ndarray, w: int, h: int) -> Optional[tuple]:
    """
    Try to detect a face in the image.
    Returns (x1, y1, x2, y2) in pixels, or None on failure.
    """
    cascade = _load_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=HAAR_MIN_SIZE,
    )
    if len(faces) == 0:
        return None
    # Take the largest detected face
    areas = [(fw * fh, fx, fy, fw, fh) for (fx, fy, fw, fh) in faces]
    _, fx, fy, fw, fh = max(areas)
    # Add 10 % padding
    pad_x = int(fw * 0.10)
    pad_y = int(fh * 0.10)
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_y)
    x2 = min(w, fx + fw + pad_x)
    y2 = min(h, fy + fh + pad_y)
    return x1, y1, x2, y2


# ─── Single Card Extraction ───────────────────────────────────────────────────

def extract_template(
    src_path: str,
    dst_path: str,
    bg_color: Optional[tuple] = None,
) -> dict:
    """
    Create a template from one real Aadhaar card image.

    Returns metadata dict:
        {
          "source":      filename,
          "template":    output filename,
          "face_region": [x1, y1, x2, y2],   ← pixel coords for paste
          "card_size":   [width, height],
          "bg_color":    [R, G, B]
        }
    """
    img_bgr = cv2.imread(src_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {src_path}")

    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Sample background colour ──────────────────────────────────────────
    if bg_color is None:
        bg_color = _sample_background_color(img_bgr)

    # OpenCV fill colour (BGR)
    fill_bgr = (bg_color[2], bg_color[1], bg_color[0])

    # ── Try face detection; fall back to config REGIONS["face"] ──────────
    face_px = _detect_face_region(gray, w, h)
    if face_px is None:
        logger.debug(f"  No face detected in {os.path.basename(src_path)}, using config region.")
        face_px = _rel_to_px(REGIONS["face"], w, h)

    # ── Mask face region ──────────────────────────────────────────────────
    x1, y1, x2, y2 = face_px
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), fill_bgr, thickness=-1)

    # ── Mask text regions ─────────────────────────────────────────────────
    for key in ("name", "dob", "gender", "aadhaar_num"):
        tx1, ty1, tx2, ty2 = _rel_to_px(REGIONS[key], w, h)
        cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), fill_bgr, thickness=-1)

    # ── Save template ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_img.save(dst_path, format="PNG")

    return {
        "source":      os.path.basename(src_path),
        "template":    os.path.basename(dst_path),
        "face_region": list(face_px),        # [x1, y1, x2, y2] px
        "card_size":   [w, h],
        "bg_color":    list(bg_color),       # [R, G, B]
    }


# ─── Batch Extraction ─────────────────────────────────────────────────────────

def extract_all_templates(real_dir: str = REAL_DIR, templates_dir: str = TEMPLATES_DIR) -> list:
    """
    Extract templates from all real Aadhaar cards in `real_dir`.
    Saves PNGs to `templates_dir` and writes metadata.json.

    Returns list of metadata dicts.
    """
    from tqdm import tqdm

    os.makedirs(templates_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(real_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    ])

    if not image_files:
        logger.warning(f"No images found in {real_dir}")
        return []

    # ── Use ONLY the first card as the single master template ─────────────
    # All 90 cards share identical layout/font/format, so one template is
    # sufficient and guarantees consistent masking across all generated cards.
    image_files = image_files[:1]
    logger.info(f"Single-template mode: using '{image_files[0]}' as master template.")

    logger.info(f"Extracting templates from {len(image_files)} real Aadhaar cards …")

    # Sample background colour once from first image and reuse (cards identical)
    first_img = cv2.imread(os.path.join(real_dir, image_files[0]))
    shared_bg = _sample_background_color(first_img) if first_img is not None else None

    metadata_list = []
    for idx, fname in enumerate(tqdm(image_files, desc="Extracting templates", unit="card"), start=1):
        src  = os.path.join(real_dir, fname)
        stem = Path(fname).stem
        dst  = os.path.join(templates_dir, f"template_{idx:03d}.png")
        try:
            meta = extract_template(src, dst, bg_color=shared_bg)
            metadata_list.append(meta)
        except Exception as e:
            logger.error(f"  ✗ Failed on {fname}: {e}")

    # ── Write metadata JSON ────────────────────────────────────────────────
    class _NumpyEncoder(json.JSONEncoder):
        """Convert numpy scalar types → native Python before serialising."""
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(TEMPLATE_META, "w") as f:
        json.dump(metadata_list, f, indent=2, cls=_NumpyEncoder)

    logger.info(f"Saved {len(metadata_list)} templates → {templates_dir}")
    logger.info(f"Metadata → {TEMPLATE_META}")
    return metadata_list


def load_template_metadata() -> list:
    """Load previously-saved template metadata from JSON."""
    if not os.path.exists(TEMPLATE_META):
        return []
    with open(TEMPLATE_META, "r") as f:
        return json.load(f)


# ─── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metas = extract_all_templates()
    print(f"\nDone. {len(metas)} templates extracted.")
    if metas:
        print("Sample metadata:", json.dumps(metas[0], indent=2))
