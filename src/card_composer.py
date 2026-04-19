"""
card_composer.py — Assemble synthetic Aadhaar card images.

Strategy:
  1. Open the cleaned template (all personal-data VALUE areas already blank)
  2. Paste the new face photo into the face region
  3. Write new VALUES only into their respective blank areas:
       - name       → left-aligned in name VALUE region
       - dob        → left-aligned in dob VALUE region  (label stays in template)
       - gender     → left-aligned in gender VALUE region
       - aadhaar_num → centred bold in aadhaar number region

Font priority: NotoSans-Bold / Regular → Arial → Verdana → PIL default
"""

import logging
import os
import urllib.request
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .config import (
    FONT_DIR,
    FONT_SIZE_AADHAAR,
    FONT_SIZE_NAME,
    FONT_SIZE_VALUE,
    REGIONS,
    TEXT_COLOR_DARK,
    TEXT_COLOR_NUM,
    TEMPLATES_DIR,
)

logger = logging.getLogger(__name__)

# ─── Font Loader ──────────────────────────────────────────────────────────────

_FONT_URLS = {
    "NotoSans-Bold.ttf":
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    "NotoSans-Regular.ttf":
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
}

_WINDOWS_FALLBACKS = [
    "C:/Windows/Fonts/Arial.ttf",
    "C:/Windows/Fonts/Verdana.ttf",
    "C:/Windows/Fonts/Calibri.ttf",
]


def _download_font(name: str) -> Optional[str]:
    os.makedirs(FONT_DIR, exist_ok=True)
    dst = os.path.join(FONT_DIR, name)
    if os.path.exists(dst):
        return dst
    url = _FONT_URLS.get(name)
    if url is None:
        return None
    try:
        logger.info(f"Downloading font {name} …")
        urllib.request.urlretrieve(url, dst)
        return dst
    except Exception as e:
        logger.warning(f"Font download failed ({name}): {e}")
        return None


def _get_font(size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_name = "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"
    font_path = _download_font(font_name)
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size=size_px)
    for sys_font in _WINDOWS_FALLBACKS:
        if os.path.exists(sys_font):
            return ImageFont.truetype(sys_font, size=size_px)
    return ImageFont.load_default()


# ─── Coordinate Helpers ───────────────────────────────────────────────────────

def _rel_to_px(region: tuple, w: int, h: int) -> tuple:
    return (
        int(region[0] * w),
        int(region[1] * h),
        int(region[2] * w),
        int(region[3] * h),
    )


# ─── Face Paste ───────────────────────────────────────────────────────────────

def _paste_face(card: Image.Image, face_img_path: str, face_region_px: tuple) -> None:
    """Resize face to fill the exact face region. No square crop."""
    x1, y1, x2, y2 = face_region_px
    target_w = x2 - x1
    target_h = y2 - y1
    try:
        face = Image.open(face_img_path).convert("RGB")
        # Fill the box directly — preserves all content, fills photo area cleanly
        face = face.resize((target_w, target_h), Image.LANCZOS)
        card.paste(face, (x1, y1))
    except Exception as e:
        logger.warning(f"Face paste failed ({face_img_path}): {e}")


# ─── White-fill Helper ────────────────────────────────────────────────────────

def _clear_region(card: Image.Image, region_px: tuple, fill=(255, 255, 255)) -> None:
    """Fill a region with solid colour (default white) before writing new text."""
    draw = ImageDraw.Draw(card)
    x1, y1, x2, y2 = region_px
    draw.rectangle([x1, y1, x2, y2], fill=fill)


# ─── Text Value Writer ────────────────────────────────────────────────────────

def _write_value(
    draw: ImageDraw.ImageDraw,
    text: str,
    region_px: tuple,
    font: ImageFont.FreeTypeFont,
    color: tuple,
    align: str = "left",   # "left" | "center"
    valign: str = "center", # "center" | "top"
) -> None:
    """
    Write text inside a pre-cleared region.
    - left:   small left padding
    - center: horizontally centred
    - top/center vertically
    """
    x1, y1, x2, y2 = region_px
    rw = x2 - x1
    rh = y2 - y1

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if align == "center":
        tx = x1 + (rw - tw) // 2
    else:
        tx = x1 + max(4, rw // 40)   # small left pad

    if valign == "top":
        ty = y1 + max(2, rh // 6)    # push slightly down from top edge
    else:
        ty = y1 + (rh - th) // 2         # always vertically centred

    draw.text((tx, ty), text, font=font, fill=color)


# ─── Main Compositor ──────────────────────────────────────────────────────────

def compose_card(
    template_path: str,
    face_img_path: str,
    name: str,
    dob: str,
    gender: str,
    aadhaar_number: str,
    output_path: str,
    face_region_px: Optional[tuple] = None,
) -> str:
    """
    Produce a synthetic Aadhaar card by placing new personal data onto a template.

    The template already has the personal VALUE areas blanked (white-filled) by
    template_extractor; this function just writes new values into those areas.
    Static labels ("नाम / Name:", "जन्म तारीख / DOB:", etc.) remain from the template.

    Args:
        template_path:  Cleaned template PNG path.
        face_img_path:  Face image to paste in.
        name:           Full name (e.g. "Ananya Sharma").
        dob:            Date of birth "DD/MM/YYYY".
        gender:         "Male" or "Female".
        aadhaar_number: "XXXX XXXX XXXX" formatted number.
        output_path:    Where to save the result.
        face_region_px: (x1,y1,x2,y2) in pixels from template metadata;
                        falls back to config REGIONS["face"] if None.

    Returns:
        output_path
    """
    card = Image.open(template_path).convert("RGB")
    w, h = card.size

    # ── 1. Font sizes (calibrated to card height) ──────────────────────────
    name_sz    = max(10, int(h * FONT_SIZE_NAME))
    value_sz   = max(8,  int(h * FONT_SIZE_VALUE))
    aadhaar_sz = max(12, int(h * FONT_SIZE_AADHAAR))

    font_name    = _get_font(name_sz,    bold=True)
    font_value   = _get_font(value_sz,   bold=False)
    font_aadhaar = _get_font(aadhaar_sz, bold=True)

    draw = ImageDraw.Draw(card)

    # ── 2. Face ────────────────────────────────────────────────────────────
    # Ignore HAAR results; use exact config block to preserve proper aspect ratio sizing
    face_region_px = _rel_to_px(REGIONS["face"], w, h)
    _paste_face(card, face_img_path, face_region_px)

    # ── 3. Name value — below the "नाम / Name:" label line ────────────────
    name_px = _rel_to_px(REGIONS["name"], w, h)
    _clear_region(card, name_px)                        # ensure clean white bg
    draw = ImageDraw.Draw(card)                          # refresh after paste
    _write_value(draw, name, name_px, font_name, TEXT_COLOR_DARK, align="left")

    # ── 4. DOB value — just the date, after "जन्म तारीख / DOB:" label ─────
    dob_px = _rel_to_px(REGIONS["dob"], w, h)
    _clear_region(card, dob_px)
    draw = ImageDraw.Draw(card)
    # The mask box is tall. Setting valign="top" aligns it perfectly inline with the label.
    _write_value(draw, dob, dob_px, font_value, TEXT_COLOR_DARK, align="left", valign="top")

    # ── 5. Gender value — standalone word ─────────────────────────────────
    gender_px = _rel_to_px(REGIONS["gender"], w, h)
    _clear_region(card, gender_px)
    draw = ImageDraw.Draw(card)
    _write_value(draw, gender.capitalize(), gender_px, font_value, TEXT_COLOR_DARK, align="left")

    # ── 6. Aadhaar number — large bold, horizontally centred ──────────────
    num_px = _rel_to_px(REGIONS["aadhaar_num"], w, h)
    _clear_region(card, num_px)
    draw = ImageDraw.Draw(card)
    _write_value(draw, aadhaar_number, num_px, font_aadhaar, TEXT_COLOR_NUM, align="center")

    # ── 7. Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    card.save(output_path, format="JPEG", quality=94)
    return output_path


# ─── Quick CLI test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, glob
    logging.basicConfig(level=logging.INFO)
    templates = sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.png")))
    if not templates:
        print("No templates found — run template_extractor first.")
        sys.exit(1)
    faces_dir = os.path.join(os.path.dirname(TEMPLATES_DIR), "..", "faces")
    faces = sorted([
        f for f in os.listdir(faces_dir)
        if f.endswith(".jpg") and os.path.getsize(os.path.join(faces_dir, f)) > 50_000
    ])
    if not faces:
        print("No face images found.")
        sys.exit(1)
    out = os.path.join(os.path.dirname(TEMPLATES_DIR), "..", "output", "test_compose.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    compose_card(
        template_path  = templates[0],
        face_img_path  = os.path.join(faces_dir, faces[0]),
        name           = "Ananya Sharma",
        dob            = "15/03/1995",
        gender         = "Female",
        aadhaar_number = "3456 7890 1234",
        output_path    = out,
    )
    print(f"Test card saved → {out}")
