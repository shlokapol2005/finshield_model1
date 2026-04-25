"""
pan_card_composer.py — Assemble synthetic PAN card images with REALISTIC rendering.

Strategy:
  1. Open the cleaned PAN template (all personal-data VALUE areas already blank)
  2. For each VALUE region:
     a. Inpaint / clone the background texture (no white patches)
     b. Render text with near-black colour variation
     c. Apply subtle blur + noise so text looks printed, not digital
  3. Paste the face photo into the face region

Font priority: NotoSans-Bold / Regular → Arial → Verdana → PIL default
"""

import logging
import os
import urllib.request
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

from .pan_config import (
    FONT_DIR,
    PAN_FONT_SIZE_DOB,
    PAN_FONT_SIZE_FATHER,
    PAN_FONT_SIZE_NAME,
    PAN_FONT_SIZE_PAN_NUM,
    PAN_REGIONS,
    PAN_TEXT_COLOR,
    PAN_TEXT_COLOR_NUM,
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


# ═══════════════════════════════════════════════════════════════════════════════
# REALISTIC BACKGROUND-AWARE FILLING  (replaces naive white fill)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_region_with_background(pil_img: Image.Image, region_px: tuple,
                                rng=None) -> Image.Image:
    """
    Fill a VALUE region by inpainting from surrounding card texture.

    Uses OpenCV's Navier-Stokes inpainting with a tight radius to seamlessly
    reconstruct the background gradient inside the region.  Only a very faint
    amount of noise is added so the fill is virtually invisible.

    Args:
        pil_img:    PIL Image (RGB).
        region_px:  (x1, y1, x2, y2) pixel coordinates of the value area.
        rng:        Optional random.Random for noise variance.

    Returns:
        Modified PIL Image.
    """
    import random as _random
    if rng is None:
        rng = _random

    x1, y1, x2, y2 = region_px
    arr = np.array(pil_img)  # H×W×3 uint8 (RGB)

    # ── Create an inpainting mask: white = region to fill ─────────────────
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    # Work in BGR for OpenCV
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ── Inpaint using Navier-Stokes (smoother gradients than Telea) ───────
    # Small radius (3) prevents the "blurry rectangle" artefact
    inpainted = cv2.inpaint(arr_bgr, mask, inpaintRadius=3,
                            flags=cv2.INPAINT_NS)

    # ── Barely-perceptible noise so the fill isn't pixel-perfect smooth ───
    region_patch = inpainted[y1:y2, x1:x2].astype(np.float32)
    noise_sigma = rng.uniform(0.5, 1.2)  # very faint
    noise = np.random.normal(0, noise_sigma, region_patch.shape)
    region_patch = np.clip(region_patch + noise, 0, 255).astype(np.uint8)
    inpainted[y1:y2, x1:x2] = region_patch

    # Convert back to RGB PIL
    result_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


# ═══════════════════════════════════════════════════════════════════════════════
# REALISTIC TEXT DRAWING  (replaces crisp digital text)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_text_realistic(
    pil_img: Image.Image,
    text: str,
    region_px: tuple,
    font: ImageFont.FreeTypeFont,
    base_color: tuple = (0, 0, 0),
    align: str = "left",
    valign: str = "center",
    tamper_shift: bool = False,
    tamper_spacing: bool = False,
    rng=None,
) -> Image.Image:
    """
    Draw text onto the card with realistic printed-ink appearance.

    Instead of drawing perfectly sharp pure-black text, this function:
      1. Varies the ink colour slightly (near-black, not pure 0,0,0)
      2. Draws text onto a transparent layer
      3. Applies a very slight Gaussian blur (simulates ink bleed)
      4. Composites the text onto the card
      5. Adds micro-noise to the text region for print grain

    Args:
        pil_img:      PIL Image (RGB).
        text:         String to render.
        region_px:    (x1, y1, x2, y2).
        font:         PIL font object.
        base_color:   Base text colour (will be slightly varied).
        align:        "left" | "center".
        valign:       "center" | "top".
        tamper_shift: If True, randomly offset text position (forgery sim).
        tamper_spacing: If True, random per-char spacing (forgery sim).
        rng:          Random instance.

    Returns:
        Modified PIL Image.
    """
    import random as _random
    if rng is None:
        rng = _random

    x1, y1, x2, y2 = region_px
    rw = x2 - x1
    rh = y2 - y1

    # ── 1. Ink colour: pure black as requested ──────────
    ink_color = base_color

    # ── 2. Create a transparent text layer the size of the full image ─────
    text_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    # ── 3. Compute text position ──────────────────────────────────────────
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if align == "center":
        tx = x1 + (rw - tw) // 2
    else:
        tx = x1 + max(2, rw // 40)

    if valign == "top":
        ty = y1 + max(1, rh // 8)
    else:
        ty = y1 + (rh - th) // 2

    # Intentional misalignment for fakes
    if tamper_shift:
        tx += rng.randint(-12, 12)
        ty += rng.randint(-8, 8)

    # ── 4. Draw text ─────────────────────────────────────────────────────
    ink_rgba = ink_color + (255,)

    if tamper_spacing:
        cx = tx
        for char in text:
            draw.text((cx, ty + rng.randint(-2, 2)), char,
                      font=font, fill=ink_rgba)
            char_w = draw.textlength(char, font=font)
            cx += char_w + rng.uniform(1.0, 5.0)
    else:
        draw.text((tx, ty), text, font=font, fill=ink_rgba)

    # ── 5. No blur as requested (direct paste) ───────────────────────────
    pass

    # ── 6. Composite text onto card ──────────────────────────────────────
    card_rgba = pil_img.convert("RGBA")
    composited = Image.alpha_composite(card_rgba, text_layer)
    result = composited.convert("RGB")

    return result


def apply_blend_effects(pil_img: Image.Image, region_px: tuple,
                        rng=None) -> Image.Image:
    """
    Apply minimal post-render effects — just barely-perceptible grain
    so the text doesn't look 100% digital.  No blur, no contrast shift.

    Args:
        pil_img:    PIL Image (RGB).
        region_px:  (x1, y1, x2, y2).
        rng:        Random instance.

    Returns:
        Modified PIL Image.
    """
    import random as _random
    if rng is None:
        rng = _random

    # No blend effects requested, keep the text clean
    return pil_img


# ─── Face Paste ───────────────────────────────────────────────────────────────

def _paste_face(card: Image.Image, face_img_path: str, face_region_px: tuple,
                brightness_adjust: bool = False) -> Image.Image:
    """Resize face to fill the exact face region with optional brightness tampering."""
    x1, y1, x2, y2 = face_region_px
    target_w = x2 - x1
    target_h = y2 - y1
    try:
        face = Image.open(face_img_path).convert("RGB")

        if brightness_adjust:
            import random
            shift = random.choice([0.65, 0.75, 1.25, 1.40])
            enhancer = ImageEnhance.Brightness(face)
            face = enhancer.enhance(shift)

        face = face.resize((target_w, target_h), Image.LANCZOS)

        # ── Blend face edges into card background for seamless paste ──────
        # Create a feathered mask so the face edges aren't razor-sharp
        mask = Image.new("L", (target_w, target_h), 255)
        mask_arr = np.array(mask)
        feather = 3  # pixels of feathering at edges
        for i in range(feather):
            alpha = int(255 * (i + 1) / (feather + 1))
            mask_arr[i, :] = min(mask_arr[i, :].min(), alpha)
            mask_arr[-(i+1), :] = min(mask_arr[-(i+1), :].min(), alpha)
            mask_arr[:, i] = np.minimum(mask_arr[:, i], alpha)
            mask_arr[:, -(i+1)] = np.minimum(mask_arr[:, -(i+1)], alpha)
        mask = Image.fromarray(mask_arr)

        card.paste(face, (x1, y1), mask)
    except Exception as e:
        logger.warning(f"Face paste failed ({face_img_path}): {e}")
    return card


# ─── Main PAN Card Compositor ─────────────────────────────────────────────────

def compose_pan_card(
    template_path: str,
    face_img_path: str,
    name: str,
    father_name: str,
    dob: str,
    pan_number: str,
    output_path: str,
    tamper_instructions: Optional[dict] = None,
) -> str:
    """
    Produce a synthetic PAN card by placing new personal data onto a template.

    Uses background-aware inpainting (no white patches) and realistic text
    rendering (ink variation, blur, noise) so the result looks like a real
    scanned document.

    Args:
        template_path:  Cleaned PAN template PNG path.
        face_img_path:  Face image to paste in.
        name:           Applicant full name.
        father_name:    Father's full name.
        dob:            Date of birth "DD/MM/YYYY".
        pan_number:     PAN in "AAAPL1234C" format.
        output_path:    Where to save the result.
        tamper_instructions: Optional dict for fake generation controls.

    Returns:
        output_path
    """
    import random as _random

    card = Image.open(template_path).convert("RGB")
    w, h = card.size

    tamper = tamper_instructions or {}
    rng = _random.Random()  # per-card randomness for rendering variation

    face_bright  = tamper.get("face_brightness", False)
    bad_fonts    = tamper.get("font_tamper_fields", [])
    shift_fields = tamper.get("text_shift_fields", [])
    space_fields = tamper.get("char_spacing_fields", [])

    # ── Font sizes (calibrated to card height) ────────────────────────────
    name_sz      = max(8, int(h * PAN_FONT_SIZE_NAME))
    father_sz    = max(8, int(h * PAN_FONT_SIZE_FATHER))
    pan_num_sz   = max(10, int(h * PAN_FONT_SIZE_PAN_NUM))
    dob_sz       = max(8, int(h * PAN_FONT_SIZE_DOB))

    def _get_target_font(field_id: str, default_sz: int, bold: bool):
        if field_id in bad_fonts:
            try:
                return ImageFont.truetype("C:/Windows/Fonts/Arial.ttf", default_sz)
            except:
                return _get_font(default_sz, bold=bold)
        return _get_font(default_sz, bold=bold)

    font_name      = _get_target_font("name", name_sz, bold=True)
    font_father    = _get_target_font("father_name", father_sz, bold=True)
    font_pan_num   = _get_target_font("pan_num", pan_num_sz, bold=True)
    font_dob       = _get_target_font("dob", dob_sz, bold=True)

    # Skip background fill as user requested direct pasting
    face_region_px = _rel_to_px(PAN_REGIONS["face"], w, h)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Paste face (with feathered edges)
    # ══════════════════════════════════════════════════════════════════════
    card = _paste_face(card, face_img_path, face_region_px,
                       brightness_adjust=face_bright)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Draw text with realistic ink rendering
    # ══════════════════════════════════════════════════════════════════════

    # PAN Number — bold, centred
    pan_px = _rel_to_px(PAN_REGIONS["pan_num"], w, h)
    card = draw_text_realistic(
        card, pan_number, pan_px, font_pan_num, PAN_TEXT_COLOR_NUM,
        align="center",
        tamper_shift=("pan_num" in shift_fields),
        tamper_spacing=("pan_num" in space_fields),
        rng=rng,
    )
    card = apply_blend_effects(card, pan_px, rng=rng)

    # Applicant Name — uppercase, left-aligned
    name_px = _rel_to_px(PAN_REGIONS["name"], w, h)
    card = draw_text_realistic(
        card, name.upper(), name_px, font_name, PAN_TEXT_COLOR,
        align="left",
        tamper_shift=("name" in shift_fields),
        tamper_spacing=("name" in space_fields),
        rng=rng,
    )
    card = apply_blend_effects(card, name_px, rng=rng)

    # Father's Name — uppercase, left-aligned
    father_px = _rel_to_px(PAN_REGIONS["father_name"], w, h)
    card = draw_text_realistic(
        card, father_name.upper(), father_px, font_father, PAN_TEXT_COLOR,
        align="left",
        tamper_shift=("father_name" in shift_fields),
        tamper_spacing=("father_name" in space_fields),
        rng=rng,
    )
    card = apply_blend_effects(card, father_px, rng=rng)

    # DOB — DD/MM/YYYY
    dob_px = _rel_to_px(PAN_REGIONS["dob"], w, h)
    card = draw_text_realistic(
        card, dob, dob_px, font_dob, PAN_TEXT_COLOR,
        align="left",
        tamper_shift=("dob" in shift_fields),
        tamper_spacing=("dob" in space_fields),
        rng=rng,
    )
    card = apply_blend_effects(card, dob_px, rng=rng)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Save
    # ══════════════════════════════════════════════════════════════════════
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    card.save(output_path, format="JPEG", quality=94)
    return output_path
