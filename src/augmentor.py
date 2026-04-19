"""
augmentor.py — Fake-sample augmentations for the KYC Aadhaar pipeline.

Each augmentation is independent and probability-gated via AUG_PROBS in config.
Functions operate on PIL Image → return PIL Image (so they can be composed).

Public API:
    apply_fake_augmentations(pil_img, rng) → PIL Image
"""

import io
import logging
import random
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from .config import REGIONS, AUG_SETTINGS

logger = logging.getLogger(__name__)


# ─── Helper: PIL ↔ NumPy ─────────────────────────────────────────────────────

def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def _rel_to_px(region: tuple, w: int, h: int) -> tuple:
    x1 = int(region[0] * w)
    y1 = int(region[1] * h)
    x2 = int(region[2] * w)
    y2 = int(region[3] * h)
    return x1, y1, x2, y2


# ─── Individual Augmentations ─────────────────────────────────────────────────

def jpeg_compress(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate JPEG compression artefacts (quality 15–45)."""
    q = rng.randint(15, 45)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def gaussian_blur_full(img: Image.Image, rng: random.Random) -> Image.Image:
    """Light gaussian blur on the entire card (simulates image smoothing)."""
    radius = rng.uniform(0.6, 1.8)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def gaussian_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    """Add random Gaussian noise."""
    arr = _to_np(img).astype(np.float32)
    sigma = rng.uniform(5, 20)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return _to_pil(arr)


def affine_warp(img: Image.Image, rng: random.Random) -> Image.Image:
    """Slight affine transform (rotation ±5°, mild shear)."""
    arr = _to_np(img)
    h, w = arr.shape[:2]
    angle = rng.uniform(-5, 5)
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Add small shear
    shear_x = rng.uniform(-0.02, 0.02)
    M[0, 1] += shear_x * h
    warped = cv2.warpAffine(arr, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return _to_pil(warped)


def face_shift(img: Image.Image, rng: random.Random) -> Image.Image:
    """Shift the face region by ±10–30 px (misalignment artefact)."""
    arr = _to_np(img)
    h, w = arr.shape[:2]

    x1, y1, x2, y2 = _rel_to_px(REGIONS["face"], w, h)
    face_crop = arr[y1:y2, x1:x2].copy()

    # Fill original region with surrounding median colour
    bg_color = np.median(arr[:10, :10], axis=(0, 1)).astype(np.uint8)
    arr[y1:y2, x1:x2] = bg_color

    # Shift amount
    dx = rng.randint(10, 30) * rng.choice([-1, 1])
    dy = rng.randint(10, 30) * rng.choice([-1, 1])

    nx1 = max(0, x1 + dx)
    ny1 = max(0, y1 + dy)
    nx2 = min(w, nx1 + (x2 - x1))
    ny2 = min(h, ny1 + (y2 - y1))

    ch = ny2 - ny1
    cw = nx2 - nx1
    arr[ny1:ny2, nx1:nx2] = face_crop[:ch, :cw]
    return _to_pil(arr)


def color_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    """Hue/Saturation/Value jitter to simulate scanned document colour shift."""
    arr = _to_np(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-12, 12)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.15), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.88, 1.12), 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return _to_pil(rgb)


def text_region_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Blur one or two text fields (name / dob / gender / aadhaar_num).
    Simulates selective text tampering / partial erasure.
    """
    arr = _to_np(img)
    h, w = arr.shape[:2]
    text_regions = ["name", "dob", "gender", "aadhaar_num"]
    # Pick 1–2 fields to blur
    targets = rng.sample(text_regions, k=rng.randint(1, 2))
    for key in targets:
        x1, y1, x2, y2 = _rel_to_px(REGIONS[key], w, h)
        patch = arr[y1:y2, x1:x2]
        ksize = rng.choice([7, 11, 15, 21])
        blurred = cv2.GaussianBlur(patch, (ksize, ksize), 0)
        arr[y1:y2, x1:x2] = blurred
    return _to_pil(arr)


def border_artifact(img: Image.Image, rng: random.Random) -> Image.Image:
    """Draw a faint rectangle around the tampered text/face zone (copy-paste trace)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    key = rng.choice(["face", "name", "aadhaar_num"])
    x1, y1, x2, y2 = _rel_to_px(REGIONS[key], w, h)
    # Very faint outline
    alpha = rng.randint(40, 100)
    color = (rng.randint(100, 200), rng.randint(100, 200), rng.randint(100, 200))
    lw = rng.randint(1, 3)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
    return img


def edge_crop(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a slightly cropped image (e.g. edge artifacts)."""
    w, h = img.size
    crop_w = int(w * rng.uniform(0.01, 0.05))
    crop_h = int(h * rng.uniform(0.01, 0.05))

    left   = rng.choice([0, crop_w])
    top    = rng.choice([0, crop_h])
    right  = w - rng.choice([0, crop_w])
    bottom = h - rng.choice([0, crop_h])

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.LANCZOS)


# ─── Augmentation Registry ────────────────────────────────────────────────────

_AUG_REGISTRY: dict[str, Callable] = {
    # Image Quality
    "jpeg_compress":   jpeg_compress,
    "gaussian_blur":   gaussian_blur_full,
    "gaussian_noise":  gaussian_noise,
    "color_jitter":    color_jitter,

    # Structural
    "affine_warp":     affine_warp,

    # Border / Crop
    "border_artifact": border_artifact,
    "edge_crop":       edge_crop,

    # Sub-region / Text
    "text_blur":       text_region_blur,
}


# ─── Public Entrypoint ────────────────────────────────────────────────────────

def apply_fake_augmentations(img: Image.Image, applied_categories: list[str], rng: random.Random = None) -> Image.Image:
    """
    Apply global image augmentations based on the chosen categories.
    """
    if rng is None:
        rng = random

    applied = []

    if "image_quality" in applied_categories:
        # Pick 1 or 2 quality hits
        attacks = rng.sample(["jpeg_compress", "gaussian_blur", "gaussian_noise", "color_jitter"], k=rng.randint(1, 2))
        for attack in attacks:
            img = _AUG_REGISTRY[attack](img, rng)
            applied.append(attack)

    if "structural" in applied_categories:
        img = _AUG_REGISTRY["affine_warp"](img, rng)
        applied.append("affine_warp")

    if "border_crop" in applied_categories:
        attack = rng.choice(["border_artifact", "edge_crop"])
        img = _AUG_REGISTRY[attack](img, rng)
        applied.append(attack)

    if "text_tampering" in applied_categories:
        # Some text tampering is handled by card_composer (fonts, spacing, shifts).
        # We also randomly apply post-process text blur here.
        if rng.random() < 0.5:
            img = _AUG_REGISTRY["text_blur"](img, rng)
            applied.append("text_blur")

    # (Note: "face_tampering", "semantic", "partial_editing" are fully handled in card_composer and pipeline)

    logger.debug(f"Applied visual augmentations: {applied}")
    return img


# ─── CLI Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python -m src.augmentor <path_to_card_image>")
        sys.exit(1)
    inp = sys.argv[1]
    out = inp.replace(".", "_augmented.")
    rng = random.Random(42)
    img = Image.open(inp)
    aug = apply_fake_augmentations(img, rng)
    aug.save(out)
    print(f"Augmented image saved: {out}")
