"""
pan_pipeline.py — Full PAN card dataset generation orchestrator.

Steps:
  1. Extract clean PAN template from the sample PAN card image
  2. Analyze face images (gender + age via DeepFace, uses existing cache)
  3. Generate PAN_TARGET_REAL real synthetic cards (label=0)
     - Semantically consistent: face gender ↔ name gender, face age ↔ DOB
     - Father and applicant share the SAME surname
     - PAN number in valid govt format (AAAPL1234C)
  4. Generate PAN_TARGET_FAKE fake/forged cards (label=1)
     - 2-3 tampering categories per card
  5. Write output_pan/pan_dataset.csv

Usage:
  python -m src.pan_pipeline                # full run (10 real + 10 fake)
  python -m src.pan_pipeline --n-real 50 --n-fake 50
"""

import argparse
import io
import json
import logging
import os
import random
import string
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from .pan_config import (
    BASE_DIR,
    FACES_DIR,
    MIN_FACE_IMG_SIZE_KB,
    PAN_AUG_SETTINGS,
    PAN_DATASET_CSV,
    PAN_ENTITY_TYPES,
    PAN_FAKE_CATEGORIES_PROBS,
    PAN_OUTPUT_DIR,
    PAN_OUTPUT_FAKE,
    PAN_OUTPUT_REAL,
    PAN_REGIONS,
    PAN_SAMPLE,
    PAN_TARGET_FAKE,
    PAN_TARGET_REAL,
    PAN_TEMPLATE_META,
    PAN_TEMPLATES_DIR,
    RANDOM_SEED,
)
from .pan_card_composer import compose_pan_card

# Reuse face analyzer + data_utils from the existing Aadhaar pipeline
from .face_analyzer import analyze_all_faces, partition_by_gender
from .data_utils import (
    MALE_FIRST_NAMES,
    FEMALE_FIRST_NAMES,
    SURNAMES,
    generate_dob,
    get_mismatched_dob,
)

logger = logging.getLogger(__name__)

# ─── Setup ────────────────────────────────────────────────────────────────────

def _setup_dirs() -> None:
    for d in (PAN_OUTPUT_REAL, PAN_OUTPUT_FAKE, PAN_TEMPLATES_DIR):
        os.makedirs(d, exist_ok=True)


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ─── PAN Number Generator ────────────────────────────────────────────────────

def generate_pan_number(surname: str, invalid: bool = False,
                        rng: random.Random = None) -> str:
    """
    Generate a PAN number in valid Indian govt format: AAAPL1234C

    Format:
      Pos 1-3: Random uppercase letters
      Pos 4:   Entity type ('P' for person)
      Pos 5:   First letter of surname (uppercase)
      Pos 6-9: 4 sequential digits (0001-9999)
      Pos 10:  Random uppercase check letter

    If invalid=True, produces malformed PAN (wrong digit count / mixed case).
    """
    if rng is None:
        rng = random

    if not invalid:
        first3 = "".join(rng.choices(string.ascii_uppercase, k=3))
        entity = rng.choice(PAN_ENTITY_TYPES)  # 'P' for person
        surname_letter = surname[0].upper() if surname else "A"
        digits = f"{rng.randint(1, 9999):04d}"
        check = rng.choice(string.ascii_uppercase)
        return f"{first3}{entity}{surname_letter}{digits}{check}"
    else:
        # Produce a clearly invalid PAN
        attack = rng.choice(["wrong_digits", "wrong_length", "lowercase"])
        first3 = "".join(rng.choices(string.ascii_uppercase, k=3))
        entity = rng.choice(PAN_ENTITY_TYPES)
        surname_letter = rng.choice(string.ascii_uppercase)

        if attack == "wrong_digits":
            # 3 or 5 digits instead of 4
            n_digits = rng.choice([3, 5])
            digits = "".join([str(rng.randint(0, 9)) for _ in range(n_digits)])
            check = rng.choice(string.ascii_uppercase)
            return f"{first3}{entity}{surname_letter}{digits}{check}"
        elif attack == "wrong_length":
            # Either 9 or 11 characters total
            if rng.random() < 0.5:
                return f"{first3}{entity}{surname_letter}{rng.randint(100,999):03d}{rng.choice(string.ascii_uppercase)}"
            else:
                digits = f"{rng.randint(1, 9999):04d}"
                extra = rng.choice(string.ascii_uppercase)
                check = rng.choice(string.ascii_uppercase)
                return f"{first3}{entity}{surname_letter}{digits}{check}{extra}"
        else:  # lowercase
            digits = f"{rng.randint(1, 9999):04d}"
            check = rng.choice(string.ascii_lowercase)  # lowercase = invalid
            return f"{first3.lower()}{entity}{surname_letter}{digits}{check}"


# ─── Indian Name Generators (PAN-specific: shared surname) ────────────────────

def generate_pan_identity(gender: str, rng: random.Random = None) -> dict:
    """
    Generate a consistent PAN card identity.

    Returns:
        {
            "name": "FIRSTNAME SURNAME",
            "father_name": "FATHERFIRST SURNAME",
            "surname": "SURNAME",
            "gender": "male" | "female"
        }

    Father and applicant share the same surname.
    Father's first name is always male.
    """
    if rng is None:
        rng = random

    gender = gender.lower().strip()
    surname = rng.choice(SURNAMES)

    # Applicant first name based on detected gender
    if gender in ("male", "man", "m"):
        first_name = rng.choice(MALE_FIRST_NAMES)
    else:
        first_name = rng.choice(FEMALE_FIRST_NAMES)

    # Father is always male
    father_first = rng.choice(MALE_FIRST_NAMES)

    return {
        "name": f"{first_name} {surname}",
        "father_name": f"{father_first} {surname}",
        "surname": surname,
        "gender": gender,
    }


def get_mismatched_pan_identity(actual_gender: str, rng: random.Random = None) -> dict:
    """Generate a PAN identity with OPPOSITE gender name (for fakes)."""
    if rng is None:
        rng = random
    opposite = "female" if actual_gender in ("male", "man", "m") else "male"
    return generate_pan_identity(opposite, rng)


# ─── Template Extraction ─────────────────────────────────────────────────────

def _sample_background_color(img: np.ndarray) -> tuple:
    """Sample median background colour from corner patches."""
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
    return (int(median[2]), int(median[1]), int(median[0]))  # BGR → RGB


def _rel_to_px(region: tuple, w: int, h: int) -> tuple:
    return (int(region[0] * w), int(region[1] * h),
            int(region[2] * w), int(region[3] * h))


def extract_pan_template(src_path: str, dst_path: str) -> dict:
    """
    Create a PAN template by masking all VALUE regions with background colour.
    QR code region is left untouched.
    """
    img_bgr = cv2.imread(src_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {src_path}")

    h, w = img_bgr.shape[:2]
    bg_color = _sample_background_color(img_bgr)
    fill_bgr = (bg_color[2], bg_color[1], bg_color[0])

    # Masking bypassed — directly use the unaltered background

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_img.save(dst_path, format="PNG")

    return {
        "source": os.path.basename(src_path),
        "template": os.path.basename(dst_path),
        "card_size": [w, h],
        "bg_color": list(bg_color),
    }


def ensure_pan_template() -> dict:
    """Extract PAN template from the sample card (or load cached metadata)."""
    if os.path.exists(PAN_TEMPLATE_META):
        with open(PAN_TEMPLATE_META, "r") as f:
            meta = json.load(f)
            if meta:
                logger.info(f"Loaded existing PAN template metadata.")
                return meta

    logger.info("Extracting PAN template from sample card …")
    os.makedirs(PAN_TEMPLATES_DIR, exist_ok=True)
    dst = os.path.join(PAN_TEMPLATES_DIR, "pan_template_001.png")
    meta = extract_pan_template(PAN_SAMPLE, dst)

    with open(PAN_TEMPLATE_META, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  ✓ PAN template saved → {dst}")
    return meta


# ─── PAN-specific Augmentor ───────────────────────────────────────────────────
# Reimplemented here so we don't touch the Aadhaar augmentor.py

def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def pan_jpeg_compress(img, rng):
    q = rng.randint(15, 45)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def pan_gaussian_blur(img, rng):
    radius = rng.uniform(0.6, 1.8)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def pan_gaussian_noise(img, rng):
    arr = _to_np(img).astype(np.float32)
    sigma = rng.uniform(5, 20)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return _to_pil(arr)


def pan_color_jitter(img, rng):
    arr = _to_np(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-12, 12)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.15), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.88, 1.12), 0, 255)
    return _to_pil(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


def pan_affine_warp(img, rng):
    arr = _to_np(img)
    h, w = arr.shape[:2]
    angle = rng.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    M[0, 1] += rng.uniform(-0.02, 0.02) * h
    warped = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return _to_pil(warped)


def pan_text_region_blur(img, rng):
    """Blur specific text regions to simulate smudge/erasure."""
    arr = _to_np(img)
    h, w = arr.shape[:2]
    targets = rng.sample(["name", "father_name", "dob", "pan_num"],
                         k=rng.randint(1, 2))
    for key in targets:
        x1, y1, x2, y2 = _rel_to_px(PAN_REGIONS[key], w, h)
        patch = arr[y1:y2, x1:x2]
        ksize = rng.choice([7, 11, 15, 21])
        blurred = cv2.GaussianBlur(patch, (ksize, ksize), 0)
        arr[y1:y2, x1:x2] = blurred
    return _to_pil(arr)


def pan_border_artifact(img, rng):
    """Draw faint rectangle around tampered region (copy-paste trace)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    key = rng.choice(["face", "name", "pan_num", "father_name"])
    x1, y1, x2, y2 = _rel_to_px(PAN_REGIONS[key], w, h)
    color = (rng.randint(100, 200), rng.randint(100, 200), rng.randint(100, 200))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=rng.randint(1, 3))
    return img


def pan_edge_crop(img, rng):
    w, h = img.size
    crop_w = int(w * rng.uniform(0.01, 0.05))
    crop_h = int(h * rng.uniform(0.01, 0.05))
    left = rng.choice([0, crop_w])
    top = rng.choice([0, crop_h])
    right = w - rng.choice([0, crop_w])
    bottom = h - rng.choice([0, crop_h])
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.LANCZOS)


def apply_pan_augmentations(img: Image.Image, categories: list,
                            rng: random.Random) -> Image.Image:
    """Apply post-render visual augmentations based on chosen forgery categories."""
    applied = []

    if "image_quality" in categories:
        attacks = rng.sample(["jpeg", "blur", "noise", "color"], k=rng.randint(1, 2))
        fn_map = {"jpeg": pan_jpeg_compress, "blur": pan_gaussian_blur,
                  "noise": pan_gaussian_noise, "color": pan_color_jitter}
        for a in attacks:
            img = fn_map[a](img, rng)
            applied.append(a)

    if "structural" in categories:
        img = pan_affine_warp(img, rng)
        applied.append("warp")

    if "border_crop" in categories:
        fn = rng.choice([pan_border_artifact, pan_edge_crop])
        img = fn(img, rng)
        applied.append("border")

    if "text_tampering" in categories:
        if rng.random() < 0.5:
            img = pan_text_region_blur(img, rng)
            applied.append("text_blur")

    logger.debug(f"PAN augmentations applied: {applied}")
    return img


# ─── Step: Generate Real PAN Samples ─────────────────────────────────────────

def generate_real_pan_samples(
    template_meta: dict,
    face_analysis: dict,
    records: list,
    rng: random.Random,
    n: int = PAN_TARGET_REAL,
) -> None:
    """
    Generate n semantically consistent real PAN samples (label=0).
    Face gender ↔ name gender, face age ↔ DOB, surname matches father.
    """
    logger.info(f"Generating {n} real PAN card samples …")
    males, females = partition_by_gender(face_analysis)
    all_faces = males + females

    if not all_faces:
        logger.warning("No usable face images — skipping.")
        return

    template_path = os.path.join(PAN_TEMPLATES_DIR, template_meta["template"])
    if not os.path.exists(template_path):
        logger.error(f"Template not found: {template_path}")
        return

    rng.shuffle(all_faces)
    generated = 0

    pbar = tqdm(total=n, desc="PAN Real", unit="card")

    while generated < n:
        face_idx = generated % len(all_faces)
        face_fn, face_info = all_faces[face_idx]
        gender = face_info["gender"]
        age = face_info["age"]

        # Consistent identity
        identity = generate_pan_identity(gender, rng)
        dob = generate_dob(age, variance=3, rng=rng)
        pan_num = generate_pan_number(identity["surname"], invalid=False, rng=rng)

        out_name = f"pan_real_{generated + 1:04d}.jpg"
        out_path = os.path.join(PAN_OUTPUT_REAL, out_name)

        try:
            compose_pan_card(
                template_path=template_path,
                face_img_path=os.path.join(FACES_DIR, face_fn),
                name=identity["name"],
                father_name=identity["father_name"],
                dob=dob,
                pan_number=pan_num,
                output_path=out_path,
            )
        except Exception as e:
            logger.debug(f"compose_pan_card failed: {e}")
            continue

        records.append({
            "image_path": os.path.relpath(out_path),
            "label": 0,
            "label_name": "real",
            "source": "synthetic",
            "gender": gender,
            "age": age,
            "name": identity["name"],
            "father_name": identity["father_name"],
            "dob": dob,
            "pan_num": pan_num,
            "face_file": face_fn,
            "notes": f"consistent|gender={gender}|age={age}|surname={identity['surname']}",
        })
        generated += 1
        pbar.update(1)

    pbar.close()
    logger.info(f"  ✓ {generated} real PAN cards generated.")


# ─── Step: Generate Fake PAN Samples ─────────────────────────────────────────

def generate_fake_pan_samples(
    template_meta: dict,
    face_analysis: dict,
    records: list,
    rng: random.Random,
    n: int = PAN_TARGET_FAKE,
) -> None:
    """
    Generate n fake/forged PAN samples (label=1).
    Applies 2-3 tampering categories per card.
    """
    logger.info(f"Generating {n} fake PAN card samples …")
    males, females = partition_by_gender(face_analysis)
    all_faces = males + females

    if not all_faces:
        logger.warning("No usable face images — skipping.")
        return

    template_path = os.path.join(PAN_TEMPLATES_DIR, template_meta["template"])
    if not os.path.exists(template_path):
        logger.error(f"Template not found: {template_path}")
        return

    cats = list(PAN_FAKE_CATEGORIES_PROBS.keys())
    probs = list(PAN_FAKE_CATEGORIES_PROBS.values())

    rng.shuffle(all_faces)
    generated = 0

    pbar = tqdm(total=n, desc="PAN Fake", unit="card")

    while generated < n:
        face_idx = generated % len(all_faces)
        face_fn, face_info = all_faces[face_idx]
        actual_gender = face_info["gender"]
        actual_age = face_info["age"]

        # Step 1: Build TRUE base identity (consistent)
        identity = generate_pan_identity(actual_gender, rng)
        dob = generate_dob(actual_age, variance=3, rng=rng)
        pan_num = generate_pan_number(identity["surname"], invalid=False, rng=rng)

        # Step 2: Choose 2-3 tampering categories
        num_cats = rng.randint(2, 3)
        chosen_cats = []
        while len(chosen_cats) < num_cats:
            c = rng.choices(cats, weights=probs, k=1)[0]
            if c not in chosen_cats:
                chosen_cats.append(c)

        tamper_instructions = {
            "face_brightness": False,
            "font_tamper_fields": [],
            "text_shift_fields": [],
            "char_spacing_fields": [],
        }

        # Step 3: Apply Semantic Inconsistencies
        if "semantic" in chosen_cats:
            subtype = rng.choice(["gender", "age", "pan"])
            if subtype == "gender":
                # Use opposite gender name but keep face
                identity = get_mismatched_pan_identity(actual_gender, rng)
            elif subtype == "age":
                dob = get_mismatched_dob(actual_age, offset_years=25, rng=rng)
            else:
                pan_num = generate_pan_number(identity["surname"],
                                              invalid=True, rng=rng)

        # Step 4: Partial Document Editing
        if "partial_editing" in chosen_cats:
            subtype = rng.choice(["name", "dob", "pan_num", "father_name"])
            if subtype == "name":
                identity["name"] = get_mismatched_pan_identity(actual_gender, rng)["name"]
                tamper_instructions["text_shift_fields"].append("name")
                tamper_instructions["font_tamper_fields"].append("name")
            elif subtype == "dob":
                dob = get_mismatched_dob(actual_age, offset_years=25, rng=rng)
                tamper_instructions["text_shift_fields"].append("dob")
            elif subtype == "pan_num":
                pan_num = generate_pan_number(identity["surname"],
                                              invalid=True, rng=rng)
                tamper_instructions["char_spacing_fields"].append("pan_num")
            else:  # father_name
                # Different surname for father (inconsistent)
                fake_father_identity = generate_pan_identity("male", rng)
                identity["father_name"] = fake_father_identity["father_name"]
                tamper_instructions["text_shift_fields"].append("father_name")
                tamper_instructions["font_tamper_fields"].append("father_name")

        # Step 5: Face Tampering
        if "face_tampering" in chosen_cats:
            tamper_instructions["face_brightness"] = True

        # Step 6: Text Tampering
        if "text_tampering" in chosen_cats:
            targets = rng.sample(["name", "dob", "father_name", "pan_num"],
                                 k=rng.randint(1, 2))
            tamper_instructions["font_tamper_fields"].extend(targets)
            if rng.random() > 0.5:
                tamper_instructions["char_spacing_fields"].append(
                    rng.choice(["name", "pan_num"]))

        out_name = f"pan_fake_{generated + 1:04d}.jpg"
        out_path = os.path.join(PAN_OUTPUT_FAKE, out_name)

        try:
            compose_pan_card(
                template_path=template_path,
                face_img_path=os.path.join(FACES_DIR, face_fn),
                name=identity["name"],
                father_name=identity["father_name"],
                dob=dob,
                pan_number=pan_num,
                output_path=out_path,
                tamper_instructions=tamper_instructions,
            )

            # Apply post-render visual augmentations
            img = Image.open(out_path)
            img = apply_pan_augmentations(img, chosen_cats, rng)
            img.save(out_path, format="JPEG", quality=rng.randint(85, 95))

        except Exception as e:
            logger.debug(f"Fake PAN generation failed: {e}")
            continue

        records.append({
            "image_path": os.path.relpath(out_path),
            "label": 1,
            "label_name": "fake",
            "source": "synthetic",
            "gender": actual_gender,
            "age": actual_age,
            "name": identity["name"],
            "father_name": identity["father_name"],
            "dob": dob,
            "pan_num": pan_num,
            "face_file": face_fn,
            "notes": f"cats={'+'.join(chosen_cats)}|face_gender={actual_gender}",
        })
        generated += 1
        pbar.update(1)

    pbar.close()
    logger.info(f"  ✓ {generated} fake PAN cards generated.")


# ─── Write Dataset CSV ────────────────────────────────────────────────────────

def write_pan_csv(records: list) -> None:
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(PAN_DATASET_CSV), exist_ok=True)
    df.to_csv(PAN_DATASET_CSV, index=False)

    n_real = (df["label"] == 0).sum()
    n_fake = (df["label"] == 1).sum()
    logger.info(f"\n{'='*55}")
    logger.info(f"  PAN Dataset saved → {PAN_DATASET_CSV}")
    logger.info(f"  Total samples : {len(df)}")
    logger.info(f"  Real  (0)     : {n_real}")
    logger.info(f"  Fake  (1)     : {n_fake}")
    logger.info(f"  Balance ratio : {n_real/(n_fake+1e-9):.2f}:1")
    logger.info(f"{'='*55}")


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def run_pan_pipeline(
    n_real: int = PAN_TARGET_REAL,
    n_fake: int = PAN_TARGET_FAKE,
) -> None:
    _setup_dirs()
    rng = random.Random(RANDOM_SEED)
    records: list = []

    # Step 1: Extract PAN template
    template_meta = ensure_pan_template()

    # Step 2: Analyse faces (reuses existing cache)
    logger.info("Analysing face images (gender + age) …")
    face_analysis = analyze_all_faces(FACES_DIR)
    if not face_analysis:
        logger.error("No usable face images found. Aborting.")
        sys.exit(1)
    logger.info(f"  ✓ {len(face_analysis)} usable face images.")

    # Step 3: Generate real PAN cards
    generate_real_pan_samples(template_meta, face_analysis, records, rng, n=n_real)

    # Step 4: Generate fake PAN cards
    generate_fake_pan_samples(template_meta, face_analysis, records, rng, n=n_fake)

    # Step 5: Write CSV
    write_pan_csv(records)
    logger.info("PAN pipeline complete! ✅")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PAN Card Dataset Generation Pipeline"
    )
    p.add_argument("--n-real", type=int, default=PAN_TARGET_REAL,
                   help=f"Number of real PAN samples (default {PAN_TARGET_REAL}).")
    p.add_argument("--n-fake", type=int, default=PAN_TARGET_FAKE,
                   help=f"Number of fake PAN samples (default {PAN_TARGET_FAKE}).")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    run_pan_pipeline(
        n_real=args.n_real,
        n_fake=args.n_fake,
    )
