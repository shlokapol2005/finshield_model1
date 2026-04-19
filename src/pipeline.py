"""
pipeline.py — Full KYC Aadhaar dataset generation orchestrator.

Steps:
  1. Copy 90 original real Aadhaar images → output/real/  (label=0)
  2. Extract clean templates from real cards → data/templates/
  3. Analyze all TPDNE face images (gender + age via DeepFace) → cache
  4. Generate TARGET_REAL_SYNTHETIC real synthetic cards (label=0)
     - Semantically consistent: face gender ↔ name gender, face age ↔ DOB
  5. Generate TARGET_FAKE fake/forged cards (label=1)
     - Intentional mismatch: gender swap OR age mismatch (or both)
     - Plus visual augmentation (blur, noise, warp, etc.)
  6. Write output/dataset.csv

Usage:
  python -m src.pipeline              # full run
  python -m src.pipeline --dry-run    # analyse faces but don't generate cards
  python -m src.pipeline --skip-extract  # reuse existing templates
"""

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import (
    DATASET_CSV,
    FACES_DIR,
    OUTPUT_FAKE,
    OUTPUT_REAL,
    RANDOM_SEED,
    REAL_DIR,
    TARGET_FAKE,
    TARGET_REAL_SYNTHETIC,
    TEMPLATES_DIR,
)
from .augmentor import apply_fake_augmentations
from .card_composer import compose_card
from .data_utils import (
    generate_aadhaar_number,
    generate_dob,
    generate_indian_name,
    get_mismatched_dob,
    get_mismatched_name,
)
from .face_analyzer import analyze_all_faces, partition_by_gender
from .template_extractor import extract_all_templates, load_template_metadata

from PIL import Image

logger = logging.getLogger(__name__)

# ─── Setup ────────────────────────────────────────────────────────────────────

def _setup_dirs() -> None:
    for d in (OUTPUT_REAL, OUTPUT_FAKE, TEMPLATES_DIR):
        os.makedirs(d, exist_ok=True)


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ─── Step 1: Copy Original Real Cards ─────────────────────────────────────────

def copy_real_cards(records: list) -> None:
    """Copy original 90 real Aadhaar images to output/real/ and record them."""
    logger.info("Step 1 — Copying original real Aadhaar cards …")
    image_files = sorted([
        f for f in os.listdir(REAL_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    ])
    for fname in tqdm(image_files, desc="Copying real cards", unit="img"):
        src = os.path.join(REAL_DIR, fname)
        dst = os.path.join(OUTPUT_REAL, f"real_orig_{fname}")
        shutil.copy2(src, dst)
        records.append({
            "image_path":    os.path.relpath(dst),
            "label":         0,
            "label_name":    "real",
            "source":        "original",
            "gender":        "",
            "age":           "",
            "dob":           "",
            "aadhaar_num":   "",
            "notes":         "original_real_card",
        })
    logger.info(f"  ✓ {len(image_files)} original real cards copied.")


# ─── Step 2: Extract Templates ────────────────────────────────────────────────

def ensure_templates(skip_extract: bool) -> list:
    """Extract templates (or load cached metadata)."""
    if skip_extract:
        meta = load_template_metadata()
        if meta:
            logger.info(f"Step 2 — Loaded {len(meta)} existing templates (--skip-extract).")
            return meta
        logger.warning("--skip-extract set but no metadata found. Extracting now.")

    logger.info("Step 2 — Extracting templates from real Aadhaar cards …")
    return extract_all_templates()


# ─── Step 3: Analyse Faces ───────────────────────────────────────────────────

def analyse_faces() -> dict:
    """DeepFace analysis of all TPDNE face images."""
    logger.info("Step 3 — Analysing face images (gender + age) …")
    face_analysis = analyze_all_faces(FACES_DIR)
    logger.info(f"  ✓ {len(face_analysis)} usable face images analysed.")
    return face_analysis


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pick_template(templates: list, rng: random.Random) -> dict:
    return rng.choice(templates)


def _face_path(filename: str) -> str:
    return os.path.join(FACES_DIR, filename)


def _gender_str(gender: str) -> str:
    return "Male" if gender == "male" else "Female"


# ─── Step 4: Generate Real Synthetic Samples ──────────────────────────────────

def generate_real_samples(
    templates: list,
    face_analysis: dict,
    records: list,
    rng: random.Random,
    n: int = TARGET_REAL_SYNTHETIC,
    dry_run: bool = False,
) -> None:
    """
    Generate n semantically consistent real samples (label=0).
    Face gender ↔ name gender ↔ DOB age ≈ face age.
    """
    logger.info(f"Step 4 — Generating {n} real synthetic samples …")
    males, females = partition_by_gender(face_analysis)
    all_faces      = males + females

    if not all_faces:
        logger.warning("No usable face images — skipping real synthetic generation.")
        return
    if not templates:
        logger.warning("No templates — skipping real synthetic generation.")
        return

    generated = 0
    attempts  = 0
    max_attempts = n * 5

    pbar = tqdm(total=n, desc="Real synthetic", unit="card")

    while generated < n and attempts < max_attempts:
        attempts += 1
        face_fn, face_info = rng.choice(all_faces)
        gender  = face_info["gender"]
        age     = face_info["age"]

        name         = generate_indian_name(gender, rng)
        dob          = generate_dob(age, variance=3, rng=rng)
        aadhaar_num  = generate_aadhaar_number(rng)
        template_meta = _pick_template(templates, rng)
        template_path = os.path.join(TEMPLATES_DIR, template_meta["template"])

        if not os.path.exists(template_path):
            continue

        out_name = f"real_synth_{generated + 1:04d}.jpg"
        out_path = os.path.join(OUTPUT_REAL, out_name)

        if not dry_run:
            try:
                face_region_px = tuple(template_meta["face_region"])
                compose_card(
                    template_path   = template_path,
                    face_img_path   = _face_path(face_fn),
                    name            = name,
                    dob             = dob,
                    gender          = _gender_str(gender),
                    aadhaar_number  = aadhaar_num,
                    output_path     = out_path,
                    face_region_px  = face_region_px,
                )
            except Exception as e:
                logger.debug(f"compose_card failed: {e}")
                continue

        records.append({
            "image_path":  os.path.relpath(out_path),
            "label":       0,
            "label_name":  "real",
            "source":      "synthetic",
            "gender":      gender,
            "age":         age,
            "dob":         dob,
            "aadhaar_num": aadhaar_num,
            "notes":       f"consistent|name_gender={gender}|face_age={age}",
        })
        generated += 1
        pbar.update(1)

    pbar.close()
    logger.info(f"  ✓ {generated} real synthetic cards generated.")


# ─── Step 5: Generate Fake Samples ────────────────────────────────────────────

def generate_fake_samples(
    templates: list,
    face_analysis: dict,
    records: list,
    rng: random.Random,
    n: int = TARGET_FAKE,
    dry_run: bool = False,
) -> None:
    """
    Generate n fake/forged samples (label=1).
    Applies one or more of:
      - Gender mismatch (male face + female name)
      - Age mismatch (young face + old DOB)
      - Visual augmentations (blur, noise, warp, etc.)
    """
    logger.info(f"Step 5 — Generating {n} fake/forged samples …")
    males, females = partition_by_gender(face_analysis)
    all_faces      = males + females

    if not all_faces:
        logger.warning("No usable face images — skipping fake generation.")
        return
    if not templates:
        logger.warning("No templates — skipping fake generation.")
        return

    # Fake type distribution: 40% gender-mismatch, 30% age-mismatch, 30% both
    generated = 0
    attempts  = 0
    max_attempts = n * 5

    pbar = tqdm(total=n, desc="Fake/forged", unit="card")

    while generated < n and attempts < max_attempts:
        attempts += 1
        face_fn, face_info = rng.choice(all_faces)
        actual_gender = face_info["gender"]
        actual_age    = face_info["age"]

        # -- Choose mismatch strategy --
        strategy = rng.choices(
            ["gender_mismatch", "age_mismatch", "both"],
            weights=[0.35, 0.30, 0.35],
        )[0]

        if strategy in ("gender_mismatch", "both"):
            name = get_mismatched_name(actual_gender, rng)
            card_gender = "Male" if actual_gender == "female" else "Female"
        else:
            name = generate_indian_name(actual_gender, rng)
            card_gender = _gender_str(actual_gender)

        if strategy in ("age_mismatch", "both"):
            dob = get_mismatched_dob(actual_age, offset_years=25, rng=rng)
        else:
            dob = generate_dob(actual_age, variance=3, rng=rng)

        aadhaar_num   = generate_aadhaar_number(rng)
        template_meta = _pick_template(templates, rng)
        template_path = os.path.join(TEMPLATES_DIR, template_meta["template"])

        if not os.path.exists(template_path):
            continue

        out_name = f"fake_{generated + 1:04d}.jpg"
        out_path = os.path.join(OUTPUT_FAKE, out_name)

        if not dry_run:
            try:
                face_region_px = tuple(template_meta["face_region"])
                compose_card(
                    template_path  = template_path,
                    face_img_path  = _face_path(face_fn),
                    name           = name,
                    dob            = dob,
                    gender         = card_gender,
                    aadhaar_number = aadhaar_num,
                    output_path    = out_path,
                    face_region_px = face_region_px,
                )
                # Apply visual augmentations
                img = Image.open(out_path)
                img = apply_fake_augmentations(img, rng)
                img.save(out_path, format="JPEG", quality=88)
            except Exception as e:
                logger.debug(f"Fake generation failed: {e}")
                continue

        records.append({
            "image_path":  os.path.relpath(out_path),
            "label":       1,
            "label_name":  "fake",
            "source":      "synthetic",
            "gender":      actual_gender,
            "age":         actual_age,
            "dob":         dob,
            "aadhaar_num": aadhaar_num,
            "notes":       f"strategy={strategy}|card_gender={card_gender}|face_gender={actual_gender}",
        })
        generated += 1
        pbar.update(1)

    pbar.close()
    logger.info(f"  ✓ {generated} fake cards generated.")


# ─── Step 6: Write Dataset CSV ────────────────────────────────────────────────

def write_csv(records: list) -> None:
    """Write the final labeled dataset CSV."""
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(DATASET_CSV), exist_ok=True)
    df.to_csv(DATASET_CSV, index=False)

    n_real = (df["label"] == 0).sum()
    n_fake = (df["label"] == 1).sum()
    logger.info(f"\n{'='*55}")
    logger.info(f"  Dataset saved → {DATASET_CSV}")
    logger.info(f"  Total samples : {len(df)}")
    logger.info(f"  Real  (0)     : {n_real}")
    logger.info(f"  Fake  (1)     : {n_fake}")
    logger.info(f"  Balance ratio : {n_real/(n_fake+1e-9):.2f}:1")
    logger.info(f"{'='*55}")


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def run_pipeline(
    dry_run: bool = False,
    skip_extract: bool = False,
    n_real: int = TARGET_REAL_SYNTHETIC,
    n_fake: int = TARGET_FAKE,
) -> None:
    _setup_dirs()
    rng = random.Random(RANDOM_SEED)
    records: list = []

    if dry_run:
        logger.info("⚠️  DRY RUN — no card images will be written.")

    # Step 1
    copy_real_cards(records)

    # Step 2
    templates = ensure_templates(skip_extract)

    # Step 3
    face_analysis = analyse_faces()
    if not face_analysis:
        logger.error("No usable face images found. Aborting.")
        sys.exit(1)

    # Step 4
    generate_real_samples(templates, face_analysis, records, rng, n=n_real, dry_run=dry_run)

    # Step 5
    generate_fake_samples(templates, face_analysis, records, rng, n=n_fake, dry_run=dry_run)

    # Step 6
    write_csv(records)
    logger.info("Pipeline complete! ✅")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KYC Aadhaar Dataset Generation Pipeline"
    )
    p.add_argument("--dry-run",      action="store_true",
                   help="Analyse faces only; do not write card images.")
    p.add_argument("--skip-extract", action="store_true",
                   help="Reuse existing templates (skip template extraction).")
    p.add_argument("--n-real",       type=int, default=TARGET_REAL_SYNTHETIC,
                   help=f"Number of real synthetic samples (default {TARGET_REAL_SYNTHETIC}).")
    p.add_argument("--n-fake",       type=int, default=TARGET_FAKE,
                   help=f"Number of fake samples (default {TARGET_FAKE}).")
    p.add_argument("--verbose",      action="store_true",
                   help="Enable DEBUG logging.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    run_pipeline(
        dry_run=args.dry_run,
        skip_extract=args.skip_extract,
        n_real=args.n_real,
        n_fake=args.n_fake,
    )
