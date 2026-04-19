"""
face_analyzer.py — Gender & age estimation for face images.

Uses the DeepFace library (auto-downloads weights on first run).
Results are cached to disk to avoid re-analyzing the same image.

Public API:
    analyze_face(img_path)  → dict | None
    analyze_all_faces(directory) → dict[filename → result]
    load_cache() → dict
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

import cv2

from .config import CACHE_FILE, MIN_FACE_IMG_SIZE_KB

logger = logging.getLogger(__name__)


# ─── DeepFace import (lazy, with clear error message) ────────────────────────

def _get_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        raise ImportError(
            "DeepFace not installed. Run: pip install deepface tf-keras"
        )


# ─── Cache Helpers ────────────────────────────────────────────────────────────

def load_cache() -> dict:
    """Load the face analysis cache from disk (returns {} if missing)."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars → native Python types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, cls=_NumpyEncoder)


# ─── Single Image Analysis ────────────────────────────────────────────────────

def analyze_face(img_path: str, cache: Optional[dict] = None) -> Optional[dict]:
    """
    Estimate gender and age for a face image.

    Returns:
        {
            "gender":      "male" | "female",
            "age":         int,
            "confidence":  float   (0.0–1.0, gender confidence)
        }
        or None if no face detected / image too small.
    """
    img_path = str(img_path)
    filename = os.path.basename(img_path)

    # ── Size gate: skip tiny thumbnails (CelebA ~2-3 KB) ──────────────────
    try:
        size_kb = os.path.getsize(img_path) / 1024
        if size_kb < MIN_FACE_IMG_SIZE_KB:
            logger.debug(f"Skipping {filename} — too small ({size_kb:.1f} KB)")
            return None
    except OSError:
        return None

    # ── Cache hit ─────────────────────────────────────────────────────────
    if cache is not None and filename in cache:
        return cache[filename]

    DeepFace = _get_deepface()

    try:
        results = DeepFace.analyze(
            img_path=img_path,
            actions=["age", "gender"],
            enforce_detection=False,   # don't crash if detection fails
            silent=True,
        )

        # DeepFace returns list[dict] or dict depending on version
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        # ── Normalise gender field (varies across DeepFace versions) ──────
        raw_gender = result.get("dominant_gender") or result.get("gender")
        if isinstance(raw_gender, dict):
            # Old format: {"Man": 92.3, "Woman": 7.7}
            raw_gender = max(raw_gender, key=raw_gender.get)

        gender_str  = raw_gender.lower()  # "man" or "woman"
        gender_norm = "male" if gender_str in ("man", "male") else "female"

        # Confidence: check gender_probability dict if available
        gender_probs = result.get("gender") if isinstance(result.get("gender"), dict) else {}
        if gender_probs:
            confidence = max(gender_probs.values()) / 100.0
        else:
            confidence = 0.85   # DeepFace doesn't always expose confidence

        age = int(result.get("age", 25))          # ensure native int
        confidence = float(confidence)              # ensure native float

        analysis = {
            "gender":     gender_norm,
            "age":        age,
            "confidence": round(confidence, 3),
        }

        # ── Update cache ──────────────────────────────────────────────────
        if cache is not None:
            cache[filename] = analysis
            _save_cache(cache)

        return analysis

    except Exception as e:
        logger.warning(f"DeepFace failed on {filename}: {e}")
        return None


# ─── Batch Analysis ───────────────────────────────────────────────────────────

def analyze_all_faces(directory: str, force_rerun: bool = False) -> dict:
    """
    Analyze every image in `directory` for gender + age.
    Skips images already in cache unless force_rerun=True.

    Returns:
        dict mapping filename → {gender, age, confidence}
        (only includes images where analysis succeeded)
    """
    from tqdm import tqdm

    directory = str(directory)
    cache = {} if force_rerun else load_cache()

    image_paths = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    logger.info(f"Analyzing {len(image_paths)} face images in {directory} …")

    results = {}
    for img_path in tqdm(image_paths, desc="Face analysis", unit="img"):
        filename = os.path.basename(img_path)
        analysis = analyze_face(img_path, cache=cache)
        if analysis:
            results[filename] = analysis
        else:
            logger.debug(f"  ✗ Skipped: {filename}")

    # Persist updated cache
    _save_cache(cache)

    n_male   = sum(1 for v in results.values() if v["gender"] == "male")
    n_female = sum(1 for v in results.values() if v["gender"] == "female")
    logger.info(
        f"Face analysis done: {len(results)} usable | "
        f"{n_male} male, {n_female} female | "
        f"{len(image_paths) - len(results)} skipped"
    )
    return results


# ─── Partition Helpers ────────────────────────────────────────────────────────

def partition_by_gender(face_analysis: dict) -> tuple[list, list]:
    """
    Split analysis results into (male_list, female_list).
    Each list contains (filename, result_dict) tuples.
    """
    males   = [(fn, r) for fn, r in face_analysis.items() if r["gender"] == "male"]
    females = [(fn, r) for fn, r in face_analysis.items() if r["gender"] == "female"]
    return males, females


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        r = analyze_face(sys.argv[1], cache=load_cache())
        print(r)
    else:
        print("Usage: python -m src.face_analyzer <image_path>")
