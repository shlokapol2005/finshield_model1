"""
run_explainability.py — Root-level runner for explainability & transparency.

Produces:
  1. Branch Ablation:  Spatial-only vs Frequency-only vs Dual-branch accuracy
  2. Grad-CAM Report:  Heatmaps for stratified test samples (real/fake × Aadhaar/PAN)
  3. Confidence Distribution: Correct vs incorrect prediction confidence
  4. SHAP (optional): Channel-level attribution on spatial branch

Usage:
  python run_explainability.py
  python run_explainability.py --skip-shap          # faster, Grad-CAM still runs
  python run_explainability.py --n-gradcam 3        # fewer Grad-CAM examples per stratum
  python run_explainability.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from training.explainability import run_explainability


def main():
    p = argparse.ArgumentParser(
        description="Explainability & Transparency Suite"
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pt)"
    )
    p.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP computation (recommended for CPU — Grad-CAM still runs)"
    )
    p.add_argument(
        "--n-gradcam", type=int, default=5,
        help="Grad-CAM examples per stratum: real-aadhaar/fake-aadhaar/real-pan/fake-pan (default: 5)"
    )
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("  EXPLAINABILITY & TRANSPARENCY SUITE")
    print("  DualBranchForgeryDetector — Document Forgery Detection")
    print("=" * 60)
    print("  Outputs → results/explainability/")
    print("    branch_ablation.png        Spatial-only vs Freq-only vs Dual")
    print("    gradcam_<id>.png           Per-sample Grad-CAM panels")
    print("    gradcam_grid.png           Tiled summary grid")
    print("    confidence_distribution.png  Correct vs incorrect confidence")
    if not args.skip_shap:
        print("    shap_summary.png           Channel-level SHAP attribution")
    print("=" * 60)

    run_explainability(
        checkpoint_path=args.checkpoint,
        skip_shap=args.skip_shap,
        n_gradcam=args.n_gradcam,
    )


if __name__ == "__main__":
    main()
