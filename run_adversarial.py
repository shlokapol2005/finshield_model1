"""
run_adversarial.py — Root-level runner for adversarial robustness evaluation.

Attacks the DualBranchForgeryDetector with FGSM and PGD.
ε values defined in [0, 1] pixel space.
DCT is recomputed from the adversarial image on every step.

Usage:
  python run_adversarial.py                        # full eval (FGSM + PGD)
  python run_adversarial.py --quick                # FGSM only, 4 ε values
  python run_adversarial.py --checkpoint checkpoints/best_model.pt
  python run_adversarial.py --batch-size 4         # reduce if low RAM/VRAM
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from training.adversarial import run_adversarial_eval


def main():
    p = argparse.ArgumentParser(
        description="Adversarial Robustness Evaluation — FGSM + PGD"
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pt)"
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick mode: FGSM only with 4 epsilon values [0, 0.05, 0.1, 0.3]"
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for evaluation (default: 8; reduce if OOM)"
    )
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("  ADVERSARIAL ROBUSTNESS EVALUATION")
    print("  DualBranchForgeryDetector — Document Forgery Detection")
    print("=" * 60)
    if args.quick:
        print("  Mode     : Quick (FGSM only)")
    else:
        print("  Mode     : Full  (FGSM + PGD)")
    print(f"  ε space  : [0, 1] pixel space (not normalised space)")
    print(f"  DCT      : recomputed from adversarial image each step")
    print(f"  Outputs  : results/adversarial/")
    print("=" * 60)

    run_adversarial_eval(
        checkpoint_path=args.checkpoint,
        quick=args.quick,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
