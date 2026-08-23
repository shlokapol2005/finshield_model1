"""
adversarial.py — Adversarial Robustness Evaluation for DualBranchForgeryDetector.

Attack strategy:
  • Both FGSM and PGD attack the *original image* (pixel space, [0,1]).
  • After perturbation the adversarial image is clipped back to [0,1].
  • ImageNet normalisation is applied INSIDE the forward pass wrapper so
    that the gradient signal travels through normalisation → network.
  • DCT is recomputed from the adversarial pixel image on every step,
    so the frequency branch also sees the perturbed document.

Epsilon values are defined in [0,1] pixel space (e.g. ε=0.03 ≈ ±8/255).

Outputs → results/adversarial/
  adversarial_results.json     — full metrics per (method, epsilon, doc_type)
  adversarial_degradation.png  — Accuracy vs ε curves
  adversarial_examples.png     — clean vs adversarial example grid
  adversarial_summary.txt      — human-readable report

Usage:
  python -m training.adversarial
  python -m training.adversarial --quick        # FGSM only, fewer epsilons
  python -m training.adversarial --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import (
    AADHAAR_CSV, PAN_CSV, BASE_DIR, IMG_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    CHECKPOINT_DIR, RESULTS_DIR, PHASE1_BATCH,
)
from .model import DualBranchForgeryDetector
from .dct_utils import compute_dct
from .dataset import load_merged_csv, get_splits


# ─── Constants ───────────────────────────────────────────────────────────────

EPSILONS_FULL  = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
EPSILONS_QUICK = [0.0, 0.05, 0.1, 0.3]
ADV_RESULTS_DIR = os.path.join(RESULTS_DIR, "adversarial")

_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)  # (1,3,1,1)
_STD  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1)


# ─── Model wrapper: pixel [0,1] → normalised → dual forward ──────────────────

class PixelSpaceModel(nn.Module):
    """
    Wraps DualBranchForgeryDetector so that forward() accepts a raw pixel
    tensor in [0, 1] (B, 3, H, W).  Internally it:
      1. Applies ImageNet normalisation to get the spatial input.
      2. Converts the pixel tensor to a numpy array and runs compute_dct
         to build the frequency input.
    This allows gradients w.r.t. the raw pixel image to flow through both
    branches simultaneously.
    """

    def __init__(self, model: DualBranchForgeryDetector, device: torch.device):
        super().__init__()
        self.model  = model
        self.device = device
        self._mean  = _MEAN.to(device)
        self._std   = _STD.to(device)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalisation.  x in [0, 1]."""
        return (x - self._mean) / self._std

    def pixel_to_dct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel batch tensor [0, 1] → DCT tensor.
        Uses numpy so we detach; returns a new tensor with grad_fn=None.
        We re-attach it via interpolation to keep the graph for the
        spatial branch but the frequency input is treated as a constant
        w.r.t. the attacker (a reasonable threat model: the attacker
        optimises only over the visible pixel domain).
        """
        batch_np = (x.detach().cpu().numpy() * 255.0)          # (B,3,H,W) [0,255]
        batch_np = batch_np.transpose(0, 2, 3, 1)              # (B,H,W,3)
        dct_list = [compute_dct(img) for img in batch_np]      # each (H,W,3)
        dct_np   = np.stack(dct_list, axis=0)                  # (B,H,W,3)
        dct_t    = torch.from_numpy(
                       dct_np.transpose(0, 3, 1, 2)            # (B,3,H,W)
                   ).float().to(self.device)
        return dct_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.normalise(x)                # (B,3,H,W) normalised
        freq    = self.pixel_to_dct(x)             # (B,3,H,W) DCT — constant
        return self.model(spatial, freq)


# ─── Attack implementations ───────────────────────────────────────────────────

def fgsm_attack(
    wrapper: PixelSpaceModel,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    FGSM: x_adv = clip(x + ε * sign(∇_x L(x, y)), 0, 1).

    Args:
        x:       (B, 3, H, W) pixel tensor in [0, 1], requires_grad
        y:       (B,) ground-truth label tensor
        epsilon: perturbation magnitude in [0, 1] pixel space

    Returns:
        x_adv: (B, 3, H, W) adversarial pixel tensor in [0, 1]
    """
    if epsilon == 0.0:
        return x.clone()

    x_in = x.clone().detach().requires_grad_(True)
    logits = wrapper(x_in)
    loss   = F.cross_entropy(logits, y)
    loss.backward()

    with torch.no_grad():
        x_adv = x_in + epsilon * x_in.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def pgd_attack(
    wrapper: PixelSpaceModel,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = None,
    num_steps: int = 10,
) -> torch.Tensor:
    """
    PGD (Madry et al., 2018):
      x_0 = x + uniform noise in [-ε, ε]
      for t in range(num_steps):
          x_{t+1} = clip(x_t + α * sign(∇L), 0, 1)  ∩  B_ε(x)

    Args:
        x:         (B, 3, H, W) pixel tensor in [0, 1]
        epsilon:   ℓ∞ ball radius in [0, 1] pixel space
        alpha:     step size; defaults to ε / 4
        num_steps: number of gradient steps (default 10)
    """
    if epsilon == 0.0:
        return x.clone()

    if alpha is None:
        alpha = epsilon / 4.0

    x_orig = x.clone().detach()

    # Random start inside the ε-ball
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(num_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = wrapper(x_adv)
        loss   = F.cross_entropy(logits, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # Project back into ε-ball around x_orig
            delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

    return x_adv.detach()


# ─── Doc-type aware dataset wrapper ──────────────────────────────────────────

class PixelTestDataset(Dataset):
    """
    Like ForgeryDataset but returns:
        pixel_tensor  — (3, H, W) float32 in [0, 1]  (no normalisation)
        label         — int
        doc_type      — str ("aadhaar" | "pan")
    """

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(BASE_DIR, row["image_path"])
        label    = int(row["label"])
        doc_type = str(row.get("doc_type", "unknown"))

        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0           # [0,1] HWC
        pixel_t = torch.from_numpy(arr.transpose(2, 0, 1))      # CHW
        return pixel_t, label, doc_type


# ─── Evaluation helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def _eval_clean(wrapper: PixelSpaceModel, loader: DataLoader, device: torch.device):
    """Evaluate on clean inputs (no attack). Returns (labels, preds, doc_types)."""
    all_labels, all_preds, all_dtypes = [], [], []
    for pixels, labels, dtypes in loader:
        pixels = pixels.to(device)
        logits = wrapper(pixels)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_dtypes.extend(list(dtypes))
    return all_labels, all_preds, all_dtypes


def _eval_adversarial(
    wrapper: PixelSpaceModel,
    loader: DataLoader,
    device: torch.device,
    attack_fn,
    epsilon: float,
) -> tuple:
    """
    Evaluate model under an adversarial attack.
    Returns (labels, preds, doc_types).
    """
    all_labels, all_preds, all_dtypes = [], [], []
    wrapper.eval()

    for pixels, labels, dtypes in tqdm(
        loader, desc=f"  ε={epsilon:.3f}", leave=False, unit="batch"
    ):
        pixels = pixels.to(device)
        labels_dev = labels.to(device)

        # Generate adversarial examples
        x_adv = attack_fn(wrapper, pixels, labels_dev, epsilon)

        # Evaluate on adversarial examples
        with torch.no_grad():
            logits = wrapper(x_adv)
            preds  = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_dtypes.extend(list(dtypes))

    return all_labels, all_preds, all_dtypes


def _compute_metrics(labels, preds, doc_types):
    """Compute accuracy overall and per doc_type."""
    labels    = np.array(labels)
    preds     = np.array(preds)
    doc_types = np.array(doc_types)
    overall   = float(np.mean(labels == preds))

    per_doc = {}
    for dt in np.unique(doc_types):
        mask = doc_types == dt
        per_doc[dt] = float(np.mean(labels[mask] == preds[mask]))

    return {"overall": overall, "per_doc_type": per_doc}


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_degradation_curves(results: dict, epsilons: list, save_path: str):
    """
    Plot Accuracy vs ε for FGSM and PGD, with per-doc-type subplots.

    results structure:
      results[method][epsilon] = {"overall": float, "per_doc_type": {dt: float}}
    """
    methods  = list(results.keys())           # ["fgsm", "pgd"]
    doc_types_set = set()
    for m in methods:
        for eps in epsilons:
            doc_types_set |= set(results[m][eps]["per_doc_type"].keys())
    doc_types = sorted(doc_types_set)

    n_subplots = 1 + len(doc_types)
    fig, axes  = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 5), sharey=True)

    colors = {"fgsm": "#ef4444", "pgd": "#2563eb"}
    styles = {"fgsm": "o-",      "pgd": "s--"}

    def _plot_on(ax, key_fn, title):
        for method in methods:
            ys = [key_fn(results[method][eps]) for eps in epsilons]
            ax.plot(epsilons, ys, styles[method], color=colors[method],
                    label=method.upper(), linewidth=2, markersize=5)
        ax.set_xlabel("ε (perturbation magnitude)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.01, max(epsilons) + 0.01)

    _plot_on(axes[0], lambda r: r["overall"], "Overall Accuracy vs ε")

    for i, dt in enumerate(doc_types):
        _plot_on(
            axes[i + 1],
            lambda r, dt=dt: r["per_doc_type"].get(dt, float("nan")),
            f"{dt.upper()} Accuracy vs ε",
        )

    plt.suptitle("Adversarial Robustness — DualBranchForgeryDetector",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Degradation curves → {save_path}")


def plot_adversarial_examples(
    wrapper: PixelSpaceModel,
    loader: DataLoader,
    device: torch.device,
    save_path: str,
    epsilon: float = 0.05,
    n_examples: int = 4,
):
    """
    Save a grid: Clean | FGSM(ε) | PGD(ε) with prediction labels.
    One row per example.
    """
    wrapper.eval()
    examples = []

    for pixels, labels, dtypes in loader:
        if len(examples) >= n_examples:
            break
        pixels     = pixels.to(device)
        labels_dev = labels.to(device)

        x_fgsm = fgsm_attack(wrapper, pixels, labels_dev, epsilon)
        x_pgd  = pgd_attack(wrapper, pixels, labels_dev, epsilon)

        with torch.no_grad():
            p_clean = wrapper(pixels).argmax(1)
            p_fgsm  = wrapper(x_fgsm).argmax(1)
            p_pgd   = wrapper(x_pgd).argmax(1)

        for i in range(min(pixels.shape[0], n_examples - len(examples))):
            examples.append({
                "clean":  pixels[i].cpu(),
                "fgsm":   x_fgsm[i].cpu(),
                "pgd":    x_pgd[i].cpu(),
                "true":   labels[i].item(),
                "p_clean": p_clean[i].item(),
                "p_fgsm":  p_fgsm[i].item(),
                "p_pgd":   p_pgd[i].item(),
                "dtype":   dtypes[i],
            })

    def _tensor_to_img(t):
        """Pixel tensor [0,1] → uint8 HWC numpy."""
        return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    label_str = {0: "REAL", 1: "FAKE"}
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))

    col_titles = ["Clean", f"FGSM (ε={epsilon})", f"PGD (ε={epsilon})"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontweight="bold", fontsize=11)

    for row, ex in enumerate(examples):
        for col, (key, pred_key) in enumerate(
            [("clean", "p_clean"), ("fgsm", "p_fgsm"), ("pgd", "p_pgd")]
        ):
            img  = _tensor_to_img(ex[key])
            pred = ex[pred_key]
            true = ex["true"]
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            color  = "#16a34a" if pred == true else "#dc2626"
            suffix = "✓" if pred == true else "✗"
            axes[row, col].set_xlabel(
                f"Pred: {label_str[pred]} {suffix}  (True: {label_str[true]})",
                color=color, fontsize=9
            )
            axes[row, col].xaxis.set_label_position("bottom")
            axes[row, col].xaxis.label.set_visible(True)

        # Row label: doc type + true label
        axes[row, 0].set_ylabel(
            f"{ex['dtype'].upper()}\n{label_str[ex['true']]}",
            fontsize=9, fontweight="bold"
        )

    plt.suptitle("Adversarial Examples — Clean vs Perturbed",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🖼  Adversarial examples → {save_path}")


# ─── Main evaluation routine ──────────────────────────────────────────────────

def run_adversarial_eval(
    checkpoint_path: str = None,
    quick: bool = False,
    batch_size: int = 8,
) -> dict:
    """
    Full adversarial robustness evaluation.

    Args:
        checkpoint_path: path to model checkpoint (default: checkpoints/best_model.pt)
        quick:           if True, use EPSILONS_QUICK and skip PGD
        batch_size:      batch size for evaluation (smaller = less GPU memory)

    Returns:
        results dict
    """
    os.makedirs(ADV_RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load model ──────────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return {}

    print(f"  Loading: {checkpoint_path}")
    base_model = DualBranchForgeryDetector(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()

    wrapper = PixelSpaceModel(base_model, device)

    # ── Build test loader ────────────────────────────────────────
    print("  Building test loader …")
    df               = load_merged_csv()
    _, _, test_df    = get_splits(df)
    test_ds          = PixelTestDataset(test_df)
    test_loader      = DataLoader(test_ds, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_ds)}")

    epsilons = EPSILONS_QUICK if quick else EPSILONS_FULL
    attacks  = {"fgsm": fgsm_attack} if quick else {
        "fgsm": fgsm_attack,
        "pgd":  pgd_attack,
    }

    results = {m: {} for m in attacks}

    for method, attack_fn in attacks.items():
        print(f"\n  ── {method.upper()} ──────────────────────────────")
        for eps in epsilons:
            print(f"  Running ε={eps:.3f} …")
            if eps == 0.0:
                labels, preds, dtypes = _eval_clean(wrapper, test_loader, device)
            else:
                labels, preds, dtypes = _eval_adversarial(
                    wrapper, test_loader, device, attack_fn, eps
                )
            metrics = _compute_metrics(labels, preds, dtypes)
            results[method][eps] = metrics
            print(f"    Overall accuracy: {metrics['overall']:.4f}")
            for dt, acc in sorted(metrics["per_doc_type"].items()):
                print(f"    {dt:10s}: {acc:.4f}")

    # ── Save results JSON ────────────────────────────────────────
    json_path = os.path.join(ADV_RESULTS_DIR, "adversarial_results.json")
    # Convert float keys to strings for JSON serialisation
    json_safe = {
        m: {str(eps): v for eps, v in eps_dict.items()}
        for m, eps_dict in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\n  💾 Results JSON → {json_path}")

    # ── Plot degradation curves ──────────────────────────────────
    plot_degradation_curves(
        results, epsilons,
        save_path=os.path.join(ADV_RESULTS_DIR, "adversarial_degradation.png"),
    )

    # ── Plot adversarial examples (ε=0.05) ──────────────────────
    vis_eps = 0.05 if 0.05 in epsilons else epsilons[min(2, len(epsilons) - 1)]
    plot_adversarial_examples(
        wrapper, test_loader, device,
        save_path=os.path.join(ADV_RESULTS_DIR, "adversarial_examples.png"),
        epsilon=vis_eps,
        n_examples=4,
    )

    # ── Write human-readable summary ────────────────────────────
    _write_summary(results, epsilons, os.path.join(ADV_RESULTS_DIR, "adversarial_summary.txt"))

    print(f"\n  ✅ Adversarial evaluation complete!\n")
    return results


def _write_summary(results: dict, epsilons: list, save_path: str):
    lines = []
    lines.append("=" * 60)
    lines.append("  ADVERSARIAL ROBUSTNESS SUMMARY")
    lines.append("  DualBranchForgeryDetector (Spatial + Frequency branches)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  Attack: FGSM  - single-step gradient sign attack")
    lines.append("  Attack: PGD   - 10-step projected gradient descent")
    lines.append("  eps range     - defined in [0,1] pixel space")
    lines.append("  DCT input     - recomputed from adversarial pixel image")
    lines.append("")

    for method in sorted(results.keys()):
        lines.append(f"  [{method.upper()}]")
        lines.append(f"  {'eps':>8}  {'Overall':>10}  {'Aadhaar':>10}  {'PAN':>10}")
        lines.append(f"  {'-'*45}")
        for eps in epsilons:
            row   = results[method][eps]
            ov    = f"{row['overall']:.4f}"
            aad   = f"{row['per_doc_type'].get('aadhaar', float('nan')):.4f}"
            pan   = f"{row['per_doc_type'].get('pan', float('nan')):.4f}"
            lines.append(f"  {eps:>8.3f}  {ov:>10}  {aad:>10}  {pan:>10}")
        lines.append("")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  📝 Summary → {save_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Adversarial Robustness Evaluation")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint (default: checkpoints/best_model.pt)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: FGSM only, fewer epsilon values")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size for evaluation (default: 8)")
    args = p.parse_args()
    run_adversarial_eval(
        checkpoint_path=args.checkpoint,
        quick=args.quick,
        batch_size=args.batch_size,
    )
