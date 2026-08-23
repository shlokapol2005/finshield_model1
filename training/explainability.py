"""
explainability.py — Explainability & Transparency for DualBranchForgeryDetector.

Produces:
  1. Branch Ablation Study
       • Accuracy of: Spatial-only | Frequency-only | Dual-branch (full model)
       • Per class (real/fake) and per doc type (Aadhaar/PAN)
       • Saved: results/explainability/branch_ablation.png + .json

  2. Grad-CAM Report
       • Batch Grad-CAM on a stratified sample of test images
         (real-Aadhaar, fake-Aadhaar, real-PAN, fake-PAN)
       • Individual panels: results/explainability/gradcam_<id>.png
       • Summary grid:     results/explainability/gradcam_grid.png

  3. Confidence Distribution
       • Histogram of softmax confidence for correct vs incorrect predictions
       • Saved: results/explainability/confidence_distribution.png

  4. SHAP Feature Attribution (optional, skipped if shap not installed)
       • DeepExplainer on spatial branch, top-20 feature regions
       • Saved: results/explainability/shap_summary.png

Usage:
  python -m training.explainability
  python -m training.explainability --checkpoint checkpoints/best_model.pt
  python -m training.explainability --skip-shap
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    BASE_DIR, IMG_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from .model import DualBranchForgeryDetector
from .dct_utils import compute_dct
from .dataset import load_merged_csv, get_splits, ForgeryDataset
from .grad_cam import GradCAM

EXPLAIN_DIR = os.path.join(RESULTS_DIR, "explainability")
_MEAN = IMAGENET_MEAN
_STD  = IMAGENET_STD


# ─── Helper: load model ──────────────────────────────────────────────────────

def _load_model(checkpoint_path: str, device: torch.device) -> DualBranchForgeryDetector:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = DualBranchForgeryDetector(pretrained=False).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ─── Helper: tensor normalise ────────────────────────────────────────────────

def _normalise(pixel_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply ImageNet normalisation to a [0,1] pixel batch (B,3,H,W)."""
    mean = torch.tensor(_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(_STD ).view(1, 3, 1, 1).to(device)
    return (pixel_batch - mean) / std


# ─── 1. Branch Ablation Study ─────────────────────────────────────────────────

class SpatialOnlyModel(nn.Module):
    """Forward pass using only the spatial branch (frequency input zeroed)."""
    def __init__(self, model: DualBranchForgeryDetector):
        super().__init__()
        self.model = model

    def forward(self, spatial, freq):
        spatial_feat = self.model.spatial(spatial)
        zero_freq    = torch.zeros(
            spatial.size(0), 128, device=spatial.device
        )
        combined = torch.cat([spatial_feat, zero_freq], dim=1)
        return self.model.fusion(combined)


class FreqOnlyModel(nn.Module):
    """Forward pass using only the frequency branch (spatial input zeroed)."""
    def __init__(self, model: DualBranchForgeryDetector):
        super().__init__()
        self.model = model

    def forward(self, spatial, freq):
        # Zero spatial: pass a blank normalised image through EfficientNet
        zero_spatial = torch.zeros_like(spatial)
        spatial_feat = self.model.spatial(zero_spatial)
        freq_feat    = self.model.frequency(freq)
        combined     = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.model.fusion(combined)


@torch.no_grad()
def _collect_preds(model, loader, device):
    """Collect (labels, preds, probs, doc_types) from a dataloader."""
    all_labels, all_preds, all_probs, all_dtypes = [], [], [], []
    for spatial, freq, labels in loader:
        if isinstance(labels, (list, tuple)) and len(labels) == 2:
            labels, dtypes = labels
        else:
            dtypes = ["unknown"] * len(labels)

        spatial = spatial.to(device)
        freq    = freq.to(device)
        logits  = model(spatial, freq)
        probs   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds   = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_dtypes.extend(list(dtypes))
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_dtypes)


def _accuracy_breakdown(labels, preds, doc_types):
    """Return dict with overall, per-class, per-doc-type accuracy."""
    result = {
        "overall": float(np.mean(labels == preds)),
        "per_class": {
            "real": float(np.mean(preds[labels == 0] == 0)) if (labels == 0).any() else None,
            "fake": float(np.mean(preds[labels == 1] == 1)) if (labels == 1).any() else None,
        },
        "per_doc_type": {},
    }
    for dt in np.unique(doc_types):
        mask = doc_types == dt
        result["per_doc_type"][dt] = float(np.mean(labels[mask] == preds[mask]))
    return result


def _build_eval_loader(test_df, batch_size=16):
    """Standard ForgeryDataset dataloader (no augment) with doc_type passed via df."""
    ds = ForgeryDatasetWithDocType(test_df)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


class ForgeryDatasetWithDocType(ForgeryDataset):
    """Extends ForgeryDataset to also return doc_type as part of labels."""

    def __getitem__(self, idx):
        spatial, freq, label = super().__getitem__(idx)
        doc_type = str(self.df.iloc[idx].get("doc_type", "unknown"))
        return spatial, freq, (label, doc_type)


def run_branch_ablation(
    model: DualBranchForgeryDetector,
    test_df,
    device: torch.device,
    save_dir: str,
) -> dict:
    """
    Evaluate Spatial-only, Frequency-only, and Dual-branch models.

    Returns ablation dict.
    """
    print("\n  ── Branch Ablation Study ──────────────────────────")

    loader = _build_eval_loader(test_df, batch_size=16)

    variants = {
        "Spatial-only":    SpatialOnlyModel(model),
        "Frequency-only":  FreqOnlyModel(model),
        "Dual-branch":     model,
    }

    ablation = {}
    for name, m in variants.items():
        m.to(device)
        m.eval()

        all_labels, all_preds, all_probs, all_dtypes = [], [], [], []
        with torch.no_grad():
            for spatial, freq, label_info in tqdm(loader, desc=f"  {name}", leave=False):
                if isinstance(label_info, (list, tuple)):
                    labels, dtypes = label_info
                else:
                    labels = label_info
                    dtypes = ["unknown"] * len(labels)

                labels = labels if isinstance(labels, torch.Tensor) else labels[0]
                spatial = spatial.to(device)
                freq    = freq.to(device)
                logits  = m(spatial, freq)
                preds   = logits.argmax(dim=1).cpu().numpy()

                # Handle label_info which might be a tuple (label, doc_type)
                if isinstance(label_info, (list, tuple)):
                    label_vals = label_info[0]
                    dtype_vals = label_info[1]
                else:
                    label_vals = label_info
                    dtype_vals = ["unknown"] * len(label_vals)

                if isinstance(label_vals, torch.Tensor):
                    all_labels.extend(label_vals.numpy().tolist())
                else:
                    all_labels.extend(label_vals)

                all_preds.extend(preds.tolist())
                if isinstance(dtype_vals, torch.Tensor):
                    all_dtypes.extend(dtype_vals.tolist())
                else:
                    all_dtypes.extend(list(dtype_vals))

        labels_np  = np.array(all_labels)
        preds_np   = np.array(all_preds)
        dtypes_np  = np.array(all_dtypes)
        breakdown  = _accuracy_breakdown(labels_np, preds_np, dtypes_np)
        ablation[name] = breakdown

        print(f"    {name:18s}  Overall={breakdown['overall']:.4f}"
              f"  Real={breakdown['per_class'].get('real', 0):.4f}"
              f"  Fake={breakdown['per_class'].get('fake', 0):.4f}")

    # ── Plot ────────────────────────────────────────────────────
    _plot_ablation(ablation, os.path.join(save_dir, "branch_ablation.png"))

    ablation_path = os.path.join(save_dir, "branch_ablation.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"  💾 Ablation JSON → {ablation_path}")

    return ablation


def _plot_ablation(ablation: dict, save_path: str):
    """Grouped bar chart: Spatial-only | Freq-only | Dual-branch × Overall/Real/Fake."""
    variants  = list(ablation.keys())
    metrics   = ["overall", "real", "fake"]
    colors    = {"overall": "#2563eb", "real": "#16a34a", "fake": "#dc2626"}

    x = np.arange(len(variants))
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, offset) in enumerate(zip(metrics, offsets)):
        vals = []
        for v in variants:
            if metric == "overall":
                vals.append(ablation[v]["overall"])
            else:
                vals.append(ablation[v]["per_class"].get(metric) or 0.0)
        bars = ax.bar(x + offset, vals, width, label=metric.capitalize(),
                      color=colors[metric], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Branch Ablation Study — Spatial-only vs Frequency-only vs Dual-branch",
                 fontweight="bold", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Ablation chart → {save_path}")


# ─── 2. Grad-CAM Report ───────────────────────────────────────────────────────

def _prepare_spatial_freq(img_path: str, device: torch.device):
    """Load image → (spatial_tensor, freq_tensor, img_np)."""
    img  = Image.open(img_path).convert("RGB")
    img  = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr  = np.array(img, dtype=np.float32)

    # Spatial: normalise
    norm = arr / 255.0
    for c in range(3):
        norm[:, :, c] = (norm[:, :, c] - _MEAN[c]) / _STD[c]
    spatial = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Frequency: DCT
    dct_arr = compute_dct(arr)
    freq    = torch.from_numpy(dct_arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    return spatial, freq, arr.astype(np.uint8)


def run_gradcam_report(
    model: DualBranchForgeryDetector,
    test_df,
    device: torch.device,
    save_dir: str,
    n_per_stratum: int = 5,
) -> None:
    """
    Generate Grad-CAM panels for a stratified sample of test images.
    Strata: (real, aadhaar), (fake, aadhaar), (real, pan), (fake, pan).
    """
    print("\n  ── Grad-CAM Report ────────────────────────────────")

    label_str   = {0: "REAL", 1: "FAKE"}
    target_layer = model.spatial.conv_head
    grad_cam     = GradCAM(model, target_layer)

    # Sample per stratum
    samples = []
    for label_val in [0, 1]:
        for dt in ["aadhaar", "pan"]:
            mask = (test_df["label"] == label_val)
            if "doc_type" in test_df.columns:
                mask &= (test_df["doc_type"] == dt)
            stratum = test_df[mask]
            if len(stratum) == 0:
                continue
            chosen = stratum.sample(n=min(n_per_stratum, len(stratum)),
                                    random_state=42)
            for _, row in chosen.iterrows():
                samples.append((row["image_path"], int(row["label"]), dt))

    saved_paths = []
    for i, (img_rel, true_label, doc_type) in enumerate(
        tqdm(samples, desc="  Grad-CAM", unit="img")
    ):
        img_path = os.path.join(BASE_DIR, img_rel)
        if not os.path.exists(img_path):
            continue

        spatial, freq, img_np = _prepare_spatial_freq(img_path, device)

        # Prediction
        with torch.no_grad():
            logits     = model(spatial, freq)
            probs      = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(1).item()
            confidence = probs[0, pred_class].item()

        # Grad-CAM heatmap
        heatmap = grad_cam.generate(spatial, freq, target_class=pred_class)

        # ── Plot single panel ─────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img_np)
        axes[0].set_title(f"Original\n({doc_type.upper()})", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap\n(Spatial Branch)", fontweight="bold")
        axes[1].axis("off")

        hm_rgb   = plt.cm.jet(heatmap)[:, :, :3]
        overlay  = img_np.astype(np.float32) / 255.0
        blended  = np.clip(0.6 * overlay + 0.4 * hm_rgb, 0, 1)
        color    = "#16a34a" if pred_class == true_label else "#dc2626"
        correct  = "✓" if pred_class == true_label else "✗"
        axes[2].imshow(blended)
        axes[2].set_title(
            f"Prediction: {label_str[pred_class]} {correct} ({confidence:.1%})\n"
            f"True: {label_str[true_label]}",
            fontweight="bold", color=color
        )
        axes[2].axis("off")

        plt.suptitle("Grad-CAM — DualBranchForgeryDetector (Spatial Focus)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()

        fname    = f"gradcam_{i:03d}_{doc_type}_{label_str[true_label].lower()}.png"
        fpath    = os.path.join(save_dir, fname)
        plt.savefig(fpath, dpi=120, bbox_inches="tight")
        plt.close()
        saved_paths.append(fpath)

    print(f"  🔍 {len(saved_paths)} Grad-CAM panels saved to {save_dir}")

    # ── Summary grid ─────────────────────────────────────────
    if saved_paths:
        _make_gradcam_grid(saved_paths, os.path.join(save_dir, "gradcam_grid.png"))


def _make_gradcam_grid(paths: list, save_path: str, cols: int = 4):
    """Tile Grad-CAM panels into a summary grid."""
    images = []
    for p in paths:
        try:
            images.append(Image.open(p))
        except Exception:
            pass
    if not images:
        return

    w, h   = images[0].size
    rows   = (len(images) + cols - 1) // cols
    grid   = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img.resize((w, h)), (c * w, r * h))

    grid.save(save_path)
    print(f"  🖼  Grad-CAM grid → {save_path}")


# ─── 3. Confidence Distribution ───────────────────────────────────────────────

def run_confidence_distribution(
    model: DualBranchForgeryDetector,
    test_df,
    device: torch.device,
    save_dir: str,
) -> None:
    """Plot confidence score histograms for correct vs incorrect predictions."""
    print("\n  ── Confidence Distribution ────────────────────────")

    loader = DataLoader(
        ForgeryDataset(test_df, augment=False),
        batch_size=16, shuffle=False, num_workers=0
    )

    correct_conf, wrong_conf = [], []

    model.eval()
    with torch.no_grad():
        for spatial, freq, labels in loader:
            spatial = spatial.to(device)
            freq    = freq.to(device)
            logits  = model(spatial, freq)
            probs   = torch.softmax(logits, dim=1)
            preds   = logits.argmax(dim=1).cpu()
            # Confidence = probability of the predicted class
            conf    = probs.max(dim=1).values.cpu().numpy()

            correct_mask = (preds == labels).numpy()
            correct_conf.extend(conf[ correct_mask].tolist())
            wrong_conf.extend(  conf[~correct_mask].tolist())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct_conf, bins=20, alpha=0.7, color="#16a34a",
            label=f"Correct (n={len(correct_conf)})", density=True)
    ax.hist(wrong_conf,   bins=20, alpha=0.7, color="#dc2626",
            label=f"Incorrect (n={len(wrong_conf)})", density=True)
    ax.axvline(0.5, color="#94a3b8", linestyle="--", alpha=0.7)
    ax.set_xlabel("Confidence (max softmax probability)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Prediction Confidence Distribution", fontweight="bold", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "confidence_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confidence distribution → {save_path}")


# ─── 4. SHAP Feature Attribution (optional) ───────────────────────────────────

def run_shap_attribution(
    model: DualBranchForgeryDetector,
    test_df,
    device: torch.device,
    save_dir: str,
    n_background: int = 50,
    n_test: int = 10,
) -> None:
    """
    SHAP DeepExplainer on the spatial branch output (1280-dim features).
    This shows which feature dimensions drive real vs fake decisions.
    Skips gracefully if shap is not installed.
    """
    print("\n  ── SHAP Feature Attribution ───────────────────────")

    try:
        import shap
    except ImportError:
        print("  ⚠ shap not installed — skipping. Install with: pip install shap")
        return

    # ── Build a simple wrapper that returns spatial features only ─
    class SpatialFeatExtractor(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, spatial):
            """Takes normalised spatial (B,3,H,W), returns logits."""
            # Use a representative zero DCT freq tensor
            freq = torch.zeros(
                spatial.size(0), 3, IMG_SIZE, IMG_SIZE, device=spatial.device
            )
            return self.m(spatial, freq)

    extractor = SpatialFeatExtractor(model).to(device)
    extractor.eval()

    # ── Sample background and test sets ───────────────────────────
    loader = DataLoader(
        ForgeryDataset(test_df, augment=False),
        batch_size=n_background + n_test, shuffle=True, num_workers=0
    )
    batch = next(iter(loader))
    spatial_all = batch[0].to(device)

    background  = spatial_all[:n_background]
    test_inputs = spatial_all[n_background: n_background + n_test]

    # ── SHAP ──────────────────────────────────────────────────────
    print(f"  Running DeepExplainer (background={n_background}, test={len(test_inputs)}) …")
    try:
        e      = shap.DeepExplainer(extractor, background)
        shap_v = e.shap_values(test_inputs)            # list of arrays per class

        # shap_v is [shap_real, shap_fake], each (n_test, 3, H, W)
        shap_fake = np.array(shap_v[1])                # class-1 (fake) attributions

        # Average across spatial dims → (n_test, 3) channel importance
        channel_imp = shap_fake.mean(axis=(0, 2, 3))   # (3,)

        fig, ax = plt.subplots(figsize=(6, 4))
        ch_names = ["R channel", "G channel", "B channel"]
        colors   = ["#ef4444", "#16a34a", "#2563eb"]
        ax.bar(ch_names, channel_imp, color=colors, alpha=0.85)
        ax.set_ylabel("Mean |SHAP| (fake class)")
        ax.set_title("SHAP Channel Importance — Spatial Branch (Fake detection)",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        shap_path = os.path.join(save_dir, "shap_summary.png")
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  🔬 SHAP summary → {shap_path}")

    except Exception as e:
        print(f"  ⚠ SHAP computation failed: {e}")
        print("    This is common on CPU with large models. "
              "Grad-CAM provides equivalent spatial explainability.")


# ─── Main runner ──────────────────────────────────────────────────────────────

def run_explainability(
    checkpoint_path: str = None,
    skip_shap: bool = False,
    n_gradcam: int = 5,
) -> None:
    """
    Full explainability suite:
      1. Branch ablation
      2. Grad-CAM report
      3. Confidence distribution
      4. SHAP (optional)
    """
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    model   = _load_model(checkpoint_path, device)
    df      = load_merged_csv()
    _, _, test_df = get_splits(df)

    print(f"  Test samples: {len(test_df)}")

    # 1. Branch ablation
    run_branch_ablation(model, test_df, device, EXPLAIN_DIR)

    # 2. Grad-CAM report
    run_gradcam_report(model, test_df, device, EXPLAIN_DIR, n_per_stratum=n_gradcam)

    # 3. Confidence distribution
    run_confidence_distribution(model, test_df, device, EXPLAIN_DIR)

    # 4. SHAP (optional)
    if not skip_shap:
        run_shap_attribution(model, test_df, device, EXPLAIN_DIR)
    else:
        print("\n  ── SHAP skipped (--skip-shap) ─────────────────────")

    print(f"\n  ✅ Explainability suite complete!  → {EXPLAIN_DIR}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Explainability & Transparency Suite")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint (default: checkpoints/best_model.pt)")
    p.add_argument("--skip-shap", action="store_true",
                   help="Skip SHAP computation (faster, Grad-CAM still runs)")
    p.add_argument("--n-gradcam", type=int, default=5,
                   help="Number of Grad-CAM examples per stratum (default: 5)")
    args = p.parse_args()
    run_explainability(
        checkpoint_path=args.checkpoint,
        skip_shap=args.skip_shap,
        n_gradcam=args.n_gradcam,
    )
