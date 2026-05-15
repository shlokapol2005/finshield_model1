"""
grad_cam.py — Grad-CAM visualization for the spatial branch.

Generates heatmap overlays showing which regions of the document
the model focuses on when making real/fake decisions.

Usage:
  python -m training.grad_cam --image path/to/document.jpg
  python -m training.grad_cam --image path/to/document.jpg --checkpoint checkpoints/best_model.pt
"""

import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CHECKPOINT_DIR, RESULTS_DIR, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .model import DualBranchForgeryDetector
from .dct_utils import compute_dct


class GradCAM:
    """Grad-CAM for the spatial branch of DualBranchForgeryDetector."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, spatial_input, freq_input, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            spatial_input: (1, 3, H, W) tensor
            freq_input:    (1, 3, H, W) tensor
            target_class:  int or None (uses predicted class if None)

        Returns:
            heatmap: numpy array (H, W) in [0, 1]
        """
        self.model.eval()
        output = self.model(spatial_input, freq_input)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Grad-CAM computation
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def prepare_image(img_path, device):
    """Load and prepare image for both branches."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32)

    # Spatial tensor (normalised)
    arr = img_np / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    spatial = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Frequency tensor (DCT)
    dct_arr = compute_dct(img_np)
    freq = torch.from_numpy(dct_arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    return spatial, freq, img_np


def visualize_grad_cam(img_path, checkpoint_path=None, save_dir=None):
    """Generate and save Grad-CAM visualization for a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}"); return

    # Load model
    model = DualBranchForgeryDetector(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Target layer = last conv block of EfficientNet-B0
    target_layer = model.spatial.conv_head

    grad_cam = GradCAM(model, target_layer)
    spatial, freq, img_np = prepare_image(img_path, device)

    # Get prediction
    with torch.no_grad():
        logits = model(spatial, freq)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    # Generate heatmap
    heatmap = grad_cam.generate(spatial, freq, target_class=pred_class)

    # Plot
    label = "FAKE" if pred_class == 1 else "REAL"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(img_np.astype(np.uint8))
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontweight="bold")
    axes[1].axis("off")

    # Overlay
    overlay = img_np.astype(np.float32) / 255.0
    heatmap_rgb = plt.cm.jet(heatmap)[:, :, :3]
    blended = 0.6 * overlay + 0.4 * heatmap_rgb
    blended = np.clip(blended, 0, 1)
    axes[2].imshow(blended)
    axes[2].set_title(f"Prediction: {label} ({confidence:.2%})", fontweight="bold",
                      color="#16a34a" if pred_class == 0 else "#dc2626")
    axes[2].axis("off")

    plt.suptitle("Grad-CAM — Spatial Branch Focus Areas", fontsize=14, fontweight="bold")
    plt.tight_layout()

    basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_dir, f"gradcam_{basename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🔍 Grad-CAM saved → {save_path}")
    print(f"     Prediction: {label} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Grad-CAM Visualization")
    p.add_argument("--image", type=str, required=True, help="Path to document image")
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()
    visualize_grad_cam(args.image, args.checkpoint)
