"""
model.py — Dual-Branch Forgery Detector architecture.

Architecture:
  Spatial Branch  : EfficientNet-B0 (pretrained ImageNet) → 1280-d features
  Frequency Branch: DCT-CNN (3-layer conv + FC) → 128-d features
  Fusion Head     : concat(1280+128) → [256] → [64] → 2 (real/fake)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from .config import (
    NUM_CLASSES,
    SPATIAL_DIM,
    FREQ_DIM,
    FUSION_HIDDEN,
    DROPOUT,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
)


# ─── Frequency Branch (DCT-CNN) ───────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Three-layer Conv2D CNN to process DCT frequency maps.
    Input:  (B, 3, H, W)  — DCT coefficients
    Output: (B, FREQ_DIM) — 128-d feature vector
    """

    def __init__(self, out_dim: int = FREQ_DIM):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 224→112
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 112→56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),    # → (B, 128, 1, 1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


# ─── Spatial Branch (EfficientNet-B0) ────────────────────────────────────────

class SpatialBranch(nn.Module):
    """
    EfficientNet-B0 backbone (timm) with classifier head removed.
    Output: (B, 1280)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for the spatial branch. "
                "Install it with: pip install timm"
            )
        backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,   # remove classifier head
            global_pool="avg",
        )
        # Expose weights as a flat Sequential-like module so state-dict keys
        # match the checkpoint format: spatial.conv_stem, spatial.blocks, etc.
        for name, module in backbone.named_children():
            setattr(self, name, module)
        self._backbone = backbone

    def forward(self, x):
        return self._backbone(x)


# ─── Dual-Branch Fusion Model ─────────────────────────────────────────────────

class DualBranchForgeryDetector(nn.Module):
    """
    Combines EfficientNet-B0 spatial features with DCT frequency features
    through a learned fusion head.

    Input:
        spatial  — (B, 3, 224, 224)  RGB image (ImageNet-normalised)
        freq     — (B, 3, 224, 224)  DCT feature map
    Output:
        logits   — (B, NUM_CLASSES)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Branches
        self.spatial   = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        ) if _TIMM_AVAILABLE else _dummy_spatial()

        self.frequency = FrequencyBranch(out_dim=FREQ_DIM)

        # Fusion head: (SPATIAL_DIM + FREQ_DIM) → FUSION_HIDDEN → NUM_CLASSES
        in_dim = SPATIAL_DIM + FREQ_DIM   # 1280 + 128 = 1408
        h1, h2 = FUSION_HIDDEN            # [256, 64]

        self.fusion = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(h2, NUM_CLASSES),
        )

    def forward(self, spatial_x, freq_x):
        spatial_feat = self.spatial(spatial_x)        # (B, 1280)
        freq_feat    = self.frequency(freq_x)         # (B, 128)
        combined     = torch.cat([spatial_feat, freq_feat], dim=1)  # (B, 1408)
        return self.fusion(combined)

    # ── Backbone freeze/unfreeze helpers ─────────────────────────────────────

    def freeze_backbone(self):
        """Freeze EfficientNet-B0 weights (Phase 1)."""
        for param in self.spatial.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters (Phase 2)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr_backbone, lr_freq, lr_fusion):
        """Return param groups for differential learning rates."""
        return [
            {"params": self.spatial.parameters(),   "lr": lr_backbone},
            {"params": self.frequency.parameters(), "lr": lr_freq},
            {"params": self.fusion.parameters(),    "lr": lr_fusion},
        ]


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Supports optional class_weights tensor for imbalanced datasets.
    """

    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA,
                 class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights  # (num_classes,) tensor or None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets,
                             weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()
