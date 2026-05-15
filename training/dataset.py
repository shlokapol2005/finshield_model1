"""
dataset.py — PyTorch Dataset for Document Forgery Detection.

CRITICAL: Uses IDENTITY-AWARE SPLITTING.
All documents sharing the same face_file go into the SAME split.
This prevents data leakage where the model memorizes faces instead
of learning forgery patterns.

Returns two views of each image:
  1. Spatial view  — resized RGB with augmentation + ImageNet normalisation
  2. Frequency view — DCT coefficients (log-scaled, normalised)
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

from .config import (
    AADHAAR_CSV,
    PAN_CSV,
    BASE_DIR,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPLIT_RATIOS,
    RANDOM_SEED,
    NUM_WORKERS,
    PIN_MEMORY,
)
from .dct_utils import compute_dct


# ─── Augmentation helpers (PIL-based, no torchvision dependency) ──────────────

def _random_rotation(img, max_deg=10):
    angle = random.uniform(-max_deg, max_deg)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def _random_brightness(img, low=0.8, high=1.2):
    return ImageEnhance.Brightness(img).enhance(random.uniform(low, high))


def _random_contrast(img, low=0.8, high=1.2):
    return ImageEnhance.Contrast(img).enhance(random.uniform(low, high))


def _random_gaussian_blur(img, p=0.3, radius_range=(0.5, 1.5)):
    if random.random() < p:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*radius_range)))
    return img


def _random_horizontal_flip(img, p=0.5):
    if random.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _pil_to_tensor(img, mean, std):
    """Convert PIL Image to normalised float32 tensor (C, H, W)."""
    arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - mean[c]) / std[c]
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _dct_to_tensor(dct_arr):
    """Convert DCT numpy array (H, W, 3) to float32 tensor (C, H, W)."""
    return torch.from_numpy(dct_arr.transpose(2, 0, 1))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ForgeryDataset(Dataset):
    """
    Dual-view dataset for document forgery detection.

    Each __getitem__ returns:
        spatial_tensor  — (3, IMG_SIZE, IMG_SIZE) augmented + normalised
        freq_tensor     — (3, IMG_SIZE, IMG_SIZE) DCT coefficients
        label           — 0 (real) or 1 (fake)
    """

    def __init__(self, df: pd.DataFrame, augment: bool = True):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(BASE_DIR, row["image_path"])
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # ── Spatial branch (with augmentation) ────────────
        if self.augment:
            img_aug = _random_rotation(img)
            img_aug = _random_horizontal_flip(img_aug)
            img_aug = _random_brightness(img_aug)
            img_aug = _random_contrast(img_aug)
            img_aug = _random_gaussian_blur(img_aug)
        else:
            img_aug = img.copy()

        spatial_tensor = _pil_to_tensor(img_aug, self.mean, self.std)

        # ── Frequency branch (DCT on clean image) ────────
        img_np = np.array(img, dtype=np.float32)
        dct_arr = compute_dct(img_np)
        freq_tensor = _dct_to_tensor(dct_arr)

        return spatial_tensor, freq_tensor, label


# ─── Data Loading Utilities ───────────────────────────────────────────────────

def load_merged_csv() -> pd.DataFrame:
    """Load and merge both Aadhaar and PAN CSVs into one DataFrame."""
    dfs = []

    if os.path.exists(AADHAAR_CSV):
        df_a = pd.read_csv(AADHAAR_CSV)
        df_a["doc_type"] = "aadhaar"
        dfs.append(df_a)
        print(f"  Aadhaar: {len(df_a)} samples  "
              f"(real={int((df_a['label']==0).sum())}, fake={int((df_a['label']==1).sum())})")

    if os.path.exists(PAN_CSV):
        df_p = pd.read_csv(PAN_CSV)
        df_p["doc_type"] = "pan"
        dfs.append(df_p)
        print(f"  PAN:     {len(df_p)} samples  "
              f"(real={int((df_p['label']==0).sum())}, fake={int((df_p['label']==1).sum())})")

    if not dfs:
        raise FileNotFoundError(
            f"No dataset CSV found at:\n  {AADHAAR_CSV}\n  {PAN_CSV}\n"
            "Run the generation pipelines first."
        )

    merged = pd.concat(dfs, ignore_index=True)

    # Ensure face_file column exists
    if "face_file" not in merged.columns:
        raise ValueError(
            "CSV is missing 'face_file' column!\n"
            "Re-run the generation pipelines to get updated CSVs:\n"
            "  python -m src.pipeline\n"
            "  python -m src.pan_pipeline"
        )

    # Validate: drop rows where image file doesn't exist
    valid_mask = merged["image_path"].apply(
        lambda p: os.path.exists(os.path.join(BASE_DIR, p))
    )
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        print(f"  ⚠ Dropped {n_missing} rows (missing image files)")
    merged = merged[valid_mask].reset_index(drop=True)

    # Keep only the columns we need
    keep_cols = ["image_path", "label", "doc_type", "face_file"]
    merged = merged[[c for c in keep_cols if c in merged.columns]]

    print(f"  ─────────────────────────")
    print(f"  Total: {len(merged)} samples  "
          f"(real={int((merged['label']==0).sum())}, fake={int((merged['label']==1).sum())})")

    return merged


def _assign_face_groups(df: pd.DataFrame) -> pd.Series:
    """
    Assign a group ID to each row based on face_file.

    Rules:
      - All rows sharing the same face_file get the SAME group ID
      - Original real cards (empty face_file) each get a UNIQUE group ID
        so they can freely go into any split
    """
    groups = pd.Series(index=df.index, dtype=str)
    counter = 0

    for idx, row in df.iterrows():
        face = str(row.get("face_file", "")).strip()
        if face == "" or face == "nan":
            # Original real card — unique group (no leakage risk)
            groups[idx] = f"__orig_{counter}"
            counter += 1
        else:
            groups[idx] = face

    return groups


def get_splits(df: pd.DataFrame):
    """
    IDENTITY-AWARE stratified split.

    All documents using the same face go into the SAME split.
    This prevents data leakage.

    Returns:
        train_df, val_df, test_df
    """
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS

    groups = _assign_face_groups(df)

    # First split: train vs (val+test)
    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=(val_ratio + test_ratio),
        random_state=RANDOM_SEED,
    )
    train_idx, temp_idx = next(gss1.split(df, df["label"], groups))

    temp_df = df.iloc[temp_idx]
    temp_groups = groups.iloc[temp_idx]

    # Second split: val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test,
        random_state=RANDOM_SEED,
    )
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, temp_df["label"], temp_groups))

    train_df = df.iloc[train_idx]
    val_df = temp_df.iloc[val_idx_rel]
    test_df = temp_df.iloc[test_idx_rel]

    # Verify no face leakage
    train_faces = set(groups.iloc[train_idx]) - {g for g in groups.iloc[train_idx] if g.startswith("__orig_")}
    val_faces = set(groups.iloc[temp_idx].iloc[val_idx_rel]) - {g for g in groups.iloc[temp_idx].iloc[val_idx_rel] if g.startswith("__orig_")}
    test_faces = set(groups.iloc[temp_idx].iloc[test_idx_rel]) - {g for g in groups.iloc[temp_idx].iloc[test_idx_rel] if g.startswith("__orig_")}

    train_val_leak = train_faces & val_faces
    train_test_leak = train_faces & test_faces
    val_test_leak = val_faces & test_faces

    print(f"\n  Split sizes (identity-aware):")
    print(f"    Train : {len(train_df)}  (real={int((train_df['label']==0).sum())}, fake={int((train_df['label']==1).sum())})")
    print(f"    Val   : {len(val_df)}  (real={int((val_df['label']==0).sum())}, fake={int((val_df['label']==1).sum())})")
    print(f"    Test  : {len(test_df)}  (real={int((test_df['label']==0).sum())}, fake={int((test_df['label']==1).sum())})")
    print(f"\n  Face leakage check:")
    print(f"    Unique faces — train: {len(train_faces)}, val: {len(val_faces)}, test: {len(test_faces)}")
    print(f"    Train↔Val leak:  {len(train_val_leak)} faces")
    print(f"    Train↔Test leak: {len(train_test_leak)} faces")
    print(f"    Val↔Test leak:   {len(val_test_leak)} faces")

    if train_val_leak or train_test_leak:
        print(f"    ⚠ WARNING: Face leakage detected! Results may be unreliable.")
    else:
        print(f"    ✅ No face leakage — splits are clean!")

    return train_df, val_df, test_df


def get_dataloaders(batch_size_train: int, batch_size_eval: int = None):
    """
    Full pipeline: load CSVs → identity-aware split → create DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size_eval is None:
        batch_size_eval = batch_size_train

    print("Loading dataset …")
    df = load_merged_csv()
    train_df, val_df, test_df = get_splits(df)

    train_ds = ForgeryDataset(train_df, augment=True)
    val_ds   = ForgeryDataset(val_df,   augment=False)
    test_ds  = ForgeryDataset(test_df,  augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size_eval, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size_eval, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
