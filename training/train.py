"""
train.py — Two-phase training loop for the Dual-Branch Forgery Detector.

Phase 1 (warm-up):
  - Freeze EfficientNet-B0 backbone
  - Train only frequency branch + fusion head
  - 5 epochs at LR = 1e-3

Phase 2 (fine-tune):
  - Unfreeze entire model
  - Differential LR:  backbone=1e-5, freq=5e-4, fusion=1e-3
  - Up to 30 epochs with early stopping (patience=7)

Checkpoint fix:
  - best_model.pt is ONLY overwritten when exact val_loss improves
  - History is written after EVERY epoch so it's never lost
  - Phase-level best checkpoints saved separately

Usage:
  python -m training.train
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    RESULTS_DIR,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE1_BATCH,
    PHASE2_EPOCHS,
    PHASE2_LR_BACKBONE,
    PHASE2_LR_FREQ,
    PHASE2_LR_FUSION,
    PHASE2_BATCH,
    WEIGHT_DECAY,
    GRAD_CLIP,
    EARLY_STOP_PATIENCE,
)
from .dataset import get_dataloaders, load_merged_csv
from .model import DualBranchForgeryDetector, FocalLoss


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_dirs():
    for d in (CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)


def _accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _compute_class_weights(train_loader, device):
    """Compute inverse-frequency class weights from training set."""
    all_labels = []
    for _, _, labels in train_loader:
        all_labels.extend(labels.numpy().tolist())
    all_labels = np.array(all_labels)
    n_total = len(all_labels)
    n_classes = 2
    weights = []
    for c in range(n_classes):
        n_c = (all_labels == c).sum()
        weights.append(n_total / (n_classes * max(n_c, 1)))
    w = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"  Class weights: Real={w[0]:.4f}, Fake={w[1]:.4f}")
    return w


# ─── Single Epoch ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=GRAD_CLIP):
    """Train for one epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for spatial, freq, labels in loader:
        spatial = spatial.to(device)
        freq = freq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(spatial, freq)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_acc += _accuracy(logits, labels)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate. Returns (avg_loss, avg_accuracy)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for spatial, freq, labels in loader:
        spatial = spatial.to(device)
        freq = freq.to(device)
        labels = labels.to(device)

        logits = model(spatial, freq)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        total_acc += _accuracy(logits, labels)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


# ─── Training Phase ───────────────────────────────────────────────────────────

def run_phase(
    phase_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    n_epochs,
    patience,
    history,
    best_val_loss,
    history_path,
):
    """
    Run a training phase.

    FIX vs old version:
      - Compares EXACT (unrounded) float val_loss so the checkpoint is never
        accidentally overwritten by a worse result from a re-run.
      - Writes history to disk after EVERY epoch.
      - Saves a phase-specific checkpoint so the best per-phase is preserved.
    """
    best_path       = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    phase_slug      = phase_name.lower().replace(" ", "_").replace("-", "").replace("/", "")
    phase_best_path = os.path.join(CHECKPOINT_DIR, f"best_{phase_slug}.pt")
    phase_best_loss = float("inf")
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        # ── Checkpoint: GLOBAL best (never overwritten by worse) ──────────
        improved_global = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,      # exact float, NOT rounded
                "val_acc":  val_acc,
                "phase":    phase_name,
                "epoch":    epoch,
            }, best_path)
            improved_global = " [BEST]"

        # ── Checkpoint: PHASE best ─────────────────────────────────────────
        if val_loss < phase_best_loss:
            phase_best_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc":  val_acc,
                "phase":    phase_name,
                "epoch":    epoch,
            }, phase_best_path)

        # ── Early stopping counter ─────────────────────────────────────────
        if improved_global:
            no_improve = 0
        else:
            no_improve += 1

        # ── Log record (write to disk every epoch) ─────────────────────────
        record = {
            "phase":      phase_name,
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
            "val_loss_exact": val_loss,         # store exact value too
            "lr":         optimizer.param_groups[0]["lr"],
            "time_s":     round(elapsed, 1),
        }
        history.append(record)

        # Write history to disk immediately after each epoch
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"  Epoch {epoch:02d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s{improved_global}"
        )

        if no_improve >= patience:
            print(f"\n  Early stopping (no improvement for {patience} epochs).")
            break

    return best_val_loss


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(
    phase1_epochs=PHASE1_EPOCHS,
    phase2_epochs=PHASE2_EPOCHS,
    phase1_batch=PHASE1_BATCH,
    phase2_batch=PHASE2_BATCH,
):
    _ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Backup old stale checkpoint if it exists ───────────────────────────
    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best_path):
        backup = best_path.replace(".pt", "_previous_run.pt")
        shutil.copy2(best_path, backup)
        print(f"  Backed up old checkpoint -> {backup}")

    # ── History path ───────────────────────────────────────────────────────
    history_path = os.path.join(RESULTS_DIR, "training_history.json")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size_train=phase1_batch,
        batch_size_eval=phase1_batch,
    )

    # ── Class weights ──────────────────────────────────────────────────────
    class_weights = _compute_class_weights(train_loader, device)

    # ── Model ──────────────────────────────────────────────────────────────
    model = DualBranchForgeryDetector(pretrained=True).to(device)
    criterion = FocalLoss(class_weights=class_weights)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    history = []
    best_val_loss = float("inf")

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: Freeze backbone — train freq + fusion only
    # ════════════════════════════════════════════════════════════════
    model.freeze_backbone()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer1 = AdamW(trainable, lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
    scheduler1 = CosineAnnealingWarmRestarts(optimizer1, T_0=2, T_mult=2)

    best_val_loss = run_phase(
        phase_name="Phase 1 - Backbone Frozen",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer1,
        scheduler=scheduler1,
        device=device,
        n_epochs=phase1_epochs,
        patience=phase1_epochs,    # no early stop in phase 1
        history=history,
        best_val_loss=best_val_loss,
        history_path=history_path,
    )

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze everything — differential LR
    # ════════════════════════════════════════════════════════════════
    model.unfreeze_backbone()
    param_groups = model.get_param_groups(
        lr_backbone=PHASE2_LR_BACKBONE,
        lr_freq=PHASE2_LR_FREQ,
        lr_fusion=PHASE2_LR_FUSION,
    )
    print(f"  Phase 2 trainable params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer2 = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=5, T_mult=2)

    if phase2_batch != phase1_batch:
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size_train=phase2_batch,
            batch_size_eval=phase2_batch,
        )

    best_val_loss = run_phase(
        phase_name="Phase 2 - Full Fine-Tune",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer2,
        scheduler=scheduler2,
        device=device,
        n_epochs=phase2_epochs,
        patience=EARLY_STOP_PATIENCE,
        history=history,
        best_val_loss=best_val_loss,
        history_path=history_path,
    )

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "history": history}, final_path)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best validation loss  : {best_val_loss:.4f}")
    print(f"  Best model saved      : {best_path}")
    print(f"  Final model saved     : {final_path}")
    print(f"  Training history      : {history_path}")
    print(f"{'='*60}\n")

    return model, history


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train Dual-Branch Forgery Detector")
    p.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS)
    p.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS)
    p.add_argument("--phase1-batch",  type=int, default=PHASE1_BATCH)
    p.add_argument("--phase2-batch",  type=int, default=PHASE2_BATCH)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_batch=args.phase1_batch,
        phase2_batch=args.phase2_batch,
    )
