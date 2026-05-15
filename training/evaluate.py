"""
evaluate.py — Evaluation & metrics for the Dual-Branch Forgery Detector.

Produces:
  1. Classification report (precision, recall, F1)
  2. Confusion matrix heatmap  → results/confusion_matrix.png
  3. ROC curve + AUC           → results/roc_curve.png
  4. Training history curves   → results/training_curves.png

Usage:
  python -m training.evaluate
"""

import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from .config import CHECKPOINT_DIR, RESULTS_DIR, PHASE1_BATCH
from .dataset import get_dataloaders
from .model import DualBranchForgeryDetector


def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)

@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for spatial, freq, labels in loader:
        logits = model(spatial.to(device), freq.to(device))
        probs = torch.softmax(logits, dim=1)
        all_labels.append(labels.numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"], linewidths=0.5)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix", fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  📊 Confusion matrix → {save_path}")

def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--")
    plt.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", fontweight="bold"); plt.legend(loc="lower right")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  📈 ROC curve        → {save_path}")
    return auc

def plot_training_curves(history_path, save_path):
    if not os.path.exists(history_path):
        print(f"  ⚠ No training history at {history_path}"); return
    with open(history_path) as f:
        history = json.load(f)
    epochs = list(range(1, len(history) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", color="#ef4444", label="Train", ms=3)
    ax1.plot(epochs, [h["val_loss"] for h in history], "o-", color="#2563eb", label="Val", ms=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss", fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h["train_acc"] for h in history], "o-", color="#ef4444", label="Train", ms=3)
    ax2.plot(epochs, [h["val_acc"] for h in history], "o-", color="#2563eb", label="Val", ms=3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy", fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)
    phase1_end = sum(1 for h in history if h["phase"].startswith("Phase 1"))
    if 0 < phase1_end < len(epochs):
        for ax in (ax1, ax2):
            ax.axvline(x=phase1_end+0.5, color="#94a3b8", linestyle="--", alpha=0.7)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  📉 Training curves  → {save_path}")

def evaluate(checkpoint_path=None):
    _ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}"); return
    print(f"\n  Loading: {checkpoint_path}")
    model = DualBranchForgeryDetector(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, test_loader = get_dataloaders(PHASE1_BATCH, PHASE1_BATCH)
    print(f"  Running inference on test set …")
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    print(f"\n{'='*50}\n  TEST RESULTS\n{'='*50}")
    print(f"  Accuracy : {acc:.4f}\n  Precision: {prec:.4f}\n  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}\n  AUC-ROC  : {auc:.4f}\n{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Real","Fake"], digits=4))
    plot_confusion_matrix(y_true, y_pred, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_training_curves(os.path.join(RESULTS_DIR, "training_history.json"),
                         os.path.join(RESULTS_DIR, "training_curves.png"))
    metrics = {"accuracy": round(acc,4), "precision": round(prec,4), "recall": round(rec,4),
               "f1_score": round(f1,4), "auc_roc": round(auc,4), "test_samples": len(y_true)}
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Evaluation complete! ✅\n")
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    evaluate(checkpoint_path=p.parse_args().checkpoint)
