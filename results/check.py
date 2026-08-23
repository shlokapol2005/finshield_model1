import torch

checkpoint = torch.load("checkpoints/best_model.pt")

print(checkpoint.keys())
import torch

ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")

print("Validation Accuracy:", ckpt['val_acc'])
print("Validation Loss:", ckpt['val_loss'])
print("Phase:", ckpt['phase'])
print("Epoch:", ckpt['epoch'])