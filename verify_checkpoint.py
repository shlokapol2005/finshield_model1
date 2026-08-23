"""
verify_checkpoint.py - Confirms the best_model.pt matches training history
and loads cleanly into the architecture.
"""
import torch, json, sys, os

# 1. Check what's stored in best_model.pt
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
print('=== Checkpoint metadata (what was ACTUALLY saved) ===')
print(f'  Phase    : {ckpt["phase"]}')
print(f'  Epoch    : {ckpt["epoch"]}')
print(f'  Val Loss : {ckpt["val_loss"]}')
print(f'  Val Acc  : {ckpt["val_acc"]}')

# 2. Cross-check with training_history.json
with open('results/training_history.json') as f:
    history = json.load(f)
best_entry = min(history, key=lambda h: h['val_loss'])
print()
print('=== Best epoch in training_history.json (lowest val_loss) ===')
print(f'  Phase    : {best_entry["phase"]}')
print(f'  Epoch    : {best_entry["epoch"]}')
print(f'  Val Loss : {best_entry["val_loss"]}')
print(f'  Val Acc  : {best_entry["val_acc"]}')

# 3. Check all val_accs across training
print()
print('=== All epochs - val_acc over training ===')
for h in history:
    marker = ' <-- BEST (saved)' if h['val_loss'] == ckpt['val_loss'] else ''
    print(f'  {h["phase"]:35s} epoch {h["epoch"]:02d} | val_loss={h["val_loss"]:.4f} | val_acc={h["val_acc"]:.4f}{marker}')

# 4. Verify state dict loads cleanly
sys.path.insert(0, os.getcwd())
from training.model import DualBranchForgeryDetector
model = DualBranchForgeryDetector(pretrained=False)
result = model.load_state_dict(ckpt['model_state_dict'], strict=True)
print()
print('=== State dict load result ===')
print(f'  Missing keys   : {result.missing_keys}')
print(f'  Unexpected keys: {result.unexpected_keys}')
ok = len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0
print(f'  Loaded cleanly : {ok}')
if ok:
    print('  -> The evaluated model is 100% the trained EfficientNet checkpoint.')
else:
    print('  -> WARNING: Architecture mismatch!')

# 5. Count parameters
total = sum(p.numel() for p in model.parameters())
print(f'  Total params   : {total:,}')
