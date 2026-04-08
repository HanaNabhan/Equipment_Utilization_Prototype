# ╔══════════════════════════════════════════════════════════════════╗
# ║  LSTM Activity Classifier — Complete Training Notebook          ║
# ║  Google Colab — GPU Runtime (T4)                                ║
# ║                                                                  ║
# ║  Steps:                                                          ║
# ║   1. Runtime → Change runtime type → T4 GPU                     ║
# ║   2. Upload f1.pt, f2.pt, f3.pt to Colab files panel            ║
# ║   3. Run all cells in order (Ctrl+F9 = run all)                 ║
# ║   4. Download best_lstm.pth when done                           ║
# ║   5. Place best_lstm.pth in your project root                   ║
# ║   6. Set use_lstm=True in run_local.py                          ║
# ╚══════════════════════════════════════════════════════════════════╝


# ── CELL 1: Setup ─────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from pathlib import Path

print("="*55)
print("  LSTM ACTIVITY CLASSIFIER — TRAINING")
print("="*55)
print(f"\nPyTorch : {torch.__version__}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
if not torch.cuda.is_available():
    print("\n⚠ No GPU — go to Runtime → Change runtime type → T4 GPU")


# ── CELL 2: Configuration ─────────────────────────────────────────
INPUT_DIM   = 6
HIDDEN_DIM  = 64
NUM_CLASSES = 4
SEQ_LEN     = 16
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10

CLASS_NAMES = ["DIGGING", "SWINGING", "DUMPING", "WAITING"]
WAITING_IDX = 3
DUMPING_IDX = 2

print(f"\nClasses : {CLASS_NAMES}")
print(f"Seq len : {SEQ_LEN} frames (~0.5s at 30fps)")
print(f"Epochs  : {EPOCHS} (early stop at {PATIENCE} patience)")


# ── CELL 3: Load feature files ────────────────────────────────────
print("\nLoading feature files...")

all_features = []
all_labels   = []

for fname in ["f1.pt", "f2.pt", "f3.pt"]:
    if not Path(fname).exists():
        print(f"  ⚠ {fname} not found — skipping")
        continue
    d = torch.load(fname, map_location="cpu")
    f, l = d["features"], d["labels"]
    all_features.append(f)
    all_labels.append(l)
    counts = Counter(l.tolist())
    print(f"\n  {fname}: {len(f)} frames")
    for i, name in enumerate(CLASS_NAMES):
        c = counts.get(i, 0)
        print(f"    {name:<12} {c:>5}  {'█' * min(c//100, 20)}")

if not all_features:
    raise RuntimeError("No .pt files found! Upload f1.pt, f2.pt, f3.pt first.")

features = torch.cat(all_features)
labels   = torch.cat(all_labels)
print(f"\nCombined: {len(features)} frames")


# ── CELL 4: Augment minority classes ─────────────────────────────
print("\nClass distribution BEFORE augmentation:")
counts_raw = Counter(labels.tolist())
for i, name in enumerate(CLASS_NAMES):
    c = counts_raw.get(i, 0)
    print(f"  {name:<12} {c:>6}  {'█' * min(c//200, 25)}")

max_count = max(counts_raw.values())

def augment_class(features, labels, class_idx, target_count, noise_std=0.005):
    """Duplicate minority class samples with small gaussian noise."""
    mask  = (labels == class_idx)
    feats = features[mask]
    current = len(feats)
    if current == 0 or current >= target_count:
        return features, labels

    n_copies = max(1, (target_count - current) // current)
    aug_list = []
    for _ in range(n_copies):
        noise = torch.randn_like(feats) * noise_std
        noise = noise.clamp(-noise_std * 4, noise_std * 4)
        aug_list.append(feats + noise)

    aug_feats  = torch.cat(aug_list)
    aug_labels = torch.full((len(aug_feats),), class_idx, dtype=torch.long)
    return (
        torch.cat([features, aug_feats]),
        torch.cat([labels,   aug_labels])
    )

# Augment WAITING to 50% of majority class
target_waiting = max(max_count // 2, counts_raw.get(WAITING_IDX, 0) * 4)
features, labels = augment_class(features, labels, WAITING_IDX, target_waiting)

# Augment DUMPING to 40% of majority class
target_dumping = max(max_count * 2 // 5, counts_raw.get(DUMPING_IDX, 0) * 3)
features, labels = augment_class(features, labels, DUMPING_IDX, target_dumping)

print("\nClass distribution AFTER augmentation:")
counts_aug = Counter(labels.tolist())
for i, name in enumerate(CLASS_NAMES):
    c = counts_aug.get(i, 0)
    print(f"  {name:<12} {c:>6}  {'█' * min(c//200, 25)}")
print(f"\nTotal frames: {len(features)}")


# ── CELL 5: Dataset ───────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        self.X       = features
        self.y       = labels
        self.seq_len = seq_len
        self.indices = list(range(seq_len - 1, len(features)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end   = self.indices[idx] + 1
        start = end - self.seq_len
        return self.X[start:end], self.y[self.indices[idx]]


dataset = SequenceDataset(features, labels, SEQ_LEN)
n_val   = int(len(dataset) * 0.20)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"\nSequences — train: {len(train_ds)}  val: {len(val_ds)}")


# ── CELL 6: Model ─────────────────────────────────────────────────
class ActivityLSTM(nn.Module):
    """Must match architecture in cv_service/motion_analyzer.py exactly."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = INPUT_DIM,
            hidden_size = HIDDEN_DIM,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


model = ActivityLSTM().to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── CELL 7: Loss with class weights ───────────────────────────────
train_labels_list = torch.tensor([
    dataset[i][1].item() for i in train_ds.indices
])
class_counts  = torch.zeros(NUM_CLASSES)
for i in range(NUM_CLASSES):
    class_counts[i] = (train_labels_list == i).sum().float()

class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(DEVICE)

print("\nClass weights:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<12} n={int(class_counts[i]):>6}  w={class_weights[i]:.3f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=4, factor=0.5, verbose=True
)


# ── CELL 8: Training ──────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    pc_correct = torch.zeros(NUM_CLASSES)
    pc_total   = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += len(y)
            for c in range(NUM_CLASSES):
                mask = (y == c)
                pc_correct[c] += (preds[mask] == c).sum().item()
                pc_total[c]   += mask.sum().item()
    return correct / total, pc_correct / (pc_total + 1e-6)


best_val_acc   = 0.0
patience_count = 0

print("\nTraining...\n")
print(f"{'Ep':>4} {'Train':>7} {'Val':>7} {'Best':>7}  "
      f"DIG / SWG / DMP / WAT")
print("─" * 62)

for epoch in range(1, EPOCHS + 1):
    model.train()
    correct = total = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds    = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += len(y)

    train_acc       = correct / total
    val_acc, per_c  = evaluate(model, val_loader)
    scheduler.step(val_acc)

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc   = val_acc
        patience_count = 0
        torch.save(model.state_dict(), "best_lstm.pth")
    else:
        patience_count += 1

    per_str = " / ".join([f"{per_c[i]*100:>4.0f}%" for i in range(NUM_CLASSES)])
    marker  = " ◄" if is_best else ""
    print(f"{epoch:>4} {train_acc:>7.3f} {val_acc:>7.3f} "
          f"{best_val_acc:>7.3f}  {per_str}{marker}")

    if patience_count >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break


# ── CELL 9: Final results ─────────────────────────────────────────
model.load_state_dict(torch.load("best_lstm.pth"))
val_acc, per_c = evaluate(model, val_loader)

print("\n" + "="*55)
print("  FINAL RESULTS")
print("="*55)
print(f"  Best val accuracy: {best_val_acc*100:.1f}%\n")
for i, name in enumerate(CLASS_NAMES):
    acc    = per_c[i] * 100
    bar    = "█" * int(acc / 5)
    status = "✓" if acc >= 70 else "⚠"
    print(f"  {status} {name:<12} {acc:>5.1f}%  {bar}")

print("\n  Guide:")
print("  > 85% → excellent, ready for production")
print("  > 70% → good, ready for prototype delivery")
print("  > 60% → acceptable")
print("  < 60% → needs more labeled data")


# ── CELL 10: Download ─────────────────────────────────────────────
try:
    from google.colab import files
    files.download("best_lstm.pth")
    print("\nbest_lstm.pth downloaded!")
except ImportError:
    print("\nNot in Colab — find best_lstm.pth in current directory")

print("\n" + "="*55)
print("  NEXT STEPS")
print("="*55)
print("  1. Place best_lstm.pth in project root")
print("  2. In run_local.py change use_lstm=False → use_lstm=True")
print("  3. python run_local.py --video data/input.mp4 --fresh")
print("="*55)
