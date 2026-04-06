"""
Idea 6: Restart fine-tuning from exp1 with warmup + higher LR
- exp1(94.58%) 기반
- features.6, 7 + classifier unfreeze (아이디어1과 동일 범위)
- LR=5e-5 (아이디어1의 1.67x), warmup 2ep, cosine decay
- 20 epochs (더 오래)
- Label smoothing 없이 CrossEntropy (overconfidence 방지)
"""

import torch
import torch.nn as nn
import os
import time
import shutil
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from PIL import Image
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_MODEL  = os.path.join(BASE_DIR, "data/models/cropdoc_ext_exp1/model.pt")
EXP_DIR      = os.path.join(BASE_DIR, "data/models/cropdoc_ext_exp6")
EXTENDED_DIR = os.path.join(BASE_DIR, "data/extended_datasets")
os.makedirs(EXP_DIR, exist_ok=True)

BATCH_SIZE   = 32
EPOCHS       = 20
LR_BASE      = 5e-5
LR_WARMUP    = 1e-6
WARMUP_EPOCHS= 2
WEIGHT_DECAY = 5e-5
SEED         = 42
MIN_IMAGES   = 50
BASELINE     = 0.9458

torch.manual_seed(SEED)
np.random.seed(SEED)

FOLDER_TO_LABEL = {
    "Coffee Leaf Rust": "Coffee Leaf Rust",
    "Coffee Leaf Miner": "Coffee Leaf Miner",
    "Coffee Phoma": "Coffee Phoma",
    "Rice Blast": "Rice Blast",
    "Rice Brown Spot": "Rice Brown Spot",
    "Rice Bacterial Blight": "Rice Bacterial Blight",
    "Rice Hispa": "Rice Hispa",
    "Rice Leaf Smut": "Rice Leaf Smut",
    "Wheat Leaf Rust": "Wheat Leaf Rust",
    "Wheat Stripe Rust": "Wheat Stripe Rust",
    "Wheat Stem Rust": "Wheat Stem Rust",
    "Wheat Loose Smut": "Wheat Loose Smut",
    "Mango Anthracnose": "Mango Anthracnose",
    "Mango Bacterial Canker": "Mango Bacterial Canker",
    "Mango Die Back": "Mango Die Back",
    "Mango Gall Midge": "Mango Gall Midge",
    "Mango Powdery Mildew": "Mango Powdery Mildew",
    "Mango Sooty Mould": "Mango Sooty Mould",
    "Cassava Bacterial Blight": "Cassava Bacterial Blight",
    "Cassava Mosaic Disease": "Cassava Mosaic Virus",
    "Cassava Brown Streak Disease": "Cassava Brown Streak",
    "Cassava Green Mottle": "Cassava Green Mottle",
    "Banana Black Sigatoka": "Banana Black Sigatoka",
    "Banana Yellow Sigatoka": "Banana Yellow Sigatoka",
    "Banana Panama Disease": "Banana Panama Disease",
    "Banana Moko Disease": "Banana Moko Disease",
    "Banana Bract Mosaic Virus": "Banana Bract Mosaic Virus",
    "Citrus Canker": "Citrus Canker",
    "Citrus Black Spot": "Citrus Black Spot",
    "Citrus Greening": "Citrus Greening",
    "Corn_Blight": "Corn Blight",
    "Corn_Common_Rust": "Corn Common Rust",
    "Corn_Gray_Leaf_Spot": "Corn Gray Leaf Spot",
    "Corn_Healthy": "Corn Healthy",
}

print("[1/5] 모델 로딩 (exp1 기반)...")
ck = torch.load(START_MODEL, map_location='cpu')
NUM_OLD   = ck['num_old']
NUM_NEW   = ck['num_new']
NUM_TOTAL = ck['num_classes']
NEW_CLASSES = ck['new_classes']
id2label  = ck['id2label']
label2id  = ck['label2id']
OLD_ACC   = ck.get('val_acc_old', '?')
print(f"  exp1 val_acc_new={ck['val_acc_new']:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_v2_s(weights=None)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, NUM_TOTAL))
model.load_state_dict(ck['model_state_dict'])

# features.6,7 + classifier (idea1과 동일 범위)
for p in model.parameters():
    p.requires_grad = False
for name, param in model.named_parameters():
    for pat in ['classifier', 'features.6', 'features.7']:
        if name.startswith(pat):
            param.requires_grad = True
            break

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable: {total_params:,}, device: {device}")
model = model.to(device)

TRAIN_TF = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("\n[2/5] 데이터셋...")
valid_new = []
for folder, label in FOLDER_TO_LABEL.items():
    fp = os.path.join(EXTENDED_DIR, folder)
    if not os.path.isdir(fp): continue
    imgs = [f for f in os.listdir(fp) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    if len(imgs) >= MIN_IMAGES:
        valid_new.append((folder, label))

class ExtendedDataset(Dataset):
    def __init__(self, valid_list, new_classes, num_old, tf):
        self.tf = tf
        self.samples = []
        for folder, label in valid_list:
            if label not in new_classes: continue
            idx = new_classes.index(label) + num_old
            fp = os.path.join(EXTENDED_DIR, folder)
            for f in os.listdir(fp):
                if f.lower().endswith(('.jpg','.jpeg','.png','.webp')):
                    self.samples.append((os.path.join(fp, f), idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try: return self.tf(Image.open(path).convert("RGB")), label
        except: return torch.zeros(3, 224, 224), label

class ValSubset(Dataset):
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf = tf
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try: return self.tf(Image.open(path).convert("RGB")), label
        except: return torch.zeros(3, 224, 224), label

full_ds = ExtendedDataset(valid_new, NEW_CLASSES, NUM_OLD, TRAIN_TF)
val_size   = len(full_ds) // 10
train_size = len(full_ds) - val_size
torch.manual_seed(SEED)
train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
val_ds_proper = ValSubset(val_ds, VAL_TF)

# Weighted sampler for hard classes
hard_classes_idx = {}
for label in ["Corn Common Rust", "Cassava Mosaic Virus", "Coffee Leaf Rust"]:
    if label in NEW_CLASSES:
        hard_classes_idx[NEW_CLASSES.index(label) + NUM_OLD] = 3.0
for label in ["Coffee Leaf Miner", "Banana Bract Mosaic Virus"]:
    if label in NEW_CLASSES:
        hard_classes_idx[NEW_CLASSES.index(label) + NUM_OLD] = 2.0

sample_weights = []
for idx in train_ds.indices:
    _, label = full_ds.samples[idx]
    w = hard_classes_idx.get(label, 1.0)
    sample_weights.append(w)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds_proper, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"  Train: {train_size}장, Val: {val_size}장 (weighted sampler)")

print(f"\n[3/5] Restart fine-tuning ({EPOCHS} epochs, LR={LR_BASE}, warmup={WARMUP_EPOCHS}ep)...")
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_BASE, weight_decay=WEIGHT_DECAY)

def warmup_cosine_lr(epoch, warmup_ep, total_ep, lr_base, lr_warmup):
    if epoch < warmup_ep:
        return lr_warmup + (lr_base - lr_warmup) * epoch / warmup_ep
    prog = (epoch - warmup_ep) / (total_ep - warmup_ep)
    return lr_warmup + 0.5 * (lr_base - lr_warmup) * (1 + math.cos(math.pi * prog))

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
best_val_acc = 0.0
start = time.time()

for epoch in range(EPOCHS):
    elapsed = (time.time() - start) / 60
    if elapsed > 14.5:
        print(f"  ⏱️ Time limit at epoch {epoch+1}")
        break

    # LR 업데이트
    lr = warmup_cosine_lr(epoch, WARMUP_EPOCHS, EPOCHS, LR_BASE, LR_WARMUP)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    model.train()
    train_correct = train_total = 0
    train_loss_sum = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        new_logits = out[:, NUM_OLD:]
        new_labels = labels - NUM_OLD
        loss = criterion(new_logits, new_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = new_logits.argmax(1) + NUM_OLD
        train_correct += (preds == labels).sum().item()
        train_total   += len(labels)
        train_loss_sum += loss.item() * len(labels)

    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[:, NUM_OLD:].argmax(1) + NUM_OLD
            val_correct += (preds == labels).sum().item()
            val_total   += len(labels)

    train_acc = train_correct / train_total
    val_acc   = val_correct   / val_total
    elapsed = (time.time() - start) / 60
    print(f"  Ep{epoch+1}/{EPOCHS} [{elapsed:.1f}m] lr={lr:.2e}  loss={train_loss_sum/train_total:.4f}  train={train_acc:.4f}  val={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "id2label": id2label, "label2id": label2id,
            "architecture": "efficientnet_v2_s",
            "num_classes": NUM_TOTAL, "num_old": NUM_OLD,
            "num_new": NUM_NEW, "new_classes": NEW_CLASSES,
            "val_acc": OLD_ACC, "val_acc_new": best_val_acc,
            "val_acc_old": OLD_ACC, "epoch": epoch + 1,
        }, os.path.join(EXP_DIR, "model.pt"))
        print(f"    ✅ Best (val_acc_new={best_val_acc:.4f})")

THRESHOLD = BASELINE + 0.020
print(f"\n[4/5] 완료: best={best_val_acc:.4f}, baseline={BASELINE:.4f}, threshold={THRESHOLD:.4f}")

if best_val_acc >= THRESHOLD:
    print(f"🎉 KEEP!")
    status = "keep"
else:
    print(f"❌ DISCARD (improvement={best_val_acc-BASELINE:+.4f})")
    status = "discard"

print(f"\n[RESULT] idea6_restart  val_acc={best_val_acc:.4f}  status={status}")
