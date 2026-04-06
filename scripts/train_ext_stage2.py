"""
Stage 2 fine-tuning: backbone 마지막 2블록 + classifier unfreeze
Idea 1: features.6, features.7 + classifier unfreeze
LR = 3e-5 (낮게), 8 epoch, cosine schedule
"""

import torch
import torch.nn as nn
import os
import json
import time
import shutil
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_MODEL = os.path.join(BASE_DIR, "data/models/cropdoc_efficientnet_v2_extended/model_baseline.pt")
EXP_DIR        = os.path.join(BASE_DIR, "data/models/cropdoc_ext_exp1")
EXTENDED_DIR   = os.path.join(BASE_DIR, "data/extended_datasets")
os.makedirs(EXP_DIR, exist_ok=True)

BATCH_SIZE   = 32
EPOCHS       = 8
LR           = 3e-5
WEIGHT_DECAY = 1e-4
SEED         = 42
MIN_IMAGES   = 50

torch.manual_seed(SEED)
np.random.seed(SEED)

FOLDER_TO_LABEL = {
    "Coffee Leaf Rust":              "Coffee Leaf Rust",
    "Coffee Leaf Miner":             "Coffee Leaf Miner",
    "Coffee Phoma":                  "Coffee Phoma",
    "Rice Blast":                    "Rice Blast",
    "Rice Brown Spot":               "Rice Brown Spot",
    "Rice Bacterial Blight":         "Rice Bacterial Blight",
    "Rice Hispa":                    "Rice Hispa",
    "Rice Leaf Smut":                "Rice Leaf Smut",
    "Wheat Leaf Rust":               "Wheat Leaf Rust",
    "Wheat Stripe Rust":             "Wheat Stripe Rust",
    "Wheat Stem Rust":               "Wheat Stem Rust",
    "Wheat Loose Smut":              "Wheat Loose Smut",
    "Mango Anthracnose":             "Mango Anthracnose",
    "Mango Bacterial Canker":        "Mango Bacterial Canker",
    "Mango Die Back":                "Mango Die Back",
    "Mango Gall Midge":              "Mango Gall Midge",
    "Mango Powdery Mildew":          "Mango Powdery Mildew",
    "Mango Sooty Mould":             "Mango Sooty Mould",
    "Cassava Bacterial Blight":      "Cassava Bacterial Blight",
    "Cassava Mosaic Disease":        "Cassava Mosaic Virus",
    "Cassava Brown Streak Disease":  "Cassava Brown Streak",
    "Cassava Green Mottle":          "Cassava Green Mottle",
    "Banana Black Sigatoka":         "Banana Black Sigatoka",
    "Banana Yellow Sigatoka":        "Banana Yellow Sigatoka",
    "Banana Panama Disease":         "Banana Panama Disease",
    "Banana Moko Disease":           "Banana Moko Disease",
    "Banana Bract Mosaic Virus":     "Banana Bract Mosaic Virus",
    "Citrus Canker":                 "Citrus Canker",
    "Citrus Black Spot":             "Citrus Black Spot",
    "Citrus Greening":               "Citrus Greening",
    "Corn_Blight":                   "Corn Blight",
    "Corn_Common_Rust":              "Corn Common Rust",
    "Corn_Gray_Leaf_Spot":           "Corn Gray Leaf Spot",
    "Corn_Healthy":                  "Corn Healthy",
}

# Baseline 모델 로드
print("[1/5] Baseline 모델 로딩...")
ck = torch.load(BASELINE_MODEL, map_location='cpu')
NUM_OLD   = ck['num_old']
NUM_NEW   = ck['num_new']
NUM_TOTAL = ck['num_classes']
NEW_CLASSES = ck['new_classes']
id2label  = ck['id2label']
label2id  = ck['label2id']
OLD_ACC   = ck.get('val_acc_old', '?')
print(f"  num_old={NUM_OLD}, num_new={NUM_NEW}, total={NUM_TOTAL}")
print(f"  baseline val_acc_new={ck['val_acc_new']:.4f}")

# 모델 재구성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}")

model = efficientnet_v2_s(weights=None)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_f, NUM_TOTAL)
)
model.load_state_dict(ck['model_state_dict'])

# Stage 2: features.6, features.7 + classifier unfreeze
# 먼저 전체 동결
for p in model.parameters():
    p.requires_grad = False

unfreeze_patterns = ['classifier', 'features.6', 'features.7']
unfrozen = 0
for name, param in model.named_parameters():
    for pat in unfreeze_patterns:
        if name.startswith(pat):
            param.requires_grad = True
            unfrozen += 1
            break

print(f"  Unfrozen params: {unfrozen} (patterns: {unfreeze_patterns})")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {total_params:,}")

model = model.to(device)

# Augmentation
TRAIN_TF = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomRotation(20),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 데이터셋
print("\n[2/5] 데이터셋 구성...")
valid_new = []
for folder, label in FOLDER_TO_LABEL.items():
    folder_path = os.path.join(EXTENDED_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    if len(images) >= MIN_IMAGES:
        valid_new.append((folder, label))

class ExtendedDataset(Dataset):
    def __init__(self, valid_list, new_classes, num_old, tf):
        self.tf = tf
        self.samples = []
        for folder, label in valid_list:
            if label not in new_classes:
                continue
            idx = new_classes.index(label) + num_old
            folder_path = os.path.join(EXTENDED_DIR, folder)
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg','.jpeg','.png','.webp')):
                    self.samples.append((os.path.join(folder_path, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.tf(img), label
        except Exception:
            return torch.zeros(3, 224, 224), label

class ValSubset(Dataset):
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf = tf
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            img = Image.open(path).convert("RGB")
            return self.tf(img), label
        except Exception:
            return torch.zeros(3, 224, 224), label

full_ds = ExtendedDataset(valid_new, NEW_CLASSES, NUM_OLD, TRAIN_TF)
val_size   = len(full_ds) // 10
train_size = len(full_ds) - val_size
torch.manual_seed(SEED)
train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
val_ds_proper = ValSubset(val_ds, VAL_TF)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds_proper, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"  Train: {train_size}장, Val: {val_size}장")

# 학습
print(f"\n[3/5] Stage 2 학습 시작 ({EPOCHS} epochs, backbone[-2:]+classifier)...")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val_acc = 0.0
start = time.time()

for epoch in range(EPOCHS):
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
            out = model(imgs)
            preds = out[:, NUM_OLD:].argmax(1) + NUM_OLD
            val_correct += (preds == labels).sum().item()
            val_total   += len(labels)

    train_acc = train_correct / train_total
    val_acc   = val_correct   / val_total
    scheduler.step()
    elapsed = (time.time() - start) / 60

    print(f"  Ep{epoch+1}/{EPOCHS} [{elapsed:.1f}m]  "
          f"loss={train_loss_sum/train_total:.4f}  "
          f"train={train_acc:.4f}  val={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "id2label":         id2label,
            "label2id":         label2id,
            "architecture":     "efficientnet_v2_s",
            "num_classes":      NUM_TOTAL,
            "num_old":          NUM_OLD,
            "num_new":          NUM_NEW,
            "new_classes":      NEW_CLASSES,
            "val_acc":          OLD_ACC,
            "val_acc_new":      best_val_acc,
            "val_acc_old":      OLD_ACC,
            "epoch":            epoch + 1,
        }, os.path.join(EXP_DIR, "model.pt"))
        print(f"    ✅ Best model saved (val_acc_new={best_val_acc:.4f})")

total_min = (time.time() - start) / 60
print(f"\n[4/5] 완료: best_val_acc={best_val_acc:.4f}, baseline=0.8281")
print(f"  소요: {total_min:.1f}분")

# 결과 판정
BASELINE = 0.8281
THRESHOLD = BASELINE + 0.020

if best_val_acc >= THRESHOLD:
    print(f"\n🎉 KEEP! {best_val_acc:.4f} >= {THRESHOLD:.4f}")
    # 메인 모델 교체
    shutil.copy(os.path.join(EXP_DIR, "model.pt"),
                os.path.join(BASE_DIR, "data/models/cropdoc_efficientnet_v2_extended/model.pt"))
    print("  ✅ data/models/cropdoc_efficientnet_v2_extended/model.pt 업데이트됨")
    status = "keep"
else:
    print(f"\n❌ DISCARD: {best_val_acc:.4f} < {THRESHOLD:.4f}")
    # 원래 모델 복원
    shutil.copy(os.path.join(BASE_DIR, "data/models/cropdoc_efficientnet_v2_extended/model_baseline.pt"),
                os.path.join(BASE_DIR, "data/models/cropdoc_efficientnet_v2_extended/model.pt"))
    print("  ↩️  baseline 모델 복원됨")
    status = "discard"

print(f"\n[RESULT] idea1_s2  val_acc={best_val_acc:.4f}  status={status}")
