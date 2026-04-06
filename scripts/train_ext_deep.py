"""
Idea 4: exp3(95.65%) 기반 — 더 깊은 fine-tuning
- features.4,5,6,7 + classifier (backbone 절반 이상 unfreeze)
- LR=1e-5 (매우 낮게), 15 epoch
- Focal Loss + 강화 augmentation 결합
- exp3 모델에서 시작 (현재 95.65%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import shutil
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# exp3에서 시작 (95.65%)
START_MODEL = os.path.join(BASE_DIR, "data/models/cropdoc_ext_exp3/model.pt")
EXP1_MODEL  = os.path.join(BASE_DIR, "data/models/cropdoc_ext_exp1/model.pt")
EXP_DIR     = os.path.join(BASE_DIR, "data/models/cropdoc_ext_exp4")
EXTENDED_DIR = os.path.join(BASE_DIR, "data/extended_datasets")
os.makedirs(EXP_DIR, exist_ok=True)

BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-5
WEIGHT_DECAY = 1e-4
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

print("[1/5] 모델 로딩 (exp3 기반)...")
# exp3 모델 시도, 없으면 exp1 사용
if os.path.exists(START_MODEL):
    ck = torch.load(START_MODEL, map_location='cpu')
    print(f"  exp3 모델: val_acc_new={ck['val_acc_new']:.4f}")
else:
    ck = torch.load(EXP1_MODEL, map_location='cpu')
    print(f"  exp3 없음, exp1 모델 사용: val_acc_new={ck['val_acc_new']:.4f}")

NUM_OLD   = ck['num_old']
NUM_NEW   = ck['num_new']
NUM_TOTAL = ck['num_classes']
NEW_CLASSES = ck['new_classes']
id2label  = ck['id2label']
label2id  = ck['label2id']
OLD_ACC   = ck.get('val_acc_old', '?')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_v2_s(weights=None)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, NUM_TOTAL))
model.load_state_dict(ck['model_state_dict'])

# features.4,5,6,7 + classifier 모두 unfreeze
for p in model.parameters():
    p.requires_grad = False
unfreeze_patterns = ['classifier', 'features.4', 'features.5', 'features.6', 'features.7']
for name, param in model.named_parameters():
    for pat in unfreeze_patterns:
        if name.startswith(pat):
            param.requires_grad = True
            break

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Unfreeze: {unfreeze_patterns}")
print(f"  Trainable params: {total_params:,}, device: {device}")
model = model.to(device)

# Augmentation (Idea 2 강화 버전)
TRAIN_TF = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomRotation(30),
    T.RandomGrayscale(p=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds_proper, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"  Train: {train_size}장, Val: {val_size}장")

print(f"\n[3/5] deep fine-tuning ({EPOCHS} epochs, feat.4-7+classifier, LR={LR})...")
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)

best_val_acc = 0.0
start = time.time()

for epoch in range(EPOCHS):
    elapsed = (time.time() - start) / 60
    if elapsed > 14.5:
        print(f"  ⏱️ Time limit at epoch {epoch+1}")
        break

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
            preds = model(imgs)[:, NUM_OLD:].argmax(1) + NUM_OLD
            val_correct += (preds == labels).sum().item()
            val_total   += len(labels)

    train_acc = train_correct / train_total
    val_acc   = val_correct   / val_total
    scheduler.step()
    elapsed = (time.time() - start) / 60
    print(f"  Ep{epoch+1}/{EPOCHS} [{elapsed:.1f}m]  loss={train_loss_sum/train_total:.4f}  train={train_acc:.4f}  val={val_acc:.4f}")

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
    # exp1 경로에 저장 (공식 best)
    shutil.copy(os.path.join(EXP_DIR, "model.pt"),
                os.path.join(BASE_DIR, "data/models/cropdoc_efficientnet_v2_extended/model_exp4.pt"))
    status = "keep"
else:
    print(f"❌ DISCARD (improvement={best_val_acc-BASELINE:+.4f})")
    status = "discard"

print(f"\n[RESULT] idea4_deep  val_acc={best_val_acc:.4f}  status={status}  (baseline {BASELINE:.4f})")
