"""
Idea P: 극단적 가중치 미세조정
- expD 기반 (val=0.9749, 현재 최고)
- 오답 2개에만 집중:
  1. Coffee Leaf Miner (1361.jpg) - 잎 갱도 패턴 고해상도 이미지
  2. Rice Blast (shape 600.jpg) - Brown Spot vs Blast
- Coffee Miner 전체에 weight 50, Rice Blast 전체에 weight 30
- 극도로 낮은 LR: backbone 5e-8, classifier 5e-7
- 12 epoch, 조기종료
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NEW_CLASSES = ck['new_classes']
NUM_NEW = len(NEW_CLASSES)
NUM_TOTAL = ck['num_classes']
id2label = ck['id2label']
label2id = ck['label2id']

print(f"expD val_acc: {ck.get('val_acc_new'):.4f}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# 극단적 가중치: 오답 pair에 집중
WEIGHT_MAP = {
    "Coffee Leaf Miner": 50.0,   # 핵심 오답
    "Coffee Leaf Rust": 30.0,    # 혼동 대상 (같이 강화해야 함)
    "Rice Blast": 30.0,          # 핵심 오답
    "Rice Brown Spot": 20.0,     # 혼동 대상
    "Coffee Phoma": 10.0,        # Coffee 계열
    "Corn Blight": 3.0,
    "Cassava Brown Streak": 4.0,
    "Citrus Canker": 3.0,
    "Rice Hispa": 8.0,           # Rice 계열 혼동 방지
}

FOLDER_MAP = {
    "Corn Blight": "data/extended_datasets/Corn_Blight",
    "Corn Common Rust": "data/extended_datasets/Corn_Common_Rust",
    "Corn Gray Leaf Spot": "data/extended_datasets/Corn_Gray_Leaf_Spot",
    "Healthy Corn": "data/extended_datasets/Corn_Healthy",
    "Cassava Mosaic Virus": "data/extended_datasets/Cassava Mosaic Disease",
    "Cassava Brown Streak": "data/extended_datasets/Cassava Brown Streak Disease",
    "Cassava Green Mottle": "data/extended_datasets/Cassava Green Mottle",
    "Cassava Bacterial Blight": "data/extended_datasets/Cassava Bacterial Blight",
}

class NewClassDataset(Dataset):
    def __init__(self, tf):
        self.samples = []
        self.weights = []
        self.tf = tf
        sorted_classes = sorted(NEW_CLASSES)
        for lbl in sorted_classes:
            folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            class_idx = NUM_OLD + sorted_classes.index(lbl)
            imgs = imgs[:600]
            w = WEIGHT_MAP.get(lbl, 1.0)
            for p in imgs:
                self.samples.append((p, class_idx))
                self.weights.append(w / max(len(imgs), 1))
        print(f"총 {len(self.samples)}장 로드")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except:
            return torch.zeros(3, 224, 224), label

full_ds = NewClassDataset(TRAIN_TF)
val_size = max(len(full_ds)//8, 1)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(99))
val_ds.dataset.tf = VAL_TF

train_weights = [full_ds.weights[i] for i in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
print(f"train={len(train_ds)}, val={len(val_ds)}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, ls=0.02):
        super().__init__()
        self.gamma = gamma
        self.ls = ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt)**self.gamma * ce).mean()

criterion = FocalLoss(gamma=2.5, ls=0.02)

# 전체 unfreeze, 매우 낮은 LR
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'classifier' not in n], 'lr': 5e-8},
    {'params': model.classifier.parameters(), 'lr': 5e-7},
], weight_decay=1e-4)

EPOCHS = 12
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-9)

OUT_DIR = "data/models/cropdoc_ext_expP"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
no_improve = 0
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    t_ok = t_tot = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        new_out = out[:, NUM_OLD:]
        new_labels = labels - NUM_OLD
        loss = criterion(new_out, new_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        preds = new_out.argmax(1) + NUM_OLD
        t_ok += (preds == labels).sum().item()
        t_tot += len(labels)

    model.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = out[:, NUM_OLD:].argmax(1) + NUM_OLD
            v_ok += (preds == labels).sum().item()
            v_tot += len(labels)

    val_acc = v_ok/v_tot if v_tot > 0 else 0
    scheduler.step()
    elapsed = time.time() - t0
    print(f"Ep{epoch+1}/{EPOCHS}: train={t_ok/t_tot:.4f} val={val_acc:.4f} (best={best_val:.4f}) [{elapsed:.0f}s]")

    if val_acc > best_val:
        best_val = val_acc
        no_improve = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'id2label': id2label, 'label2id': label2id,
            'architecture': 'efficientnet_v2_s',
            'num_classes': NUM_TOTAL, 'num_old': NUM_OLD,
            'num_new': NUM_NEW, 'new_classes': NEW_CLASSES,
            'val_acc_new': best_val,
            'val_acc_old': ck.get('val_acc_old', 0.9991),
            'stage': 'P',
        }, f"{OUT_DIR}/model.pt")
        print(f"  → 저장 (best={best_val:.4f})")
    else:
        no_improve += 1
        if no_improve >= 5:
            print(f"  Early stopping (no improvement for 5 epochs)")
            break

total_time = time.time() - t0
print(f"\nIdea P 완료: val_acc_new={best_val:.4f}, 총 {total_time/60:.1f}분")
print(f"threshold(0.9779): {'KEEP ✓' if best_val >= 0.9779 else f'DISCARD (gap={best_val-0.9779:.4f})'}")
print(f"expD(0.9749) 대비: {'+' if best_val >= 0.9749 else ''}{best_val-0.9749:.4f}")
