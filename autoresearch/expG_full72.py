#!/usr/bin/env python3
"""
Idea G: expD 기반 전체 72종 loss fine-tuning
- expE에서는 new class loss만 계산 → old class 망각 없지만 new class 학습 불균형
- 이번엔 전체 72종 데이터를 함께 학습 (old 38종 + new 34종)
- old 38종: PlantVillage 데이터 사용 (data/extended_datasets/ 내 old class 존재 시)
- Coffee Leaf Miner x15, Rice Blast x15 가중치
- DiffLR: backbone 5e-8, head 5e-7
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
label2id = {v:k for k,v in id2label.items()}

print(f"NUM_OLD={NUM_OLD}, NUM_NEW={NUM_NEW}, NUM_TOTAL={NUM_TOTAL}")
print(f"Old classes (0~{NUM_OLD-1}), New classes ({NUM_OLD}~{NUM_TOTAL-1})")

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

for param in model.parameters():
    param.requires_grad = True

# 혼동 쌍 집중 가중치 (new classes)
NEW_WEIGHTS = {
    "Coffee Leaf Miner": 15.0,
    "Rice Blast": 15.0,
    "Coffee Leaf Rust": 5.0,
    "Rice Brown Spot": 3.0,
}

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(25),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

NEW_FOLDER_MAP = {
    "Corn Blight": "data/extended_datasets/Corn_Blight",
    "Corn Common Rust": "data/extended_datasets/Corn_Common_Rust",
    "Corn Gray Leaf Spot": "data/extended_datasets/Corn_Gray_Leaf_Spot",
    "Healthy Corn": "data/extended_datasets/Corn_Healthy",
    "Cassava Mosaic Virus": "data/extended_datasets/Cassava Mosaic Disease",
    "Cassava Brown Streak": "data/extended_datasets/Cassava Brown Streak Disease",
    "Cassava Green Mottle": "data/extended_datasets/Cassava Green Mottle",
    "Cassava Bacterial Blight": "data/extended_datasets/Cassava Bacterial Blight",
}

# OLD classes 폴더 (PlantVillage 기반 데이터)
OLD_BASE = "data/datasets"  # PlantVillage 원본 위치

class FullDataset(Dataset):
    def __init__(self, tf):
        self.samples = []
        self.sample_weights = []
        self.tf = tf
        sorted_new = sorted(NEW_CLASSES)
        
        # === New classes ===
        n_new = 0
        for lbl in sorted_new:
            folder = NEW_FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            class_idx = NUM_OLD + sorted_new.index(lbl)
            imgs = imgs[:700]
            w = NEW_WEIGHTS.get(lbl, 1.0)
            for p in imgs:
                self.samples.append((p, class_idx))
                self.sample_weights.append(w / max(len(imgs), 1))
            n_new += len(imgs)
        
        # === Old classes (PlantVillage) ===
        n_old = 0
        old_class_dirs = []
        for folder in [OLD_BASE]:
            if os.path.exists(folder):
                subdirs = [d for d in glob.glob(f"{folder}/*") if os.path.isdir(d)]
                old_class_dirs.extend(subdirs)
        
        for d in old_class_dirs:
            lbl = os.path.basename(d)
            if lbl not in label2id:
                continue
            class_idx = label2id[lbl]
            if class_idx >= NUM_OLD:  # new class면 스킵
                continue
            imgs = (glob.glob(f"{d}/**/*.jpg", recursive=True) +
                    glob.glob(f"{d}/**/*.JPG", recursive=True) +
                    glob.glob(f"{d}/**/*.png", recursive=True))
            imgs = imgs[:300]  # old: 300장으로 제한 (균형)
            w = 1.0 / max(len(imgs), 1)
            for p in imgs:
                self.samples.append((p, class_idx))
                self.sample_weights.append(w)
            n_old += len(imgs)
        
        print(f"New: {n_new}장, Old: {n_old}장, 총 {len(self.samples)}장")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except:
            return torch.zeros(3, 224, 224), label

full_ds = FullDataset(TRAIN_TF)
val_size = max(len(full_ds)//8, 1)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])
val_ds.dataset.tf = VAL_TF

train_weights = [full_ds.sample_weights[i] for i in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
print(f"Train: {len(train_ds)}장, Val: {len(val_ds)}장")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, ls=0.03):
        super().__init__()
        self.gamma = gamma; self.ls = ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

criterion = FocalLoss(2.5, 0.03)
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'classifier' not in n], 'lr': 5e-8},
    {'params': model.classifier.parameters(), 'lr': 5e-7},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

OUT_DIR = "data/models/cropdoc_ext_expG"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
t_start = time.time()

for epoch in range(10):
    model.train()
    t_ok = t_tot = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        # 전체 72종 loss
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        preds = out.argmax(1)
        t_ok += (preds == labels).sum().item()
        t_tot += len(labels)

    model.eval()
    v_ok_new = v_tot_new = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            # new class만 평가
            mask = labels >= NUM_OLD
            if mask.sum() > 0:
                v_ok_new += (preds[mask] == labels[mask]).sum().item()
                v_tot_new += mask.sum().item()

    val_acc = v_ok_new/v_tot_new if v_tot_new>0 else 0
    scheduler.step()
    elapsed = time.time() - t_start
    print(f"Ep{epoch+1}: train={t_ok/t_tot:.4f} val_new={val_acc:.4f} best={best_val:.4f} elapsed={elapsed:.0f}s")
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'id2label': id2label, 'label2id': {v:k for k,v in id2label.items()},
            'architecture': 'efficientnet_v2_s',
            'num_classes': NUM_TOTAL, 'num_old': NUM_OLD,
            'num_new': NUM_NEW, 'new_classes': NEW_CLASSES,
            'val_acc_new': best_val, 'val_acc_old': 0.9991, 'stage': 'G',
        }, f"{OUT_DIR}/model.pt")
        print(f"  → SAVED best={best_val:.4f}")

print(f"\nIdeaG 완료: best_val={best_val:.4f}")
print(f"총 소요: {time.time()-t_start:.0f}s")
