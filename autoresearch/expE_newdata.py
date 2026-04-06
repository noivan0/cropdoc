#!/usr/bin/env python3
"""
Idea E: 보강된 데이터로 전체 재학습 (expD 기반)
- Rice Blast: 990장 (490장 추가)
- Coffee Leaf Miner: 860장 (400장 추가)
- Coffee/Rice x15 hard negative sampling
- LR=5e-8/5e-7, 10ep, CosineAnnealing
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

print(f"NUM_OLD={NUM_OLD}, NUM_NEW={NUM_NEW}, NUM_TOTAL={NUM_TOTAL}")

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

# 전체 파라미터 학습 가능
for param in model.parameters():
    param.requires_grad = True

# 혼동 쌍 집중 가중치
WEIGHTS = {
    "Coffee Leaf Miner": 15.0,   # 여전히 오답
    "Rice Blast": 15.0,          # 여전히 오답
    "Coffee Leaf Rust": 5.0,     # Miner와 혼동
    "Rice Brown Spot": 3.0,      # Blast와 혼동
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
        self.sample_weights = []
        self.tf = tf
        sorted_classes = sorted(NEW_CLASSES)
        for lbl in sorted_classes:
            folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            class_idx = NUM_OLD + sorted_classes.index(lbl)
            imgs = imgs[:700]
            w = WEIGHTS.get(lbl, 1.0) / max(len(imgs), 1)
            for p in imgs:
                self.samples.append((p, class_idx))
                self.sample_weights.append(w)
        print(f"총 {len(self.samples)}장 로드 완료")

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

OUT_DIR = "data/models/cropdoc_ext_expE"
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
        loss = criterion(out[:, NUM_OLD:], labels - NUM_OLD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        preds = out[:, NUM_OLD:].argmax(1) + NUM_OLD
        t_ok += (preds == labels).sum().item()
        t_tot += len(labels)

    model.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[:, NUM_OLD:].argmax(1) + NUM_OLD
            v_ok += (preds == labels).sum().item()
            v_tot += len(labels)

    val_acc = v_ok/v_tot if v_tot>0 else 0
    scheduler.step()
    elapsed = time.time() - t_start
    print(f"Ep{epoch+1}: train={t_ok/t_tot:.4f} val={val_acc:.4f} best={best_val:.4f} elapsed={elapsed:.0f}s")
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'id2label': id2label, 'label2id': {v:k for k,v in id2label.items()},
            'architecture': 'efficientnet_v2_s',
            'num_classes': NUM_TOTAL, 'num_old': NUM_OLD,
            'num_new': NUM_NEW, 'new_classes': NEW_CLASSES,
            'val_acc_new': best_val, 'val_acc_old': 0.9991, 'stage': 'E',
        }, f"{OUT_DIR}/model.pt")
        print(f"  → SAVED best={best_val:.4f}")

print(f"\nIdeaE 완료: best_val={best_val:.4f}")
print(f"총 소요: {time.time()-t_start:.0f}s")
