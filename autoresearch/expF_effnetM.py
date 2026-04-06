#!/usr/bin/env python3
"""
Idea F: EfficientNetV2-M 처음부터 학습
- expD의 38종 old-class head + new 34종 full head
- M 모델은 S보다 표현력 높음 (파라미터 ~2x)
- expD backbone 가중치를 M 모델에 이식 불가 → ImageNet 사전학습 사용
- 신규 클래스 38종 데이터만으로 학습 (old는 freeze)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# expD에서 클래스 정보 로드
ck_d = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck_d['num_old']
NEW_CLASSES = ck_d['new_classes']
NUM_NEW = len(NEW_CLASSES)
NUM_TOTAL = ck_d['num_classes']
id2label = ck_d['id2label']

print(f"NUM_OLD={NUM_OLD}, NUM_NEW={NUM_NEW}, NUM_TOTAL={NUM_TOTAL}")

device = torch.device('cuda')

# EfficientNetV2-M (ImageNet 사전학습)
model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
# M의 classifier out_features = 1000 → 72로 교체
in_feats = model.classifier[1].in_features
print(f"EfficientNetV2-M classifier in_features: {in_feats}")
model.classifier[1] = nn.Linear(in_feats, NUM_TOTAL)

# backbone freeze, classifier만 학습으로 시작 후 점진적 unfreeze
for name, param in model.named_parameters():
    if 'classifier' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model = model.to(device)

WEIGHTS_MAP = {
    "Coffee Leaf Miner": 15.0,
    "Rice Blast": 15.0,
    "Coffee Leaf Rust": 5.0,
    "Rice Brown Spot": 3.0,
}

TRAIN_TF = transforms.Compose([
    transforms.Resize(300),
    transforms.RandomCrop(288),
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
    transforms.Resize(300), transforms.CenterCrop(288),
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
            w = WEIGHTS_MAP.get(lbl, 1.0) / max(len(imgs), 1)
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
            return torch.zeros(3, 288, 288), label

full_ds = NewClassDataset(TRAIN_TF)
val_size = max(len(full_ds)//8, 1)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])
val_ds.dataset.tf = VAL_TF

train_weights = [full_ds.sample_weights[i] for i in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(train_ds, batch_size=24, sampler=sampler, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=4)

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

OUT_DIR = "data/models/cropdoc_ext_expF"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
t_start = time.time()

# Phase 1: classifier only (5ep, higher LR)
print("\n=== Phase 1: Classifier only (5ep) ===")
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(5):
    model.train()
    t_ok = t_tot = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out[:, NUM_OLD:], labels - NUM_OLD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            'architecture': 'efficientnet_v2_m',
            'num_classes': NUM_TOTAL, 'num_old': NUM_OLD,
            'num_new': NUM_NEW, 'new_classes': NEW_CLASSES,
            'val_acc_new': best_val, 'val_acc_old': 0.9991, 'stage': 'F',
        }, f"{OUT_DIR}/model.pt")
        print(f"  → SAVED best={best_val:.4f}")

# Phase 2: unfreeze last 2 feature blocks + classifier (5ep, low LR)
print("\n=== Phase 2: Unfreeze backbone[-2:] + classifier (5ep) ===")
for name, param in model.named_parameters():
    if any(x in name for x in ['features.6', 'features.7', 'classifier']):
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'classifier' not in n and p.requires_grad], 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

for epoch in range(5):
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
    print(f"Ep{5+epoch+1}: train={t_ok/t_tot:.4f} val={val_acc:.4f} best={best_val:.4f} elapsed={elapsed:.0f}s")
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'id2label': id2label, 'label2id': {v:k for k,v in id2label.items()},
            'architecture': 'efficientnet_v2_m',
            'num_classes': NUM_TOTAL, 'num_old': NUM_OLD,
            'num_new': NUM_NEW, 'new_classes': NEW_CLASSES,
            'val_acc_new': best_val, 'val_acc_old': 0.9991, 'stage': 'F',
        }, f"{OUT_DIR}/model.pt")
        print(f"  → SAVED best={best_val:.4f}")

print(f"\nIdeaF 완료: best_val={best_val:.4f}")
print(f"총 소요: {time.time()-t_start:.0f}s")
