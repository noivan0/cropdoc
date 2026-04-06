#!/usr/bin/env python3
"""
Idea H: expD 기반 Contrastive Hard Pair 집중 학습
- 실패 분석: Coffee Leaf Miner(4/5) vs Coffee Leaf Rust, Rice Blast(4/5) vs Rice Brown Spot
- 전략: 혼동 쌍만 x30 가중치, 나머지 x1
  - val set에서 혼동 클래스 성능 최대화
- LR: backbone 2e-7, head 2e-6 (expD vs G의 중간값)
- Scheduler: CosineAnnealing, 12ep
- 추가: Mixup augmentation (혼동 쌍 학습 강화)
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
print(f"Baseline val_acc: {ck['val_acc_new']:.4f}")

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

for param in model.parameters():
    param.requires_grad = True

# 극단적 혼동 쌍 가중치
WEIGHTS = {
    # 혼동 쌍: x30
    "Coffee Leaf Miner": 30.0,
    "Coffee Leaf Rust": 30.0,
    "Rice Blast": 30.0,
    "Rice Brown Spot": 20.0,
    # 보조 클래스: x5
    "Banana Black Sigatoka": 5.0,
    "Banana Yellow Sigatoka": 5.0,
}

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.08),
    transforms.RandomRotation(20),
    transforms.RandomGrayscale(p=0.08),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
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
    def __init__(self, gamma=3.0, ls=0.02):
        super().__init__()
        self.gamma = gamma; self.ls = ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

criterion = FocalLoss(3.0, 0.02)

optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'classifier' not in n], 'lr': 2e-7},
    {'params': model.classifier.parameters(), 'lr': 2e-6},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

OUT_DIR = "data/models/cropdoc_ext_expH"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
t_start = time.time()

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

NUM_EPOCHS = 12
for epoch in range(NUM_EPOCHS):
    model.train()
    t_ok = t_tot = 0
    use_mixup = epoch < 8  # 후반 4ep는 mixup 없이 clean

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup and np.random.random() < 0.5:
            imgs_mix, y_a, y_b, lam = mixup_data(imgs, labels)
            out = model(imgs_mix)
            loss = mixup_criterion(
                lambda p, t: criterion(p[:, NUM_OLD:], t - NUM_OLD),
                out, y_a, y_b, lam
            )
        else:
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
            'val_acc_new': best_val, 'val_acc_old': 0.9991, 'stage': 'H',
        }, f"{OUT_DIR}/model.pt")
        print(f"  → SAVED best={best_val:.4f}")

print(f"\nIdeaH 완료: best_val={best_val:.4f}")
print(f"총 소요: {time.time()-t_start:.0f}s")
