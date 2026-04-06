"""
Idea D: Coffee Leaf Rust 강화 학습
- expC 기반 (val=0.9715)
- Coffee Leaf Rust: weight x20, 전체 데이터 사용 (400장)
- Coffee Leaf Miner: weight x15
- Corn Common Rust vs Wheat Stripe Rust: weight x10
- 매우 낮은 LR (전체 구조 거의 고정, classifier만 미세조정)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# expC 로드 (best checkpoint, val=0.9715)
ck = torch.load('data/models/cropdoc_ext_expC/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NEW_CLASSES = ck['new_classes']
NUM_NEW = len(NEW_CLASSES)
NUM_TOTAL = ck['num_classes']
id2label = ck['id2label']
label2id = {v:k for k,v in id2label.items()}

print(f"expC val_acc: {ck.get('val_acc_new'):.4f}")

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

# Coffee 문제에 집중: 더 강한 weight
WEIGHT_MAP = {
    "Coffee Leaf Rust": 20.0,   # 400장, 테스트 3/5 오답 핵심
    "Coffee Leaf Miner": 15.0,  # 860장, 혼동 대상
    "Corn Common Rust": 10.0,   # vs Wheat Stripe Rust
    "Wheat Stripe Rust": 5.0,
    "Cassava Brown Streak": 8.0,
    "Cassava Mosaic Virus": 5.0,
    "Citrus Canker": 6.0,
    "Rice Brown Spot": 6.0,
    "Corn Blight": 4.0,  # expC에서 -1장
}

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(30),
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
        self.weights = []
        self.tf = tf
        sorted_classes = sorted(NEW_CLASSES)
        for lbl in sorted_classes:
            folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            class_idx = NUM_OLD + sorted_classes.index(lbl)
            imgs = imgs[:600]  # 최대 600장
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
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(77))
val_ds.dataset.tf = VAL_TF

train_weights = [full_ds.weights[i] for i in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
print(f"train={len(train_ds)}, val={len(val_ds)}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, ls=0.03):
        super().__init__()
        self.gamma = gamma
        self.ls = ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt)**self.gamma * ce).mean()

criterion = FocalLoss(gamma=2.5, ls=0.03)

# 전체 unfreeze, 극도로 낮은 LR
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'classifier' not in n], 'lr': 1e-7},
    {'params': model.classifier.parameters(), 'lr': 1e-6},
], weight_decay=1e-4)

EPOCHS = 8
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-9)

OUT_DIR = "data/models/cropdoc_ext_expD"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
        torch.save({
            'model_state_dict': model.state_dict(),
            'id2label': id2label, 'label2id': label2id,
            'architecture': 'efficientnet_v2_s',
            'num_classes': NUM_TOTAL, 'num_old': NUM_OLD,
            'num_new': NUM_NEW, 'new_classes': NEW_CLASSES,
            'val_acc_new': best_val,
            'val_acc_old': ck.get('val_acc_old', 0.9991),
            'stage': 'D',
        }, f"{OUT_DIR}/model.pt")
        print(f"  → 저장 (best={best_val:.4f})")

total_time = time.time() - t0
print(f"\nIdea D 완료: val_acc_new={best_val:.4f}, 총 {total_time/60:.1f}분")
print(f"threshold(0.9715): {'KEEP ✓' if best_val >= 0.9715 else f'DISCARD (gap={best_val-0.9715:.4f})'}")
