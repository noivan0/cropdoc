"""
Idea O: 고해상도 448 fine-tuning
- expD 기반 (val=0.9749, 현재 최고)
- Coffee Leaf Miner 1361.jpg: 2048×1024 고해상도 → 448 입력으로 세부 패턴 학습
- Rice Blast shape600.jpg: 256×256 → 448은 업스케일이지만 Brown Spot 구분에 도움될 수 있음
- Backbone 동결 + classifier fine-tuning
- LR=1e-4 (classifier만), 5 epoch

전략:
1. classifier 레이어만 unfreeze (빠른 수렴)
2. 448×448 입력 → 세부 텍스처 패턴 학습
3. Coffee Leaf Miner/Rust, Rice Blast/Brown Spot에 가중치 집중
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# expD 로드 (best, val=0.9749)
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

# 고해상도 학습 transform (448)
TRAIN_TF_448 = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF_448 = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# 오답 집중 가중치
WEIGHT_MAP = {
    "Coffee Leaf Miner": 20.0,   # 핵심 오답: 1361.jpg
    "Coffee Leaf Rust": 15.0,    # 혼동 대상
    "Rice Blast": 15.0,          # 핵심 오답: shape600.jpg
    "Rice Brown Spot": 10.0,     # 혼동 대상
    "Coffee Phoma": 5.0,
    "Corn Blight": 4.0,
    "Cassava Brown Streak": 5.0,
    "Citrus Canker": 4.0,
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
            return torch.zeros(3, 448, 448), label

full_ds = NewClassDataset(TRAIN_TF_448)
val_size = max(len(full_ds)//8, 1)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# val dataset은 val transform 사용
class ValWrapper(Dataset):
    def __init__(self, base_ds, indices, tf):
        self.base_ds = base_ds
        self.indices = indices
        self.tf = tf
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        path, label = self.base_ds.samples[self.indices[idx]]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except:
            return torch.zeros(3, 448, 448), label

val_ds_proper = ValWrapper(full_ds, val_ds.indices, VAL_TF_448)

train_weights = [full_ds.weights[i] for i in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights))

# batch_size=16 (448 입력이라 메모리 더 사용)
train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds_proper, batch_size=16, shuffle=False, num_workers=4)
print(f"train={len(train_ds)}, val={len(val_ds_proper)}")

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

# Backbone 동결, classifier만 학습
for name, param in model.named_parameters():
    if 'classifier' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
print(f"학습 파라미터: {sum(p.numel() for p in classifier_params):,}")

optimizer = torch.optim.AdamW(classifier_params, lr=1e-4, weight_decay=1e-4)

EPOCHS = 5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

OUT_DIR = "data/models/cropdoc_ext_expO"
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            'stage': 'O',
            'input_size': 448,
        }, f"{OUT_DIR}/model.pt")
        print(f"  → 저장 (best={best_val:.4f})")

total_time = time.time() - t0
print(f"\nIdea O 완료: val_acc_new={best_val:.4f}, 총 {total_time/60:.1f}분")
print(f"threshold(0.9779): {'KEEP ✓' if best_val >= 0.9779 else f'DISCARD (gap={best_val-0.9779:.4f})'}")
