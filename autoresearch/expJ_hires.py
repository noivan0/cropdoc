"""
Idea J: 고해상도 fine-tuning
- expD 기반 (val=0.9749)
- Resize(512) → RandomCrop(384) → 모델 입력 (EfficientNetV2는 adaptive avg pool로 가변 해상도 지원)
- 특히 Coffee 1361.jpg (2048x1024) 같은 고해상도 이미지 개선
- Coffee Leaf Rust weight 강화 유지
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob, time, numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# expD 로드
ck = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NEW_CLASSES = ck['new_classes']
NUM_NEW = len(NEW_CLASSES)
NUM_TOTAL = ck['num_classes']
id2label = ck['id2label']
label2id = {v: k for k, v in id2label.items()}

print(f"expD val_acc_new: {ck.get('val_acc_new'):.4f}")
print(f"NUM_OLD={NUM_OLD}, NUM_NEW={NUM_NEW}, NUM_TOTAL={NUM_TOTAL}")

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

# 고해상도 입력: Resize(512) → RandomCrop(384)
# EfficientNetV2는 adaptive avg pool로 384x384도 처리 가능
TRAIN_TF_HIRES = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop(384),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(30),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF_HIRES = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# expD와 동일한 weight map
WEIGHT_MAP = {
    "Coffee Leaf Rust": 20.0,
    "Coffee Leaf Miner": 15.0,
    "Corn Common Rust": 10.0,
    "Wheat Stripe Rust": 5.0,
    "Cassava Brown Streak": 8.0,
    "Cassava Mosaic Virus": 5.0,
    "Citrus Canker": 6.0,
    "Rice Brown Spot": 6.0,
    "Corn Blight": 4.0,
    "Rice Hispa": 8.0,    # 오답 클래스 강화
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
            return torch.zeros(3, 384, 384), label

full_ds = NewClassDataset(TRAIN_TF_HIRES)
val_size = max(len(full_ds) // 8, 1)
train_size = len(full_ds) - val_size
gen = torch.Generator().manual_seed(77)
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=gen)

# val은 고해상도 VAL_TF
class ValWrapper(Dataset):
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf = tf
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except:
            return torch.zeros(3, 384, 384), label

val_ds_proper = ValWrapper(val_ds, VAL_TF_HIRES)

train_weights = [full_ds.weights[i] for i in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights))
# 배치 사이즈를 줄임 (384x384 고해상도 → 메모리 증가)
BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds_proper, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"train={len(train_ds)}, val={len(val_ds)}, batch={BATCH_SIZE}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, ls=0.03):
        super().__init__()
        self.gamma = gamma
        self.ls = ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

criterion = FocalLoss(gamma=2.5, ls=0.03)

# 전체 unfreeze, 극도로 낮은 LR (expD와 동일)
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': 1e-7},
    {'params': model.classifier.parameters(), 'lr': 1e-6},
], weight_decay=1e-4)

EPOCHS = 8
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-9)

OUT_DIR = "data/models/cropdoc_ext_expJ"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
baseline = ck['val_acc_new']  # 0.9749
t0 = time.time()

print(f"\n학습 시작 ({EPOCHS} epochs, 고해상도 384x384)")
print(f"baseline: {baseline:.4f}, target: 0.9779")

for epoch in range(EPOCHS):
    model.train()
    t_ok = t_tot = 0
    t_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        preds = out.argmax(dim=1)
        t_ok += (preds == labels).sum().item()
        t_tot += labels.size(0)
        t_loss += loss.item()

    train_acc = t_ok / t_tot

    # val
    model.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1)
            v_ok += (preds == labels).sum().item()
            v_tot += labels.size(0)

    val_acc = v_ok / v_tot
    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1}/{EPOCHS}: train={train_acc:.4f} val={val_acc:.4f} loss={t_loss/len(train_loader):.4f} [{elapsed:.0f}s]")

    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'id2label': id2label,
            'label2id': label2id,
            'architecture': 'EfficientNetV2-S-hires384',
            'num_classes': NUM_TOTAL,
            'num_old': NUM_OLD,
            'num_new': NUM_NEW,
            'new_classes': NEW_CLASSES,
            'val_acc_new': best_val,
            'val_acc_old': ck['val_acc_old'],
            'stage': 'J',
        }, f'{OUT_DIR}/model.pt')
        print(f"    ✅ Best saved (val={best_val:.4f})")

    scheduler.step()

print(f"\n=== Idea J 결과 ===")
print(f"best val_acc_new: {best_val:.4f}")
print(f"baseline (expD):  {baseline:.4f}")
print(f"개선: {best_val - baseline:+.4f}")
keep = best_val >= 0.9779
print(f"{'✅ KEEP' if keep else '❌ DISCARD'} (기준: 0.9779)")
