"""
Idea L: 혼동 쌍 집중 fine-tuning
- expD 기반 (val=0.9749)
- 7개 오답 클래스 + 혼동 대상 클래스에 극도 oversampling
- 224×224 유지 (고해상도 실험 실패)
- 매우 낮은 LR, 짧은 epoch
오답 케이스:
  1. Coffee Leaf Rust ↔ Coffee Leaf Miner
  2. Rice Hispa ↔ Rice Blast
  3. Corn Common Rust ↔ Wheat Stripe Rust
  4. Corn Blight ↔ Corn Gray Leaf Spot
  5. Cassava Brown Streak ↔ Cassava Mosaic Virus
  6. Citrus Canker ↔ Citrus Black Spot
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob, time, numpy as np
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
label2id = {v: k for k, v in id2label.items()}

print(f"expD val_acc_new: {ck.get('val_acc_new'):.4f}")

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

# 224×224 기반 (고해상도 실험 실패 → 원래 해상도)
TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.15),
    transforms.RandomRotation(30),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 혼동 쌍 기반 강화 weight:
# - 오답 클래스와 혼동 대상 모두 높은 weight
WEIGHT_MAP = {
    # 핵심 오답 클래스 (극도 강화)
    "Coffee Leaf Rust": 40.0,      # 400장, 핵심 오답
    "Coffee Leaf Miner": 30.0,     # 혼동 대상
    "Rice Hispa": 35.0,            # 오답 (2위와 3.25%p차)
    "Rice Blast": 20.0,            # 혼동 대상
    "Corn Common Rust": 35.0,      # 오답
    "Wheat Stripe Rust": 20.0,     # 혼동 대상
    "Corn Blight": 30.0,           # 오답
    "Corn Gray Leaf Spot": 20.0,   # 혼동 대상
    "Cassava Brown Streak": 30.0,  # 오답
    "Cassava Mosaic Virus": 20.0,  # 혼동 대상
    "Citrus Canker": 30.0,         # 오답
    "Citrus Black Spot": 20.0,     # 혼동 대상
    # 기타 (낮은 weight)
    "Rice Brown Spot": 5.0,
    "Wheat Leaf Rust": 3.0,
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
val_size = max(len(full_ds) // 8, 1)
train_size = len(full_ds) - val_size
gen = torch.Generator().manual_seed(77)
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=gen)

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
            return torch.zeros(3, 224, 224), label

val_ds_proper = ValWrapper(val_ds, VAL_TF)
train_weights = [full_ds.weights[i] for i in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds_proper, batch_size=64, shuffle=False, num_workers=4)
print(f"train={len(train_ds)}, val={len(val_ds)}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, ls=0.02):
        super().__init__()
        self.gamma = gamma
        self.ls = ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

criterion = FocalLoss(gamma=3.0, ls=0.02)

# 전체 unfreeze, 매우 낮은 LR
for param in model.parameters():
    param.requires_grad = True

# 분류기는 더 높은 LR, backbone은 극도로 낮게
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier' not in n and 'features.7' not in n and 'features.6' not in n], 'lr': 5e-8},
    {'params': [p for n, p in model.named_parameters() if 'features.6' in n or 'features.7' in n], 'lr': 2e-7},
    {'params': model.classifier.parameters(), 'lr': 2e-6},
], weight_decay=1e-4)

EPOCHS = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-9)

OUT_DIR = "data/models/cropdoc_ext_expL"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
baseline = ck['val_acc_new']  # 0.9749
t0 = time.time()

print(f"\n학습 시작 ({EPOCHS} epochs, 224×224, 혼동쌍 집중 weight)")
print(f"baseline: {baseline:.4f}, target: 0.9779")

for epoch in range(EPOCHS):
    model.train()
    t_ok = t_tot = 0
    t_loss = 0.0
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
            'architecture': 'EfficientNetV2-S',
            'num_classes': NUM_TOTAL,
            'num_old': NUM_OLD,
            'num_new': NUM_NEW,
            'new_classes': NEW_CLASSES,
            'val_acc_new': best_val,
            'val_acc_old': ck['val_acc_old'],
            'stage': 'L',
        }, f'{OUT_DIR}/model.pt')
        print(f"    ✅ Best saved (val={best_val:.4f})")

    scheduler.step()

print(f"\n=== Idea L 결과 ===")
print(f"best val_acc_new: {best_val:.4f}")
print(f"baseline (expD):  {baseline:.4f}")
print(f"개선: {best_val - baseline:+.4f}")
keep = best_val >= 0.9779
print(f"{'✅ KEEP' if keep else '❌ DISCARD'} (기준: 0.9779)")

# 85장 테스트도 실행
if best_val > baseline * 0.99:
    print("\n85장 테스트 실행...")
    ck_best = torch.load(f'{OUT_DIR}/model.pt', map_location='cpu')
    model_best = efficientnet_v2_s()
    model_best.classifier[1] = nn.Linear(1280, NUM_TOTAL)
    model_best.load_state_dict(ck_best['model_state_dict'])
    model_best.eval().to(device)

    import sys
    sys.path.insert(0, 'scripts')
    test_cases = [
        ('data/extended_datasets/Coffee Leaf Rust', 'Coffee Leaf Rust'),
        ('data/extended_datasets/Coffee Leaf Miner', 'Coffee Leaf Miner'),
        ('data/extended_datasets/Coffee Phoma', 'Coffee Phoma'),
        ('data/extended_datasets/Rice Blast', 'Rice Blast'),
        ('data/extended_datasets/Rice Brown Spot', 'Rice Brown Spot'),
        ('data/extended_datasets/Rice Hispa', 'Rice Hispa'),
        ('data/extended_datasets/Mango Anthracnose', 'Mango Anthracnose'),
        ('data/extended_datasets/Mango Powdery Mildew', 'Mango Powdery Mildew'),
        ('data/extended_datasets/Wheat Stripe Rust', 'Wheat Stripe Rust'),
        ('data/extended_datasets/Wheat Leaf Rust', 'Wheat Leaf Rust'),
        ('data/extended_datasets/Corn_Common_Rust', 'Corn Common Rust'),
        ('data/extended_datasets/Corn_Blight', 'Corn Blight'),
        ('data/extended_datasets/Banana Black Sigatoka', 'Banana Black Sigatoka'),
        ('data/extended_datasets/Cassava Mosaic Disease', 'Cassava Mosaic Virus'),
        ('data/extended_datasets/Cassava Brown Streak Disease', 'Cassava Brown Streak'),
        ('data/extended_datasets/Citrus Canker', 'Citrus Canker'),
        ('data/extended_datasets/Citrus Black Spot', 'Citrus Black Spot'),
    ]

    NORM = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    TTA_TFs = [
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), NORM]),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), NORM]),
        transforms.Compose([transforms.Resize(280), transforms.CenterCrop(224), transforms.ToTensor(), NORM]),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), NORM]),
    ]

    total_ok, total = 0, 0
    for folder, true_label in test_cases:
        imgs = sorted(
            glob.glob(f'{folder}/*.jpg') + glob.glob(f'{folder}/*.JPG') + glob.glob(f'{folder}/*.png')
        )[:5]
        ok = 0
        for img_path in imgs:
            img = Image.open(img_path).convert('RGB')
            tensors = torch.stack([tf(img) for tf in TTA_TFs]).to(device)
            with torch.no_grad():
                probs = torch.softmax(model_best(tensors), -1).mean(0).cpu().numpy()
            new_probs = [(float(probs[i]), id2label[i]) for i in range(NUM_OLD, NUM_TOTAL)]
            top = sorted(new_probs, reverse=True)[0]
            hit = (true_label.lower() in top[1].lower() or top[1].lower() in true_label.lower())
            ok += hit
            total += 1
        total_ok += ok
        icon = '✅' if ok == len(imgs) else ('⚠️' if ok > 0 else '❌')
        print(f"  {icon} {true_label}: {ok}/{len(imgs)}")

    print(f"\n85장 테스트: {total_ok}/{total} = {total_ok/total:.1%}")
