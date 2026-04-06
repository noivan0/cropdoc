#!/usr/bin/env python3.10
"""
Stage 2 Fine-tuning: EfficientNetV2-S backbone[-3:] unfreeze
목표: 신규 34종 val_acc 82.8% → 88%+
"""
import torch, torch.nn as nn, os, glob, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

print("=" * 60)
print("Stage 2 Fine-tuning 시작")
print("=" * 60)

# ── Task 1: 실패 케이스 분석 ──────────────────────────────────
print("\n[Task 1] 실패 케이스 분석")

ck = torch.load('data/models/cropdoc_efficientnet_v2_extended/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']  # 38
NEW_CLASSES = ck['new_classes']
NUM_NEW = len(NEW_CLASSES)
NUM_TOTAL = ck['num_classes']
id2label = ck['id2label']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

problem_classes = {
    "Corn Common Rust": "data/extended_datasets/Corn_Common_Rust",
    "Cassava Mosaic Virus": "data/extended_datasets/Cassava Mosaic Disease",
    "Coffee Leaf Rust": "data/extended_datasets/Coffee Leaf Rust",
}

for label, folder in problem_classes.items():
    imgs = (glob.glob(f"{folder}/*.jpg") +
            glob.glob(f"{folder}/*.JPG") +
            glob.glob(f"{folder}/*.png") +
            glob.glob(f"{folder}/**/*.jpg", recursive=True) +
            glob.glob(f"{folder}/**/*.JPG", recursive=True))
    imgs = list(set(imgs))[:10]
    print(f"\n=== {label} ({len(imgs)}장) ===")
    for img_path in imgs[:5]:
        try:
            img = Image.open(img_path).convert('RGB')
            t = TF(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(t)[0], -1).cpu().numpy()
            top3_all = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
            top3_new = sorted([(i,p) for i,p in enumerate(probs) if i>=NUM_OLD], key=lambda x:-x[1])[:3]
            print(f"  {os.path.basename(img_path)[:30]}")
            print(f"    전체 top1: {id2label[top3_all[0][0]]}({top3_all[0][1]:.3f})")
            print(f"    신규 top1: {id2label[top3_new[0][0]]}({top3_new[0][1]:.3f}), top2: {id2label[top3_new[1][0]]}({top3_new[1][1]:.3f})")
        except Exception as e:
            print(f"  [오류] {img_path}: {e}")

# ── Task 2: Stage 2 Fine-tuning ───────────────────────────────
print("\n" + "=" * 60)
print("[Task 2] Stage 2 Fine-tuning 시작")
print("=" * 60)

# 새 모델 로드 (Stage 1 가중치에서 시작)
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

# Stage 2: 전부 동결 후 상위 블록만 unfreeze
for name, param in model.named_parameters():
    param.requires_grad = False

unfreeze_patterns = ['classifier', 'features.6', 'features.7']
for name, param in model.named_parameters():
    if any(p in name for p in unfreeze_patterns):
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"학습 파라미터: {trainable:,} / {total:,} ({trainable/total:.1%})")

# 데이터셋 정의
TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class NewClassDataset(Dataset):
    def __init__(self, tf):
        self.samples = []
        self.tf = tf
        folder_map = {
            "Corn Blight": "data/extended_datasets/Corn_Blight",
            "Corn Common Rust": "data/extended_datasets/Corn_Common_Rust",
            "Corn Gray Leaf Spot": "data/extended_datasets/Corn_Gray_Leaf_Spot",
            "Healthy Corn": "data/extended_datasets/Corn_Healthy",
            "Cassava Mosaic Virus": "data/extended_datasets/Cassava Mosaic Disease",
            "Cassava Brown Streak": "data/extended_datasets/Cassava Brown Streak Disease",
        }
        sorted_classes = sorted(NEW_CLASSES)
        for lbl in sorted_classes:
            folder = folder_map.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            imgs = list(set(imgs))
            class_idx = NUM_OLD + sorted_classes.index(lbl)
            count = 0
            for p in imgs[:500]:
                self.samples.append((p, class_idx))
                count += 1
            if count == 0:
                print(f"  ⚠️  {lbl}: 이미지 없음 (폴더: {folder})")
            else:
                print(f"  {lbl}: {count}장")
        print(f"총 {len(self.samples)}장 로드")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.tf(img), label
        except:
            return torch.zeros(3, 224, 224), label

print("\n데이터셋 구성:")
full_ds = NewClassDataset(TRAIN_TF)
val_size = max(len(full_ds) // 10, 1)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))
val_ds.dataset.tf = VAL_TF

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)
print(f"Train: {train_size}, Val: {val_size}")

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-5, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

OUT_DIR = "data/models/cropdoc_efficientnet_v2_extended_v2"
os.makedirs(OUT_DIR, exist_ok=True)

best_val = 0.0
print("\n학습 시작...")
for epoch in range(8):
    t0 = time.time()
    model.train()
    t_ok = t_tot = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        # 신규 클래스만 loss
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

    val_acc = v_ok / v_tot if v_tot > 0 else 0
    scheduler.step()
    elapsed = time.time() - t0
    print(f"Ep{epoch+1}/8: train={t_ok/t_tot:.4f} val={val_acc:.4f} ({elapsed:.0f}s)")

    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'id2label': id2label,
            'label2id': {v:k for k,v in id2label.items()},
            'architecture': 'efficientnet_v2_s',
            'num_classes': NUM_TOTAL,
            'num_old': NUM_OLD,
            'num_new': NUM_NEW,
            'new_classes': NEW_CLASSES,
            'val_acc_new': best_val,
            'val_acc_old': ck['val_acc_old'],
            'stage': 2,
        }, f"{OUT_DIR}/model.pt")
        print(f"  ✅ 저장 (best={best_val:.4f})")

print(f"\n✅ Stage 2 완료! val_acc_new={best_val:.4f}")
print(f"저장: {OUT_DIR}/model.pt")

# ── Task 4: 전체 신규 클래스 정확도 재측정 ─────────────────────
print("\n" + "=" * 60)
print("[Task 4] 전체 신규 클래스 정확도 재측정")
print("=" * 60)

# best model 로드
ck2 = torch.load(f"{OUT_DIR}/model.pt", map_location='cpu')
model2 = efficientnet_v2_s()
model2.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model2.load_state_dict(ck2['model_state_dict'])
model2.eval().to(device)

folder_map_all = {
    "Corn Blight": "data/extended_datasets/Corn_Blight",
    "Corn Common Rust": "data/extended_datasets/Corn_Common_Rust",
    "Corn Gray Leaf Spot": "data/extended_datasets/Corn_Gray_Leaf_Spot",
    "Healthy Corn": "data/extended_datasets/Corn_Healthy",
    "Cassava Mosaic Virus": "data/extended_datasets/Cassava Mosaic Disease",
    "Cassava Brown Streak": "data/extended_datasets/Cassava Brown Streak Disease",
}

sorted_classes = sorted(NEW_CLASSES)
total_ok = total_n = 0
print(f"\n{'클래스':<35} {'정확도':>8} {'(ok/n)':>10}")
print("-" * 60)
for lbl in sorted_classes:
    folder = folder_map_all.get(lbl, f"data/extended_datasets/{lbl}")
    imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
            glob.glob(f"{folder}/**/*.JPG", recursive=True) +
            glob.glob(f"{folder}/**/*.png", recursive=True))
    imgs = list(set(imgs))[:5]  # 각 5장
    if not imgs:
        print(f"{'  ' + lbl:<35} {'N/A':>8}")
        continue

    class_idx = NUM_OLD + sorted_classes.index(lbl)
    ok = 0
    for img_path in imgs:
        try:
            img = Image.open(img_path).convert('RGB')
            t = TF(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model2(t)
                pred = out[:, NUM_OLD:].argmax(1).item() + NUM_OLD
            if pred == class_idx:
                ok += 1
        except:
            pass
    acc = ok / len(imgs) if imgs else 0
    total_ok += ok
    total_n += len(imgs)
    status = "✅" if acc >= 0.6 else ("⚠️ " if acc >= 0.4 else "❌")
    print(f"{status} {lbl:<33} {acc:>7.1%} ({ok}/{len(imgs)})")

print("-" * 60)
overall = total_ok / total_n if total_n > 0 else 0
print(f"{'전체 평균':<35} {overall:>7.1%} ({total_ok}/{total_n})")
print(f"\nStage 1 val_acc: 82.81% → Stage 2 val_acc: {best_val:.2%}")
print("DONE")
