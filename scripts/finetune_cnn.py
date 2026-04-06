"""
CropDoc CNN 파인튜닝 스크립트
================================
MobileNetV2 (PlantVillage 학습) → PlantDoc 현장 이미지로 도메인 적응 파인튜닝

전략: Feature Extractor 동결 + Classifier 레이어만 학습 (빠른 수렴)
      이후 전체 언프리즈 후 낮은 lr로 fine-grained 조정

목표: 현장 이미지 정확도 12~81% → 85%+ 달성
"""

import os, sys, json, time, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification
from collections import defaultdict
import random

# ── 경로 ─────────────────────────────────────────────────────────────────────
CNN_PATH = (
    "/root/.cache/huggingface/hub/cropdoc_cnn/"
    "models--linkanjarad--mobilenet_v2_1.0_224-plant-disease-identification/"
    "snapshots/c1861579a670fb6232258805b801cd4137cb7176"
)
FIELD_BASE  = "data/field_images"
OUTPUT_PATH = "data/models/cropdoc_cnn_finetuned"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── CNN 레이블 id 로드 ────────────────────────────────────────────────────────
proc  = MobileNetV2ImageProcessor.from_pretrained(CNN_PATH, local_files_only=True)
model = MobileNetV2ForImageClassification.from_pretrained(CNN_PATH, local_files_only=True)
id2label = model.config.id2label
label2id = {v: int(k) for k, v in id2label.items()}
NUM_CLASSES = len(id2label)

# ── 현장 이미지 → CNN 레이블 매핑 ─────────────────────────────────────────────
FIELD_TO_CNN = {
    ("tomato", "healthy"):         "Healthy Tomato Plant",
    ("tomato", "early_blight"):    "Tomato with Early Blight",
    ("tomato", "late_blight"):     "Tomato with Late Blight",
    ("tomato", "bacterial_spot"):  "Tomato with Bacterial Spot",
    ("tomato", "leaf_mold"):       "Tomato with Leaf Mold",
    ("tomato", "septoria"):        "Tomato with Septoria Leaf Spot",
    ("potato", "healthy"):         "Healthy Potato Plant",
    ("potato", "early_blight"):    "Potato with Early Blight",
    ("potato", "late_blight"):     "Potato with Late Blight",
    ("pepper", "healthy"):         "Healthy Bell Pepper Plant",
    ("pepper", "bacterial_spot"):  "Bell Pepper with Bacterial Spot",
    ("apple",  "healthy"):         "Healthy Apple",
    ("apple",  "scab"):            "Apple Scab",
    ("apple",  "rust"):            "Cedar Apple Rust",
    ("corn",   "gray_leaf_spot"):  "Corn (Maize) with Cercospora and Gray Leaf Spot",
    ("corn",   "common_rust"):     "Corn (Maize) with Common Rust",
    ("corn",   "northern_blight"): "Corn (Maize) with Northern Leaf Blight",
    ("grape",  "healthy"):         "Healthy Grape Plant",
    ("grape",  "black_rot"):       "Grape with Black Rot",
    ("strawberry", "healthy"):     "Healthy Strawberry Plant",
    ("blueberry",  "healthy"):     "Healthy Blueberry Plant",
}


# ── Dataset ──────────────────────────────────────────────────────────────────
class FieldDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items      # [(img_path, class_id), ...]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label_id = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label_id


def build_dataset():
    items = []
    for (plant, cond), cnn_lbl in FIELD_TO_CNN.items():
        d = os.path.join(FIELD_BASE, plant, cond)
        if not os.path.exists(d):
            continue
        class_id = label2id.get(cnn_lbl)
        if class_id is None:
            print(f"WARNING: no id for {cnn_lbl}")
            continue
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append((os.path.join(d, f), class_id))
    random.shuffle(items)
    return items


# augmentation transforms for training
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # MobileNetV2 norm
])

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    all_items = build_dataset()
    print(f"전체 데이터: {len(all_items)}장")

    # 80/20 train/val split (stratified)
    by_class = defaultdict(list)
    for item in all_items:
        by_class[item[1]].append(item)

    train_items, val_items = [], []
    for cls_id, cls_items in by_class.items():
        random.shuffle(cls_items)
        n_val = max(1, int(len(cls_items) * 0.2))
        val_items.extend(cls_items[:n_val])
        train_items.extend(cls_items[n_val:])

    print(f"Train: {len(train_items)}장, Val: {len(val_items)}장")

    train_ds = FieldDataset(train_items, TRAIN_TF)
    val_ds   = FieldDataset(val_items,   VAL_TF)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)

    # ── 모델 준비 ─────────────────────────────────────────────────────────────
    model.to(device)

    # Phase 1: Classifier만 학습 (backbone 동결)
    for param in model.mobilenet_v2.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    best_val_acc = 0.0

    def run_epoch(dl, train=True):
        model.train() if train else model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, labels in dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs).logits
                loss = criterion(out, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * len(labels)
                correct  += (out.argmax(1) == labels).sum().item()
                total    += len(labels)
        return loss_sum / total, correct / total

    print("\n=== Phase 1: Classifier-only (5 epochs) ===")
    for ep in range(5):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_dl, train=True)
        va_loss, va_acc = run_epoch(val_dl,   train=False)
        scheduler.step()
        print(f"Ep{ep+1}: train={tr_acc:.3f} val={va_acc:.3f} loss={va_loss:.3f} ({time.time()-t0:.0f}s)")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            model.save_pretrained(OUTPUT_PATH)
            proc.save_pretrained(OUTPUT_PATH)
            print(f"  → 저장 (best val={best_val_acc:.3f})")

    # Phase 2: 전체 언프리즈 후 낮은 lr
    print("\n=== Phase 2: Full fine-tune (10 epochs) ===")
    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)

    for ep in range(10):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_dl, train=True)
        va_loss, va_acc = run_epoch(val_dl,   train=False)
        scheduler2.step()
        print(f"Ep{ep+1}: train={tr_acc:.3f} val={va_acc:.3f} loss={va_loss:.3f} ({time.time()-t0:.0f}s)")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            model.save_pretrained(OUTPUT_PATH)
            proc.save_pretrained(OUTPUT_PATH)
            print(f"  → 저장 (best val={best_val_acc:.3f})")

    print(f"\n✅ 파인튜닝 완료. Best val accuracy: {best_val_acc:.3f}")
    print(f"모델 저장: {OUTPUT_PATH}")
    return best_val_acc


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    best = train()
    print(f"\nFinal best val accuracy: {best:.3f}")
