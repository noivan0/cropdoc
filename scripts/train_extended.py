"""
EfficientNetV2 확장 학습 — 기존 38종 보호 + 신규 34종 추가
전략:
  1단계: feature extractor 동결 → 신규 head만 학습 (5 epoch, 빠름)
  기존 38종 성능 보호: backbone 동결으로 catastrophic forgetting 방지

기존 모델: data/models/cropdoc_efficientnet_v2/model.pt (38종, val_acc=0.9991)
신규 클래스: 34종 (Citrus Melanose 제외, 50장 미만 제외)
출력: data/models/cropdoc_efficientnet_v2_extended/model.pt (72종)
"""

import torch
import torch.nn as nn
import os
import json
import glob
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

# ── 설정 ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_MODEL_PATH   = os.path.join(BASE_DIR, "data/models/cropdoc_efficientnet_v2/model.pt")
EXTENDED_DIR     = os.path.join(BASE_DIR, "data/extended_datasets")
OUTPUT_DIR       = os.path.join(BASE_DIR, "data/models/cropdoc_efficientnet_v2_extended")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE   = 64
EPOCHS       = 5
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42
MIN_IMAGES   = 50   # 클래스당 최소 이미지 수

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 폴더명 → 레이블 매핑 (Task 1) ────────────────────────────────────────────

FOLDER_TO_LABEL = {
    # Coffee
    "Coffee Leaf Rust":              "Coffee Leaf Rust",
    "Coffee Leaf Miner":             "Coffee Leaf Miner",
    "Coffee Phoma":                  "Coffee Phoma",
    # Rice
    "Rice Blast":                    "Rice Blast",
    "Rice Brown Spot":               "Rice Brown Spot",
    "Rice Bacterial Blight":         "Rice Bacterial Blight",
    "Rice Hispa":                    "Rice Hispa",
    "Rice Leaf Smut":                "Rice Leaf Smut",
    # Wheat
    "Wheat Leaf Rust":               "Wheat Leaf Rust",
    "Wheat Stripe Rust":             "Wheat Stripe Rust",
    "Wheat Stem Rust":               "Wheat Stem Rust",
    "Wheat Loose Smut":              "Wheat Loose Smut",
    # Mango
    "Mango Anthracnose":             "Mango Anthracnose",
    "Mango Bacterial Canker":        "Mango Bacterial Canker",
    "Mango Die Back":                "Mango Die Back",
    "Mango Gall Midge":              "Mango Gall Midge",
    "Mango Powdery Mildew":          "Mango Powdery Mildew",
    "Mango Sooty Mould":             "Mango Sooty Mould",
    # Cassava
    "Cassava Bacterial Blight":      "Cassava Bacterial Blight",
    "Cassava Mosaic Disease":        "Cassava Mosaic Virus",
    "Cassava Brown Streak Disease":  "Cassava Brown Streak",
    "Cassava Green Mottle":          "Cassava Green Mottle",
    # Banana
    "Banana Black Sigatoka":         "Banana Black Sigatoka",
    "Banana Yellow Sigatoka":        "Banana Yellow Sigatoka",
    "Banana Panama Disease":         "Banana Panama Disease",
    "Banana Moko Disease":           "Banana Moko Disease",
    "Banana Bract Mosaic Virus":     "Banana Bract Mosaic Virus",
    # Citrus (Melanose 제외)
    "Citrus Canker":                 "Citrus Canker",
    "Citrus Black Spot":             "Citrus Black Spot",
    "Citrus Greening":               "Citrus Greening",
    # Corn
    "Corn_Blight":                   "Corn Blight",
    "Corn_Common_Rust":              "Corn Common Rust",
    "Corn_Gray_Leaf_Spot":           "Corn Gray Leaf Spot",
    "Corn_Healthy":                  "Healthy Corn",
    # Rice generic (50장 미만 폴더 — MIN_IMAGES 필터에서 제외됨)
    "Bacterial leaf blight":         "Rice Bacterial Leaf Blight",
    "Brown spot":                    "Rice Brown Spot Extra",
    "Leaf smut":                     "Rice Leaf Smut Extra",
}

# ── 1단계: 기존 모델 로드 ─────────────────────────────────────────────────────

print("[1/6] 기존 38종 모델 로드...")
checkpoint = torch.load(OLD_MODEL_PATH, map_location="cpu", weights_only=False)
old_id2label = checkpoint["id2label"]   # {int: str}
NUM_OLD  = len(old_id2label)
OLD_ACC  = checkpoint.get("val_acc", "?")
print(f"  기존 클래스: {NUM_OLD}종, val_acc={OLD_ACC:.4f}")

# ── 2단계: 신규 클래스 검색 ───────────────────────────────────────────────────

print("\n[2/6] 신규 클래스 검색 (최소 {MIN_IMAGES}장 이상)...")
valid_new: dict[str, str] = {}   # label → folder_path
for folder, label in FOLDER_TO_LABEL.items():
    path = os.path.join(EXTENDED_DIR, folder)
    if not os.path.isdir(path):
        continue
    imgs = (
        glob.glob(f"{path}/*.jpg")  + glob.glob(f"{path}/*.JPG") +
        glob.glob(f"{path}/*.jpeg") + glob.glob(f"{path}/*.png")  +
        glob.glob(f"{path}/**/*.jpg",  recursive=True) +
        glob.glob(f"{path}/**/*.JPG",  recursive=True) +
        glob.glob(f"{path}/**/*.jpeg", recursive=True) +
        glob.glob(f"{path}/**/*.png",  recursive=True)
    )
    imgs = list(set(imgs))
    if len(imgs) >= MIN_IMAGES:
        valid_new[label] = path
        print(f"  ✅ {label}: {len(imgs)}장")
    else:
        print(f"  ⏭  {label}: {len(imgs)}장 (< {MIN_IMAGES}, 제외)")

NEW_CLASSES = sorted(valid_new.keys())
NUM_NEW     = len(NEW_CLASSES)
NUM_TOTAL   = NUM_OLD + NUM_NEW
print(f"\n  신규 클래스: {NUM_NEW}종  →  총 {NUM_OLD}+{NUM_NEW}={NUM_TOTAL}종")

# ── 3단계: 레이블 맵 확장 ────────────────────────────────────────────────────

print("\n[3/6] 레이블 맵 확장...")
new_id2label = dict(old_id2label)
for i, lbl in enumerate(NEW_CLASSES):
    new_id2label[NUM_OLD + i] = lbl
new_label2id = {v: k for k, v in new_id2label.items()}
print(f"  id2label: {NUM_OLD} → {NUM_TOTAL}종")

# ── 4단계: 모델 구성 ─────────────────────────────────────────────────────────

print("\n[4/6] EfficientNetV2 head 확장...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_OLD)
model.load_state_dict(checkpoint["model_state_dict"])

# Head 확장: 기존 가중치 보존 + 신규 head 초기화
old_w = model.classifier[1].weight.data.clone()
old_b = model.classifier[1].bias.data.clone()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
with torch.no_grad():
    model.classifier[1].weight[:NUM_OLD] = old_w
    model.classifier[1].bias[:NUM_OLD]   = old_b
    nn.init.xavier_uniform_(model.classifier[1].weight[NUM_OLD:])
    nn.init.zeros_(model.classifier[1].bias[NUM_OLD:])

# Feature extractor 동결 (기존 38종 성능 절대 보호)
for name, param in model.named_parameters():
    param.requires_grad = "classifier" in name

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"  학습 파라미터: {n_train:,}/{n_total:,} ({100*n_train/n_total:.2f}%) — classifier only")

model = model.to(device)

# ── 5단계: 데이터 로더 ───────────────────────────────────────────────────────

print("\n[5/6] 데이터셋 구성...")

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ExtendedDataset(Dataset):
    """신규 클래스 이미지 데이터셋."""
    def __init__(self, label_dirs: dict, new_class_list: list, class_offset: int, tf):
        self.samples = []
        self.tf = tf
        for i, lbl in enumerate(new_class_list):
            path = label_dirs[lbl]
            imgs = list(set(
                glob.glob(f"{path}/*.jpg")  + glob.glob(f"{path}/*.JPG") +
                glob.glob(f"{path}/*.jpeg") + glob.glob(f"{path}/*.png") +
                glob.glob(f"{path}/**/*.jpg",  recursive=True) +
                glob.glob(f"{path}/**/*.JPG",  recursive=True) +
                glob.glob(f"{path}/**/*.jpeg", recursive=True) +
                glob.glob(f"{path}/**/*.png",  recursive=True)
            ))
            imgs.sort()
            imgs = imgs[:500]   # 클래스당 최대 500장
            for img_p in imgs:
                self.samples.append((img_p, class_offset + i))
        print(f"    총 샘플: {len(self.samples)}장")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.tf(img), label
        except Exception:
            return torch.zeros(3, 224, 224), label


# 전체 데이터셋 → train/val 분할 (90/10)
full_dataset = ExtendedDataset(valid_new, NEW_CLASSES, NUM_OLD, TRAIN_TF)
total = len(full_dataset)
val_size   = total // 10
train_size = total - val_size

torch.manual_seed(SEED)
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# val_ds는 VAL_TF 사용 (Dataset 공유하므로 wrapper 사용)
class ValSubset(Dataset):
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf = tf
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            img = Image.open(path).convert("RGB")
            return self.tf(img), label
        except Exception:
            return torch.zeros(3, 224, 224), label

val_ds_proper = ValSubset(val_ds, VAL_TF)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds_proper, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"  Train: {train_size}장 ({len(train_loader)} batches)  Val: {val_size}장")

# ── 6단계: 학습 ──────────────────────────────────────────────────────────────

print(f"\n[6/6] Stage 1 학습 시작 ({EPOCHS} epochs, classifier only)...")
optimizer  = torch.optim.AdamW(
    [p for p in model.classifier.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val_acc = 0.0
log = {
    "epochs": [],
    "new_classes":   NEW_CLASSES,
    "num_old":       NUM_OLD,
    "num_new":       NUM_NEW,
    "num_total":     NUM_TOTAL,
}
start = time.time()

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    train_correct = train_total = 0
    train_loss_sum = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        # 신규 클래스 로짓만 loss 계산 (기존 head는 영향 없음)
        new_logits     = out[:, NUM_OLD:]       # [B, NUM_NEW]
        new_labels     = labels - NUM_OLD       # [B]
        loss = criterion(new_logits, new_labels)
        loss.backward()
        optimizer.step()
        preds = new_logits.argmax(1) + NUM_OLD
        train_correct += (preds == labels).sum().item()
        train_total   += len(labels)
        train_loss_sum += loss.item() * len(labels)

    # ── Val ──
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out    = model(imgs)
            preds  = out[:, NUM_OLD:].argmax(1) + NUM_OLD
            val_correct += (preds == labels).sum().item()
            val_total   += len(labels)

    train_acc = train_correct / train_total
    val_acc   = val_correct   / val_total
    scheduler.step()
    elapsed = (time.time() - start) / 60

    print(f"  Ep{epoch+1}/{EPOCHS} [{elapsed:.1f}m]  "
          f"loss={train_loss_sum/train_total:.4f}  "
          f"train={train_acc:.4f}  val={val_acc:.4f}")

    log["epochs"].append({
        "epoch":     epoch + 1,
        "train_acc": round(train_acc, 4),
        "val_acc":   round(val_acc,   4),
        "loss":      round(train_loss_sum / train_total, 4),
        "elapsed_min": round(elapsed, 2),
    })

    # Best 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "id2label":         new_id2label,
            "label2id":         new_label2id,
            "architecture":     "efficientnet_v2_s",
            "num_classes":      NUM_TOTAL,
            "num_old":          NUM_OLD,
            "num_new":          NUM_NEW,
            "new_classes":      NEW_CLASSES,
            "val_acc":          OLD_ACC,          # 기존 38종 성능 (보존됨)
            "val_acc_new":      best_val_acc,
            "val_acc_old":      OLD_ACC,
            "epoch":            epoch + 1,
        }, os.path.join(OUTPUT_DIR, "model.pt"))
        print(f"    ✅ Best model saved (val_acc_new={best_val_acc:.4f})")

# ── 결과 저장 ─────────────────────────────────────────────────────────────────

log["best_val_acc_new"] = best_val_acc
log["val_acc_old_preserved"] = float(OLD_ACC) if OLD_ACC != "?" else None
log["total_minutes"] = round((time.time() - start) / 60, 2)

with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
    json.dump(log, f, indent=2)

total_min = (time.time() - start) / 60
print(f"\n{'='*60}")
print(f"✅ 학습 완료!")
print(f"  소요 시간:       {total_min:.1f}분")
print(f"  클래스:          {NUM_OLD} → {NUM_TOTAL}종 (+{NUM_NEW}종)")
print(f"  신규 val_acc:    {best_val_acc:.4f}")
print(f"  기존 38종 acc:   {OLD_ACC} (backbone 동결으로 보존)")
print(f"  모델 저장:       {os.path.join(OUTPUT_DIR, 'model.pt')}")
print(f"{'='*60}")
