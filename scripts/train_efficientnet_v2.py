"""
CropDoc EfficientNetV2-S 학습 스크립트
=======================================
EfficientNetV2-S → New Plant Diseases Dataset (38클래스) 학습
목표: val_acc >= 99%

전략:
  - ImageNet 가중치로 시작 (transfer learning)
  - 전체 레이어 fine-tune (lr=1e-4)
  - 배치 64, 10에폭, Early Stopping (patience=3)
  - Label Smoothing 0.1, CosineAnnealingLR
  - 데이터 augmentation (ColorJitter, Flip)
  - GPU: CUDA (A6000)
"""

import os, sys, json, time, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
from transformers import MobileNetV2ForImageClassification
import random

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_BASE = (
    "/root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/"
    "versions/2/New Plant Diseases Dataset(Augmented)/"
    "New Plant Diseases Dataset(Augmented)"
)
TRAIN_DIR = os.path.join(DATASET_BASE, "train")
VALID_DIR = os.path.join(DATASET_BASE, "valid")
CNN_V2_PATH = os.path.join(PROJECT_ROOT, "data/models/cropdoc_cnn_v2")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/models/cropdoc_efficientnet_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CNN v2 에서 id2label 로드 (레이블 일관성 유지) ─────────────────────────────
print("CNN v2 id2label 로드 중...")
cnn_v2 = MobileNetV2ForImageClassification.from_pretrained(CNN_V2_PATH)
id2label = cnn_v2.config.id2label  # {0: "Apple Scab", ...}
label2id = {v: int(k) for k, v in id2label.items()}
NUM_CLASSES = len(id2label)
del cnn_v2  # 메모리 해제
print(f"클래스 수: {NUM_CLASSES}")

# ── 폴더명 → id2label 레이블 매핑 (train_cnn_v2.py 기준) ─────────────────────
FOLDER_TO_LABEL = {
    "Apple___Apple_scab":           "Apple Scab",
    "Apple___Black_rot":            "Apple with Black Rot",
    "Apple___Cedar_apple_rust":     "Cedar Apple Rust",
    "Apple___healthy":              "Healthy Apple",
    "Blueberry___healthy":          "Healthy Blueberry Plant",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry with Powdery Mildew",
    "Cherry_(including_sour)___healthy":        "Healthy Cherry Plant",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn (Maize) with Cercospora and Gray Leaf Spot",
    "Corn_(maize)___Common_rust_":  "Corn (Maize) with Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn (Maize) with Northern Leaf Blight",
    "Corn_(maize)___healthy":       "Healthy Corn (Maize) Plant",
    "Grape___Black_rot":            "Grape with Black Rot",
    "Grape___Esca_(Black_Measles)": "Grape with Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape with Isariopsis Leaf Spot",
    "Grape___healthy":              "Healthy Grape Plant",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange with Citrus Greening",
    "Peach___Bacterial_spot":       "Peach with Bacterial Spot",
    "Peach___healthy":              "Healthy Peach Plant",
    "Pepper,_bell___Bacterial_spot":"Bell Pepper with Bacterial Spot",
    "Pepper,_bell___healthy":       "Healthy Bell Pepper Plant",
    "Potato___Early_blight":        "Potato with Early Blight",
    "Potato___Late_blight":         "Potato with Late Blight",
    "Potato___healthy":             "Healthy Potato Plant",
    "Raspberry___healthy":          "Healthy Raspberry Plant",
    "Soybean___healthy":            "Healthy Soybean Plant",
    "Squash___Powdery_mildew":      "Squash with Powdery Mildew",
    "Strawberry___Leaf_scorch":     "Strawberry with Leaf Scorch",
    "Strawberry___healthy":         "Healthy Strawberry Plant",
    "Tomato___Bacterial_spot":      "Tomato with Bacterial Spot",
    "Tomato___Early_blight":        "Tomato with Early Blight",
    "Tomato___Late_blight":         "Tomato with Late Blight",
    "Tomato___Leaf_Mold":           "Tomato with Leaf Mold",
    "Tomato___Septoria_leaf_spot":  "Tomato with Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato with Spider Mites or Two-spotted Spider Mite",
    "Tomato___Target_Spot":         "Tomato with Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy":             "Healthy Tomato Plant",
}

# ── Transforms ────────────────────────────────────────────────────────────────
# ImageNet normalization (EfficientNetV2 표준)
IMAGENET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    IMAGENET_NORM,
])

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    IMAGENET_NORM,
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, label2id, transform=None):
        self.items = []
        self.transform = transform
        missing = []
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            cnn_label = FOLDER_TO_LABEL.get(folder_name)
            if cnn_label is None:
                missing.append(folder_name)
                continue
            class_id = label2id.get(cnn_label)
            if class_id is None:
                print(f"WARNING: 레이블 ID 없음: '{cnn_label}' (folder={folder_name})")
                continue
            count = 0
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.items.append((os.path.join(folder_path, fname), class_id))
                    count += 1
        if missing:
            print(f"WARNING: 매핑 없는 폴더 {len(missing)}개: {missing}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label_id = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"WARNING: 이미지 로드 실패 {path}: {e}")
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label_id


def main():
    random.seed(42)
    torch.manual_seed(42)

    # ── GPU 설정 ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        free0 = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
        if torch.cuda.device_count() > 1:
            free1 = torch.cuda.get_device_properties(1).total_memory - torch.cuda.memory_reserved(1)
            print(f"GPU0: {free0//1024**3}GB free, GPU1: {free1//1024**3}GB free")
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"학습 장치: {device}")

    # ── EfficientNetV2-S 모델 생성 ────────────────────────────────────────────
    print("EfficientNetV2-S (ImageNet1K_V1) 가중치 로드 중...")
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)
    print(f"EfficientNetV2-S 준비 완료 (in_features={in_features}, num_classes={NUM_CLASSES})")

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    print("\n데이터셋 로드 중...")
    t0 = time.time()
    train_ds = PlantDiseaseDataset(TRAIN_DIR, label2id, TRAIN_TF)
    val_ds   = PlantDiseaseDataset(VALID_DIR, label2id, VAL_TF)
    print(f"Train: {len(train_ds)}장, Val: {len(val_ds)}장 ({time.time()-t0:.1f}s)")

    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # ── 학습 설정 ─────────────────────────────────────────────────────────────
    MAX_EPOCHS = 10
    PATIENCE = 3
    LR = 1e-4

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_acc = 0.0
    patience_cnt = 0
    training_log = []

    print(f"\n학습 시작 (최대 {MAX_EPOCHS}에폭, patience={PATIENCE})")
    print("=" * 65)
    t_train = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_dl):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)  # EfficientNet: 직접 logits 반환
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * len(labels)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total   += len(labels)

            if (batch_idx + 1) % 200 == 0:
                elapsed = time.time() - t_ep
                pct = (batch_idx + 1) / len(train_dl) * 100
                print(f"  Ep{epoch} [{pct:.0f}%] loss={tr_loss/tr_total:.4f} "
                      f"acc={tr_correct/tr_total:.4f} ({elapsed:.0f}s)")

        tr_acc  = tr_correct / tr_total
        tr_loss = tr_loss / tr_total

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                va_loss    += loss.item() * len(labels)
                va_correct += (logits.argmax(1) == labels).sum().item()
                va_total   += len(labels)

        va_acc  = va_correct / va_total
        va_loss = va_loss / va_total

        elapsed_ep    = time.time() - t_ep
        elapsed_total = time.time() - t_train
        cur_lr = scheduler.get_last_lr()[0]

        print(f"\nEp{epoch}/{MAX_EPOCHS}: "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} | "
              f"lr={cur_lr:.2e} ({elapsed_ep:.0f}s / total {elapsed_total:.0f}s)")

        training_log.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc, 6),
            "val_loss":   round(va_loss, 6),
            "val_acc":    round(va_acc, 6),
            "lr":         cur_lr,
            "elapsed_s":  round(elapsed_ep, 1),
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_cnt = 0
            print(f"  ★ Best val_acc: {best_val_acc:.4f} — 모델 저장 중...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'id2label': id2label,
                'label2id': label2id,
                'architecture': 'efficientnet_v2_s',
                'num_classes': NUM_CLASSES,
                'val_acc': best_val_acc,
                'epoch': epoch,
            }, os.path.join(OUTPUT_DIR, "model.pt"))
            print(f"  ★ 저장 완료: {OUTPUT_DIR}/model.pt")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{PATIENCE})")
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        scheduler.step()

    # ── 학습 로그 저장 ─────────────────────────────────────────────────────
    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "total_epochs": len(training_log),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "num_classes": NUM_CLASSES,
            "id2label": {str(k): v for k, v in id2label.items()},
            "epochs": training_log,
        }, f, indent=2)
    print(f"\n학습 로그 저장: {log_path}")

    print(f"\n{'='*65}")
    print(f"✅ 학습 완료!")
    print(f"   Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   총 학습 시간: {(time.time()-t_train)/60:.1f}분")
    print(f"   모델 저장: {OUTPUT_DIR}/model.pt")

    return best_val_acc


if __name__ == "__main__":
    best = main()
    sys.exit(0 if best >= 0.99 else 1)
