"""
CropDoc CNN v2 - 대규모 재학습 스크립트
=======================================
MobileNetV2 → New Plant Diseases Dataset (175,767장, 38클래스) 전체 재학습

전략:
  - 기존 PlantVillage 학습 가중치 시작 (transfer learning)
  - 전체 레이어 fine-tune (lr=1e-4)
  - 배치 크기 64, 최대 15에폭, Early Stopping
  - GPU: CUDA (A6000)
"""

import os, sys, json, time, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import MobileNetV2ImageProcessor, MobileNetV2ForImageClassification
import random

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
CNN_ORIGINAL_PATH = (
    "/root/.cache/huggingface/hub/cropdoc_cnn/"
    "models--linkanjarad--mobilenet_v2_1.0_224-plant-disease-identification/"
    "snapshots/c1861579a670fb6232258805b801cd4137cb7176"
)
DATASET_BASE = (
    "/root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/"
    "versions/2/New Plant Diseases Dataset(Augmented)/"
    "New Plant Diseases Dataset(Augmented)"
)
TRAIN_DIR = os.path.join(DATASET_BASE, "train")
VALID_DIR = os.path.join(DATASET_BASE, "valid")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/models/cropdoc_cnn_v2")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── 폴더명 → CNN 레이블 매핑 ─────────────────────────────────────────────────
FOLDER_TO_CNN_LABEL = {
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
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato with Spider Mites",
    "Tomato___Target_Spot":         "Tomato with Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy":             "Healthy Tomato Plant",
}

# ── transforms ────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, label2id, transform=None):
        self.items = []  # [(path, class_id), ...]
        self.transform = transform

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            cnn_label = FOLDER_TO_CNN_LABEL.get(folder_name)
            if cnn_label is None:
                print(f"WARNING: 폴더 매핑 없음: {folder_name}")
                continue
            class_id = label2id.get(cnn_label)
            if class_id is None:
                print(f"WARNING: 레이블 ID 없음: {cnn_label}")
                continue
            count = 0
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                    self.items.append((os.path.join(folder_path, fname), class_id))
                    count += 1
            # print(f"  {folder_name}: {count}장 → class {class_id}")

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GPU 0 우선 사용 (GPU 1은 Gemma가 점유 가능)
    if torch.cuda.is_available():
        # GPU 0 메모리 확인
        free0 = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
        free1 = torch.cuda.get_device_properties(1).total_memory - torch.cuda.memory_reserved(1)
        print(f"GPU0 free: {free0//1024**3}GB, GPU1 free: {free1//1024**3}GB")
        # GPU 0 사용 (더 많은 메모리)
        device = "cuda:0"
    print(f"학습 장치: {device}")

    # ── 모델 로드 (기존 가중치 시작) ─────────────────────────────────────────
    print("기존 CNN 모델 로드 중...")
    proc = MobileNetV2ImageProcessor.from_pretrained(CNN_ORIGINAL_PATH, local_files_only=True)
    model = MobileNetV2ForImageClassification.from_pretrained(
        CNN_ORIGINAL_PATH, local_files_only=True)

    # id2label 확인
    id2label = model.config.id2label
    label2id = {v: int(k) for k, v in id2label.items()}
    num_classes = len(id2label)
    print(f"클래스 수: {num_classes}")

    # Spider Mites 레이블 확인 (원본 vs 데이터셋 차이)
    # 원본: "Tomato with Spider Mites or Two-spotted Spider Mite"
    # 데이터셋 폴더 → "Tomato with Spider Mites"
    # label2id에서 실제 키 확인
    spider_key = None
    for k in label2id.keys():
        if "Spider" in k or "spider" in k:
            spider_key = k
            print(f"Spider 레이블: '{k}' → id={label2id[k]}")
    # FOLDER_TO_CNN_LABEL의 Spider Mites를 실제 모델 레이블로 수정
    if spider_key and spider_key != "Tomato with Spider Mites":
        FOLDER_TO_CNN_LABEL["Tomato___Spider_mites Two-spotted_spider_mite"] = spider_key
        print(f"Spider Mites 레이블 수정: → '{spider_key}'")

    model = model.to(device)
    model.train()

    # ── 데이터셋 준비 ─────────────────────────────────────────────────────────
    print("\n데이터셋 로드 중...")
    t_ds_start = time.time()
    train_ds = PlantDiseaseDataset(TRAIN_DIR, label2id, TRAIN_TF)
    val_ds   = PlantDiseaseDataset(VALID_DIR, label2id, VAL_TF)
    print(f"Train: {len(train_ds)}장, Val: {len(val_ds)}장 ({time.time()-t_ds_start:.1f}s)")

    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=2
    )
    val_dl = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ── 학습 설정 ─────────────────────────────────────────────────────────────
    MAX_EPOCHS = 15
    PATIENCE = 3  # early stopping
    LR = 1e-4

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_acc = 0.0
    patience_cnt = 0
    training_log = []

    print(f"\n학습 시작 (최대 {MAX_EPOCHS}에폭, Early Stop patience={PATIENCE})")
    print("=" * 60)
    t_train_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_dl):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += len(labels)

            # 진행 상황 출력 (200배치마다)
            if (batch_idx + 1) % 200 == 0:
                elapsed = time.time() - t_ep_start
                pct = (batch_idx + 1) / len(train_dl) * 100
                print(f"  Ep{epoch} [{pct:.0f}%] loss={train_loss/train_total:.4f} "
                      f"acc={train_correct/train_total:.4f} ({elapsed:.0f}s)")

        train_acc  = train_correct / train_total
        train_loss = train_loss / train_total

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs).logits
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += len(labels)

        val_acc  = val_correct / val_total
        val_loss = val_loss / val_total

        elapsed_ep = time.time() - t_ep_start
        elapsed_total = time.time() - t_train_start
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEp{epoch}/{MAX_EPOCHS}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={current_lr:.2e} ({elapsed_ep:.0f}s / total {elapsed_total:.0f}s)")

        # ── 로그 기록 ──────────────────────────────────────────────────────
        training_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 6),
            "lr":         current_lr,
            "elapsed_s":  round(elapsed_ep, 1),
        })

        # ── 최고 모델 저장 ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            print(f"  ★ Best val acc: {best_val_acc:.4f} → 모델 저장 중...")
            model.save_pretrained(OUTPUT_PATH)
            proc.save_pretrained(OUTPUT_PATH)
            print(f"  ★ 저장 완료: {OUTPUT_PATH}")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{PATIENCE})")
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}!")
                break

        scheduler.step()

    # ── 학습 로그 저장 ─────────────────────────────────────────────────────────
    log_path = os.path.join(OUTPUT_PATH, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "total_epochs": len(training_log),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "num_classes": num_classes,
            "epochs": training_log,
        }, f, indent=2)
    print(f"\n학습 로그 저장: {log_path}")

    print(f"\n{'='*60}")
    print(f"✅ 학습 완료!")
    print(f"   Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   총 학습 시간: {(time.time()-t_train_start)/60:.1f}분")
    print(f"   모델 저장 위치: {OUTPUT_PATH}")

    return best_val_acc


if __name__ == "__main__":
    best = main()
    sys.exit(0 if best >= 0.85 else 1)
