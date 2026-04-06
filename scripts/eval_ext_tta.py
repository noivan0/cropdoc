"""
Idea 6: TTA (Test-Time Augmentation) for extended model
- exp3 모델(95.65%)에 TTA 적용
- 4가지 변환 평균으로 예측 개선
- 학습 없음 — 추론 전략만 변경
"""

import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 최고 성능 모델
MODEL_PATH   = os.path.join(BASE_DIR, "data/models/cropdoc_ext_exp3/model.pt")
EXP1_MODEL   = os.path.join(BASE_DIR, "data/models/cropdoc_ext_exp1/model.pt")
EXTENDED_DIR = os.path.join(BASE_DIR, "data/extended_datasets")

SEED = 42
MIN_IMAGES = 50
BATCH_SIZE = 32
BASELINE = 0.9458

FOLDER_TO_LABEL = {
    "Coffee Leaf Rust": "Coffee Leaf Rust",
    "Coffee Leaf Miner": "Coffee Leaf Miner",
    "Coffee Phoma": "Coffee Phoma",
    "Rice Blast": "Rice Blast",
    "Rice Brown Spot": "Rice Brown Spot",
    "Rice Bacterial Blight": "Rice Bacterial Blight",
    "Rice Hispa": "Rice Hispa",
    "Rice Leaf Smut": "Rice Leaf Smut",
    "Wheat Leaf Rust": "Wheat Leaf Rust",
    "Wheat Stripe Rust": "Wheat Stripe Rust",
    "Wheat Stem Rust": "Wheat Stem Rust",
    "Wheat Loose Smut": "Wheat Loose Smut",
    "Mango Anthracnose": "Mango Anthracnose",
    "Mango Bacterial Canker": "Mango Bacterial Canker",
    "Mango Die Back": "Mango Die Back",
    "Mango Gall Midge": "Mango Gall Midge",
    "Mango Powdery Mildew": "Mango Powdery Mildew",
    "Mango Sooty Mould": "Mango Sooty Mould",
    "Cassava Bacterial Blight": "Cassava Bacterial Blight",
    "Cassava Mosaic Disease": "Cassava Mosaic Virus",
    "Cassava Brown Streak Disease": "Cassava Brown Streak",
    "Cassava Green Mottle": "Cassava Green Mottle",
    "Banana Black Sigatoka": "Banana Black Sigatoka",
    "Banana Yellow Sigatoka": "Banana Yellow Sigatoka",
    "Banana Panama Disease": "Banana Panama Disease",
    "Banana Moko Disease": "Banana Moko Disease",
    "Banana Bract Mosaic Virus": "Banana Bract Mosaic Virus",
    "Citrus Canker": "Citrus Canker",
    "Citrus Black Spot": "Citrus Black Spot",
    "Citrus Greening": "Citrus Greening",
    "Corn_Blight": "Corn Blight",
    "Corn_Common_Rust": "Corn Common Rust",
    "Corn_Gray_Leaf_Spot": "Corn Gray Leaf Spot",
    "Corn_Healthy": "Corn Healthy",
}

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 4가지 TTA 변환
EXT_TTA = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), NORM]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomVerticalFlip(p=1.0), T.ToTensor(), NORM]),
]

print("[1/4] 모델 로딩...")
if os.path.exists(MODEL_PATH):
    ck = torch.load(MODEL_PATH, map_location='cpu')
    print(f"  exp3: val_acc_new={ck['val_acc_new']:.4f}")
else:
    ck = torch.load(EXP1_MODEL, map_location='cpu')
    print(f"  exp1: val_acc_new={ck['val_acc_new']:.4f}")

NUM_OLD   = ck['num_old']
NUM_NEW   = ck['num_new']
NUM_TOTAL = ck['num_classes']
NEW_CLASSES = ck['new_classes']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_v2_s(weights=None)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, NUM_TOTAL))
model.load_state_dict(ck['model_state_dict'])
model = model.to(device).eval()
print(f"  device: {device}")

print("\n[2/4] Val 데이터 (raw paths)...")
valid_new = []
for folder, label in FOLDER_TO_LABEL.items():
    fp = os.path.join(EXTENDED_DIR, folder)
    if not os.path.isdir(fp): continue
    imgs = [f for f in os.listdir(fp) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    if len(imgs) >= MIN_IMAGES:
        valid_new.append((folder, label))

# 동일 val split 재생성
class AllDataset(Dataset):
    def __init__(self, valid_list, new_classes, num_old):
        self.samples = []
        for folder, label in valid_list:
            if label not in new_classes: continue
            idx = new_classes.index(label) + num_old
            fp = os.path.join(EXTENDED_DIR, folder)
            for f in os.listdir(fp):
                if f.lower().endswith(('.jpg','.jpeg','.png','.webp')):
                    self.samples.append((os.path.join(fp, f), idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

all_ds = AllDataset(valid_new, NEW_CLASSES, NUM_OLD)
val_size = len(all_ds) // 10
train_size = len(all_ds) - val_size
torch.manual_seed(SEED)
train_idx, val_idx = torch.utils.data.random_split(range(len(all_ds)), [train_size, val_size])
val_paths = [all_ds.samples[i] for i in val_idx.indices]
print(f"  Val samples: {len(val_paths)}장")

print("\n[3/4] TTA 평가...")

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return Image.new("RGB", (224, 224), 0)

# 배치 단위로 TTA 적용
correct = total = 0

# 이미지별로 4개 TTA 변환 logit 평균
tta_results = []  # (avg_logits, label)

batch_paths = []
batch_labels = []

for path, label in val_paths:
    batch_paths.append(path)
    batch_labels.append(label)
    if len(batch_paths) == BATCH_SIZE:
        # 각 TTA 변환으로 4번 추론
        avg_logits = None
        for tf in EXT_TTA:
            imgs = []
            for p in batch_paths:
                imgs.append(tf(load_image(p)))
            imgs_t = torch.stack(imgs).to(device)
            with torch.no_grad():
                logits = model(imgs_t)
            if avg_logits is None:
                avg_logits = logits.cpu()
            else:
                avg_logits += logits.cpu()
        avg_logits /= len(EXT_TTA)
        labels_t = torch.tensor(batch_labels)
        preds = avg_logits[:, NUM_OLD:].argmax(1) + NUM_OLD
        correct += (preds == labels_t).sum().item()
        total   += len(batch_labels)
        batch_paths = []
        batch_labels = []

# 나머지
if batch_paths:
    avg_logits = None
    for tf in EXT_TTA:
        imgs = [tf(load_image(p)) for p in batch_paths]
        imgs_t = torch.stack(imgs).to(device)
        with torch.no_grad():
            logits = model(imgs_t)
        if avg_logits is None:
            avg_logits = logits.cpu()
        else:
            avg_logits += logits.cpu()
    avg_logits /= len(EXT_TTA)
    labels_t = torch.tensor(batch_labels)
    preds = avg_logits[:, NUM_OLD:].argmax(1) + NUM_OLD
    correct += (preds == labels_t).sum().item()
    total   += len(batch_labels)

tta_acc = correct / total
THRESHOLD = BASELINE + 0.020

print(f"\n[4/4] TTA 결과:")
print(f"  TTA val_acc = {tta_acc:.4f}")
print(f"  baseline    = {BASELINE:.4f}")
print(f"  threshold   = {THRESHOLD:.4f}")
print(f"  improvement = {tta_acc - BASELINE:+.4f}")

if tta_acc >= THRESHOLD:
    print(f"🎉 KEEP! TTA로 threshold 달성")
    status = "keep"
else:
    print(f"❌ DISCARD: {tta_acc:.4f} < {THRESHOLD:.4f}")
    status = "discard"

print(f"\n[RESULT] idea5_tta  val_acc={tta_acc:.4f}  status={status}")
