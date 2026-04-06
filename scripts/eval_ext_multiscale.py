"""
eval_ext_multiscale.py
- expD val set을 동일한 split으로 재구성
- 단일스케일 vs 멀티스케일 비교
"""

import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os, glob, numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTENDED_DIR = os.path.join(BASE_DIR, "data/extended_datasets")

SEED = 42
MIN_IMAGES = 50
BATCH_SIZE = 64

torch.manual_seed(SEED)
np.random.seed(SEED)

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

ck = torch.load(os.path.join(BASE_DIR, 'data/models/cropdoc_ext_expD/model.pt'), map_location='cpu')
NUM_OLD = ck['num_old']
NUM_CLASSES = ck['num_classes']
id2label = ck['id2label']
label2id = ck['label2id']
NEW_CLASSES = ck['new_classes']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = efficientnet_v2_s(weights=None)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, NUM_CLASSES))
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

VAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])

MULTI_SCALE_TFs = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(448), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(512), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), NORM]),
]

# expD의 valid_new 구성 동일하게
valid_new = []
for folder, label in FOLDER_TO_LABEL.items():
    fp = os.path.join(EXTENDED_DIR, folder)
    if not os.path.isdir(fp):
        continue
    imgs = [f for f in os.listdir(fp) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    if len(imgs) >= MIN_IMAGES:
        valid_new.append((folder, label))

class ExtendedDataset(Dataset):
    def __init__(self, valid_list, new_classes, num_old, tf):
        self.tf = tf
        self.samples = []
        for folder, label in valid_list:
            if label not in new_classes:
                continue
            idx = new_classes.index(label) + num_old
            fp = os.path.join(EXTENDED_DIR, folder)
            for f in os.listdir(fp):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(fp, f), idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.tf(Image.open(path).convert("RGB")), label
        except:
            return torch.zeros(3, 224, 224), label

class ValSubset(Dataset):
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf = tf
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            return self.tf(Image.open(path).convert("RGB")), label
        except:
            return torch.zeros(3, 224, 224), label

# expD의 val split (train_ext_deep.py or train_ext_polish2.py 확인 필요)
# expD: Corn_Healthy는 "Corn Healthy"
full_ds = ExtendedDataset(valid_new, NEW_CLASSES, NUM_OLD, VAL_TF)
val_size = len(full_ds) // 10
train_size = len(full_ds) - val_size
torch.manual_seed(SEED)
_, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
val_ds_proper = ValSubset(val_ds, VAL_TF)
val_ds_ms = ValSubset(val_ds, VAL_TF)  # 멀티스케일은 개별 처리

print(f'Full dataset: {len(full_ds)}, Val: {val_size}, Train: {train_size}')

# 단일 스케일 evaluation (배치)
val_loader = DataLoader(val_ds_proper, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
correct_single = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        preds = out.argmax(dim=1)
        correct_single += (preds == labels).sum().item()
        total += labels.size(0)

val_acc_single = correct_single / total
print(f'\n[단일 스케일] val_acc_new: {correct_single}/{total} = {val_acc_single:.4f}')
print(f'  (저장된 expD val_acc_new: {ck["val_acc_new"]:.4f})')

# 멀티 스케일 evaluation (개별 처리)
print('\n[멀티 스케일 TTA] 평가 중...')
correct_ms = 0
val_indices = val_ds.indices  # val split의 실제 인덱스

for i, idx in enumerate(val_indices):
    path, true_lid = full_ds.samples[idx]
    try:
        img = Image.open(path).convert('RGB')
        tensors = torch.stack([tf(img) for tf in MULTI_SCALE_TFs]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensors), -1).mean(0).cpu().numpy()
        pred = int(np.argmax(probs))
        if pred == true_lid:
            correct_ms += 1
    except Exception as e:
        pass

    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(val_indices)} done... ({correct_ms/(i+1):.4f})')

val_acc_ms = correct_ms / len(val_indices)
print(f'\n[멀티 스케일] val_acc_new: {correct_ms}/{len(val_indices)} = {val_acc_ms:.4f}')
print(f'\n결론:')
print(f'  단일: {val_acc_single:.4f} | 멀티: {val_acc_ms:.4f} | Δ={val_acc_ms-val_acc_single:+.4f}')
print(f'  Keep 기준: 0.9779')
keep = val_acc_ms >= 0.9779
print(f'  멀티스케일 {"✅ KEEP" if keep else "❌ DISCARD"}')
