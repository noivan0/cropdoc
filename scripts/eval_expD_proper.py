"""
expD 정확한 val_acc 측정 (expD_coffee.py와 동일한 split)
- val_size = len(full_ds)//8
- generator=torch.Generator().manual_seed(77)
- sorted(NEW_CLASSES), imgs[:600]
"""

import torch, torch.nn as nn, torch.nn.functional as F
import os, glob, numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NUM_CLASSES = ck['num_classes']
id2label = ck['id2label']
label2id = ck['label2id']
NEW_CLASSES = ck['new_classes']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
VAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])
TRAIN_TF = T.Compose([
    T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomRotation(30),
    T.RandomGrayscale(p=0.1), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(), NORM,
])

MULTI_SCALE_TFs = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(448), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(512), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), NORM]),
]

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
        self.tf = tf
        sorted_classes = sorted(NEW_CLASSES)
        for lbl in sorted_classes:
            folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            class_idx = NUM_OLD + sorted_classes.index(lbl)
            imgs = imgs[:600]
            for p in imgs:
                self.samples.append((p, class_idx))

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

print(f'Full: {len(full_ds)}, Train: {train_size}, Val: {val_size}')

# Val dataset에 VAL_TF 적용을 위한 wrapper
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

val_wrapped = ValWrapper(val_ds, VAL_TF)
val_loader = DataLoader(val_wrapped, batch_size=64, shuffle=False, num_workers=4)

# 1. 단일 스케일
print('\n[1] 단일 스케일 평가...')
correct_s, total = 0, 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct_s += (preds == labels).sum().item()
        total += labels.size(0)
val_acc_s = correct_s / total
print(f'  단일스케일: {correct_s}/{total} = {val_acc_s:.4f} (저장: {ck["val_acc_new"]:.4f})')

# 2. 멀티 스케일
print('\n[2] 멀티 스케일 TTA 평가...')
correct_ms = 0
for i, idx in enumerate(val_ds.indices):
    path, true_lid = full_ds.samples[idx]
    try:
        img = Image.open(path).convert('RGB')
        tensors = torch.stack([tf(img) for tf in MULTI_SCALE_TFs]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensors), -1).mean(0).cpu().numpy()
        pred = int(np.argmax(probs))
        if pred == true_lid:
            correct_ms += 1
    except:
        pass
    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(val_ds.indices)} ({correct_ms/(i+1):.4f})')

val_acc_ms = correct_ms / len(val_ds.indices)
print(f'  멀티스케일: {correct_ms}/{len(val_ds.indices)} = {val_acc_ms:.4f}')

print(f'\n최종 결과:')
print(f'  단일: {val_acc_s:.4f} | 멀티: {val_acc_ms:.4f} | Δ={val_acc_ms-val_acc_s:+.4f}')
print(f'  baseline expD: 0.9749')
print(f'  keep 기준: 0.9779')
keep = val_acc_ms >= 0.9779
print(f'  IdeaI 멀티스케일 {"✅ KEEP" if keep else "❌ DISCARD"} (val={val_acc_ms:.4f})')
