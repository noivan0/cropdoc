"""
expQ: Swin Transformer V2-S — 신규 34종 분류
ImageNet pretrain, DiffLR (backbone 1e-5 / head 1e-3), FocalLoss(2.0, 0.1), 10ep
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import swin_v2_s, Swin_V2_S_Weights
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck_base = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck_base['num_old']
NEW_CLASSES = ck_base['new_classes']
NUM_NEW = len(NEW_CLASSES)
id2label = ck_base['id2label']
device = torch.device('cuda')

print(f"NUM_OLD={NUM_OLD}, NUM_NEW={NUM_NEW}, device={device}")

# Swin V2 Small — ImageNet pretrained
model_swin = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
in_features = model_swin.head.in_features  # 768
model_swin.head = nn.Linear(in_features, NUM_NEW)
model_swin = model_swin.to(device)
print(f"Swin V2-S head: in_features={in_features}, out={NUM_NEW}")

# Swin은 256×256 입력에 최적화됨
TRAIN_TF = transforms.Compose([
    transforms.Resize(260),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(260),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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
        self.sorted_classes = sorted(NEW_CLASSES)
        for i, lbl in enumerate(self.sorted_classes):
            folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            cnt = len(imgs[:600])
            if cnt == 0:
                print(f"  [WARN] {lbl}: 0장 (경로={folder})")
            else:
                print(f"  {lbl}: {cnt}장")
            for p in imgs[:600]:
                self.samples.append((p, i))
        print(f"총 {len(self.samples)}장")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except Exception as e:
            return torch.zeros(3, 256, 256), label

full_ds = NewClassDataset(TRAIN_TF)
val_size = max(len(full_ds) // 8, 1)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                 generator=torch.Generator().manual_seed(42))
val_ds.dataset.tf = VAL_TF

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
print(f"Train: {train_size}, Val: {val_size}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ls=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = ls

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

criterion = FocalLoss(2.0, 0.1)

# Differential LR: backbone 낮게, head 높게
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model_swin.named_parameters() if 'head' not in n], 'lr': 1e-5},
    {'params': model_swin.head.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

OUT_DIR = "data/models/cropdoc_ext_expQ"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
sorted_classes = sorted(NEW_CLASSES)

for epoch in range(10):
    model_swin.train()
    t_ok = t_tot = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model_swin(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_swin.parameters(), 1.0)
        optimizer.step()
        t_ok += (out.argmax(1) == labels).sum().item()
        t_tot += len(labels)

    model_swin.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            v_ok += (model_swin(imgs).argmax(1) == labels).sum().item()
            v_tot += len(labels)

    val_acc = v_ok / v_tot if v_tot > 0 else 0
    scheduler.step()
    print(f"Swin Ep{epoch+1:02d}: train={t_ok/t_tot:.4f} val={val_acc:.4f} best={best_val:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model_swin.state_dict(),
            'architecture': 'swin_v2_s',
            'sorted_classes': sorted_classes,
            'num_new': NUM_NEW,
            'val_acc_new': best_val,
            'stage': 'Q',
        }, f"{OUT_DIR}/model.pt")
        print(f"  → 저장 완료 (best={best_val:.4f})")

print(f"\nSwin 완료: best_val={best_val:.4f}")
print(f"RESULT:{best_val:.6f}")
