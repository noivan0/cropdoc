"""
Idea T: ConvNeXt-Base for 34 new classes
ConvNeXt는 ViT의 아이디어를 CNN에 적용 — 고해상도 이미지 처리에 강점
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck_base = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck_base['num_old']
NEW_CLASSES = ck_base['new_classes']
NUM_NEW = len(NEW_CLASSES)
NUM_TOTAL = ck_base['num_classes']
id2label = ck_base['id2label']
sorted_classes = sorted(NEW_CLASSES)
device = torch.device('cuda')

print(f"NUM_NEW={NUM_NEW}, NUM_OLD={NUM_OLD}, NUM_TOTAL={NUM_TOTAL}")
print(f"sorted_classes[:5]: {sorted_classes[:5]}")

# ConvNeXt-Base — ImageNet-1K pretrained
model_cnx = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
# ConvNeXt-Base classifier: Sequential → 마지막 Linear 교체
in_features = model_cnx.classifier[2].in_features  # 1024
model_cnx.classifier[2] = nn.Linear(in_features, NUM_NEW)
model_cnx = model_cnx.to(device)
print(f"ConvNeXt-Base in_features={in_features}, out={NUM_NEW}")

TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
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
        for i, lbl in enumerate(sorted_classes):
            folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            for p in imgs[:600]:
                self.samples.append((p, i))
        print(f"총 {len(self.samples)}장")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except:
            return torch.zeros(3, 224, 224), label

full_ds = NewClassDataset(TRAIN_TF)
val_size = max(len(full_ds)//8, 1)
train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_size, val_size])
val_ds.dataset.tf = VAL_TF
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ls=0.1):
        super().__init__()
        self.gamma = gamma; self.ls = ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

criterion = FocalLoss(2.0, 0.1)
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model_cnx.named_parameters() if 'classifier.2' not in n], 'lr': 5e-6},
    {'params': model_cnx.classifier[2].parameters(), 'lr': 5e-4},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

OUT_DIR = "data/models/cropdoc_ext_expT"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0

for epoch in range(10):
    model_cnx.train()
    t_ok = t_tot = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model_cnx(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_cnx.parameters(), 1.0)
        optimizer.step()
        t_ok += (out.argmax(1) == labels).sum().item()
        t_tot += len(labels)

    model_cnx.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            v_ok += (model_cnx(imgs).argmax(1) == labels).sum().item()
            v_tot += len(labels)

    val_acc = v_ok/v_tot if v_tot>0 else 0
    scheduler.step()
    print(f"ConvNeXt Ep{epoch+1}: train={t_ok/t_tot:.4f} val={val_acc:.4f} best={best_val:.4f}", flush=True)
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model_cnx.state_dict(),
            'architecture': 'convnext_base',
            'sorted_classes': sorted_classes,
            'num_new': NUM_NEW,
            'val_acc_new': best_val,
            'stage': 'T',
        }, f"{OUT_DIR}/model.pt")

print(f"ConvNeXt 완료: best_val={best_val:.4f}")
