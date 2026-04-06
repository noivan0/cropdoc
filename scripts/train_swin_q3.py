"""
Task 1: Swin V2-S Q3 — 추가 15에포크, LR 더 낮게, FocalLoss(ls=0.05)
"""
import ssl, os, glob, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import swin_v2_s
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck = torch.load('data/models/cropdoc_ext_expQ/model.pt', map_location='cpu')
sorted_classes = ck['sorted_classes']
NUM_NEW = len(sorted_classes)
device = torch.device('cuda:0')

model = swin_v2_s(weights=None)
model.head = nn.Linear(model.head.in_features, NUM_NEW)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)
print(f"Swin 이전 best: {ck['val_acc_new']:.4f}, stage={ck.get('stage')}")

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

TRAIN_TF = transforms.Compose([
    transforms.Resize(260), transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.RandomRotation(20),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(260), transforms.CenterCrop(256),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

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
            return torch.zeros(3, 256, 256), label

full_ds = NewClassDataset(TRAIN_TF)
val_size = max(len(full_ds)//8, 1)
train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_size, val_size])
val_ds.dataset.tf = VAL_TF
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ls=0.05):
        super().__init__(); self.gamma=gamma; self.ls=ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        return ((1-torch.exp(-ce))**self.gamma * ce).mean()

criterion = FocalLoss(2.0, 0.05)
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'head' not in n], 'lr': 5e-7},
    {'params': model.head.parameters(), 'lr': 5e-5},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

best_val = ck['val_acc_new']
OUT_PATH = "data/models/cropdoc_ext_expQ/model.pt"

for epoch in range(15):
    model.train()
    t_ok = t_tot = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_ok += (out.argmax(1)==labels).sum().item(); t_tot += len(labels)

    model.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            v_ok += (model(imgs).argmax(1)==labels).sum().item(); v_tot += len(labels)

    val_acc = v_ok/v_tot if v_tot>0 else 0
    scheduler.step()
    print(f"Swin Ep{epoch+1:02d}: train={t_ok/t_tot:.4f} val={val_acc:.4f} best={best_val:.4f}", flush=True)
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': 'swin_v2_s',
            'sorted_classes': sorted_classes,
            'num_new': NUM_NEW,
            'val_acc_new': best_val,
            'stage': 'Q3',
        }, OUT_PATH)
        print(f"  ✅ 저장: {best_val:.4f}", flush=True)

print(f"Swin 최종: {best_val:.4f}")
