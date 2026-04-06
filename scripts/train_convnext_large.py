"""
Task 2: ConvNeXt-Large — 10에포크, DiffLR, FocalLoss(2.0)
GPU1 사용
"""
import ssl, os, glob, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# sorted_classes는 expQ 체크포인트에서 가져옴
ck_ref = torch.load('data/models/cropdoc_ext_expQ/model.pt', map_location='cpu')
sorted_classes = ck_ref['sorted_classes']
NUM_NEW = len(sorted_classes)
device = torch.device('cuda:0')

model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_NEW)
model = model.to(device)
print(f"ConvNeXt-Large 파라미터: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"NUM_NEW={NUM_NEW}, classes={sorted_classes[:3]}...")

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
    def __init__(self, gamma=2.0, ls=0.1):
        super().__init__(); self.gamma=gamma; self.ls=ls
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        return ((1-torch.exp(-ce))**self.gamma * ce).mean()

criterion = FocalLoss(2.0, 0.1)
# DiffLR: backbone 낮게, head 높게
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'classifier' not in n], 'lr': 5e-6},
    {'params': model.classifier.parameters(), 'lr': 5e-4},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

best_val = 0.0
OUT_PATH = "data/models/cropdoc_ext_convnext_large/model.pt"
os.makedirs("data/models/cropdoc_ext_convnext_large", exist_ok=True)

for epoch in range(10):
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
    print(f"CNX-L Ep{epoch+1:02d}: train={t_ok/t_tot:.4f} val={val_acc:.4f} best={best_val:.4f}", flush=True)
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': 'convnext_large',
            'sorted_classes': sorted_classes,
            'num_new': NUM_NEW,
            'val_acc_new': best_val,
            'stage': 'U1',
        }, OUT_PATH)
        print(f"  ✅ 저장: {best_val:.4f}", flush=True)

print(f"ConvNeXt-Large 최종: {best_val:.4f}")
