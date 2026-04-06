"""
Task 4: DINOv2-Large 34종 계속 학습
현재 val_acc=95.50% → 목표: 98%+
"""
import ssl, os, glob, torch, torch.nn as nn, torch.nn.functional as F
import timm
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import random

ssl._create_default_https_context = ssl._create_unverified_context
os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

device = torch.device('cuda')
print(f"Device: {device}")

# 기존 DINOv2-Large 체크포인트에서 이어서 학습
ck = torch.load('data/models/cropdoc_ext_dinov2/model.pt', map_location='cpu')
sorted_classes = ck['sorted_classes']
NUM_NEW = len(sorted_classes)
print(f"DINOv2-Large 이전 best: {ck['val_acc_new']:.4f}, {NUM_NEW}클래스")

model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, 
                          num_classes=NUM_NEW, img_size=518)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

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
    transforms.Resize(560),
    transforms.RandomCrop(518),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(560),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class NewClassDataset(Dataset):
    def __init__(self, samples, tf):
        self.samples = samples
        self.tf = tf
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except:
            return torch.zeros(3, 518, 518), label

# 전체 샘플 수집
all_samples = []
for i, lbl in enumerate(sorted_classes):
    folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
    imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
            glob.glob(f"{folder}/**/*.JPG", recursive=True) +
            glob.glob(f"{folder}/**/*.png", recursive=True) +
            glob.glob(f"{folder}/**/*.jpeg", recursive=True))
    if not imgs:
        print(f"  ⚠️ {lbl}: 이미지 없음 (폴더: {folder})")
    for p in imgs[:500]:
        all_samples.append((p, i))

print(f"총 {len(all_samples)}장")
random.shuffle(all_samples)

val_size = max(len(all_samples) // 8, 1)
train_size = len(all_samples) - val_size
train_samples = all_samples[:train_size]
val_samples = all_samples[train_size:]

train_ds = NewClassDataset(train_samples, TRAIN_TF)
val_ds = NewClassDataset(val_samples, VAL_TF)

print(f"train: {len(train_ds)}, val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, 
                          drop_last=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ls=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = ls
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        return ((1 - torch.exp(-ce))**self.gamma * ce).mean()

criterion = FocalLoss(2.0, 0.1)

# LR 낮춰서 안정적 fine-tuning
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'head' not in n], 'lr': 5e-7},
    {'params': model.head.parameters(), 'lr': 5e-5},
], weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

best_val = ck['val_acc_new']
OUT_DIR = "data/models/cropdoc_ext_dinov2"
print(f"\n목표: val_acc >= 0.9800 (현재 {best_val:.4f})")

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
        t_ok += (out.argmax(1) == labels).sum().item()
        t_tot += len(labels)
    
    model.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            v_ok += (model(imgs).argmax(1) == labels).sum().item()
            v_tot += len(labels)
    
    val_acc = v_ok / v_tot if v_tot > 0 else 0
    scheduler.step()
    
    train_acc = t_ok / t_tot if t_tot > 0 else 0
    print(f"DINOv2-L Ep{epoch+1:02d}: train={train_acc:.4f} val={val_acc:.4f} best={best_val:.4f}")
    
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': 'vit_large_patch14_reg4_dinov2',
            'sorted_classes': sorted_classes,
            'id2label_new': ck.get('id2label_new', {}),
            'num_new': NUM_NEW,
            'input_size': 518,
            'val_acc_new': best_val,
            'stage': 'DINOv2-L-continued',
        }, f"{OUT_DIR}/model.pt")
        print(f"  ✅ 저장: best={best_val:.4f}")

print(f"\nDINOv2-Large 34종 최종: {best_val:.4f}")
if best_val >= 0.9800:
    print("🎉 목표 달성! (>=98.00%)")
else:
    print(f"⚠️ 목표 미달성 ({best_val:.4f} < 0.9800)")
