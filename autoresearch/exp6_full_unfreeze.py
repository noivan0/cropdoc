"""
Idea 6: 전체 backbone unfreeze + very small LR (1e-6)
기반: exp3 모델 (val_acc_new=0.9609)
새 baseline = 0.9609, keep 기준: >= 0.9709
전략: 모든 파라미터를 매우 작은 LR로 fine-tune
backbone LR = 1e-6, classifier LR = 5e-6 (differential LR)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob, time
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck = torch.load('data/models/cropdoc_ext_exp3/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NEW_CLASSES = ck['new_classes']
NUM_NEW = len(NEW_CLASSES)
NUM_TOTAL = ck['num_classes']
id2label = ck['id2label']
print(f"exp3 baseline: {ck['val_acc_new']:.4f} → keep 기준: >=0.9709")

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device)

# 전체 unfreeze
for param in model.parameters():
    param.requires_grad = True

# Differential LR: backbone vs classifier
backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
print(f"backbone params: {sum(p.numel() for p in backbone_params):,}")
print(f"classifier params: {sum(p.numel() for p in classifier_params):,}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

TRAIN_TF = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(30), transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class NewClassDataset(Dataset):
    def __init__(self, tf):
        self.samples = []
        self.tf = tf
        folder_map = {
            "Corn Blight": "data/extended_datasets/Corn_Blight",
            "Corn Common Rust": "data/extended_datasets/Corn_Common_Rust",
            "Corn Gray Leaf Spot": "data/extended_datasets/Corn_Gray_Leaf_Spot",
            "Healthy Corn": "data/extended_datasets/Corn_Healthy",
            "Cassava Mosaic Virus": "data/extended_datasets/Cassava Mosaic Disease",
            "Cassava Brown Streak": "data/extended_datasets/Cassava Brown Streak Disease",
            "Cassava Green Mottle": "data/extended_datasets/Cassava Green Mottle",
            "Cassava Bacterial Blight": "data/extended_datasets/Cassava Bacterial Blight",
        }
        self.class_counts = {}
        for lbl in sorted(NEW_CLASSES):
            folder = folder_map.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            class_idx = NUM_OLD + sorted(NEW_CLASSES).index(lbl)
            imgs = imgs[:500]
            self.class_counts[class_idx] = len(imgs)
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
val_size = max(len(full_ds)//10, 1)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])
val_ds.dataset.tf = VAL_TF

sample_weights = []
for path, label in train_ds.dataset.samples:
    count = full_ds.class_counts.get(label, 100)
    sample_weights.append(1.0 / max(count, 1))
train_weights = [sample_weights[i] for i in train_ds.indices]
sampler = WeightedRandomSampler(train_weights, len(train_weights))

EPOCHS = 8
BATCH = 32
train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-6, 'weight_decay': 1e-4},
    {'params': classifier_params, 'lr': 5e-6, 'weight_decay': 1e-4},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

OUT_DIR = "data/models/cropdoc_ext_exp6"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0
t_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    t_ok = t_tot = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        new_out = out[:, NUM_OLD:]
        new_labels = labels - NUM_OLD
        loss = criterion(new_out, new_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = new_out.argmax(1) + NUM_OLD
        t_ok += (preds == labels).sum().item()
        t_tot += len(labels)

    model.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = out[:, NUM_OLD:].argmax(1) + NUM_OLD
            v_ok += (preds == labels).sum().item()
            v_tot += len(labels)

    val_acc = v_ok/v_tot if v_tot > 0 else 0
    elapsed = (time.time() - t_start) / 60
    scheduler.step()
    print(f"Ep{epoch+1}/{EPOCHS}: train={t_ok/t_tot:.4f} val={val_acc:.4f} [{elapsed:.1f}min]")
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'id2label': id2label, 'label2id': {v:k for k,v in id2label.items()},
            'architecture': 'efficientnet_v2_s',
            'num_classes': NUM_TOTAL, 'num_old': NUM_OLD,
            'num_new': NUM_NEW, 'new_classes': NEW_CLASSES,
            'val_acc_new': best_val, 'val_acc_old': ck.get('val_acc_old', 0.9991),
            'stage': 6,
        }, f"{OUT_DIR}/model.pt")
        print(f"  → 저장 (best_val={best_val:.4f})")

total_time = (time.time() - t_start) / 60
print(f"\nIdea6 완료: val_acc_new={best_val:.4f}, 소요={total_time:.1f}min")
print(f"keep 기준(>=0.9709): {'KEEP ✓' if best_val >= 0.9709 else 'DISCARD ✗'}")
