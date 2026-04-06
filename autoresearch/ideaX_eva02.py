"""
Idea X: EVA-02-Large 448px — 신규 34종 분류
305M 파라미터, FocalLoss, 차등 LR (backbone 1e-6 / head 1e-4), 10ep
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os, glob, torch, torch.nn as nn, torch.nn.functional as F
os.environ['CURL_CA_BUNDLE'] = ''

import timm
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck_base = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NEW_CLASSES = ck_base['new_classes']
NUM_NEW = len(NEW_CLASSES)
sorted_classes = sorted(NEW_CLASSES)
device = torch.device('cuda')

print(f"신규 클래스 수: {NUM_NEW}")
print(f"클래스 목록: {sorted_classes[:5]}...")

# ── EVA-02-Large 로드 ────────────────────────────────────────────────────────
print("EVA-02-Large 로드 중...")
arch_name = None
try:
    model = timm.create_model(
        'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        pretrained=True,
        num_classes=NUM_NEW
    )
    arch_name = 'EVA02Large448'
    print("✅ EVA-02-Large 448 로드 성공!")
except Exception as e:
    print(f"1차 실패: {e}")
    try:
        model = timm.create_model(
            'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
            pretrained=True,
            num_classes=NUM_NEW
        )
        arch_name = 'EVA02Base448'
        print("✅ EVA-02-Base 448 사용")
    except Exception as e2:
        print(f"2차 실패: {e2}")
        model = timm.create_model(
            'beit_large_patch16_224.in22k_ft_in22k_in1k',
            pretrained=True,
            num_classes=NUM_NEW
        )
        arch_name = 'BEiTLarge224'
        print("✅ BEiT-Large 224 사용")

model = model.to(device)
params = sum(p.numel() for p in model.parameters())
print(f"파라미터: {params/1e6:.1f}M")

# ── 입력 크기 확인 ────────────────────────────────────────────────────────────
data_config = timm.data.resolve_model_data_config(model)
input_size = data_config.get('input_size', (3, 448, 448))[-1]
print(f"입력 크기: {input_size}px")

# ── Transforms ───────────────────────────────────────────────────────────────
resize_size = int(input_size * 1.15)
TRAIN_TF = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.RandomCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

FOLDER_MAP = {
    "Corn Blight":              "data/extended_datasets/Corn_Blight",
    "Corn Common Rust":         "data/extended_datasets/Corn_Common_Rust",
    "Corn Gray Leaf Spot":      "data/extended_datasets/Corn_Gray_Leaf_Spot",
    "Healthy Corn":             "data/extended_datasets/Corn_Healthy",
    "Cassava Mosaic Virus":     "data/extended_datasets/Cassava Mosaic Disease",
    "Cassava Brown Streak":     "data/extended_datasets/Cassava Brown Streak Disease",
    "Cassava Green Mottle":     "data/extended_datasets/Cassava Green Mottle",
    "Cassava Bacterial Blight": "data/extended_datasets/Cassava Bacterial Blight",
}

class NewClassDataset(Dataset):
    def __init__(self, tf):
        self.samples = []
        self.tf = tf
        for i, lbl in enumerate(sorted_classes):
            folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (
                glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                glob.glob(f"{folder}/**/*.png", recursive=True)
            )
            for p in imgs[:500]:
                self.samples.append((p, i))
        print(f"총 {len(self.samples)}장")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except:
            return torch.zeros(3, input_size, input_size), label

full_ds = NewClassDataset(TRAIN_TF)
val_size = max(len(full_ds) // 8, 1)
train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
val_ds.dataset.tf = VAL_TF

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4)

print(f"Train: {len(train_ds)}장, Val: {len(val_ds)}장")

# ── FocalLoss ────────────────────────────────────────────────────────────────
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

# ── 차등 LR (head vs backbone) ───────────────────────────────────────────────
head_params = []
backbone_params = []
for n, p in model.named_parameters():
    if any(x in n for x in ['head', 'classifier', 'fc']):
        head_params.append(p)
    else:
        backbone_params.append(p)

print(f"Head params: {sum(p.numel() for p in head_params)/1e3:.1f}K")
print(f"Backbone params: {sum(p.numel() for p in backbone_params)/1e6:.1f}M")

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-6},
    {'params': head_params,     'lr': 1e-4},
], weight_decay=0.05)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

OUT_DIR = "data/models/cropdoc_ext_eva02"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0

print(f"\n=== 학습 시작 (arch={arch_name}, input={input_size}px) ===")
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

    print(f"EVA02 Ep{epoch+1}/10: train={t_ok/t_tot:.4f} val={val_acc:.4f} best={best_val:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict':  model.state_dict(),
            'architecture':      arch_name,
            'sorted_classes':    sorted_classes,
            'num_new':           NUM_NEW,
            'input_size':        input_size,
            'val_acc_new':       best_val,
            'stage':             'EVA02',
        }, f"{OUT_DIR}/model.pt")
        print(f"  💾 저장됨 (val_acc={best_val:.4f})")

print(f"\n✅ EVA-02 학습 완료: best_val={best_val:.4f}")
print(f"RESULT:{best_val:.4f}")  # 파싱용 마커
