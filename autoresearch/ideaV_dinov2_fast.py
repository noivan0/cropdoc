"""
ideaV: DINOv2-Large 518px + CutMix + FocalLoss + Differential LR
최적화 버전: AMP + 이미지 200장/클래스 제한 + 3 epochs
- vit_large_patch14_reg4_dinov2.lvd142m (304M params)
- AMP(FP16) → 속도 ~2x, 메모리 절반
- 200장/class × 34 = 6800장 → epoch당 ~7분
- 3 epochs × ~7분 = ~21분 (60분 내 완료)
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import time

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# 기존 expD에서 클래스 정보 로드
ck_base = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck_base['num_old']
NEW_CLASSES = ck_base['new_classes']
NUM_NEW = len(NEW_CLASSES)
sorted_classes = sorted(NEW_CLASSES)
id2label_new = {i: lbl for i, lbl in enumerate(sorted_classes)}
print(f"클래스 수: {NUM_NEW}")

device = torch.device('cuda:0')

# DINOv2-Large 로드
print("DINOv2-Large 로드 중...")
model = timm.create_model(
    'vit_large_patch14_reg4_dinov2.lvd142m',
    pretrained=True,
    num_classes=NUM_NEW
)
model = model.to(device)
params = sum(p.numel() for p in model.parameters())
print(f"파라미터: {params/1e6:.1f}M")

# 518×518 입력 transforms
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

MAX_PER_CLASS = 300  # 500 → 300으로 제한

class NewClassDataset(Dataset):
    def __init__(self, tf):
        self.samples = []
        self.tf = tf
        for i, lbl in enumerate(sorted_classes):
            folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
            imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                    glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                    glob.glob(f"{folder}/**/*.png", recursive=True))
            for p in imgs[:MAX_PER_CLASS]:
                self.samples.append((p, i))
        print(f"총 {len(self.samples)}장 로드 ({MAX_PER_CLASS}장/클래스 제한)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return self.tf(Image.open(path).convert('RGB')), label
        except Exception:
            return torch.zeros(3, 518, 518), label

full_ds = NewClassDataset(TRAIN_TF)
val_size = max(len(full_ds)//8, 1)
train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_size, val_size])
val_ds.dataset.tf = VAL_TF

BATCH = 8
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Steps/epoch: {len(train_loader)}")

# CutMix
def cutmix(images, labels, alpha=1.0):
    lam = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha)).sample()
    rand_idx = torch.randperm(images.size(0))
    mixed_images = images.clone()
    W, H = images.size(3), images.size(2)
    cut_w = int(W * (1 - lam).sqrt())
    cut_h = int(H * (1 - lam).sqrt())
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    x1 = max(cx - cut_w//2, 0)
    y1 = max(cy - cut_h//2, 0)
    x2 = min(cx + cut_w//2, W)
    y2 = min(cy + cut_h//2, H)
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2-x1)*(y2-y1)/(W*H)
    return mixed_images, labels, labels[rand_idx], float(lam_actual)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ls=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = ls

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

criterion = FocalLoss(2.0, 0.1)

NUM_EPOCHS = 5  # 빠른 실험

# Differential LR
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if 'head' not in n], 'lr': 2e-6},
    {'params': model.head.parameters(), 'lr': 2e-4},
], weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# AMP (자동 혼합 정밀도) - 속도 2x, 메모리 절반
scaler = torch.cuda.amp.GradScaler()

OUT_DIR = "data/models/cropdoc_ext_dinov2"
os.makedirs(OUT_DIR, exist_ok=True)
best_val = 0.0

print(f"\n=== DINOv2-Large 학습 시작 (AMP, {NUM_EPOCHS}ep, batch={BATCH}) ===")
start_total = time.time()

for epoch in range(NUM_EPOCHS):
    ep_start = time.time()
    model.train()
    t_ok = t_tot = 0

    for step, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.cuda.amp.autocast():
            if torch.rand(1).item() > 0.5:
                imgs, lbl_a, lbl_b, lam = cutmix(imgs, labels)
                out = model(imgs)
                loss = lam * criterion(out, lbl_a) + (1-lam) * criterion(out, lbl_b)
            else:
                out = model(imgs)
                loss = criterion(out, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        t_ok += (out.argmax(1) == labels).sum().item()
        t_tot += len(labels)

        # 중간 진행상황 출력 (25% 단위)
        n_steps = len(train_loader)
        if (step + 1) % max(n_steps // 4, 1) == 0:
            elapsed = time.time() - ep_start
            pct = (step+1)/n_steps
            eta = elapsed/pct * (1-pct)
            print(f"  Ep{epoch+1} [{step+1}/{n_steps}] train_acc={t_ok/t_tot:.4f} eta={eta:.0f}s", flush=True)

    model.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                v_ok += (model(imgs).argmax(1) == labels).sum().item()
            v_tot += len(labels)

    val_acc = v_ok/v_tot if v_tot > 0 else 0
    scheduler.step()
    ep_time = time.time() - ep_start
    train_acc = t_ok/t_tot if t_tot > 0 else 0
    total_elapsed = (time.time() - start_total)/60

    print(f"DINOv2 Ep{epoch+1}/{NUM_EPOCHS}: train={train_acc:.4f} val={val_acc:.4f} best={best_val:.4f} [{ep_time:.0f}s, 총{total_elapsed:.1f}분]", flush=True)

    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': 'vit_large_patch14_reg4_dinov2',
            'sorted_classes': sorted_classes,
            'id2label_new': id2label_new,
            'num_new': NUM_NEW,
            'input_size': 518,
            'val_acc_new': best_val,
            'stage': 'DINOv2',
        }, f"{OUT_DIR}/model.pt")
        print(f"  ✅ 저장: {best_val:.4f}", flush=True)

    # 60분 타임아웃 방어: 50분 이상이면 중단
    if (time.time() - start_total)/60 > 50 and epoch < NUM_EPOCHS - 1:
        print(f"⏰ 50분 초과 → 조기 종료 (val_acc={best_val:.4f})", flush=True)
        break

total_time = time.time() - start_total
print(f"\nDINOv2 완료: best_val={best_val:.4f} 총시간={total_time/60:.1f}분", flush=True)

# ============================
# Task 2: DINOv2 단독 테스트
# ============================
print("\n=== Task 2: DINOv2 단독 테스트 ===", flush=True)
import torchvision.transforms as T
import numpy as np

if not os.path.exists(f'{OUT_DIR}/model.pt'):
    print("모델 파일 없음 — 학습 실패")
    import sys; sys.exit(1)

ck = torch.load(f'{OUT_DIR}/model.pt', map_location='cpu')
model_test = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=ck['num_new'])
model_test.load_state_dict(ck['model_state_dict'])
model_test.eval().to(device)
sorted_classes_test = ck['sorted_classes']

NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
TF518 = T.Compose([T.Resize(560), T.CenterCrop(518), T.ToTensor(), NORM])

def predict_dinov2(img_path):
    img = Image.open(img_path).convert('RGB')
    t = TF518(img).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            probs = torch.softmax(model_test(t)[0], -1).cpu().float().numpy()
    best = probs.argmax()
    return sorted_classes_test[best], float(probs[best])

test_cases = [
    ('data/extended_datasets/Coffee Leaf Rust', 'Coffee Leaf Rust'),
    ('data/extended_datasets/Coffee Leaf Miner', 'Coffee Leaf Miner'),
    ('data/extended_datasets/Coffee Phoma', 'Coffee Phoma'),
    ('data/extended_datasets/Rice Blast', 'Rice Blast'),
    ('data/extended_datasets/Rice Brown Spot', 'Rice Brown Spot'),
    ('data/extended_datasets/Rice Hispa', 'Rice Hispa'),
    ('data/extended_datasets/Mango Anthracnose', 'Mango Anthracnose'),
    ('data/extended_datasets/Mango Powdery Mildew', 'Mango Powdery Mildew'),
    ('data/extended_datasets/Wheat Stripe Rust', 'Wheat Stripe Rust'),
    ('data/extended_datasets/Wheat Leaf Rust', 'Wheat Leaf Rust'),
    ('data/extended_datasets/Corn_Common_Rust', 'Corn Common Rust'),
    ('data/extended_datasets/Corn_Blight', 'Corn Blight'),
    ('data/extended_datasets/Banana Black Sigatoka', 'Banana Black Sigatoka'),
    ('data/extended_datasets/Cassava Mosaic Disease', 'Cassava Mosaic Virus'),
    ('data/extended_datasets/Cassava Brown Streak Disease', 'Cassava Brown Streak'),
    ('data/extended_datasets/Citrus Canker', 'Citrus Canker'),
    ('data/extended_datasets/Citrus Black Spot', 'Citrus Black Spot'),
]

total_ok = total = 0
results_per_class = []
for folder, true_label in test_cases:
    imgs = (glob.glob(f'{folder}/*.jpg') + glob.glob(f'{folder}/*.JPG') + glob.glob(f'{folder}/*.png'))[:5]
    ok = 0
    for img in imgs:
        pred, conf = predict_dinov2(img)
        ok += int(true_label.lower() in pred.lower() or pred.lower() in true_label.lower())
        total += 1
    total_ok += ok
    results_per_class.append((true_label, ok, len(imgs)))
    icon = '✅' if ok==len(imgs) else ('⚠️' if ok>0 else '❌')
    print(f'  {icon} {true_label}: {ok}/{len(imgs)}', flush=True)

dinov2_test_acc = total_ok/total if total > 0 else 0
print(f'\nDINOv2 합계: {total_ok}/{total} = {dinov2_test_acc:.1%}', flush=True)

# ============================
# Task 3: EfficientNetV2(expD) + DINOv2 앙상블 테스트
# ============================
print("\n=== Task 3: 앙상블 테스트 (EfficientNetV2 + DINOv2) ===", flush=True)
from torchvision.models import efficientnet_v2_s

# EfficientNetV2 로드
ck_eff = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
model_eff = efficientnet_v2_s()
model_eff.classifier[1] = nn.Linear(1280, ck_eff['num_classes'])
model_eff.load_state_dict(ck_eff['model_state_dict'])
model_eff.eval().to(device)
NUM_OLD_EFF = ck_eff['num_old']

model_dino = model_test  # 이미 로드됨

TFS_EFF = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1), T.ToTensor(), NORM]),
]

sorted_classes_ens = sorted(ck_eff['new_classes'])

def predict_ensemble(img_path):
    img = Image.open(img_path).convert('RGB')
    t_eff = torch.stack([tf(img) for tf in TFS_EFF]).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            p_eff = torch.softmax(model_eff(t_eff), -1).mean(0).cpu().float().numpy()
    p_new_eff = p_eff[NUM_OLD_EFF:]
    t_dino = TF518(img).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            p_dino = torch.softmax(model_dino(t_dino)[0], -1).cpu().float().numpy()
    ensemble = p_new_eff * 0.5 + p_dino * 0.5
    best = ensemble.argmax()
    return sorted_classes_ens[best], float(ensemble[best])

total_ok_ens = total_ens = 0
for folder, true_label in test_cases:
    imgs = (glob.glob(f'{folder}/*.jpg') + glob.glob(f'{folder}/*.JPG') + glob.glob(f'{folder}/*.png'))[:5]
    ok = 0
    for img in imgs:
        pred, conf = predict_ensemble(img)
        ok += int(true_label.lower() in pred.lower() or pred.lower() in true_label.lower())
        total_ens += 1
    total_ok_ens += ok
    icon = '✅' if ok==len(imgs) else ('⚠️' if ok>0 else '❌')
    print(f'  {icon} {true_label}: {ok}/{len(imgs)}', flush=True)

ensemble_test_acc = total_ok_ens/total_ens if total_ens > 0 else 0
print(f'\n앙상블 합계: {total_ok_ens}/{total_ens} = {ensemble_test_acc:.1%}', flush=True)

# ============================
# Task 4: 결과 기록
# ============================
print("\n=== Task 4: 결과 기록 ===", flush=True)
dinov2_status = "keep" if best_val >= 0.9779 else "discard"
ensemble_status = "keep" if total_ok_ens >= 84 else "discard"
print(f"DINOv2 val_acc={best_val:.4f} → {dinov2_status}", flush=True)
print(f"앙상블 {total_ok_ens}/{total_ens}={ensemble_test_acc:.1%} → {ensemble_status}", flush=True)

with open('autoresearch/ext_results.tsv', 'a') as f:
    f.write(f"ideaV_dinov2\t{best_val:.4f}\t{dinov2_status}\tDINOv2-Large 518px + CutMix + AMP, {NUM_EPOCHS}ep, DiffLR(2e-6/2e-4), {MAX_PER_CLASS}장/클래스\n")
    f.write(f"ideaW_eff_dino\t{ensemble_test_acc:.4f}\t{ensemble_status}\tEfficientNetV2(expD,50%)+DINOv2(50%) 앙상블+TTA3변환, {total_ok_ens}/{total_ens}={ensemble_test_acc:.1%}\n")

print("TSV 기록 완료", flush=True)
print("\nDone!", flush=True)
