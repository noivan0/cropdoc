"""
DINOv2-Large 38종 PlantVillage 학습 스크립트 (v2 — Large 모델 사용)
목표: val_acc ≥ 0.9950 달성 후 model.pt 저장
사용 모델: vit_large_patch14_reg4_dinov2.lvd142m (이미 로컬 캐시됨)
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os, sys, glob, time, json, torch, torch.nn as nn, re
import timm
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# GPU 1에 여유 공간이 더 많음 (GPU 0: ~3.5GB free, GPU 1: ~13GB free)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda')
PV_BASE  = "/root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2"
PV_TRAIN = f"{PV_BASE}/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
OUT_DIR  = "data/models/cropdoc_dinov2_38cls"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 클래스 목록 ───────────────────────────────────────────────────────────────
classes = sorted([d for d in os.listdir(PV_TRAIN) if os.path.isdir(f"{PV_TRAIN}/{d}")])
NUM_CLASSES = len(classes)
print(f"클래스 수: {NUM_CLASSES}", flush=True)

# ── class → 레이블 매핑 ───────────────────────────────────────────────────────
def cls2label(cls: str) -> str:
    s = cls.replace("___", " ").replace("_", " ")
    s = re.sub(r'\(.*?\)', '', s).strip()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

class2label = {cls: cls2label(cls) for cls in classes}
print("클래스→레이블 예시:", flush=True)
for k, v in list(class2label.items())[:5]:
    print(f"  {k} → {v}", flush=True)

# ── DINOv2-Large (로컬 캐시 사용) ─────────────────────────────────────────────
print("\nDINOv2-Large 로드 중 (로컬 캐시)...", flush=True)
model = timm.create_model(
    'vit_large_patch14_reg4_dinov2.lvd142m',
    pretrained=True,
    num_classes=NUM_CLASSES,
    img_size=224,
)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"파라미터: {total_params:.1f}M", flush=True)
print(f"GPU 메모리: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class PVDataset(Dataset):
    def __init__(self, tf, max_per_class=1000):
        self.samples = []
        self.tf = tf
        for i, cls in enumerate(classes):
            imgs = (glob.glob(f"{PV_TRAIN}/{cls}/*.jpg") +
                    glob.glob(f"{PV_TRAIN}/{cls}/*.JPG") +
                    glob.glob(f"{PV_TRAIN}/{cls}/*.png"))
            imgs = imgs[:max_per_class]
            for p in imgs:
                self.samples.append((p, i))
        print(f"총 {len(self.samples)}장 로드됨", flush=True)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.tf(img), label
        except:
            return torch.zeros(3, 224, 224), label

class SubsetWithTransform(Dataset):
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf = tf
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            img = Image.open(path).convert('RGB')
            return self.tf(img), label
        except:
            return torch.zeros(3, 224, 224), label

full_ds = PVDataset(TRAIN_TF, max_per_class=500)  # GPU 메모리 절약위해 제한
val_size  = max(len(full_ds) // 10, NUM_CLASSES * 2)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])
val_ds_with_tf = SubsetWithTransform(val_ds, VAL_TF)

# Large 모델 + GPU 공유 환경: 배치 크기 최소화
BATCH_SIZE = 8
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds_with_tf, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)

print(f"Train: {len(train_ds)}장, Val: {len(val_ds_with_tf)}장", flush=True)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
head_params     = list(model.head.parameters())

NUM_EPOCHS = 6
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': head_params,     'lr': 3e-4},
], weight_decay=0.05)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 3e-4],
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.1,
    anneal_strategy='cos',
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler()

# ── 학습 루프 ─────────────────────────────────────────────────────────────────
best_val = 0.0
torch.save({'classes': classes, 'class2label': class2label}, f"{OUT_DIR}/metadata.pt")

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    model.train()
    t_ok = t_tot = t_loss_sum = 0

    for batch_idx, (imgs, labels_batch) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out  = model(imgs)
            loss = criterion(out, labels_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        t_ok       += (out.argmax(1) == labels_batch).sum().item()
        t_tot      += len(labels_batch)
        t_loss_sum += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  [{batch_idx+1}/{len(train_loader)}] "
                  f"train_acc={t_ok/t_tot:.4f} loss={t_loss_sum/(batch_idx+1):.4f}", flush=True)

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    v_ok = v_tot = 0
    with torch.no_grad():
        for imgs, labels_batch in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(imgs)
            v_ok  += (out.argmax(1) == labels_batch).sum().item()
            v_tot += len(labels_batch)

    val_acc   = v_ok  / v_tot  if v_tot  > 0 else 0
    train_acc = t_ok  / t_tot  if t_tot  > 0 else 0
    avg_loss  = t_loss_sum / len(train_loader)
    elapsed   = time.time() - t0

    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Ep{epoch+1:02d}/{NUM_EPOCHS}: train={train_acc:.4f} val={val_acc:.4f} "
          f"loss={avg_loss:.4f} [{elapsed:.0f}s] GPU={gpu_mem_gb:.1f}GB", flush=True)

    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture':     'vit_large_patch14_reg4_dinov2',
            'img_size':         224,
            'classes':          classes,
            'class2label':      class2label,
            'num_classes':      NUM_CLASSES,
            'val_acc':          best_val,
            'epoch':            epoch + 1,
        }, f"{OUT_DIR}/model.pt")
        print(f"  ✓ 저장됨 (best={best_val:.4f})", flush=True)

    if epoch >= 3 and val_acc < 0.96:
        print("조기 종료: val_acc 너무 낮음", flush=True)
        break

print(f"\nDINOv2-L 38종 완료: best_val={best_val:.4f}", flush=True)

result = {
    'model': 'DINOv2-Large (vit_large_patch14_reg4_dinov2)',
    'best_val_acc': best_val,
    'num_classes': NUM_CLASSES,
    'img_size': 224,
    'epochs_trained': epoch + 1,
}
with open(f"{OUT_DIR}/training_result.json", 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)
