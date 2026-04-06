"""
Idea 4: TTA (Test Time Augmentation) 적용 — 추론 시점 앙상블
기반: exp3 모델 (val_acc_new=0.9609)
TTA 적용 전후 비교
"""
import torch, torch.nn as nn, glob, os, random, time
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import torchvision.transforms as T
import numpy as np

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck = torch.load('data/models/cropdoc_ext_exp3/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
id2label = ck['id2label']
NUM_TOTAL = ck['num_classes']
NEW_CLASSES = ck['new_classes']

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device).eval()

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 단일 추론 (기존 방식)
SINGLE_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])

# TTA 4변환
TTA_TFS = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), NORM]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomVerticalFlip(p=1.0), T.ToTensor(), NORM]),
]

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

# 전체 신규 클래스 50장 샘플로 val_acc 측정
print("TTA 효과 평가 (신규 34종, 클래스당 최대 20장)")
print("=" * 60)

single_ok = single_tot = 0
tta_ok = tta_tot = 0

for lbl in sorted(NEW_CLASSES):
    folder = folder_map.get(lbl, f"data/extended_datasets/{lbl}")
    imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
            glob.glob(f"{folder}/**/*.JPG", recursive=True) +
            glob.glob(f"{folder}/**/*.png", recursive=True))
    # 후반 이미지 사용 (학습에 사용된 앞 500장 이후)
    imgs_test = imgs[500:520] if len(imgs) > 500 else imgs[-20:]
    if not imgs_test:
        imgs_test = random.sample(imgs, min(10, len(imgs)))

    class_idx = NUM_OLD + sorted(NEW_CLASSES).index(lbl)
    s_ok = t_ok = 0
    for img_path in imgs_test:
        try:
            img = Image.open(img_path).convert('RGB')
            # Single inference
            t = SINGLE_TF(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t)[0]
                probs_s = torch.softmax(out, -1).cpu().numpy()
            pred_s = probs_s[NUM_OLD:].argmax() + NUM_OLD
            if pred_s == class_idx:
                s_ok += 1

            # TTA inference
            tensors = torch.stack([tf(img) for tf in TTA_TFS]).to(device)
            with torch.no_grad():
                logits = model(tensors)
                probs_tta = torch.softmax(logits, -1).mean(0).cpu().numpy()
            pred_t = probs_tta[NUM_OLD:].argmax() + NUM_OLD
            if pred_t == class_idx:
                t_ok += 1
        except Exception as e:
            pass

    n = len(imgs_test)
    single_ok += s_ok; single_tot += n
    tta_ok += t_ok; tta_tot += n

    diff = t_ok - s_ok
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    print(f"  {lbl:<35} Single={s_ok}/{n} TTA={t_ok}/{n} ({diff_str})")

print("=" * 60)
s_acc = single_ok/single_tot if single_tot else 0
t_acc = tta_ok/tta_tot if tta_tot else 0
delta = t_acc - s_acc
print(f"Single: {single_ok}/{single_tot} = {s_acc:.4f}")
print(f"TTA:    {tta_ok}/{tta_tot} = {t_acc:.4f}")
print(f"Delta:  {delta:+.4f}")
print(f"keep 기준(>=0.9599+0.010=0.9699 기준으로 TTA 단독 판단 불가 — 추론 방식만 변경)")
print(f"exp3 기준 학습 val: {ck['val_acc_new']:.4f}")

# TTA 개선 여부
if delta > 0:
    print(f"\nTTA: 개선 확인 ({delta:+.4f}) → cropdoc_extended.py에 적용 권장")
elif delta == 0:
    print(f"\nTTA: 동일 성능 — 적용 여부는 판단에 따라")
else:
    print(f"\nTTA: 성능 저하 ({delta:+.4f}) → 미적용 권장")
