"""
Idea M: 다중 모델 앙상블 (expD + expC + expG 소프트 보팅) + Multi-Scale TTA
"""
import sys, os, glob, torch, torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# 여러 모델 로드
model_paths = [
    ('data/models/cropdoc_ext_expD/model.pt', 0.5),   # 최고, 가중치 50%
    ('data/models/cropdoc_ext_expC/model.pt', 0.3),   # 2위, 30%
    ('data/models/cropdoc_ext_expG/model.pt', 0.2),   # 3위, 20%
]

models = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

for path, w in model_paths:
    ck = torch.load(path, map_location='cpu')
    m = efficientnet_v2_s()
    m.classifier[1] = nn.Linear(1280, ck['num_classes'])
    m.load_state_dict(ck['model_state_dict'])
    m.eval().to(device)
    models.append((m, w, ck['num_old'], ck['id2label']))
    print(f"  Loaded {path} (weight={w})")

NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# Multi-Scale TTA
TFS = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(448), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), NORM]),
]

def predict_ensemble(img_path):
    img = Image.open(img_path).convert('RGB')
    tensors = torch.stack([tf(img) for tf in TFS]).to(device)

    # 각 모델의 신규 클래스 확률 합산
    ensemble_probs = None
    for model, weight, NUM_OLD, id2label in models:
        with torch.no_grad():
            probs = torch.softmax(model(tensors), -1).mean(0).cpu().numpy()
        new_probs = np.array([probs[i] for i in range(NUM_OLD, len(id2label))])
        if ensemble_probs is None:
            ensemble_probs = new_probs * weight
            num_old = NUM_OLD
            id2lbl = id2label
        else:
            ensemble_probs += new_probs * weight

    best_idx = ensemble_probs.argmax()
    best_label = id2lbl[num_old + best_idx]
    best_conf = float(ensemble_probs[best_idx])
    return best_label, best_conf

# 전체 테스트 케이스
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

total_ok, total = 0, 0
results_per_class = []
print('\n=== 앙상블 + Multi-Scale TTA 결과 ===')
for folder, true_label in test_cases:
    imgs = (glob.glob(f'{folder}/*.jpg') + glob.glob(f'{folder}/*.JPG') + glob.glob(f'{folder}/*.png'))[:5]
    if not imgs:
        print(f'  ⚠️ {true_label}: 이미지 없음')
        continue
    ok = 0
    details = []
    for img in imgs:
        pred, conf = predict_ensemble(img)
        correct = (true_label.lower() in pred.lower() or pred.lower() in true_label.lower())
        ok += correct
        total += 1
        if not correct:
            details.append(f"    ❌ {os.path.basename(img)}: pred={pred}({conf:.3f})")
    total_ok += ok
    icon = '✅' if ok==len(imgs) else ('⚠️' if ok>0 else '❌')
    print(f'  {icon} {true_label}: {ok}/{len(imgs)}')
    for d in details:
        print(d)
    results_per_class.append((true_label, ok, len(imgs)))

print(f'\n앙상블+TTA 합계: {total_ok}/{total} = {total_ok/total:.1%}')
print(f'기준: 84/85 이상이면 KEEP')
