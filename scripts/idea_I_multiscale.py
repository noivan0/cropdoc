"""
Idea I: Multi-Scale TTA (Test-Time Augmentation)
- 학습 없음, 추론 전략만 변경
- 5가지 해상도/변환에서 추론 후 평균
- 고해상도 이미지(Coffee 1361.jpg) 대응
"""

import sys, os, glob, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import efficientnet_v2_s

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
sys.path.insert(0, 'scripts')

# 모델 로드
ck = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NUM_CLASSES = ck['num_classes']
id2label = ck['id2label']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Multi-scale TTA: 여러 해상도에서 추론 후 평균
MULTI_SCALE_TFs = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),                                           # 기본
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),                                           # 중간
    T.Compose([T.Resize(448), T.CenterCrop(224), T.ToTensor(), NORM]),                                           # 고해상도
    T.Compose([T.Resize(512), T.CenterCrop(224), T.ToTensor(), NORM]),                                           # 최고해상도
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), NORM]),            # hflip
]

def predict_multiscale(img_path):
    img = Image.open(img_path).convert('RGB')
    tensors = torch.stack([tf(img) for tf in MULTI_SCALE_TFs]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensors), -1).mean(0).cpu().numpy()
    new_probs = [(float(probs[i]), id2label[i]) for i in range(NUM_OLD, NUM_CLASSES)]
    return sorted(new_probs, reverse=True)[0]

# 기존 단일 스케일 (baseline 비교용)
BASE_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])

def predict_single(img_path):
    img = Image.open(img_path).convert('RGB')
    t = BASE_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), -1)[0].cpu().numpy()
    new_probs = [(float(probs[i]), id2label[i]) for i in range(NUM_OLD, NUM_CLASSES)]
    return sorted(new_probs, reverse=True)[0]

# 85장 전체 테스트
test_cases = [
    ('data/extended_datasets/Coffee Leaf Rust',           'Coffee Leaf Rust'),
    ('data/extended_datasets/Coffee Leaf Miner',          'Coffee Leaf Miner'),
    ('data/extended_datasets/Coffee Phoma',               'Coffee Phoma'),
    ('data/extended_datasets/Rice Blast',                 'Rice Blast'),
    ('data/extended_datasets/Rice Brown Spot',            'Rice Brown Spot'),
    ('data/extended_datasets/Rice Hispa',                 'Rice Hispa'),
    ('data/extended_datasets/Mango Anthracnose',          'Mango Anthracnose'),
    ('data/extended_datasets/Mango Powdery Mildew',       'Mango Powdery Mildew'),
    ('data/extended_datasets/Wheat Stripe Rust',          'Wheat Stripe Rust'),
    ('data/extended_datasets/Wheat Leaf Rust',            'Wheat Leaf Rust'),
    ('data/extended_datasets/Corn_Common_Rust',           'Corn Common Rust'),
    ('data/extended_datasets/Corn_Blight',                'Corn Blight'),
    ('data/extended_datasets/Banana Black Sigatoka',      'Banana Black Sigatoka'),
    ('data/extended_datasets/Cassava Mosaic Disease',     'Cassava Mosaic Virus'),
    ('data/extended_datasets/Cassava Brown Streak Disease','Cassava Brown Streak'),
    ('data/extended_datasets/Citrus Canker',              'Citrus Canker'),
    ('data/extended_datasets/Citrus Black Spot',          'Citrus Black Spot'),
]

print('=== Idea I: Multi-Scale TTA 결과 ===')
print(f'{"폴더":<35} {"기본":>8} {"멀티스케일":>12}')
print('-' * 60)

total_ok_base, total_ok_ms, total = 0, 0, 0
wrong_base = []
wrong_ms = []

for folder, true_label in test_cases:
    imgs = sorted(
        glob.glob(f'{folder}/*.jpg') +
        glob.glob(f'{folder}/*.JPG') +
        glob.glob(f'{folder}/*.png')
    )[:5]

    if not imgs:
        print(f'  ⚠️  {true_label}: 이미지 없음')
        continue

    ok_base, ok_ms = 0, 0
    for img in imgs:
        img_name = os.path.basename(img)

        # 기본 단일 스케일
        conf_b, pred_b = predict_single(img)
        hit_b = (true_label.lower() in pred_b.lower() or pred_b.lower() in true_label.lower())
        ok_base += hit_b

        # 멀티 스케일
        conf_m, pred_m = predict_multiscale(img)
        hit_m = (true_label.lower() in pred_m.lower() or pred_m.lower() in true_label.lower())
        ok_ms += hit_m

        if not hit_b:
            wrong_base.append((img_name, true_label, pred_b, conf_b))
        if not hit_m:
            wrong_ms.append((img_name, true_label, pred_m, conf_m))

    total += len(imgs)
    total_ok_base += ok_base
    total_ok_ms += ok_ms

    icon_b = '✅' if ok_base == len(imgs) else ('⚠️' if ok_base > 0 else '❌')
    icon_m = '✅' if ok_ms == len(imgs) else ('⚠️' if ok_ms > 0 else '❌')
    print(f'  {true_label:<33} {icon_b} {ok_base}/{len(imgs)}   {icon_m} {ok_ms}/{len(imgs)}')

print('-' * 60)
print(f'\n기본 단일스케일: {total_ok_base}/{total} = {total_ok_base/total:.1%}')
print(f'멀티스케일 TTA:  {total_ok_ms}/{total} = {total_ok_ms/total:.1%}')
print(f'개선: +{total_ok_ms - total_ok_base}장')

if wrong_ms:
    print('\n오답 (멀티스케일):')
    for img_name, true, pred, conf in wrong_ms:
        print(f'  ❌ {img_name}: "{true}" → "{pred}" ({conf:.2%})')
else:
    print('\n✅ 멀티스케일 오답 없음!')

if wrong_base and not wrong_ms:
    print('\n🎉 개선된 케이스:')
    for img_name, true, pred, conf in wrong_base:
        print(f'  ✅ {img_name}: "{true}" ← 기존 "{pred}" ({conf:.2%})')
