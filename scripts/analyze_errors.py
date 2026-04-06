"""오답 케이스 상세 분석"""
import torch, torch.nn as nn, glob, os
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck = torch.load('data/models/cropdoc_ext_expD/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NUM_CLASSES = ck['num_classes']
id2label = ck['id2label']
NEW_CLASSES = ck['new_classes']

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
VAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])

# 85장 전체 상세 분석
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

print('=== 오답 상세 분석 (expD 단일스케일) ===\n')
total_ok, total = 0, 0
errors = []

for folder, true_label in test_cases:
    imgs = sorted(
        glob.glob(f'{folder}/*.jpg') + glob.glob(f'{folder}/*.JPG') + glob.glob(f'{folder}/*.png')
    )[:5]

    for img_path in imgs:
        img_name = os.path.basename(img_path)
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            t = VAL_TF(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(t), -1)[0].cpu().numpy()
            new_probs = [(float(probs[i]), id2label[i]) for i in range(NUM_OLD, NUM_CLASSES)]
            top3 = sorted(new_probs, reverse=True)[:3]
            hit = (true_label.lower() in top3[0][1].lower() or top3[0][1].lower() in true_label.lower())
            total += 1
            if hit:
                total_ok += 1
            else:
                errors.append({
                    'img': img_name,
                    'true': true_label,
                    'pred': top3[0][1],
                    'conf': top3[0][0],
                    'top2': top3[1],
                    'top3': top3[2],
                    'size': f'{w}x{h}',
                })
        except Exception as e:
            print(f'  ERROR {img_name}: {e}')

print(f'총 정확도: {total_ok}/{total} = {total_ok/total:.1%}')
print(f'\n❌ 오답 목록:')
for e in errors:
    print(f"\n  [{e['img']}] ({e['size']})")
    print(f"    정답: {e['true']}")
    print(f"    예측: {e['pred']} ({e['conf']:.2%})")
    print(f"    2위: {e['top2'][1]} ({e['top2'][0]:.2%})")
    print(f"    3위: {e['top3'][1]} ({e['top3'][0]:.2%})")
