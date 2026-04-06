#!/usr/bin/env python3
"""expE 85장 테스트 평가"""
import torch, torch.nn as nn
import glob, os
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck = torch.load('data/models/cropdoc_ext_expE/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NUM_TOTAL = ck['num_classes']
id2label = ck['id2label']
label2id = {v:k for k,v in id2label.items()}

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

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

total_ok = total_cnt = 0
for folder, label in test_cases:
    imgs = sorted(glob.glob(f"{folder}/**/*.jpg", recursive=True) +
                  glob.glob(f"{folder}/**/*.JPG", recursive=True) +
                  glob.glob(f"{folder}/**/*.png", recursive=True))
    # 최대 5장
    imgs = imgs[:5]
    ok = 0
    for p in imgs:
        try:
            img = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img)
            pred_id = out.argmax(1).item()
            pred_label = id2label[pred_id]
            if pred_label == label:
                ok += 1
        except Exception as e:
            print(f"  ERROR: {p} — {e}")
    total_ok += ok
    total_cnt += len(imgs)
    status = "✓" if ok == len(imgs) else f"✗ {ok}/{len(imgs)}"
    print(f"{label}: {status}")

print(f"\n총 {total_ok}/{total_cnt} = {total_ok/total_cnt*100:.1f}%")
print(f"val_acc_new(saved): {ck['val_acc_new']:.4f}")
