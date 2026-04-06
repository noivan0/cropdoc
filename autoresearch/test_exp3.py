"""
exp3 모델로 12종 × 5장 테스트
"""
import torch, torch.nn as nn, glob, os
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

ck = torch.load('data/models/cropdoc_ext_exp3/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
id2label = ck['id2label']
NUM_TOTAL = ck['num_classes']

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model.load_state_dict(ck['model_state_dict'])
model = model.to(device).eval()

VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_cases = [
    ('data/extended_datasets/Coffee Leaf Rust', 'Coffee Leaf Rust'),
    ('data/extended_datasets/Coffee Leaf Miner', 'Coffee Leaf Miner'),
    ('data/extended_datasets/Rice Blast', 'Rice Blast'),
    ('data/extended_datasets/Rice Brown Spot', 'Rice Brown Spot'),
    ('data/extended_datasets/Mango Anthracnose', 'Mango Anthracnose'),
    ('data/extended_datasets/Wheat Stripe Rust', 'Wheat Stripe Rust'),
    ('data/extended_datasets/Corn_Common_Rust', 'Corn Common Rust'),
    ('data/extended_datasets/Cassava Mosaic Disease', 'Cassava Mosaic Virus'),
    ('data/extended_datasets/Banana Black Sigatoka', 'Banana Black Sigatoka'),
    ('data/extended_datasets/Citrus Canker', 'Citrus Canker'),
    ('data/extended_datasets/Wheat Leaf Rust', 'Wheat Leaf Rust'),
    ('data/extended_datasets/Mango Powdery Mildew', 'Mango Powdery Mildew'),
]

total_ok = total_tot = 0
print(f"{'클래스':<30} {'정답/5':>6} {'정확도':>8}")
print("-" * 50)

for folder, expected_label in test_cases:
    imgs = (glob.glob(f"{folder}/**/*.jpg", recursive=True) +
            glob.glob(f"{folder}/**/*.JPG", recursive=True) +
            glob.glob(f"{folder}/**/*.png", recursive=True))
    imgs = imgs[:5]
    if not imgs:
        print(f"{expected_label:<30} {'N/A':>6}")
        continue

    ok = 0
    for img_path in imgs:
        try:
            tensor = VAL_TF(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                pred_idx = out.argmax(1).item()
                pred_label = id2label.get(pred_idx, f"class_{pred_idx}")
            if pred_label == expected_label:
                ok += 1
        except Exception as e:
            pass

    total_ok += ok
    total_tot += len(imgs)
    acc = ok / len(imgs)
    mark = "✓" if ok == len(imgs) else ("△" if ok > 0 else "✗")
    print(f"{expected_label:<30} {ok}/{len(imgs):>1}    {acc:.2f}  {mark}")

print("-" * 50)
print(f"{'총합':<30} {total_ok}/{total_tot}   {total_ok/total_tot:.4f}")
print(f"\nexp3 val_acc_new (학습 시): {ck['val_acc_new']:.4f}")
