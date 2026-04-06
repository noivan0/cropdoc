"""
16종 × 5장 = 80장 테스트 (expA vs exp7 비교)
태스크 지시서의 80장 테스트 기준과 동일
"""
import torch, torch.nn as nn, glob, os
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# 두 모델 모두 로드
ckA = torch.load('data/models/cropdoc_ext_expA/model.pt', map_location='cpu')
ck7 = torch.load('data/models/cropdoc_ext_exp7/model.pt', map_location='cpu')
NUM_OLD = ckA['num_old']
id2label = ckA['id2label']
NUM_TOTAL = ckA['num_classes']

device = torch.device('cuda')

modelA = efficientnet_v2_s()
modelA.classifier[1] = nn.Linear(1280, NUM_TOTAL)
modelA.load_state_dict(ckA['model_state_dict'])
modelA = modelA.to(device).eval()

model7 = efficientnet_v2_s()
model7.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model7.load_state_dict(ck7['model_state_dict'])
model7 = model7.to(device).eval()

VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# 16종 × 5장 = 80장 (지시서 기준)
test_cases = [
    ('data/extended_datasets/Coffee Leaf Rust', 'Coffee Leaf Rust'),
    ('data/extended_datasets/Coffee Leaf Miner', 'Coffee Leaf Miner'),
    ('data/extended_datasets/Coffee Phoma', 'Coffee Phoma'),
    ('data/extended_datasets/Rice Blast', 'Rice Blast'),
    ('data/extended_datasets/Rice Brown Spot', 'Rice Brown Spot'),
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

def infer(model, img_path):
    img = Image.open(img_path).convert('RGB')
    t = VAL_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        preds_new = out[0, NUM_OLD:].softmax(0)
        top_idx = preds_new.argmax().item()
        pred_label = id2label[NUM_OLD + top_idx]
        conf = preds_new[top_idx].item()
    return pred_label, conf

total_okA = total_ok7 = total = 0
print(f"{'클래스':<30} {'expA':>8} {'exp7':>8}")
print("-" * 50)

for folder, true_label in test_cases:
    imgs = (glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.JPG") +
            glob.glob(f"{folder}/*.png") + glob.glob(f"{folder}/*.PNG"))
    imgs = sorted(imgs)[:5]
    if not imgs:
        print(f"{true_label:<30} {'N/A':>8} {'N/A':>8}")
        continue

    okA = ok7 = 0
    details = []
    for img_path in imgs:
        try:
            predA, confA = infer(modelA, img_path)
            pred7, conf7 = infer(model7, img_path)
            matchA = (true_label.lower() in predA.lower() or predA.lower() in true_label.lower())
            match7 = (true_label.lower() in pred7.lower() or pred7.lower() in true_label.lower())
            okA += matchA
            ok7 += match7
            total += 1
            if not matchA or not match7:
                details.append(f"  {os.path.basename(img_path)}: A={predA[:20]}({'✓' if matchA else '✗'}) 7={pred7[:20]}({'✓' if match7 else '✗'})")
        except Exception as e:
            print(f"  오류: {img_path}: {e}")
            total += 1

    total_okA += okA
    total_ok7 += ok7
    iconA = '✅' if okA==len(imgs) else ('⚠️' if okA>0 else '❌')
    icon7 = '✅' if ok7==len(imgs) else ('⚠️' if ok7>0 else '❌')
    delta = f" [{'↑' if okA>ok7 else '↓' if okA<ok7 else '='}{abs(okA-ok7)}]" if okA!=ok7 else ""
    print(f"{iconA} {true_label:<28} {okA}/{len(imgs):>3}   {icon7} {ok7}/{len(imgs):>3}{delta}")
    for d in details:
        print(d)

print(f"\n{'='*50}")
print(f"80장 테스트 결과 (16종 × 5장):")
print(f"  expA: {total_okA}/{total} = {total_okA/total:.1%}")
print(f"  exp7: {total_ok7}/{total} = {total_ok7/total:.1%}")
print(f"  개선: {'+' if total_okA>=total_ok7 else ''}{total_okA-total_ok7}장")
print(f"\nexpA val_acc_new: {ckA.get('val_acc_new'):.4f}")
print(f"exp7 val_acc_new: {ck7.get('val_acc_new'):.4f}")
