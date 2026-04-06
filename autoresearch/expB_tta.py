import sys, os, glob, time
os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
sys.path.insert(0, 'scripts')

import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

ck = torch.load('data/models/cropdoc_ext_exp7/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NEW_CLASSES = ck['new_classes']
NUM_NEW = len(NEW_CLASSES)
id2label = ck['id2label']
print(f"NUM_OLD={NUM_OLD}, NUM_NEW={NUM_NEW}")

device = torch.device('cuda')
model = efficientnet_v2_s()
model.classifier[1] = nn.Linear(1280, ck['num_classes'])
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
TTA_TFs = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), NORM]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomVerticalFlip(p=1.0), T.ToTensor(), NORM]),
]

def predict_tta(img_path):
    img = Image.open(img_path).convert('RGB')
    tensors = torch.stack([tf(img) for tf in TTA_TFs]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensors), -1).mean(0).cpu().numpy()
    new_probs = [(float(probs[i]), id2label[i]) for i in range(NUM_OLD, len(id2label))]
    return sorted(new_probs, reverse=True)[0]

def predict_single(img_path):
    img = Image.open(img_path).convert('RGB')
    t = TTA_TFs[0](img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), -1)[0].cpu().numpy()
    new_probs = [(float(probs[i]), id2label[i]) for i in range(NUM_OLD, len(id2label))]
    return sorted(new_probs, reverse=True)[0]

# 전체 신규 클래스 5장씩 테스트 (TTA vs single)
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

sorted_classes = sorted(NEW_CLASSES)
test_cases = []
for lbl in sorted_classes:
    folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
    test_cases.append((folder, lbl))

total_ok_tta, total_ok_single, total = 0, 0, 0
print('=== TTA vs Single 테스트 결과 ===')
for folder, true_label in test_cases:
    imgs = (glob.glob(f'{folder}/*.jpg') + glob.glob(f'{folder}/*.JPG') +
            glob.glob(f'{folder}/*.png') + glob.glob(f'{folder}/*.PNG'))
    imgs = sorted(imgs)[:5]
    if not imgs:
        print(f"  ⚠️ {true_label}: 이미지 없음 ({folder})")
        continue
    ok_tta = ok_single = 0
    for img in imgs:
        try:
            conf_t, pred_t = predict_tta(img)
            conf_s, pred_s = predict_single(img)
            match_tta = (true_label.lower() in pred_t.lower() or pred_t.lower() in true_label.lower())
            match_single = (true_label.lower() in pred_s.lower() or pred_s.lower() in true_label.lower())
            ok_tta += match_tta
            ok_single += match_single
            total += 1
            if not match_tta:
                print(f"    ❌ TTA 오답: {os.path.basename(img)} → {pred_t} ({conf_t:.3f})")
            if not match_single and match_tta:
                print(f"    ✅ TTA가 수정: {os.path.basename(img)} single→{pred_s}, tta→{pred_t}")
        except Exception as e:
            print(f"    오류: {img}: {e}")
            total += 1
    total_ok_tta += ok_tta
    total_ok_single += ok_single
    icon = '✅' if ok_tta==len(imgs) else ('⚠️' if ok_tta>0 else '❌')
    icon_s = '✅' if ok_single==len(imgs) else ('⚠️' if ok_single>0 else '❌')
    if ok_tta != ok_single:
        delta = f" [single={ok_single}/{len(imgs)}→TTA={ok_tta}/{len(imgs)} {'↑' if ok_tta>ok_single else '↓'}]"
    else:
        delta = ""
    print(f"  {icon} {true_label}: TTA {ok_tta}/{len(imgs)}{delta}")

print(f"\n=== 결과 요약 ===")
print(f"TTA   합계: {total_ok_tta}/{total} = {total_ok_tta/total:.1%}")
print(f"Single합계: {total_ok_single}/{total} = {total_ok_single/total:.1%}")
if total_ok_tta > total_ok_single:
    print(f"TTA 개선: +{total_ok_tta-total_ok_single}장 → KEEP 후보")
elif total_ok_tta == total_ok_single:
    print("TTA 동일 → discard")
else:
    print(f"TTA 악화: {total_ok_tta-total_ok_single}장 → discard")
