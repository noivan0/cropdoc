"""
expA 모델 전체 34종 × 5장 테스트 (80장이 아닌 34×5=170장)
기존 exp7 80장 테스트와 비교하기 위해 field_images 사용
"""
import torch, torch.nn as nn, glob, os, json
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# field_images 디렉토리 확인
field_dir = 'data/field_images'
print("field_images 구조:", os.listdir(field_dir) if os.path.exists(field_dir) else "없음")

# expA 모델 로드
ck = torch.load('data/models/cropdoc_ext_expA/model.pt', map_location='cpu')
NUM_OLD = ck['num_old']
NEW_CLASSES = ck['new_classes']
NUM_TOTAL = ck['num_classes']
id2label = ck['id2label']
print(f"expA val_acc_new: {ck.get('val_acc_new'):.4f}")
print(f"NUM_OLD={NUM_OLD}, NUM_NEW={len(NEW_CLASSES)}, NUM_TOTAL={NUM_TOTAL}")

# exp7 모델도 비교용 로드
ck7 = torch.load('data/models/cropdoc_ext_exp7/model.pt', map_location='cpu')

device = torch.device('cuda')
# expA 모델
modelA = efficientnet_v2_s()
modelA.classifier[1] = nn.Linear(1280, NUM_TOTAL)
modelA.load_state_dict(ck['model_state_dict'])
modelA = modelA.to(device).eval()

# exp7 모델
model7 = efficientnet_v2_s()
model7.classifier[1] = nn.Linear(1280, NUM_TOTAL)
model7.load_state_dict(ck7['model_state_dict'])
model7 = model7.to(device).eval()

VAL_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
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

sorted_classes = sorted(NEW_CLASSES)
test_cases = []
for lbl in sorted_classes:
    folder = FOLDER_MAP.get(lbl, f"data/extended_datasets/{lbl}")
    test_cases.append((folder, lbl))

def infer(model, img_path, num_old):
    img = Image.open(img_path).convert('RGB')
    t = VAL_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        preds_new = out[0, num_old:].softmax(0)
        top_idx = preds_new.argmax().item()
        pred_label = id2label[num_old + top_idx]
        conf = preds_new[top_idx].item()
    return pred_label, conf

total_okA = total_ok7 = total = 0
print(f"\n{'클래스':<35} {'expA':>8} {'exp7':>8}")
print("-" * 55)

for folder, true_label in test_cases:
    imgs = (glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.JPG") +
            glob.glob(f"{folder}/*.png") + glob.glob(f"{folder}/*.PNG"))
    imgs = sorted(imgs)[:5]
    if not imgs:
        print(f"{true_label:<35} {'N/A':>8} {'N/A':>8}")
        continue

    okA = ok7 = 0
    wrongs = []
    for img_path in imgs:
        try:
            predA, confA = infer(modelA, img_path, NUM_OLD)
            pred7, conf7 = infer(model7, img_path, NUM_OLD)
            matchA = (true_label.lower() in predA.lower() or predA.lower() in true_label.lower())
            match7 = (true_label.lower() in pred7.lower() or pred7.lower() in true_label.lower())
            okA += matchA
            ok7 += match7
            total += 1
            if not matchA:
                wrongs.append(f"    ❌A: {os.path.basename(img_path)} → {predA}")
        except Exception as e:
            print(f"  오류: {img_path}: {e}")
            total += 1
    
    total_okA += okA
    total_ok7 += ok7
    
    iconA = '✅' if okA==len(imgs) else ('⚠️' if okA>0 else '❌')
    icon7 = '✅' if ok7==len(imgs) else ('⚠️' if ok7>0 else '❌')
    delta = ""
    if okA != ok7:
        delta = f" {'↑' if okA>ok7 else '↓'}{abs(okA-ok7)}"
    print(f"{iconA} {true_label:<33} {okA}/{len(imgs):>3}   {icon7} {ok7}/{len(imgs):>3}{delta}")
    for w in wrongs:
        print(w)

print(f"\n{'='*55}")
print(f"expA 합계: {total_okA}/{total} = {total_okA/total:.1%}")
print(f"exp7 합계: {total_ok7}/{total} = {total_ok7/total:.1%}")
delta_n = total_okA - total_ok7
print(f"개선: {'+' if delta_n>=0 else ''}{delta_n}장 ({'+' if delta_n>=0 else ''}{delta_n/total:.1%})")
print(f"\nexpA val_acc_new: {ck.get('val_acc_new'):.4f}")
print(f"exp7 val_acc_new: {ck7.get('val_acc_new'):.4f}")
