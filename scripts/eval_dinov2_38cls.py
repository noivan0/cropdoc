"""
Task 1: DINOv2-Small 38종 단독 평가
- 기존 오답 2개(inat_250907864.jpg)를 DINOv2가 맞히는지 확인
"""
import ssl, os, sys, torch, json, timm, re
import torchvision.transforms as T
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# DINOv2-Small 로드
ck = torch.load('data/models/cropdoc_dinov2_38cls/model.pt', map_location='cpu')
classes = ck['classes']
class2label = ck['class2label']
print(f"DINOv2-Small: val_acc={ck['val_acc']:.4f}, {len(classes)}클래스")

img_size = ck.get('img_size', 224)
model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(classes), img_size=img_size)
print(f"모델 img_size={img_size}")
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])
TFS = [
    TF,
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1), T.ToTensor(), NORM]),
]

# PlantVillage eval_labels 로드
with open('data/plantvillage/eval_labels.json') as f:
    eval_labels = json.load(f)

# eval_harness의 레이블 정규화 로직 복사
def normalize_label(text):
    text = text.lower().strip()
    text = re.sub(r"\b(no|not|without|healthy|none)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# DINOv2 클래스명 → eval 레이블 변환
def folder_to_eval_label(folder_name):
    name = folder_name.replace('___', ' ').replace('_', ' ')
    words = name.split()
    return ' '.join(w.capitalize() for w in words)

# eval_labels 예시 확인
sample_labels = list(set(eval_labels.values()))[:10]
print(f"eval 레이블 예시: {sample_labels[:5]}")

# 300장 전체 평가
correct = total = 0
wrong_cases = []
inat_results = []

for img_rel_path, true_label in eval_labels.items():
    img_path = f"data/plantvillage/eval_set/{img_rel_path}"
    if not os.path.exists(img_path):
        continue
    try:
        img = Image.open(img_path).convert('RGB')
        t = torch.stack([tf(img) for tf in TFS]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(t), -1).mean(0).cpu().numpy()
        pred_idx = probs.argmax()
        pred_folder = classes[pred_idx]
        pred_label = folder_to_eval_label(pred_folder)
        
        true_norm = normalize_label(true_label)
        pred_norm = normalize_label(pred_label)
        hit = (true_norm in pred_norm) or (pred_norm in true_norm)
        
        if '250907864' in img_rel_path:
            inat_results.append((img_rel_path, true_label, pred_label, float(probs[pred_idx]), hit))
        
        if hit:
            correct += 1
        else:
            wrong_cases.append((img_rel_path, true_label, pred_label, float(probs[pred_idx])))
        total += 1
    except Exception as e:
        print(f"에러 {img_rel_path}: {e}")

print(f"\n=== DINOv2-Small 38종 eval 결과 ===")
print(f"정확도: {correct}/{total} = {correct/total:.4f}")
print(f"\n📌 inat_250907864.jpg 결과:")
for path, true, pred, conf, hit in inat_results:
    status = "✅" if hit else "❌"
    print(f"  {status} {path}: 정답={true}, 예측={pred} ({conf:.3f})")

print(f"\n오답 {len(wrong_cases)}개:")
for path, true, pred, conf in wrong_cases:
    print(f"  ❌ {path.split('/')[-1]}: 정답={true}, 예측={pred} ({conf:.3f})")
