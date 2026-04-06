"""
DINOv2-Large 38종 단독 평가 (eval_harness의 is_correct 로직 사용)
- inat_250907864.jpg 오답 여부 확인
- DINOv2 diagnose 함수를 만들어 공식 판정 방식 적용
"""
import ssl, os, sys, torch, json, timm, re
import torchvision.transforms as T
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
sys.path.insert(0, 'scripts')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# eval_harness의 is_correct 로직 복사
DISEASE_WORDS = {
    "blight", "spot", "mold", "mildew", "rust", "scab",
    "wilt", "rot", "mosaic", "virus", "bacterial", "septoria",
    "alternaria", "phytophthora", "fungal", "infection", "disease",
    "necrosis", "lesion", "lesions",
}

LABEL_RULES = {
    "Tomato Early Blight":       ({"tomato"},  {"early", "blight"},   False),
    "Tomato Late Blight":        ({"tomato"},  {"late",  "blight"},   False),
    "Tomato Bacterial Spot":     ({"tomato"},  {"bacterial", "spot"}, False),
    "Tomato Leaf Mold":          ({"tomato"},  {"mold"},              False),
    "Tomato Septoria Leaf Spot": ({"tomato"},  {"septoria"},          False),
    "Healthy Tomato":            ({"tomato"},  set(),                 True),
    "Potato Early Blight":       ({"potato"},  {"early", "blight"},   False),
    "Potato Late Blight":        ({"potato"},  {"late",  "blight"},   False),
    "Healthy Potato":            ({"potato"},  set(),                 True),
    "Pepper Bacterial Spot":     ({"pepper"},  {"bacterial", "spot"}, False),
    "Healthy Pepper":            ({"pepper"},  set(),                 True),
    "Apple Scab":                ({"apple"},   {"scab"},              False),
    "Apple Black Rot":           ({"apple"},   {"black", "rot"},      False),
    "Apple Cedar Rust":          ({"apple"},   {"rust"},              False),
    "Healthy Apple":             ({"apple"},   set(),                 True),
    "Corn Gray Leaf Spot":       ({"corn"},    {"gray", "spot"},      False),
    "Corn Common Rust":          ({"corn"},    {"rust"},              False),
    "Corn Northern Blight":      ({"corn"},    {"blight"},            False),
    "Healthy Corn":              ({"corn"},    set(),                 True),
    "Grape Black Rot":           ({"grape"},   {"black", "rot"},      False),
    "Grape Esca":                ({"grape"},   {"esca"},              False),
    "Grape Leaf Spot":           ({"grape"},   {"spot"},              False),
    "Healthy Grape":             ({"grape"},   set(),                 True),
    "Peach Bacterial Spot":      ({"peach"},   {"bacterial", "spot"}, False),
    "Healthy Peach":             ({"peach"},   set(),                 True),
    "Strawberry Leaf Scorch":    ({"strawberry"}, {"scorch"},         False),
    "Healthy Strawberry":        ({"strawberry"}, set(),              True),
    "Cherry Powdery Mildew":     ({"cherry"},  {"powdery", "mildew"}, False),
    "Healthy Cherry":            ({"cherry"},  set(),                 True),
    "Squash Powdery Mildew":     ({"squash"},  {"powdery", "mildew"}, False),
    "Healthy Blueberry":         ({"blueberry"}, set(),              True),
    "Healthy Raspberry":         ({"raspberry"}, set(),              True),
    "Healthy Soybean":           ({"soybean"},   set(),              True),
    "Orange Citrus Greening":    ({"orange"},  {"citrus", "greening"}, False),
    "Tomato Spider Mites":       ({"tomato"},  {"spider", "mites"},   False),
    "Tomato Target Spot":        ({"tomato"},  {"target", "spot"},    False),
    "Tomato Yellow Leaf Curl":   ({"tomato"},  {"yellow", "curl"},    False),
    "Tomato Mosaic Virus":       ({"tomato"},  {"mosaic"},            False),
}

_NEG_RE = re.compile(
    r'\b(no|not|without|never|absent|lack of|lacking|free of|free from)\b(\s+\w+){1,3}',
    re.IGNORECASE,
)

def _strip_negations(text: str) -> str:
    return _NEG_RE.sub(" [NEGATED] ", text)

def is_correct(prediction: str, label: str) -> bool:
    if label not in LABEL_RULES:
        return label.lower() in prediction.lower()
    plant_genera, disease_kws, is_healthy = LABEL_RULES[label]
    pred_lower = prediction.lower()
    if not any(g in pred_lower for g in plant_genera):
        return False
    cleaned = _strip_negations(pred_lower)
    if is_healthy:
        return "healthy" in pred_lower and not any(kw in cleaned for kw in DISEASE_WORDS)
    else:
        return all(kw in cleaned for kw in disease_kws)

# DINOv2 모델 로드
ck = torch.load('data/models/cropdoc_dinov2_38cls/model.pt', map_location='cpu')
classes = ck['classes']
class2label = ck['class2label']
img_size = ck.get('img_size', 224)
print(f"DINOv2-Large 38종: val_acc={ck['val_acc']:.4f}, img_size={img_size}")

model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False, 
                          num_classes=len(classes), img_size=img_size)
model.load_state_dict(ck['model_state_dict'])
model.eval().to(device)

NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
TFS = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1), T.ToTensor(), NORM]),
]

# DINOv2용 diagnose 함수
LANG_TEMPLATES = {
    "en": "DIAGNOSIS: {disease}\nPLANT: {plant}\nTREATMENT: Apply appropriate fungicide. Monitor closely.",
}

def folder_to_label(folder_name):
    """PV 폴더명 → class2label 또는 직접 변환"""
    if folder_name in class2label:
        return class2label[folder_name]
    name = folder_name.replace('___', ' ').replace('_', ' ')
    words = name.split()
    return ' '.join(w.capitalize() for w in words)

def dinov2_diagnose(image_path: str, lang: str = "en") -> str:
    img = Image.open(image_path).convert('RGB')
    t = torch.stack([tf(img) for tf in TFS]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), -1).mean(0).cpu().numpy()
    pred_idx = probs.argmax()
    pred_folder = classes[pred_idx]
    pred_label = folder_to_label(pred_folder)
    # 식물명과 질병명 분리
    parts = pred_label.split(' ', 1)
    plant = parts[0]
    disease = parts[1] if len(parts) > 1 else 'Healthy'
    tmpl = LANG_TEMPLATES.get(lang, LANG_TEMPLATES["en"])
    return tmpl.format(disease=disease, plant=plant)

# 평가
with open('data/plantvillage/eval_labels.json') as f:
    eval_labels = json.load(f)

correct = total = 0
wrong_cases = []
inat_results = []

for img_rel_path, true_label in eval_labels.items():
    img_path = f"data/plantvillage/eval_set/{img_rel_path}"
    if not os.path.exists(img_path):
        continue
    try:
        prediction = dinov2_diagnose(img_path)
        ok = is_correct(prediction, true_label)
        
        if '250907864' in img_rel_path:
            inat_results.append((img_rel_path, true_label, prediction, ok))
        
        if ok:
            correct += 1
        else:
            wrong_cases.append((img_rel_path, true_label, prediction))
        total += 1
    except Exception as e:
        print(f"에러 {img_rel_path}: {e}")

print(f"\n=== DINOv2-Large 38종 eval (공식 판정) ===")
print(f"정확도: {correct}/{total} = {correct/total:.4f}")

print(f"\n📌 inat_250907864.jpg 결과:")
for path, true, pred, ok in inat_results:
    status = "✅" if ok else "❌"
    print(f"  {status} {path}")
    print(f"     정답={true}")
    print(f"     예측={pred}")

print(f"\n오답 {len(wrong_cases)}개:")
for path, true, pred in wrong_cases[:20]:
    print(f"  ❌ {path.split('/')[-1]}: 정답={true}")
    print(f"     예측={pred}")
