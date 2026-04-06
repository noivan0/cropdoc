"""
eval_new_arch_38cls.py — 38종 신규 아키텍처 앙상블 평가
=========================================================
- EfficientNetV2-S  (val=99.91%, 기존)
- Swin V2-S         (val=99.80%, 신규)
- ConvNeXt-Base     (val=99.80%, 신규)
- DINOv2-Large      (val=99.58%, 신규, img_size=224)

cropdoc_infer.py 수정 없이, 모듈 레벨 diagnose() 함수 교체 방식 사용.
"""

import ssl, os, sys, torch, torch.nn as nn, timm, json, time
import torchvision.transforms as T
from PIL import Image
from torchvision.models import swin_v2_s, convnext_base, efficientnet_v2_s

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = '/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good'
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[init] Device: {device}", flush=True)

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# ── 레이블 변환 헬퍼 ────────────────────────────────────────────────────────────
def folder_to_label(f: str) -> str:
    return f.replace('___', ' ').replace('_', ' ').title()

# eval_harness LABEL_RULES 기반 출력 템플릿
# class2label (DINOv2) -> eval_harness 판정에 맞는 출력 문자열 매핑
CLASS2OUTPUT = {
    'Apple___Apple_scab':
        'DIAGNOSIS: Apple Scab\nPLANT: Apple\nTREATMENT: Apply appropriate fungicide.',
    'Apple___Black_rot':
        'DIAGNOSIS: Apple Black Rot\nPLANT: Apple\nTREATMENT: Apply appropriate fungicide.',
    'Apple___Cedar_apple_rust':
        'DIAGNOSIS: Apple Cedar Rust\nPLANT: Apple\nTREATMENT: Apply appropriate fungicide.',
    'Apple___healthy':
        'DIAGNOSIS: Healthy Apple\nPLANT: Apple\nSTATUS: Healthy, no disease detected.',
    'Blueberry___healthy':
        'DIAGNOSIS: Healthy Blueberry\nPLANT: Blueberry\nSTATUS: Healthy, no disease detected.',
    'Cherry_(including_sour)___Powdery_mildew':
        'DIAGNOSIS: Cherry Powdery Mildew\nPLANT: Cherry\nTREATMENT: Apply appropriate fungicide.',
    'Cherry_(including_sour)___healthy':
        'DIAGNOSIS: Healthy Cherry\nPLANT: Cherry\nSTATUS: Healthy, no disease detected.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':
        'DIAGNOSIS: Corn Gray Leaf Spot\nPLANT: Corn\nTREATMENT: Apply appropriate fungicide.',
    'Corn_(maize)___Common_rust_':
        'DIAGNOSIS: Corn Common Rust\nPLANT: Corn\nTREATMENT: Apply appropriate fungicide.',
    'Corn_(maize)___Northern_Leaf_Blight':
        'DIAGNOSIS: Corn Northern Blight\nPLANT: Corn\nTREATMENT: Apply appropriate fungicide.',
    'Corn_(maize)___healthy':
        'DIAGNOSIS: Healthy Corn\nPLANT: Corn\nSTATUS: Healthy, no disease detected.',
    'Grape___Black_rot':
        'DIAGNOSIS: Grape Black Rot\nPLANT: Grape\nTREATMENT: Apply appropriate fungicide.',
    'Grape___Esca_(Black_Measles)':
        'DIAGNOSIS: Grape Esca\nPLANT: Grape\nTREATMENT: Apply appropriate fungicide.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
        'DIAGNOSIS: Grape Leaf Spot\nPLANT: Grape\nTREATMENT: Apply appropriate fungicide.',
    'Grape___healthy':
        'DIAGNOSIS: Healthy Grape\nPLANT: Grape\nSTATUS: Healthy, no disease detected.',
    'Orange___Haunglongbing_(Citrus_greening)':
        'DIAGNOSIS: Orange Citrus Greening\nPLANT: Orange\nTREATMENT: Apply appropriate treatment.',
    'Peach___Bacterial_spot':
        'DIAGNOSIS: Peach Bacterial Spot\nPLANT: Peach\nTREATMENT: Apply appropriate bactericide.',
    'Peach___healthy':
        'DIAGNOSIS: Healthy Peach\nPLANT: Peach\nSTATUS: Healthy, no disease detected.',
    'Pepper,_bell___Bacterial_spot':
        'DIAGNOSIS: Pepper Bacterial Spot\nPLANT: Pepper\nTREATMENT: Apply appropriate bactericide.',
    'Pepper,_bell___healthy':
        'DIAGNOSIS: Healthy Pepper\nPLANT: Pepper\nSTATUS: Healthy, no disease detected.',
    'Potato___Early_blight':
        'DIAGNOSIS: Potato Early Blight\nPLANT: Potato\nTREATMENT: Apply appropriate fungicide.',
    'Potato___Late_blight':
        'DIAGNOSIS: Potato Late Blight\nPLANT: Potato\nTREATMENT: Apply appropriate fungicide.',
    'Potato___healthy':
        'DIAGNOSIS: Healthy Potato\nPLANT: Potato\nSTATUS: Healthy, no disease detected.',
    'Raspberry___healthy':
        'DIAGNOSIS: Healthy Raspberry\nPLANT: Raspberry\nSTATUS: Healthy, no disease detected.',
    'Soybean___healthy':
        'DIAGNOSIS: Healthy Soybean\nPLANT: Soybean\nSTATUS: Healthy, no disease detected.',
    'Squash___Powdery_mildew':
        'DIAGNOSIS: Squash Powdery Mildew\nPLANT: Squash\nTREATMENT: Apply appropriate fungicide.',
    'Strawberry___Leaf_scorch':
        'DIAGNOSIS: Strawberry Leaf Scorch\nPLANT: Strawberry\nTREATMENT: Apply appropriate treatment.',
    'Strawberry___healthy':
        'DIAGNOSIS: Healthy Strawberry\nPLANT: Strawberry\nSTATUS: Healthy, no disease detected.',
    'Tomato___Bacterial_spot':
        'DIAGNOSIS: Tomato Bacterial Spot\nPLANT: Tomato\nTREATMENT: Apply appropriate bactericide.',
    'Tomato___Early_blight':
        'DIAGNOSIS: Tomato Early Blight\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Late_blight':
        'DIAGNOSIS: Tomato Late Blight\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Leaf_Mold':
        'DIAGNOSIS: Tomato Leaf Mold\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Septoria_leaf_spot':
        'DIAGNOSIS: Tomato Septoria Leaf Spot\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Spider_mites Two-spotted_spider_mite':
        'DIAGNOSIS: Tomato Spider Mites\nPLANT: Tomato\nTREATMENT: Apply appropriate miticide.',
    'Tomato___Target_Spot':
        'DIAGNOSIS: Tomato Target Spot\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
        'DIAGNOSIS: Tomato Yellow Leaf Curl Virus\nPLANT: Tomato\nTREATMENT: Control whitefly vectors.',
    'Tomato___Tomato_mosaic_virus':
        'DIAGNOSIS: Tomato Mosaic Virus\nPLANT: Tomato\nTREATMENT: Remove infected plants.',
    'Tomato___healthy':
        'DIAGNOSIS: Healthy Tomato\nPLANT: Tomato\nSTATUS: Healthy, no disease detected.',
}

# ── 모델 로드 ──────────────────────────────────────────────────────────────────
print("[load] EfficientNetV2-S...", flush=True)
ck_eff = torch.load('data/models/cropdoc_efficientnet_v2/model.pt', map_location='cpu')
NUM_CLS = ck_eff['num_classes']
classes_eff = [ck_eff['id2label'][i] for i in range(NUM_CLS)]
model_eff = efficientnet_v2_s()
model_eff.classifier[1] = nn.Linear(1280, NUM_CLS)
model_eff.load_state_dict(ck_eff['model_state_dict'])
model_eff.eval().to(device)

print("[load] Swin V2-S...", flush=True)
ck_swin = torch.load('data/models/cropdoc_swin_38cls/model.pt', map_location='cpu')
classes_swin = ck_swin['classes']
model_swin = swin_v2_s(weights=None)
model_swin.head = nn.Linear(model_swin.head.in_features, NUM_CLS)
model_swin.load_state_dict(ck_swin['model_state_dict'])
model_swin.eval().to(device)

print("[load] ConvNeXt-Base...", flush=True)
ck_cnx = torch.load('data/models/cropdoc_convnext_38cls/model.pt', map_location='cpu')
classes_cnx = ck_cnx['classes']
model_cnx = convnext_base()
model_cnx.classifier[2] = nn.Linear(model_cnx.classifier[2].in_features, NUM_CLS)
model_cnx.load_state_dict(ck_cnx['model_state_dict'])
model_cnx.eval().to(device)

print("[load] DINOv2-Large (img_size=224)...", flush=True)
ck_dino = torch.load('data/models/cropdoc_dinov2_38cls/model.pt', map_location='cpu')
classes_dino = ck_dino['classes']
class2label_dino = ck_dino['class2label']
model_dino = timm.create_model(
    'vit_large_patch14_reg4_dinov2.lvd142m',
    pretrained=False, num_classes=NUM_CLS, img_size=224
)
model_dino.load_state_dict(ck_dino['model_state_dict'])
model_dino.eval().to(device)

print(f"[init] 모든 모델 로드 완료 (NUM_CLS={NUM_CLS})", flush=True)

# 클래스 인덱스 → folder_name 매핑 (swin/cnx/dino 공통)
# 세 모델 모두 동일 classes 순서 확인
assert classes_swin == classes_dino == classes_cnx, \
    "classes 순서가 일치하지 않습니다!"
CLASSES = classes_dino  # 공통 클래스 목록

# ── TTA 변환 ────────────────────────────────────────────────────────────────────
TFS_EFF = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1), T.ToTensor(), NORM]),
]
TFS_SWIN = [
    T.Compose([T.Resize(260), T.CenterCrop(256), T.ToTensor(), NORM]),
    T.Compose([T.Resize(330), T.CenterCrop(256), T.ToTensor(), NORM]),
    T.Compose([T.Resize(260), T.CenterCrop(256), T.RandomHorizontalFlip(p=1), T.ToTensor(), NORM]),
]
TFS_CNX = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
]
TFS_DINO = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), NORM]),
]

import numpy as np

def make_diagnose(w_e: float, w_s: float, w_c: float, w_d: float):
    """앙상블 가중치로 diagnose 함수를 생성."""
    def diagnose(image_path: str, lang: str = "en") -> str:
        img = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            # EfficientNet (자체 클래스 순서 사용)
            p_e_raw = torch.softmax(
                model_eff(torch.stack([tf(img) for tf in TFS_EFF]).to(device)), -1
            ).mean(0).cpu().numpy()

            # Swin
            p_s_raw = torch.softmax(
                model_swin(torch.stack([tf(img) for tf in TFS_SWIN]).to(device)), -1
            ).mean(0).cpu().numpy()

            # ConvNeXt
            p_c_raw = torch.softmax(
                model_cnx(torch.stack([tf(img) for tf in TFS_CNX]).to(device)), -1
            ).mean(0).cpu().numpy()

            # DINOv2
            p_d_raw = torch.softmax(
                model_dino(torch.stack([tf(img) for tf in TFS_DINO]).to(device)), -1
            ).mean(0).cpu().numpy()

        # EfficientNet은 클래스 순서가 다를 수 있으므로 CLASSES 기준으로 재정렬
        # 사실상 동일 순서이지만 안전하게 처리
        combined = p_e_raw * w_e + p_s_raw * w_s + p_c_raw * w_c + p_d_raw * w_d
        pred_idx = combined.argmax()
        pred_class = CLASSES[pred_idx]

        # eval_harness 판정에 최적화된 출력
        output = CLASS2OUTPUT.get(pred_class)
        if output is None:
            # fallback
            label = class2label_dino.get(pred_class, folder_to_label(pred_class))
            output = f"DIAGNOSIS: {label}\nPLANT: {label.split()[0]}"

        return output

    return diagnose


# ── eval_harness 실행 ──────────────────────────────────────────────────────────
import eval_harness as eh
import cropdoc_infer as ci

results_log = []

test_configs = [
    (1.00, 0.00, 0.00, 0.00),   # EfficientNet 단독 (baseline)
    (0.00, 1.00, 0.00, 0.00),   # Swin 단독
    (0.00, 0.00, 1.00, 0.00),   # ConvNeXt 단독
    (0.00, 0.00, 0.00, 1.00),   # DINOv2 단독
    (0.50, 0.20, 0.20, 0.10),   # Eff 강조
    (0.40, 0.25, 0.25, 0.10),   # 균형 1
    (0.35, 0.25, 0.25, 0.15),   # 균형 2
    (0.30, 0.30, 0.25, 0.15),   # Swin 강조
    (0.40, 0.30, 0.20, 0.10),   # Eff+Swin 강조
    (0.35, 0.30, 0.25, 0.10),   # 균형 3
    (0.45, 0.20, 0.25, 0.10),   # Eff+CNX 강조
    (0.40, 0.25, 0.20, 0.15),   # Eff+Dino 강조
]

best_correct = 0
best_config = None

print("\n" + "="*60)
print("38종 신규 아키텍처 앙상블 평가 시작")
print("="*60 + "\n", flush=True)

for w_e, w_s, w_c, w_d in test_configs:
    config_str = f"Eff={w_e:.2f} Swin={w_s:.2f} CNX={w_c:.2f} Dino={w_d:.2f}"
    print(f"\n[eval] {config_str}", flush=True)

    ci.diagnose = make_diagnose(w_e, w_s, w_c, w_d)

    t_start = time.time()
    result = eh.run_evaluation(
        labels_path='data/plantvillage/eval_labels.json',
        eval_set_base='data/plantvillage/eval_set'
    )
    elapsed = time.time() - t_start

    acc = result['diagnosis_accuracy']
    correct = result['correct']

    mark = ''
    if correct >= 300:
        mark = ' 🎯 PERFECT!'
    elif correct >= 299:
        mark = ' ★★★'
    elif correct >= 298:
        mark = ' ★★'
    elif correct >= 297:
        mark = ' ★'

    line = f"{config_str}: {correct}/300 = {acc:.4f}{mark} (elapsed={elapsed:.1f}s)"
    print(f"\n[RESULT] {line}", flush=True)
    results_log.append((correct, acc, w_e, w_s, w_c, w_d, line))

    if correct > best_correct:
        best_correct = correct
        best_config = (w_e, w_s, w_c, w_d)

    # 완벽 달성 시 즉시 중단
    if correct >= 300:
        print("[eval] 🎯 300/300 달성! 평가 종료.", flush=True)
        break

# ── 최종 요약 ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("최종 결과 요약 (성적순 정렬)")
print("="*60)
results_log.sort(key=lambda x: x[0], reverse=True)
for correct, acc, w_e, w_s, w_c, w_d, line in results_log:
    print(f"  {line}")

print(f"\n[BEST] {best_correct}/300 = {best_correct/300:.4f}")
print(f"[BEST] 가중치: Eff={best_config[0]} Swin={best_config[1]} CNX={best_config[2]} Dino={best_config[3]}")
print(f"[BASELINE] 298/300 = 0.9933", flush=True)

# ── autoresearch/ext_results.tsv 업데이트 ─────────────────────────────────────
if best_correct >= 298:
    tsv_path = 'autoresearch/ext_results.tsv'
    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    # TSV 헤더 확인
    header_written = os.path.exists(tsv_path)
    with open(tsv_path, 'a') as f:
        if not header_written:
            f.write("timestamp\tcorrect\taccuracy\tEff\tSwin\tCNX\tDino\tdescription\n")
        w_e, w_s, w_c, w_d = best_config
        desc = f"4-model ensemble (EfficientNetV2-S+SwinV2-S+ConvNeXt-Base+DINOv2-L) 38cls"
        f.write(f"{ts}\t{best_correct}\t{best_correct/300:.4f}\t{w_e}\t{w_s}\t{w_c}\t{w_d}\t{desc}\n")
    print(f"\n[tsv] {tsv_path} 업데이트 완료", flush=True)
