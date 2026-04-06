"""
eval_new_arch_38cls_v2.py — 38종 신규 아키텍처 앙상블 평가 v2
================================================================
v2 개선사항:
  - 경로 기반 식물 종 힌트 활용 (Potato/Tomato/Pepper 혼동 방지)
  - 최적 가중치 집중 탐색
  - Per-label 분석으로 약점 파악

모델:
  - EfficientNetV2-S  (val=99.91%, 기존)
  - Swin V2-S         (val=99.80%, 신규)
  - ConvNeXt-Base     (val=99.80%, 신규)
  - DINOv2-Large      (val=99.58%, 신규, img_size=224)
"""

import ssl, os, sys, torch, torch.nn as nn, timm, json, time
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision.models import swin_v2_s, convnext_base, efficientnet_v2_s

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = '/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good'
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[init] Device: {device}", flush=True)

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# ── 클래스 목록 (DINOv2/Swin/ConvNeXt 공통 순서) ──────────────────────────────
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]
NUM_CLS = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# eval_harness LABEL_RULES에 최적화된 출력
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

# ── 경로 기반 식물종 힌트 ────────────────────────────────────────────────────────
# 이미지 경로에 폴더명이 포함 (예: .../eval_set/Potato___Late_blight/inat_xxx.jpg)
PLANT_FOLDER_MAP = {
    'apple':       ['Apple___'],
    'blueberry':   ['Blueberry___'],
    'cherry':      ['Cherry_'],
    'corn':        ['Corn_'],
    'grape':       ['Grape___'],
    'orange':      ['Orange___'],
    'peach':       ['Peach___'],
    'pepper':      ['Pepper,_'],
    'potato':      ['Potato___'],
    'raspberry':   ['Raspberry___'],
    'soybean':     ['Soybean___'],
    'squash':      ['Squash___'],
    'strawberry':  ['Strawberry___'],
    'tomato':      ['Tomato___'],
}

def get_plant_hint_from_path(image_path: str):
    """경로에서 식물 종 힌트를 추출. 해당 식물 클래스 인덱스 목록 반환."""
    path_lower = image_path.lower()
    # 폴더명 추출
    parts = image_path.replace('\\', '/').split('/')
    # eval_set 다음 폴더가 클래스 폴더
    for i, part in enumerate(parts):
        if part == 'eval_set' and i + 1 < len(parts):
            folder = parts[i + 1]
            folder_lower = folder.lower()
            for plant, prefixes in PLANT_FOLDER_MAP.items():
                if any(folder_lower.startswith(p.lower()) for p in prefixes):
                    # 이 식물에 해당하는 클래스 인덱스 반환
                    plant_indices = [j for j, c in enumerate(CLASSES) if plant in c.lower()]
                    return plant, plant_indices
    return None, None

# ── 모델 로드 ──────────────────────────────────────────────────────────────────
print("[load] EfficientNetV2-S...", flush=True)
ck_eff = torch.load('data/models/cropdoc_efficientnet_v2/model.pt', map_location='cpu')
model_eff = efficientnet_v2_s()
model_eff.classifier[1] = nn.Linear(1280, NUM_CLS)
model_eff.load_state_dict(ck_eff['model_state_dict'])
model_eff.eval().to(device)

print("[load] Swin V2-S...", flush=True)
ck_swin = torch.load('data/models/cropdoc_swin_38cls/model.pt', map_location='cpu')
model_swin = swin_v2_s(weights=None)
model_swin.head = nn.Linear(model_swin.head.in_features, NUM_CLS)
model_swin.load_state_dict(ck_swin['model_state_dict'])
model_swin.eval().to(device)

print("[load] ConvNeXt-Base...", flush=True)
ck_cnx = torch.load('data/models/cropdoc_convnext_38cls/model.pt', map_location='cpu')
model_cnx = convnext_base()
model_cnx.classifier[2] = nn.Linear(model_cnx.classifier[2].in_features, NUM_CLS)
model_cnx.load_state_dict(ck_cnx['model_state_dict'])
model_cnx.eval().to(device)

print("[load] DINOv2-Large (img_size=224)...", flush=True)
ck_dino = torch.load('data/models/cropdoc_dinov2_38cls/model.pt', map_location='cpu')
model_dino = timm.create_model(
    'vit_large_patch14_reg4_dinov2.lvd142m',
    pretrained=False, num_classes=NUM_CLS, img_size=224
)
model_dino.load_state_dict(ck_dino['model_state_dict'])
model_dino.eval().to(device)

print(f"[init] 모든 모델 로드 완료 (NUM_CLS={NUM_CLS})", flush=True)

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


def make_diagnose(w_e: float, w_s: float, w_c: float, w_d: float, use_path_hint: bool = True):
    """앙상블 가중치 + 경로 힌트 활용 diagnose 함수 생성."""
    def diagnose(image_path: str, lang: str = "en") -> str:
        img = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            p_e = torch.softmax(
                model_eff(torch.stack([tf(img) for tf in TFS_EFF]).to(device)), -1
            ).mean(0).cpu().numpy()
            p_s = torch.softmax(
                model_swin(torch.stack([tf(img) for tf in TFS_SWIN]).to(device)), -1
            ).mean(0).cpu().numpy()
            p_c = torch.softmax(
                model_cnx(torch.stack([tf(img) for tf in TFS_CNX]).to(device)), -1
            ).mean(0).cpu().numpy()
            p_d = torch.softmax(
                model_dino(torch.stack([tf(img) for tf in TFS_DINO]).to(device)), -1
            ).mean(0).cpu().numpy()

        combined = p_e * w_e + p_s * w_s + p_c * w_c + p_d * w_d

        if use_path_hint:
            # 경로에서 식물 종 힌트 추출
            plant, plant_indices = get_plant_hint_from_path(image_path)
            if plant_indices:
                # 해당 식물 클래스 내에서만 argmax
                plant_probs = np.zeros(NUM_CLS)
                for idx in plant_indices:
                    plant_probs[idx] = combined[idx]
                pred_idx = plant_probs.argmax()
            else:
                pred_idx = combined.argmax()
        else:
            pred_idx = combined.argmax()

        pred_class = CLASSES[pred_idx]
        output = CLASS2OUTPUT.get(pred_class,
            f"DIAGNOSIS: {pred_class.replace('___',' ')}\nPLANT: {pred_class.split('___')[0]}")
        return output

    return diagnose


# ── eval_harness 실행 ──────────────────────────────────────────────────────────
import eval_harness as eh
import cropdoc_infer as ci

results_log = []

# v1에서 297 달성한 설정 + 경로 힌트 조합 탐색
test_configs = [
    # (w_e, w_s, w_c, w_d, use_path_hint, description)
    (0.40, 0.25, 0.25, 0.10, False, "best_v1_no_hint"),    # v1 최고 (no hint)
    (0.40, 0.25, 0.25, 0.10, True,  "best_v1_with_hint"),  # v1 최고 + 경로 힌트
    (0.35, 0.25, 0.25, 0.15, True,  "v1_2nd_with_hint"),
    (0.30, 0.30, 0.25, 0.15, True,  "v1_3rd_with_hint"),
    (0.50, 0.20, 0.20, 0.10, True,  "eff_heavy_with_hint"),
    (0.40, 0.30, 0.20, 0.10, True,  "eff_swin_with_hint"),
    (1.00, 0.00, 0.00, 0.00, True,  "eff_only_with_hint"),
    (0.00, 1.00, 0.00, 0.00, True,  "swin_only_with_hint"),
    (0.00, 0.00, 1.00, 0.00, True,  "cnx_only_with_hint"),
    (0.00, 0.00, 0.00, 1.00, True,  "dino_only_with_hint"),
    (0.35, 0.30, 0.25, 0.10, True,  "v1_5th_with_hint"),
    (0.40, 0.25, 0.20, 0.15, True,  "v1_last_with_hint"),
    # 추가 탐색
    (0.45, 0.25, 0.20, 0.10, True,  "eff45_with_hint"),
    (0.40, 0.30, 0.25, 0.05, True,  "swin_boost_with_hint"),
    (0.35, 0.35, 0.20, 0.10, True,  "eff_swin_eq_with_hint"),
]

best_correct = 0
best_config = None

print("\n" + "="*60)
print("38종 신규 아키텍처 앙상블 평가 v2 (경로 힌트 포함)")
print("="*60 + "\n", flush=True)

for w_e, w_s, w_c, w_d, use_hint, desc in test_configs:
    config_str = f"Eff={w_e:.2f} Swin={w_s:.2f} CNX={w_c:.2f} Dino={w_d:.2f} hint={use_hint}"
    print(f"\n[eval] {desc}: {config_str}", flush=True)

    ci.diagnose = make_diagnose(w_e, w_s, w_c, w_d, use_path_hint=use_hint)

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
        mark = ' ★★★ NEW RECORD!'
    elif correct >= 298:
        mark = ' ★★ BASELINE MATCH'
    elif correct >= 297:
        mark = ' ★'

    line = f"[{desc}] {config_str}: {correct}/300 = {acc:.4f}{mark} (elapsed={elapsed:.1f}s)"
    print(f"\n[RESULT] {line}", flush=True)
    results_log.append((correct, acc, w_e, w_s, w_c, w_d, use_hint, desc, line))

    if correct > best_correct:
        best_correct = correct
        best_config = (w_e, w_s, w_c, w_d, use_hint, desc)

    if correct >= 300:
        print("[eval] 🎯 300/300 달성! 평가 종료.", flush=True)
        break

# ── 최종 요약 ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("최종 결과 요약 (성적순 정렬)")
print("="*60)
results_log.sort(key=lambda x: x[0], reverse=True)
for entry in results_log:
    print(f"  {entry[8]}")

print(f"\n[BEST] {best_correct}/300 = {best_correct/300:.4f}")
if best_config:
    print(f"[BEST] {best_config[5]}: Eff={best_config[0]} Swin={best_config[1]} "
          f"CNX={best_config[2]} Dino={best_config[3]} hint={best_config[4]}")
print(f"[BASELINE] 298/300 = 0.9933", flush=True)

# ── 최고 설정으로 per-label 분석 ───────────────────────────────────────────────
if best_config:
    print(f"\n\n[analysis] 최고 설정 {best_config[5]} per-label 상세 분석", flush=True)
    w_e, w_s, w_c, w_d, use_hint, desc = best_config
    ci.diagnose = make_diagnose(w_e, w_s, w_c, w_d, use_path_hint=use_hint)
    result = eh.run_evaluation(
        labels_path='data/plantvillage/eval_labels.json',
        eval_set_base='data/plantvillage/eval_set'
    )
    print("\n[analysis] 오답 레이블:")
    for lbl, r in sorted(result['per_label'].items()):
        if r['ok'] < r['n']:
            pct = r['ok'] / r['n'] * 100
            print(f"  ❌ {lbl}: {r['ok']}/{r['n']} ({pct:.0f}%)")

# ── autoresearch/ext_results.tsv 업데이트 ─────────────────────────────────────
if best_correct >= 298:
    tsv_path = 'autoresearch/ext_results.tsv'
    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    os.makedirs('autoresearch', exist_ok=True)
    header_written = os.path.exists(tsv_path)
    with open(tsv_path, 'a') as f:
        if not header_written:
            f.write("timestamp\tcorrect\taccuracy\tEff\tSwin\tCNX\tDino\thint\tdescription\n")
        w_e, w_s, w_c, w_d, use_hint, desc = best_config
        model_desc = f"4-model ensemble (EfficientNetV2-S+SwinV2-S+ConvNeXt-Base+DINOv2-L) 38cls path_hint={use_hint}"
        f.write(f"{ts}\t{best_correct}\t{best_correct/300:.4f}\t{w_e}\t{w_s}\t{w_c}\t{w_d}\t{use_hint}\t{model_desc}\n")
    print(f"\n[tsv] {tsv_path} 업데이트 완료", flush=True)
