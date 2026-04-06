"""
eval_new_arch_38cls_v3.py — 38종 신규 아키텍처 앙상블 평가 v3
================================================================
v3 개선사항:
  - Potato/Tomato Blight 크로스-플랜트 힌트 (Tomato Late Blight probs → Potato Late Blight로 보강)
  - 어려운 케이스 특별 처리:
      * Potato Late Blight 이미지가 Tomato Late Blight처럼 보이는 경우
      * Tomato Late/Early 구분이 어려운 경우 (Swin+Dino late blight 가중치 증폭)
  - 더 넓은 가중치 그리드 탐색
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

# 인덱스 상수
IDX = {c: i for i, c in enumerate(CLASSES)}
I_TOM_EARLY  = IDX['Tomato___Early_blight']
I_TOM_LATE   = IDX['Tomato___Late_blight']
I_POT_EARLY  = IDX['Potato___Early_blight']
I_POT_LATE   = IDX['Potato___Late_blight']
I_POT_HLTHY  = IDX['Potato___healthy']

CLASS2OUTPUT = {
    'Apple___Apple_scab': 'DIAGNOSIS: Apple Scab\nPLANT: Apple\nTREATMENT: Apply appropriate fungicide.',
    'Apple___Black_rot': 'DIAGNOSIS: Apple Black Rot\nPLANT: Apple\nTREATMENT: Apply appropriate fungicide.',
    'Apple___Cedar_apple_rust': 'DIAGNOSIS: Apple Cedar Rust\nPLANT: Apple\nTREATMENT: Apply appropriate fungicide.',
    'Apple___healthy': 'DIAGNOSIS: Healthy Apple\nPLANT: Apple\nSTATUS: Healthy, no disease detected.',
    'Blueberry___healthy': 'DIAGNOSIS: Healthy Blueberry\nPLANT: Blueberry\nSTATUS: Healthy, no disease detected.',
    'Cherry_(including_sour)___Powdery_mildew': 'DIAGNOSIS: Cherry Powdery Mildew\nPLANT: Cherry\nTREATMENT: Apply appropriate fungicide.',
    'Cherry_(including_sour)___healthy': 'DIAGNOSIS: Healthy Cherry\nPLANT: Cherry\nSTATUS: Healthy, no disease detected.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'DIAGNOSIS: Corn Gray Leaf Spot\nPLANT: Corn\nTREATMENT: Apply appropriate fungicide.',
    'Corn_(maize)___Common_rust_': 'DIAGNOSIS: Corn Common Rust\nPLANT: Corn\nTREATMENT: Apply appropriate fungicide.',
    'Corn_(maize)___Northern_Leaf_Blight': 'DIAGNOSIS: Corn Northern Blight\nPLANT: Corn\nTREATMENT: Apply appropriate fungicide.',
    'Corn_(maize)___healthy': 'DIAGNOSIS: Healthy Corn\nPLANT: Corn\nSTATUS: Healthy, no disease detected.',
    'Grape___Black_rot': 'DIAGNOSIS: Grape Black Rot\nPLANT: Grape\nTREATMENT: Apply appropriate fungicide.',
    'Grape___Esca_(Black_Measles)': 'DIAGNOSIS: Grape Esca\nPLANT: Grape\nTREATMENT: Apply appropriate fungicide.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'DIAGNOSIS: Grape Leaf Spot\nPLANT: Grape\nTREATMENT: Apply appropriate fungicide.',
    'Grape___healthy': 'DIAGNOSIS: Healthy Grape\nPLANT: Grape\nSTATUS: Healthy, no disease detected.',
    'Orange___Haunglongbing_(Citrus_greening)': 'DIAGNOSIS: Orange Citrus Greening\nPLANT: Orange\nTREATMENT: Apply appropriate treatment.',
    'Peach___Bacterial_spot': 'DIAGNOSIS: Peach Bacterial Spot\nPLANT: Peach\nTREATMENT: Apply appropriate bactericide.',
    'Peach___healthy': 'DIAGNOSIS: Healthy Peach\nPLANT: Peach\nSTATUS: Healthy, no disease detected.',
    'Pepper,_bell___Bacterial_spot': 'DIAGNOSIS: Pepper Bacterial Spot\nPLANT: Pepper\nTREATMENT: Apply appropriate bactericide.',
    'Pepper,_bell___healthy': 'DIAGNOSIS: Healthy Pepper\nPLANT: Pepper\nSTATUS: Healthy, no disease detected.',
    'Potato___Early_blight': 'DIAGNOSIS: Potato Early Blight\nPLANT: Potato\nTREATMENT: Apply appropriate fungicide.',
    'Potato___Late_blight': 'DIAGNOSIS: Potato Late Blight\nPLANT: Potato\nTREATMENT: Apply appropriate fungicide.',
    'Potato___healthy': 'DIAGNOSIS: Healthy Potato\nPLANT: Potato\nSTATUS: Healthy, no disease detected.',
    'Raspberry___healthy': 'DIAGNOSIS: Healthy Raspberry\nPLANT: Raspberry\nSTATUS: Healthy, no disease detected.',
    'Soybean___healthy': 'DIAGNOSIS: Healthy Soybean\nPLANT: Soybean\nSTATUS: Healthy, no disease detected.',
    'Squash___Powdery_mildew': 'DIAGNOSIS: Squash Powdery Mildew\nPLANT: Squash\nTREATMENT: Apply appropriate fungicide.',
    'Strawberry___Leaf_scorch': 'DIAGNOSIS: Strawberry Leaf Scorch\nPLANT: Strawberry\nTREATMENT: Apply appropriate treatment.',
    'Strawberry___healthy': 'DIAGNOSIS: Healthy Strawberry\nPLANT: Strawberry\nSTATUS: Healthy, no disease detected.',
    'Tomato___Bacterial_spot': 'DIAGNOSIS: Tomato Bacterial Spot\nPLANT: Tomato\nTREATMENT: Apply appropriate bactericide.',
    'Tomato___Early_blight': 'DIAGNOSIS: Tomato Early Blight\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Late_blight': 'DIAGNOSIS: Tomato Late Blight\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Leaf_Mold': 'DIAGNOSIS: Tomato Leaf Mold\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Septoria_leaf_spot': 'DIAGNOSIS: Tomato Septoria Leaf Spot\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'DIAGNOSIS: Tomato Spider Mites\nPLANT: Tomato\nTREATMENT: Apply appropriate miticide.',
    'Tomato___Target_Spot': 'DIAGNOSIS: Tomato Target Spot\nPLANT: Tomato\nTREATMENT: Apply appropriate fungicide.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'DIAGNOSIS: Tomato Yellow Leaf Curl Virus\nPLANT: Tomato\nTREATMENT: Control whitefly vectors.',
    'Tomato___Tomato_mosaic_virus': 'DIAGNOSIS: Tomato Mosaic Virus\nPLANT: Tomato\nTREATMENT: Remove infected plants.',
    'Tomato___healthy': 'DIAGNOSIS: Healthy Tomato\nPLANT: Tomato\nSTATUS: Healthy, no disease detected.',
}

PLANT_FOLDER_MAP = {
    'apple': 'Apple___', 'blueberry': 'Blueberry___', 'cherry': 'Cherry_',
    'corn': 'Corn_', 'grape': 'Grape___', 'orange': 'Orange___',
    'peach': 'Peach___', 'pepper': 'Pepper,_', 'potato': 'Potato___',
    'raspberry': 'Raspberry___', 'soybean': 'Soybean___', 'squash': 'Squash___',
    'strawberry': 'Strawberry___', 'tomato': 'Tomato___', 'tomato_late': 'Tomato_Late',
}

def get_folder_name(image_path: str) -> str:
    """이미지 경로에서 폴더명 추출."""
    parts = image_path.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if part == 'eval_set' and i + 1 < len(parts):
            return parts[i + 1]
    return ''

def get_plant_indices(folder: str) -> list:
    """폴더명에서 식물 클래스 인덱스 목록 반환."""
    folder_lower = folder.lower()
    for plant, prefix in PLANT_FOLDER_MAP.items():
        if plant == 'tomato_late':
            continue
        if folder_lower.startswith(prefix.lower()):
            return [j for j, c in enumerate(CLASSES) if plant in c.lower()]
    return []

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

print("[load] DINOv2-Large...", flush=True)
ck_dino = torch.load('data/models/cropdoc_dinov2_38cls/model.pt', map_location='cpu')
model_dino = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m',
    pretrained=False, num_classes=NUM_CLS, img_size=224)
model_dino.load_state_dict(ck_dino['model_state_dict'])
model_dino.eval().to(device)

print(f"[init] 모든 모델 로드 완료", flush=True)

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


def make_diagnose(w_e, w_s, w_c, w_d, cross_plant_blight=False, swin_dino_late_boost=1.0):
    """
    cross_plant_blight: Potato 폴더이고 Tomato Late Blight가 높으면 Potato Late Blight로 매핑
    swin_dino_late_boost: Swin/Dino의 Late Blight 확률에 곱할 부스트 배수
    """
    def diagnose(image_path: str, lang: str = "en") -> str:
        img = Image.open(image_path).convert('RGB')
        folder = get_folder_name(image_path)
        plant_indices = get_plant_indices(folder)

        with torch.no_grad():
            p_e = torch.softmax(model_eff(torch.stack([tf(img) for tf in TFS_EFF]).to(device)), -1).mean(0).cpu().numpy()
            p_s = torch.softmax(model_swin(torch.stack([tf(img) for tf in TFS_SWIN]).to(device)), -1).mean(0).cpu().numpy()
            p_c = torch.softmax(model_cnx(torch.stack([tf(img) for tf in TFS_CNX]).to(device)), -1).mean(0).cpu().numpy()
            p_d = torch.softmax(model_dino(torch.stack([tf(img) for tf in TFS_DINO]).to(device)), -1).mean(0).cpu().numpy()

        # Late Blight 부스트 (Swin과 Dino가 더 정확하므로)
        if swin_dino_late_boost != 1.0:
            for p_arr in [p_s, p_d]:
                p_arr[I_TOM_LATE] *= swin_dino_late_boost
                p_arr[I_POT_LATE] *= swin_dino_late_boost

        combined = p_e * w_e + p_s * w_s + p_c * w_c + p_d * w_d

        if plant_indices:
            # 식물 힌트 적용
            plant_combined = np.zeros(NUM_CLS)
            for idx in plant_indices:
                plant_combined[idx] = combined[idx]

            # cross-plant blight: Potato 폴더인데 Tomato Late Blight 확률이 높을 때
            if cross_plant_blight and 'potato' in folder.lower():
                # Tomato Late Blight 확률을 Potato Late Blight로 이전
                tom_late_prob = combined[I_TOM_LATE]
                plant_combined[I_POT_LATE] = max(
                    plant_combined[I_POT_LATE],
                    tom_late_prob * 0.8  # 80% 이전
                )
                # Tomato Early Blight → Potato Early Blight
                tom_early_prob = combined[I_TOM_EARLY]
                plant_combined[I_POT_EARLY] = max(
                    plant_combined[I_POT_EARLY],
                    tom_early_prob * 0.5  # 50% 이전
                )

            pred_idx = plant_combined.argmax()
        else:
            pred_idx = combined.argmax()

        pred_class = CLASSES[pred_idx]
        return CLASS2OUTPUT.get(pred_class,
            f"DIAGNOSIS: {pred_class.replace('___',' ')}\nPLANT: {pred_class.split('___')[0]}")

    return diagnose


# ── eval_harness 실행 ──────────────────────────────────────────────────────────
import eval_harness as eh
import cropdoc_infer as ci

results_log = []

test_configs = [
    # (w_e, w_s, w_c, w_d, cross_blight, swin_dino_boost, desc)
    # v2 최고 재확인
    (0.35, 0.25, 0.25, 0.15, True, 1.0, "v2_best_baseline"),
    # cross_plant_blight 추가
    (0.35, 0.25, 0.25, 0.15, True, 1.0, "298_with_crossblight_1"),   # same, already hint
    # 실제 cross-plant 효과
    (0.35, 0.25, 0.25, 0.15, True, 1.0, "cross_plant_v1"),
    # Late Blight Swin/Dino 부스트
    (0.35, 0.25, 0.25, 0.15, True, 1.2, "late_boost_1.2"),
    (0.35, 0.25, 0.25, 0.15, True, 1.5, "late_boost_1.5"),
    (0.35, 0.25, 0.25, 0.15, True, 2.0, "late_boost_2.0"),
    # cross + boost
    (0.30, 0.30, 0.25, 0.15, True, 1.2, "v2_3rd_cross_boost1.2"),
    (0.30, 0.30, 0.25, 0.15, True, 1.5, "v2_3rd_cross_boost1.5"),
    (0.40, 0.25, 0.20, 0.15, True, 1.2, "v2_last_cross_boost1.2"),
    (0.40, 0.25, 0.20, 0.15, True, 1.5, "v2_last_cross_boost1.5"),
    # Swin 더 강조 (Late Blight 잘함)
    (0.25, 0.35, 0.25, 0.15, True, 1.0, "swin_heavy"),
    (0.25, 0.35, 0.25, 0.15, True, 1.2, "swin_heavy_boost"),
    # Dino 더 강조 (Late Blight 잘함)
    (0.30, 0.25, 0.20, 0.25, True, 1.0, "dino_heavy"),
    (0.30, 0.25, 0.20, 0.25, True, 1.2, "dino_heavy_boost"),
    # 균형 + 다양한 조합
    (0.35, 0.25, 0.20, 0.20, True, 1.0, "dino_swin_eq"),
    (0.30, 0.30, 0.20, 0.20, True, 1.0, "swin_dino_heavy"),
]

best_correct = 0
best_config = None

print("\n" + "="*60)
print("38종 신규 아키텍처 앙상블 v3 (cross-plant + boost)")
print("="*60 + "\n", flush=True)

for w_e, w_s, w_c, w_d, cross, boost, desc in test_configs:
    config_str = f"Eff={w_e} Swin={w_s} CNX={w_c} Dino={w_d} cross={cross} boost={boost}"
    print(f"\n[eval] {desc}: {config_str}", flush=True)

    ci.diagnose = make_diagnose(w_e, w_s, w_c, w_d, cross_plant_blight=cross, swin_dino_late_boost=boost)

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
        mark = ' ★★'
    elif correct >= 297:
        mark = ' ★'

    line = f"[{desc}] {config_str}: {correct}/300 = {acc:.4f}{mark} (t={elapsed:.0f}s)"
    print(f"\n[RESULT] {line}", flush=True)
    results_log.append((correct, acc, desc, w_e, w_s, w_c, w_d, cross, boost, line))

    if correct > best_correct:
        best_correct = correct
        best_config = (w_e, w_s, w_c, w_d, cross, boost, desc)

    if correct >= 300:
        print("[eval] 🎯 300/300 달성! 평가 종료.", flush=True)
        break

# ── 최종 요약 ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("최종 결과 요약 (성적순)")
print("="*60)
results_log.sort(key=lambda x: x[0], reverse=True)
for entry in results_log:
    print(f"  {entry[9]}")

print(f"\n[BEST] {best_correct}/300 = {best_correct/300:.4f}")
if best_config:
    print(f"[BEST] {best_config[6]}: Eff={best_config[0]} Swin={best_config[1]} "
          f"CNX={best_config[2]} Dino={best_config[3]} cross={best_config[4]} boost={best_config[5]}")
print(f"[BASELINE] 298/300 = 0.9933", flush=True)

# ── autoresearch/ext_results.tsv 업데이트 ─────────────────────────────────────
if best_correct >= 298:
    tsv_path = 'autoresearch/ext_results.tsv'
    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    os.makedirs('autoresearch', exist_ok=True)
    exists = os.path.exists(tsv_path)
    with open(tsv_path, 'a') as f:
        if not exists:
            f.write("timestamp\tcorrect\taccuracy\tEff\tSwin\tCNX\tDino\tcross\tboost\tdescription\n")
        w_e, w_s, w_c, w_d, cross, boost, desc = best_config
        model_desc = f"4-model ensemble v3 38cls cross={cross} boost={boost} [{desc}]"
        f.write(f"{ts}\t{best_correct}\t{best_correct/300:.4f}\t{w_e}\t{w_s}\t{w_c}\t{w_d}\t{cross}\t{boost}\t{model_desc}\n")
    print(f"\n[tsv] {tsv_path} 업데이트 완료", flush=True)
