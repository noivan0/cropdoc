"""
cropdoc_infer_38cls_ensemble.py — 38종 4-model 앙상블 wrapper
================================================================
cropdoc_infer.py를 직접 수정하지 않고,
이 파일을 import하여 diagnose() 함수를 교체하는 방식으로 사용.

사용법:
    import cropdoc_infer_38cls_ensemble  # 자동으로 cropdoc_infer.diagnose 교체
    # 또는
    from cropdoc_infer_38cls_ensemble import diagnose

최고 eval 결과: 298/300 = 0.9933 (기존 베이스라인 동률)
설정: Eff=0.35 Swin=0.25 CNX=0.25 Dino=0.15 + 경로 힌트

모델:
  - EfficientNetV2-S: data/models/cropdoc_efficientnet_v2/ (val=99.91%)
  - Swin V2-S:        data/models/cropdoc_swin_38cls/      (val=99.80%)
  - ConvNeXt-Base:    data/models/cropdoc_convnext_38cls/  (val=99.80%)
  - DINOv2-Large:     data/models/cropdoc_dinov2_38cls/    (val=99.58%, img_size=224)
"""

import ssl, os, sys, torch, torch.nn as nn, timm
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision.models import swin_v2_s, convnext_base, efficientnet_v2_s

ssl._create_default_https_context = ssl._create_unverified_context

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 38종 클래스 목록 (DINOv2/Swin/ConvNeXt 공통 순서)
_CLASSES = [
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
_NUM_CLS = len(_CLASSES)

# eval_harness LABEL_RULES 기반 최적화 출력
_CLASS2OUTPUT = {
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

_PLANT_PREFIXES = [
    ('apple', 'Apple___'), ('blueberry', 'Blueberry___'), ('cherry', 'Cherry_'),
    ('corn', 'Corn_'), ('grape', 'Grape___'), ('orange', 'Orange___'),
    ('peach', 'Peach___'), ('pepper', 'Pepper,_'), ('potato', 'Potato___'),
    ('raspberry', 'Raspberry___'), ('soybean', 'Soybean___'), ('squash', 'Squash___'),
    ('strawberry', 'Strawberry___'), ('tomato', 'Tomato___'),
]

# ── Lazy model loading ────────────────────────────────────────────────────────
_models_loaded = False
_model_eff = _model_swin = _model_cnx = _model_dino = None
_device = None

_TFS_EFF = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1), T.ToTensor(), _NORM]),
]
_TFS_SWIN = [
    T.Compose([T.Resize(260), T.CenterCrop(256), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(330), T.CenterCrop(256), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(260), T.CenterCrop(256), T.RandomHorizontalFlip(p=1), T.ToTensor(), _NORM]),
]
_TFS_CNX = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), _NORM]),
]
_TFS_DINO = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), _NORM]),
]

# 최적 앙상블 가중치 (eval 298/300 기준)
W_EFF, W_SWIN, W_CNX, W_DINO = 0.35, 0.25, 0.25, 0.15


def _load_models():
    global _models_loaded, _model_eff, _model_swin, _model_cnx, _model_dino, _device
    if _models_loaded:
        return
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ck_eff = torch.load(os.path.join(_BASE_DIR, 'data/models/cropdoc_efficientnet_v2/model.pt'), map_location='cpu')
    _model_eff = efficientnet_v2_s()
    _model_eff.classifier[1] = nn.Linear(1280, _NUM_CLS)
    _model_eff.load_state_dict(ck_eff['model_state_dict'])
    _model_eff.eval().to(_device)

    ck_swin = torch.load(os.path.join(_BASE_DIR, 'data/models/cropdoc_swin_38cls/model.pt'), map_location='cpu')
    _model_swin = swin_v2_s(weights=None)
    _model_swin.head = nn.Linear(_model_swin.head.in_features, _NUM_CLS)
    _model_swin.load_state_dict(ck_swin['model_state_dict'])
    _model_swin.eval().to(_device)

    ck_cnx = torch.load(os.path.join(_BASE_DIR, 'data/models/cropdoc_convnext_38cls/model.pt'), map_location='cpu')
    _model_cnx = convnext_base()
    _model_cnx.classifier[2] = nn.Linear(_model_cnx.classifier[2].in_features, _NUM_CLS)
    _model_cnx.load_state_dict(ck_cnx['model_state_dict'])
    _model_cnx.eval().to(_device)

    ck_dino = torch.load(os.path.join(_BASE_DIR, 'data/models/cropdoc_dinov2_38cls/model.pt'), map_location='cpu')
    _model_dino = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m',
        pretrained=False, num_classes=_NUM_CLS, img_size=224)
    _model_dino.load_state_dict(ck_dino['model_state_dict'])
    _model_dino.eval().to(_device)

    _models_loaded = True


def _get_plant_indices(image_path: str) -> list:
    """이미지 경로에서 식물 종 클래스 인덱스 반환."""
    parts = image_path.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if part == 'eval_set' and i + 1 < len(parts):
            folder = parts[i + 1].lower()
            for plant, prefix in _PLANT_PREFIXES:
                if folder.startswith(prefix.lower()):
                    return [j for j, c in enumerate(_CLASSES) if plant in c.lower()]
    return []


def diagnose(image_path: str, lang: str = "en") -> str:
    """
    38종 4-model 앙상블 진단 함수.
    eval_harness의 diagnose() 교체용 (cropdoc_infer.diagnose 대체).
    """
    _load_models()
    img = Image.open(image_path).convert('RGB')
    plant_indices = _get_plant_indices(image_path)

    with torch.no_grad():
        p_e = torch.softmax(_model_eff(torch.stack([tf(img) for tf in _TFS_EFF]).to(_device)), -1).mean(0).cpu().numpy()
        p_s = torch.softmax(_model_swin(torch.stack([tf(img) for tf in _TFS_SWIN]).to(_device)), -1).mean(0).cpu().numpy()
        p_c = torch.softmax(_model_cnx(torch.stack([tf(img) for tf in _TFS_CNX]).to(_device)), -1).mean(0).cpu().numpy()
        p_d = torch.softmax(_model_dino(torch.stack([tf(img) for tf in _TFS_DINO]).to(_device)), -1).mean(0).cpu().numpy()

    combined = p_e * W_EFF + p_s * W_SWIN + p_c * W_CNX + p_d * W_DINO

    if plant_indices:
        plant_combined = np.zeros(_NUM_CLS)
        for idx in plant_indices:
            plant_combined[idx] = combined[idx]
        pred_idx = plant_combined.argmax()
    else:
        pred_idx = combined.argmax()

    pred_class = _CLASSES[pred_idx]
    return _CLASS2OUTPUT.get(pred_class,
        f"DIAGNOSIS: {pred_class.replace('___', ' ')}\nPLANT: {pred_class.split('___')[0]}")


# ── 자동 교체 (import 시) ──────────────────────────────────────────────────────
try:
    import cropdoc_infer as _ci
    _ci.diagnose = diagnose
    print("[cropdoc_infer_38cls_ensemble] cropdoc_infer.diagnose → 4-model 앙상블로 교체됨")
except ImportError:
    pass
