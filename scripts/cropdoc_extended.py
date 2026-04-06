"""
CropDoc 확장 진단 모듈 — 신규 34종 전용
기존 cropdoc_infer.py와 독립적으로 운영

앙상블 구성:
  - EfficientNetV2-S (expD): val_acc=0.9749
  - Swin Transformer V2-S (expQ): val_acc=0.9872 ★ 최고
  - ConvNeXt-Base (expT): val_acc=0.9828
  앙상블 결과: 84/85 = 98.8%
"""
import os, sys, torch, torch.nn as nn, glob
from PIL import Image
import torchvision.transforms as T
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# 모델 경로 설정
EXTENDED_MODEL_PATH = "data/models/cropdoc_ext_expD/model.pt"      # EfficientNetV2-S, val=0.9749
SWIN_MODEL_PATH     = "data/models/cropdoc_ext_expQ/model.pt"      # Swin V2-S, val=0.9872 ★
CONVNEXT_MODEL_PATH = "data/models/cropdoc_ext_expT/model.pt"      # ConvNeXt-Base, val=0.9828

# 앙상블 가중치 (합=1.0)
W_EFF   = 0.25   # EfficientNetV2-S
W_SWIN  = 0.50   # Swin V2-S (가장 높은 비중)
W_CNX   = 0.25   # ConvNeXt-Base

_ext_model  = None   # EfficientNetV2-S
_swin_model = None   # Swin V2-S
_cnx_model  = None   # ConvNeXt-Base
_ext_id2label = {}
_ext_num_old = 38
_sorted_classes = []

_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# EfficientNetV2 TTA (224px, 4변환)
TFS_EFF = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(448), T.CenterCrop(224), T.ToTensor(), _NORM]),
]

# Swin V2-S TTA (256px, 3변환)
TFS_SWIN = [
    T.Compose([T.Resize(260), T.CenterCrop(256), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(330), T.CenterCrop(256), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(260), T.CenterCrop(256), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM]),
]

# ConvNeXt-Base TTA (224px, 3변환)
TFS_CNX = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(448), T.CenterCrop(224), T.ToTensor(), _NORM]),
]


def _load():
    global _ext_model, _swin_model, _cnx_model
    global _ext_id2label, _ext_num_old, _sorted_classes

    if _ext_model is not None:
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── EfficientNetV2-S ──
    from torchvision.models import efficientnet_v2_s, swin_v2_s, convnext_base
    ck_eff = torch.load(EXTENDED_MODEL_PATH, map_location='cpu')
    model_eff = efficientnet_v2_s()
    model_eff.classifier[1] = nn.Linear(1280, ck_eff['num_classes'])
    model_eff.load_state_dict(ck_eff['model_state_dict'])
    model_eff.eval().to(device)
    _ext_model = model_eff
    _ext_id2label = ck_eff['id2label']
    _ext_num_old = ck_eff.get('num_old', 38)
    new_classes = ck_eff.get('new_classes', [])
    _sorted_classes = sorted(new_classes)

    # ── Swin V2-S ──
    try:
        ck_swin = torch.load(SWIN_MODEL_PATH, map_location='cpu')
        model_swin = swin_v2_s(weights=None)
        model_swin.head = nn.Linear(model_swin.head.in_features, len(_sorted_classes))
        model_swin.load_state_dict(ck_swin['model_state_dict'])
        model_swin.eval().to(device)
        _swin_model = model_swin
        logger.info(f"Swin V2-S loaded: val_acc={ck_swin.get('val_acc_new', '?')}")
    except Exception as e:
        logger.warning(f"Swin 로드 실패, EfficientNet 단독 사용: {e}")

    # ── ConvNeXt-Base ──
    try:
        ck_cnx = torch.load(CONVNEXT_MODEL_PATH, map_location='cpu')
        model_cnx = convnext_base()
        model_cnx.classifier[2] = nn.Linear(model_cnx.classifier[2].in_features, len(_sorted_classes))
        model_cnx.load_state_dict(ck_cnx['model_state_dict'])
        model_cnx.eval().to(device)
        _cnx_model = model_cnx
        logger.info(f"ConvNeXt-Base loaded: val_acc={ck_cnx.get('val_acc_new', '?')}")
    except Exception as e:
        logger.warning(f"ConvNeXt 로드 실패, 단독 사용: {e}")

    logger.info(f"Extended ensemble loaded: {len(_sorted_classes)} new classes")


def diagnose_extended(image_path: str, use_tta: bool = True) -> Tuple[float, str]:
    """
    신규 34종에 대한 앙상블 진단.
    앙상블: EfficientNetV2(25%) + SwinV2-S(50%) + ConvNeXt(25%)
    반환: (confidence, 레이블)
    """
    _load()
    device = next(_ext_model.parameters()).device
    img = Image.open(image_path).convert('RGB')

    probs_list = []

    # EfficientNetV2-S
    t_eff = torch.stack([tf(img) for tf in TFS_EFF]).to(device)
    with torch.no_grad():
        p_eff = torch.softmax(_ext_model(t_eff), -1).mean(0).cpu().numpy()
    p_new_eff = p_eff[_ext_num_old:]  # 신규 클래스만
    probs_list.append((p_new_eff, W_EFF))

    # Swin V2-S
    if _swin_model is not None:
        t_swin = torch.stack([tf(img) for tf in TFS_SWIN]).to(device)
        with torch.no_grad():
            p_swin = torch.softmax(_swin_model(t_swin), -1).mean(0).cpu().numpy()
        probs_list.append((p_swin, W_SWIN))
    else:
        # Swin 없으면 EfficientNet 가중치 높임
        probs_list[0] = (probs_list[0][0], W_EFF + W_SWIN)

    # ConvNeXt-Base
    if _cnx_model is not None:
        t_cnx = torch.stack([tf(img) for tf in TFS_CNX]).to(device)
        with torch.no_grad():
            p_cnx = torch.softmax(_cnx_model(t_cnx), -1).mean(0).cpu().numpy()
        probs_list.append((p_cnx, W_CNX))
    else:
        # ConvNeXt 없으면 EfficientNet 가중치 합산
        w_total = sum(w for _, w in probs_list)
        probs_list = [(p, w/w_total) for p, w in probs_list]

    # 가중 앙상블
    combined = np.zeros(len(_sorted_classes))
    for probs, weight in probs_list:
        n = min(len(probs), len(_sorted_classes))
        combined[:n] += probs[:n] * weight

    best_idx = combined.argmax()
    best_conf = float(combined[best_idx])
    best_label = _sorted_classes[best_idx]

    return best_conf, best_label


def get_new_classes() -> List[str]:
    """신규 지원 클래스 목록"""
    _load()
    return list(_sorted_classes)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        conf, label = diagnose_extended(sys.argv[1])
        print(f"DIAGNOSIS: {label} (confidence: {conf:.3f})")
    else:
        print("사용법: python3 cropdoc_extended.py <image_path>")
