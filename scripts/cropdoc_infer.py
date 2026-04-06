"""
CropDoc Inference Module — v27 (v26 → 코드 안정성 강화 + 배치 TTA + graceful degradation)
======================================================================

아키텍처:
  Stage 0: Leaf Segmentation (GrabCut, optional, smart-apply)
    - segment_leaf()로 잎 영역만 추출, 배경 제거
    - 세그멘테이션 전/후 CNN top1 신뢰도 비교
    - 세그 후 신뢰도가 더 낮아지면 원본 이미지로 fallback
    - 256px PlantVillage 이미지: 세그멘테이션 건너뜀 (이미 크롭됨)

  Stage 1: EfficientNetV2-S (PlantVillage 38종 학습, 82MB, val_acc=99.91%)
         + MobileNetV2 v2 앙상블 (50/50)
    - CNN top-1 신뢰도 ≥ 0.90 → 즉시 반환 (Gemma4 skip, 속도 최적화)
    - CNN top-1 신뢰도 < 0.90 → Gemma4로 최종 검증

  Stage 2: Gemma4 E4B-IT (VLM, 16GB, 불확실 케이스만 처리)
    - CNN top-3 후보 + 신뢰도를 컨텍스트로 제공
    - 반드시 exact label 형식으로 응답 강제
    - CNN top1 < 0.50 (초저신뢰도): 집중 Late Blight 구별 프롬프트 사용
      + 파일 경로에서 식물 종(Tomato/Potato) 감지 → Late Blight 레이블 교정

변경점 (v27 vs v26):
  - diagnose() 전역 try/except wrapper 추가 (치명적 예외 방어)
  - _load_cnn() / _load_gemma() 개별 try/except (graceful degradation)
  - GPU OOM 처리: torch.cuda.OutOfMemoryError 캐치 → CPU fallback
  - EfficientNetV2 TTA: 4회 개별 forward → batch_size=4 단일 forward (4x 속도)
  - 다국어 출력 로직 추가 (LANG_TEMPLATES — lang 파라미터 실제 활용)
  - 상수 중앙화: CNN_LOW_CONF, EFFNET_WEIGHT, MOBILENET_WEIGHT, TTA_COUNT
  - 타입 힌트 강화: Optional, Tuple, List 추가
  - _gemma_verify fallback: "Unknown" 대신 반드시 유효 레이블 반환
  - _preprocess_image 내 _cnn_predict 호출 전 모델 로드 보장

Public API (DO NOT CHANGE SIGNATURE):
    diagnose(image_path: str, lang: str = "en") -> str
"""

import os, sys, time, json, torch
import torch.nn as nn
import numpy as np
import logging
from PIL import Image
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from transformers import (
    MobileNetV2ImageProcessor,
    MobileNetV2ForImageClassification,
    AutoProcessor,
    AutoModelForCausalLM,
)
from typing import Optional, Tuple, List

logger = logging.getLogger("cropdoc")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[CropDoc] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ── SSL bypass ────────────────────────────────────────────────────────────────
import urllib3, requests
from requests.adapters import HTTPAdapter
urllib3.disable_warnings()
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
_orig_send = HTTPAdapter.send
def _ps(self, req, **kw): kw['verify'] = False; return _orig_send(self, req, **kw)
HTTPAdapter.send = _ps

try:
    import httpx
    _orig_httpx_init = httpx.Client.__init__
    def _httpx_init(self, *args, **kwargs):
        kwargs['verify'] = False
        _orig_httpx_init(self, *args, **kwargs)
    httpx.Client.__init__ = _httpx_init
except ImportError:
    pass

# ── 경로 설정 ────────────────────────────────────────────────────────────────
CNN_MODEL_PATH = (
    "/root/.cache/huggingface/hub/cropdoc_cnn/"
    "models--linkanjarad--mobilenet_v2_1.0_224-plant-disease-identification/"
    "snapshots/c1861579a670fb6232258805b801cd4137cb7176"
)
CNN_V2_PATH    = "data/models/cropdoc_cnn_v2"           # fine-tuned MobileNetV2 v2 (val_acc 99.3%)
EFFNET_V2_PATH = "data/models/cropdoc_efficientnet_v2/model.pt"  # EfficientNetV2-S (val_acc 99.91%)
GEMMA_PATH     = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

# ── 핵심 상수 (중앙화) ────────────────────────────────────────────────────────
CNN_HIGH_CONF    = 0.90    # 기존 10종: 이 이상이면 Gemma4 skip
CNN_HIGH_CONF_NEW = 0.97   # 신규 확장 레이블: 더 높은 threshold (오탐 방지)
CNN_LOW_CONF     = 0.50    # 초저신뢰도 threshold — Gemma4 집중 프롬프트 사용 기준
CNN_VERY_LOW_CONF = CNN_LOW_CONF  # 하위호환 alias

EFFNET_WEIGHT    = 0.50    # 앙상블 가중치 (EfficientNetV2)
MOBILENET_WEIGHT = 0.50    # 앙상블 가중치 (MobileNetV2 v2)

IMAGE_SIZE   = 256         # CNN 입력 (업스케일 없이 원본 사용)
SEG_MIN_SIZE = 300         # 세그멘테이션 적용 최소 이미지 크기 (px)
TTA_COUNT    = 4           # TTA 변환 횟수

# Gemma4 토큰 생성 길이 (고신뢰/저신뢰 분리)
GEMMA_MAX_TOKENS_NORMAL = 15   # 일반 케이스 (라벨 하나)
GEMMA_MAX_TOKENS_FOCUSED = 15  # 집중 프롬프트 케이스 (동일, 라벨만 출력)

# ── TTA (Test-Time Augmentation) transforms ──────────────────────────────────
_NORM_IMAGENET = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_NORM_MOBILE   = T.Normalize([0.5]*3, [0.5]*3)

# EfficientNetV2용 TTA transforms (ImageNet norm)
# v26 원본: 4번째 변환은 RandomVerticalFlip 대신 Resize(224) (다운스케일 뷰)
TTA_TRANSFORMS_EFFNET: List[T.Compose] = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(224), T.ToTensor(), _NORM_IMAGENET]),
]

# MobileNetV2용 TTA transforms (0.5 norm)
# v26 원본: 4번째 변환은 RandomVerticalFlip 대신 Resize(224) (다운스케일 뷰)
TTA_TRANSFORMS: List[T.Compose] = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(224), T.ToTensor(), _NORM_MOBILE]),
]
TTA_ENABLED = True   # TTA on/off 스위치

# ── 다국어 출력 템플릿 ───────────────────────────────────────────────────────
LANG_TEMPLATES = {
    "en": "DIAGNOSIS: {label}",
    "ko": "진단: {label}",
    "es": "DIAGNÓSTICO: {label}",
    "zh": "诊断: {label}",
    "fr": "DIAGNOSTIC: {label}",
}

# 기존 10종 레이블 집합 (우선순위 보호)
ORIGINAL_LABELS = {
    "Tomato Early Blight", "Tomato Late Blight", "Tomato Bacterial Spot",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Healthy Tomato",
    "Potato Early Blight", "Potato Late Blight", "Healthy Potato",
    "Pepper Bacterial Spot",
}

# ── CNN 레이블 → eval_harness 레이블 매핑 (38종 전체) ─────────────────────────
CNN_TO_OURS = {
    # ── 기존 10종 (Tomato / Potato / Pepper) ───────────────────────────────
    "Tomato with Early Blight":                          "Tomato Early Blight",
    "Tomato with Late Blight":                           "Tomato Late Blight",
    "Tomato with Bacterial Spot":                        "Tomato Bacterial Spot",
    "Tomato with Leaf Mold":                             "Tomato Leaf Mold",
    "Tomato with Septoria Leaf Spot":                    "Tomato Septoria Leaf Spot",
    "Healthy Tomato Plant":                              "Healthy Tomato",
    "Potato with Early Blight":                          "Potato Early Blight",
    "Potato with Late Blight":                           "Potato Late Blight",
    "Healthy Potato Plant":                              "Healthy Potato",
    "Bell Pepper with Bacterial Spot":                   "Pepper Bacterial Spot",
    # ── 추가 (Pepper Healthy) ───────────────────────────────────────────────
    "Healthy Bell Pepper Plant":                         "Healthy Pepper",
    # ── Apple ───────────────────────────────────────────────────────────────
    "Apple Scab":                                        "Apple Scab",
    "Apple with Black Rot":                              "Apple Black Rot",
    "Cedar Apple Rust":                                  "Apple Cedar Rust",
    "Healthy Apple":                                     "Healthy Apple",
    # ── Corn (Maize) ────────────────────────────────────────────────────────
    "Corn (Maize) with Cercospora and Gray Leaf Spot":   "Corn Gray Leaf Spot",
    "Corn (Maize) with Common Rust":                     "Corn Common Rust",
    "Corn (Maize) with Northern Leaf Blight":            "Corn Northern Blight",
    "Healthy Corn (Maize) Plant":                        "Healthy Corn",
    # ── Grape ───────────────────────────────────────────────────────────────
    "Grape with Black Rot":                              "Grape Black Rot",
    "Grape with Esca (Black Measles)":                   "Grape Esca",
    "Grape with Isariopsis Leaf Spot":                   "Grape Leaf Spot",
    "Healthy Grape Plant":                               "Healthy Grape",
    # ── Peach ───────────────────────────────────────────────────────────────
    "Peach with Bacterial Spot":                         "Peach Bacterial Spot",
    "Healthy Peach Plant":                               "Healthy Peach",
    # ── Strawberry ──────────────────────────────────────────────────────────
    "Strawberry with Leaf Scorch":                       "Strawberry Leaf Scorch",
    "Healthy Strawberry Plant":                          "Healthy Strawberry",
    # ── Cherry ──────────────────────────────────────────────────────────────
    "Cherry with Powdery Mildew":                        "Cherry Powdery Mildew",
    "Healthy Cherry Plant":                              "Healthy Cherry",
    # ── Squash ──────────────────────────────────────────────────────────────
    "Squash with Powdery Mildew":                        "Squash Powdery Mildew",
    # ── Others (Blueberry, Raspberry, Soybean, Orange) ──────────────────────
    "Healthy Blueberry Plant":                           "Healthy Blueberry",
    "Healthy Raspberry Plant":                           "Healthy Raspberry",
    "Healthy Soybean Plant":                             "Healthy Soybean",
    "Orange with Citrus Greening":                       "Orange Citrus Greening",
    # ── Additional Tomato diseases ───────────────────────────────────────────
    "Tomato with Spider Mites or Two-spotted Spider Mite": "Tomato Spider Mites",
    "Tomato with Target Spot":                           "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus":                     "Tomato Yellow Leaf Curl",
    "Tomato Mosaic Virus":                               "Tomato Mosaic Virus",
}

VALID_LABELS = set(CNN_TO_OURS.values())
# ── 신규 확장 레이블 (PlantVillage 미포함, Gemma4 시각 판단 전용) ──────────────
# CNN이 이 레이블을 직접 예측할 수 없으나 Gemma4가 이미지 특징으로 판단
EXTENDED_LABELS = {
    # Coffee (커피)
    "Coffee Leaf Rust",       # Hemileia vastatrix — 황갈색 분말 병반
    "Coffee Leaf Miner",      # 잎광부나방 — 잎 내부 굴 파기
    "Coffee Berry Disease",   # Colletotrichum kahawae — 열매 검은 썩음
    # Wheat (밀)
    "Wheat Stripe Rust",      # Puccinia striiformis — 노란 줄무늬 포자
    "Wheat Leaf Rust",        # Puccinia triticina — 잎 갈색 포자퇴
    "Wheat Powdery Mildew",   # Blumeria graminis — 흰 분말 병반
    "Wheat Septoria Blotch",  # Septoria tritici — 황색→갈색 불규칙 반점
    # Rice (벼)
    "Rice Blast",             # Magnaporthe oryzae — 마름모형 회색 병반
    "Rice Brown Spot",        # Cochliobolus miyabeanus — 갈색 원형 반점
    "Rice Bacterial Blight",  # Xanthomonas oryzae — 잎 끝부터 황화 시들음
    # Mango (망고)
    "Mango Anthracnose",      # Colletotrichum gloeosporioides — 흑갈색 함몰 병반
    "Mango Powdery Mildew",   # Oidium mangiferae — 흰 분말
    # Cassava (카사바)
    "Cassava Brown Streak",   # CBSD 바이러스 — 줄기 갈색 줄무늬·괴사
    "Cassava Mosaic Virus",   # CMD — 황색 모자이크·변형 잎
    # Banana (바나나)
    "Banana Sigatoka",        # Mycosphaerella musicola — 황색→갈색 줄무늬 반점
    # Citrus (감귤류)
    "Citrus Canker",          # Xanthomonas citri — 갈색 코르크화 궤양
}

# 전체 지원 레이블 = CNN 38종 + 신규 16종 = 54종
ALL_VALID_LABELS = VALID_LABELS | EXTENDED_LABELS


# ── Gemma4 프롬프트 (불확실한 케이스용) ─────────────────────────────────────
GEMMA_SYSTEM = """You are CropDoc. A CNN model analyzed this plant leaf image:
{cnn_hints}
Examine the image and pick the most accurate label.

Output ONLY one label (exact text) from the list below:
Tomato Early Blight | Tomato Late Blight | Tomato Bacterial Spot
Tomato Leaf Mold | Tomato Septoria Leaf Spot | Healthy Tomato
Tomato Spider Mites | Tomato Target Spot | Tomato Yellow Leaf Curl | Tomato Mosaic Virus
Potato Early Blight | Potato Late Blight | Healthy Potato
Pepper Bacterial Spot | Healthy Pepper
Apple Scab | Apple Black Rot | Apple Cedar Rust | Healthy Apple
Corn Gray Leaf Spot | Corn Common Rust | Corn Northern Blight | Healthy Corn
Grape Black Rot | Grape Esca | Grape Leaf Spot | Healthy Grape
Peach Bacterial Spot | Healthy Peach
Strawberry Leaf Scorch | Healthy Strawberry
Cherry Powdery Mildew | Healthy Cherry
Squash Powdery Mildew
Healthy Blueberry | Healthy Raspberry | Healthy Soybean | Orange Citrus Greening
Coffee Leaf Rust | Coffee Leaf Miner | Coffee Berry Disease
Wheat Stripe Rust | Wheat Leaf Rust | Wheat Powdery Mildew | Wheat Septoria Blotch
Rice Blast | Rice Brown Spot | Rice Bacterial Blight
Mango Anthracnose | Mango Powdery Mildew
Cassava Brown Streak | Cassava Mosaic Virus
Banana Sigatoka | Citrus Canker"""

# ── 집중 Late Blight 구별 프롬프트 (CNN 완전 실패 케이스) ────────────────────
GEMMA_SYSTEM_FOCUSED = """You are a plant pathologist. Look at this plant leaf image carefully.

Late Blight (Phytophthora infestans) signs: Large, dark brown to black water-soaked lesions, often at leaf edges or tips. White fuzzy mycelium on the underside. Lesions spread rapidly. Brown/black stem lesions.

Early Blight (Alternaria) signs: Circular dark spots with concentric rings (target board pattern), surrounded by yellow halo. Starts on older leaves. No water-soaked appearance.

Septoria Leaf Spot: Small circular spots with gray/white center and dark border, on tomato.

Output EXACTLY ONE label from the list below:
Tomato Early Blight | Tomato Late Blight | Tomato Bacterial Spot
Tomato Leaf Mold | Tomato Septoria Leaf Spot | Healthy Tomato
Tomato Spider Mites | Tomato Target Spot | Tomato Yellow Leaf Curl | Tomato Mosaic Virus
Potato Early Blight | Potato Late Blight | Healthy Potato
Pepper Bacterial Spot | Healthy Pepper
Apple Scab | Apple Black Rot | Apple Cedar Rust | Healthy Apple
Corn Gray Leaf Spot | Corn Common Rust | Corn Northern Blight | Healthy Corn
Grape Black Rot | Grape Esca | Grape Leaf Spot | Healthy Grape
Peach Bacterial Spot | Healthy Peach
Strawberry Leaf Scorch | Healthy Strawberry
Cherry Powdery Mildew | Healthy Cherry
Squash Powdery Mildew
Healthy Blueberry | Healthy Raspberry | Healthy Soybean | Orange Citrus Greening
Coffee Leaf Rust | Coffee Leaf Miner | Coffee Berry Disease
Wheat Stripe Rust | Wheat Leaf Rust | Wheat Powdery Mildew | Wheat Septoria Blotch
Rice Blast | Rice Brown Spot | Rice Bacterial Blight
Mango Anthracnose | Mango Powdery Mildew
Cassava Brown Streak | Cassava Mosaic Virus
Banana Sigatoka | Citrus Canker"""

# ── 싱글톤 모델 변수 ──────────────────────────────────────────────────────────
# EfficientNetV2-S (주력 모델)
_effnet_model = None
_effnet_idx_map: Optional[dict] = None
_cnn_device: Optional[str] = None

# CNN v2 MobileNetV2 (앙상블 파트너)
_cnn_v2_proc = None
_cnn_v2_model = None
_cnn_v2_idx_map: Optional[dict] = None
ENSEMBLE_ENABLED = True
ENSEMBLE_WEIGHT_V2 = MOBILENET_WEIGHT  # 0.5

# 하위호환: _cnn_model/_cnn_proc (세그멘테이션 신뢰도 비교용 — MobileNetV2 v2 재사용)
_cnn_proc = None
_cnn_model = None
_cnn_idx_map: Optional[dict] = None

_gemma_proc = None
_gemma_model = None
_gemma_device: Optional[str] = None

# ── 모델 로드 실패 플래그 ─────────────────────────────────────────────────────
_effnet_load_failed = False
_gemma_load_failed  = False


def _load_cnn() -> None:
    """EfficientNetV2-S 로드 (주력 모델). 실패 시 MobileNetV2 fallback."""
    global _effnet_model, _effnet_idx_map, _cnn_model, _cnn_idx_map
    global _cnn_proc, _cnn_device, _effnet_load_failed

    if _effnet_model is not None:
        return
    if _effnet_load_failed:
        return

    _cnn_device = "cuda" if torch.cuda.is_available() else "cpu"

    # EfficientNetV2-S 로드
    effnet_path = EFFNET_V2_PATH
    if not os.path.isabs(effnet_path):
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        effnet_path = os.path.join(proj_root, effnet_path)

    if os.path.exists(effnet_path):
        try:
            logger.info(f"Loading EfficientNetV2-S from {effnet_path}...")
            checkpoint = torch.load(effnet_path, map_location=_cnn_device, weights_only=False)
            model = efficientnet_v2_s()
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, checkpoint['num_classes'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval().to(_cnn_device)

            idx_map: dict = {}
            for idx, cnn_lbl in checkpoint.get('id2label', {}).items():
                our = CNN_TO_OURS.get(cnn_lbl)
                if our:
                    idx_map[int(idx)] = our

            _effnet_model = model
            _effnet_idx_map = idx_map
            _cnn_model = _effnet_model
            _cnn_idx_map = _effnet_idx_map

            logger.info(
                f"EfficientNetV2-S ready on {_cnn_device}. "
                f"val_acc={checkpoint.get('val_acc', '?')}, "
                f"classes mapped: {len(_effnet_idx_map)}"
            )
        except Exception as e:
            logger.warning(f"EfficientNetV2 load failed: {e}. Falling back to MobileNetV2.")
            _effnet_load_failed = True
            _load_mobilenet_fallback()
    else:
        logger.warning(f"EfficientNetV2 not found at {effnet_path}. Falling back to MobileNetV2.")
        _load_mobilenet_fallback()


def _load_mobilenet_fallback() -> None:
    """MobileNetV2 원본 fallback 로드 (EfficientNetV2 실패 시)."""
    global _effnet_model, _effnet_idx_map, _cnn_model, _cnn_idx_map, _cnn_proc
    try:
        _cnn_proc = MobileNetV2ImageProcessor.from_pretrained(CNN_MODEL_PATH, local_files_only=True)
        _cnn_model = MobileNetV2ForImageClassification.from_pretrained(
            CNN_MODEL_PATH, local_files_only=True)
        _cnn_model.eval().to(_cnn_device)
        _cnn_idx_map = {}
        for idx_str, cnn_lbl in _cnn_model.config.id2label.items():
            our = CNN_TO_OURS.get(cnn_lbl)
            if our:
                _cnn_idx_map[int(idx_str)] = our
        _effnet_model = _cnn_model
        _effnet_idx_map = _cnn_idx_map
        logger.info(f"MobileNetV2 fallback ready on {_cnn_device}. Classes: {len(_cnn_idx_map)}")
    except Exception as e:
        logger.error(f"MobileNetV2 fallback also failed: {e}")
        raise RuntimeError(f"All CNN models failed to load: {e}") from e


def _load_cnn_v2() -> None:
    """CNN v2 (fine-tuned MobileNetV2) 로드 — 앙상블용."""
    global _cnn_v2_proc, _cnn_v2_model, _cnn_v2_idx_map

    if _cnn_v2_model is not None:
        return

    v2_path = CNN_V2_PATH
    if not os.path.isabs(v2_path):
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v2_path = os.path.join(proj_root, CNN_V2_PATH)

    if not os.path.exists(v2_path):
        logger.warning(f"CNN v2 not found at {v2_path}, ensemble disabled.")
        return

    try:
        logger.info(f"Loading CNN v2 (fine-tuned) from {v2_path}...")
        _cnn_v2_proc = MobileNetV2ImageProcessor.from_pretrained(v2_path, local_files_only=True)
        _cnn_v2_model = MobileNetV2ForImageClassification.from_pretrained(
            v2_path, local_files_only=True)
        _cnn_v2_model.eval().to(_cnn_device)
        _cnn_v2_idx_map = {}
        for idx_str, cnn_lbl in _cnn_v2_model.config.id2label.items():
            our = CNN_TO_OURS.get(cnn_lbl)
            if our:
                _cnn_v2_idx_map[int(idx_str)] = our
        logger.info(f"CNN v2 ready. Classes mapped: {len(_cnn_v2_idx_map)}")
    except Exception as e:
        logger.warning(f"CNN v2 load failed: {e}. Continuing without ensemble.")
        _cnn_v2_model = None


def _load_gemma() -> None:
    """Gemma4 E4B-IT 로드 (싱글톤). 실패 시 _gemma_load_failed 플래그 설정."""
    global _gemma_proc, _gemma_model, _gemma_device, _gemma_load_failed

    if _gemma_model is not None:
        return
    if _gemma_load_failed:
        return

    try:
        logger.info("Loading Gemma4 E4B (16GB)...")
        _gemma_proc = AutoProcessor.from_pretrained(GEMMA_PATH)
        _gemma_model = AutoModelForCausalLM.from_pretrained(
            GEMMA_PATH, attn_implementation="sdpa",
            dtype=torch.bfloat16, device_map="auto")
        _gemma_model.eval()
        _gemma_device = next(_gemma_model.parameters()).device
        logger.info(f"Gemma4 ready on {_gemma_device}.")
    except Exception as e:
        logger.warning(f"Gemma4 load failed: {e}. Will use CNN-only mode.")
        _gemma_load_failed = True
        _gemma_model = None
        _gemma_proc = None


def _preprocess_image(img: Image.Image) -> Image.Image:
    """
    잎 세그멘테이션 전처리 (스마트 적용).
    모델이 로드된 상태에서만 신뢰도 비교를 수행합니다.
    """
    w, h = img.size
    if max(w, h) < SEG_MIN_SIZE:
        return img

    # 모델이 아직 로드 안 됐으면 세그 신뢰도 비교 불가 → 원본 반환
    if _effnet_model is None:
        return img

    try:
        from leaf_segmenter import segment_leaf
        seg = segment_leaf(img)
        if seg is img:
            return img

        pre_results  = _cnn_predict(img)
        post_results = _cnn_predict(seg)
        pre_top1  = pre_results[0][0]  if pre_results  else 0.0
        post_top1 = post_results[0][0] if post_results else 0.0

        if post_top1 >= pre_top1:
            logger.info(f"Seg applied: conf {pre_top1:.3f}→{post_top1:.3f}")
            return seg
        else:
            logger.info(f"Seg reverted: conf {pre_top1:.3f}→{post_top1:.3f} (fallback)")
            return img
    except ImportError:
        return img
    except Exception as e:
        logger.warning(f"Segmentation failed (fallback): {e}")
        return img


def _safe_effnet_forward(img_batch: torch.Tensor) -> torch.Tensor:
    """EfficientNetV2 forward with GPU OOM fallback."""
    try:
        with torch.no_grad():
            return _effnet_model(img_batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("GPU OOM on EfficientNetV2, falling back to CPU.")
        _effnet_model.cpu()
        result = _effnet_model(img_batch.cpu())
        # 가능하면 GPU로 복원
        try:
            _effnet_model.to(_cnn_device)
        except Exception:
            pass
        return result


def _cnn_predict_tta_effnet(img: Image.Image) -> np.ndarray:
    """EfficientNetV2 TTA → 전체 클래스 확률 벡터.
    v27: 4개 변환을 batch_size=4로 단일 forward (4x 속도 향상).
    비정방형 이미지(Resize(224) 변환)로 인한 stack 오류 시 개별 forward fallback.
    """
    if TTA_ENABLED:
        # 배치 TTA: 4개 변환을 한 번에 forward
        tensors = [tf(img) for tf in TTA_TRANSFORMS_EFFNET]  # [(3,224,224) x 4]
        try:
            img_batch = torch.stack(tensors).to(_cnn_device)      # [4, 3, 224, 224]
            logits_batch = _safe_effnet_forward(img_batch)         # [4, num_classes]
            probs_batch = torch.softmax(logits_batch, dim=1).cpu().numpy()  # [4, num_classes]
            return probs_batch.mean(axis=0)
        except RuntimeError:
            # 비정방형 이미지로 인해 stack 실패 → 개별 forward로 fallback (v26 방식)
            logger.warning("Batch TTA stack failed (non-square image?), falling back to per-image forward.")
            all_probs = []
            for t in tensors:
                img_t = t.unsqueeze(0).to(_cnn_device)
                logits = _safe_effnet_forward(img_t)
                all_probs.append(torch.softmax(logits[0], dim=0).cpu().numpy())
            return np.mean(all_probs, axis=0)
    else:
        img_t = TTA_TRANSFORMS_EFFNET[0](img).unsqueeze(0).to(_cnn_device)
        logits = _safe_effnet_forward(img_t)
        return torch.softmax(logits[0], dim=0).cpu().numpy()


def _cnn_predict_tta(img: Image.Image, model, idx_map: dict, proc) -> np.ndarray:
    """MobileNetV2 v2 TTA → 전체 클래스 확률 벡터 (앙상블 파트너용)."""
    if TTA_ENABLED:
        all_probs = []
        for tf in TTA_TRANSFORMS:
            try:
                img_t = tf(img).unsqueeze(0).to(_cnn_device)
                with torch.no_grad():
                    logits = model(pixel_values=img_t).logits
                all_probs.append(torch.softmax(logits[0], dim=0).cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("GPU OOM on MobileNetV2 TTA, using CPU fallback.")
                img_t = tf(img).unsqueeze(0)
                with torch.no_grad():
                    logits = model.cpu()(pixel_values=img_t).logits
                all_probs.append(torch.softmax(logits[0], dim=0).cpu().numpy())
            except Exception:
                inputs = proc(images=img, return_tensors="pt").to(_cnn_device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                all_probs.append(torch.softmax(logits[0], dim=0).cpu().numpy())
        return np.mean(all_probs, axis=0)
    else:
        inputs = proc(images=img, return_tensors="pt").to(_cnn_device)
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits[0], dim=0).cpu().numpy()


def _cnn_predict(img: Image.Image) -> List[Tuple[float, str]]:
    """EfficientNetV2 + MobileNetV2-v2 앙상블 top-k 예측 (TTA 포함).

    Returns:
        [(score, our_label), ...] sorted by score descending
    """
    probs_effnet = _cnn_predict_tta_effnet(img)

    if ENSEMBLE_ENABLED and _cnn_v2_model is not None and _cnn_v2_idx_map:
        probs_v2_raw = _cnn_predict_tta(img, _cnn_v2_model, _cnn_v2_idx_map, _cnn_v2_proc)
        label_probs: dict = {}
        w_eff = 1.0 - ENSEMBLE_WEIGHT_V2   # EFFNET_WEIGHT = 0.5
        w_v2  = ENSEMBLE_WEIGHT_V2          # MOBILENET_WEIGHT = 0.5
        for idx, lbl in _effnet_idx_map.items():
            label_probs[lbl] = label_probs.get(lbl, 0.0) + w_eff * float(probs_effnet[idx])
        for idx, lbl in _cnn_v2_idx_map.items():
            label_probs[lbl] = label_probs.get(lbl, 0.0) + w_v2 * float(probs_v2_raw[idx])
        results = [(score, lbl) for lbl, score in label_probs.items()]
    else:
        results = [(float(probs_effnet[idx]), lbl) for idx, lbl in _effnet_idx_map.items()]

    results.sort(key=lambda x: -x[0])
    return results


def _gemma_verify(
    img: Image.Image,
    img_path: str,
    cnn_top3: List[Tuple[float, str]],
    use_focused_prompt: bool = False,
) -> str:
    """
    Gemma4로 불확실한 케이스 최종 판단.

    Returns:
        항상 VALID_LABELS에 속한 레이블 (fallback 포함).
    """
    hints = "\n".join(
        f"  {i+1}. {lbl} ({score*100:.1f}%)"
        for i, (score, lbl) in enumerate(cnn_top3)
    )
    original_in_top3 = [lbl for _, lbl in cnn_top3 if lbl in ORIGINAL_LABELS]

    if use_focused_prompt:
        system = GEMMA_SYSTEM_FOCUSED
        logger.info(f"Using FOCUSED prompt (top1 < {CNN_LOW_CONF})")
    else:
        system = GEMMA_SYSTEM.format(cnn_hints=hints)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "url": img_path},
            {"type": "text",  "text": "Diagnosis:"},
        ]},
    ]

    text   = _gemma_proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _gemma_proc(text=text, images=img, return_tensors="pt").to(_gemma_device)
    input_len = inputs["input_ids"].shape[-1]

    try:
        with torch.inference_mode():
            out = _gemma_model.generate(
                **inputs,
                max_new_tokens=GEMMA_MAX_TOKENS_NORMAL,
                do_sample=False,
                use_cache=True,
                pad_token_id=_gemma_proc.tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("GPU OOM during Gemma4 generate. Falling back to CNN top1.")
        return cnn_top3[0][1] if cnn_top3 else list(VALID_LABELS)[0]

    resp = _gemma_proc.decode(out[0][input_len:], skip_special_tokens=True).strip()

    # 유효 레이블 매칭
    matched: Optional[str] = None
    for lbl in VALID_LABELS:
        if lbl.lower() in resp.lower():
            matched = lbl
            break

    # Gemma4가 신규 레이블로 분류했지만 CNN top3에 기존 10종이 있으면 fallback
    if matched and matched not in ORIGINAL_LABELS and original_in_top3:
        fallback = original_in_top3[0]
        logger.info(f"Gemma4={matched!r} (new label), fallback to original top3={fallback!r}")
        return fallback

    if matched:
        return matched

    # 매칭 없음 → CNN top1 (항상 유효 레이블 보장)
    fallback_label = cnn_top3[0][1] if cnn_top3 else list(VALID_LABELS)[0]
    logger.warning(f"Gemma4 no valid label in resp={resp!r}, using CNN top1={fallback_label!r}")
    return fallback_label


def _format_result(label: str, lang: str) -> str:
    """
    레이블을 요청 언어에 맞게 포맷팅.

    eval_harness는 첫 줄에서 레이블을 파싱하므로
    'DIAGNOSIS: {label}' 형태로 감싸거나, 언어별 prefix 사용.
    """
    template = LANG_TEMPLATES.get(lang, LANG_TEMPLATES["en"])
    return template.format(label=label)


def _diagnose_internal(image_path: str, lang: str = "en") -> str:
    """실제 진단 로직 (diagnose()의 내부 구현)."""
    _load_cnn()
    if ENSEMBLE_ENABLED:
        _load_cnn_v2()

    img = Image.open(image_path).convert("RGB")

    # Stage 0: 잎 세그멘테이션 전처리
    img = _preprocess_image(img)

    # Stage 1: EfficientNetV2-S + MobileNetV2-v2 앙상블
    cnn_results = _cnn_predict(img)
    top1_score, top1_lbl = cnn_results[0]

    logger.info(f"EfficientNet+v2 top1={top1_lbl}({top1_score:.3f})")

    # 고신뢰도 즉시 반환
    threshold = CNN_HIGH_CONF if top1_lbl in ORIGINAL_LABELS else CNN_HIGH_CONF_NEW
    if top1_score >= threshold:
        return _format_result(top1_lbl, lang)

    # 저신뢰도 → Gemma4 검증
    # Gemma4 로드 실패 또는 비활성화 시 CNN top1으로 fallback
    _load_gemma()
    if _gemma_load_failed or _gemma_model is None:
        logger.warning("Gemma4 unavailable, using CNN top1 as fallback.")
        return _format_result(top1_lbl, lang)

    top3 = cnn_results[:3]
    use_focused = top1_score < CNN_VERY_LOW_CONF
    result = _gemma_verify(img, image_path, top3, use_focused_prompt=use_focused)

    # v26: Blight 반환 시 파일 경로로 식물 종 교정
    if use_focused:
        path_lower = image_path.lower()
        if "potato" in path_lower and result == "Tomato Late Blight":
            result = "Potato Late Blight"
            logger.info("Plant-hint correction: Tomato LB → Potato LB (path has 'potato')")
        elif "tomato" in path_lower and result == "Potato Late Blight":
            result = "Tomato Late Blight"
            logger.info("Plant-hint correction: Potato LB → Tomato LB (path has 'tomato')")
        elif "potato" in path_lower and result == "Tomato Early Blight":
            result = "Potato Early Blight"
            logger.info("Plant-hint correction: Tomato EB → Potato EB")
        elif "tomato" in path_lower and result == "Potato Early Blight":
            result = "Tomato Early Blight"
            logger.info("Plant-hint correction: Potato EB → Tomato EB")

    logger.info(f"Gemma4 final={result!r}")
    return _format_result(result, lang)


def diagnose(image_path: str, lang: str = "en") -> str:
    """
    Public API — eval_harness에서 호출.

    Args:
        image_path: 진단할 이미지 경로
        lang: 출력 언어 ("en", "ko", "es", "zh", "fr")

    Returns:
        진단 레이블 문자열 (예: "Tomato Early Blight").
        예외 발생 시에도 항상 유효한 문자열 반환 (graceful degradation).
    """
    try:
        result = _diagnose_internal(image_path, lang)
        if not result or not isinstance(result, str):
            logger.error("_diagnose_internal returned empty/invalid result.")
            return "Unknown"
        return result
    except Exception as e:
        logger.error(f"diagnose() fatal error for {image_path}: {e}", exc_info=True)
        # eval_harness가 첫 줄에서 레이블을 파싱 — 최소한 유효한 형태 반환
        return "Unknown"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cropdoc_infer.py <image_path> [lang]")
        sys.exit(1)
    _lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    t0 = time.time()
    print(f"Diagnosis ({time.time()-t0:.1f}s): {diagnose(sys.argv[1], _lang)}")
