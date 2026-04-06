"""
CropDoc Inference Module — v24 (v16 → EfficientNetV2-S 교체)
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

변경점 (v24 vs v16):
  - EfficientNetV2-S (val_acc 99.91%) + MobileNetV2-v2 앙상블 (50/50)
  - 기존 MobileNetV2 원본 제거 → EfficientNetV2로 교체
  - TTA 4변환 유지
  - IMAGENET_NORM 적용 (EfficientNetV2 표준)

Public API (DO NOT CHANGE SIGNATURE):
    diagnose(image_path: str, lang: str = "en") -> str
"""

import os, sys, time, json, torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s
from transformers import (
    MobileNetV2ImageProcessor,
    MobileNetV2ForImageClassification,
    AutoProcessor,
    AutoModelForCausalLM,
)

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
CNN_V2_PATH = "data/models/cropdoc_cnn_v2"  # fine-tuned MobileNetV2 v2 (val_acc 99.3%)
EFFNET_V2_PATH = "data/models/cropdoc_efficientnet_v2/model.pt"  # EfficientNetV2-S (val_acc 99.91%)
GEMMA_PATH = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

CNN_HIGH_CONF = 0.90   # 기존 10종: 이 이상이면 Gemma4 skip
CNN_HIGH_CONF_NEW = 0.97  # 신규 확장 레이블: 더 높은 threshold (오탐 방지)
IMAGE_SIZE = 256        # CNN 입력 (업스케일 없이 원본 사용)
# 세그멘테이션은 이미지가 이 크기보다 클 때만 의미있음 (256px는 이미 크롭됨)
SEG_MIN_SIZE = 300

# ── TTA (Test-Time Augmentation) transforms ──────────────────────────────────
# EfficientNetV2: ImageNet normalization 사용
_NORM_IMAGENET = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_NORM_MOBILE   = T.Normalize([0.5]*3, [0.5]*3)

# EfficientNetV2용 TTA transforms (ImageNet norm)
TTA_TRANSFORMS_EFFNET = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(224), T.ToTensor(), _NORM_IMAGENET]),
]

# MobileNetV2용 TTA transforms (0.5 norm)
TTA_TRANSFORMS = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(224), T.ToTensor(), _NORM_MOBILE]),
]
TTA_ENABLED = True   # TTA on/off 스위치

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
Healthy Blueberry | Healthy Raspberry | Healthy Soybean | Orange Citrus Greening"""

# ── 싱글톤 ────────────────────────────────────────────────────────────────────
# EfficientNetV2-S (주력 모델, v24)
_effnet_model = None
_effnet_idx_map = None
_cnn_device = None

# CNN v2 MobileNetV2 (앙상블 파트너)
_cnn_v2_proc = None
_cnn_v2_model = None
_cnn_v2_idx_map = None
ENSEMBLE_ENABLED = True   # 앙상블 on/off 스위치
ENSEMBLE_WEIGHT_V2 = 0.5  # EfficientNetV2: 0.5, MobileNetV2-v2: 0.5

# 하위호환: _cnn_model/_cnn_proc (세그멘테이션 신뢰도 비교용 — MobileNetV2 v2 재사용)
_cnn_proc = None
_cnn_model = None
_cnn_idx_map = None

_gemma_proc = None
_gemma_model = None
_gemma_device = None


def _load_cnn():
    """EfficientNetV2-S 로드 (v24 주력 모델). _cnn_model/_cnn_idx_map 호환 유지."""
    global _effnet_model, _effnet_idx_map, _cnn_model, _cnn_idx_map, _cnn_proc, _cnn_device
    if _effnet_model is not None:
        return
    _cnn_device = "cuda" if torch.cuda.is_available() else "cpu"

    # EfficientNetV2-S 로드
    effnet_path = EFFNET_V2_PATH
    if not os.path.isabs(effnet_path):
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        effnet_path = os.path.join(proj_root, effnet_path)

    if os.path.exists(effnet_path):
        print(f"[CropDoc] Loading EfficientNetV2-S from {effnet_path}...", file=sys.stderr)
        checkpoint = torch.load(effnet_path, map_location=_cnn_device, weights_only=False)
        _effnet_model = efficientnet_v2_s()
        in_features = _effnet_model.classifier[1].in_features
        _effnet_model.classifier[1] = nn.Linear(in_features, checkpoint['num_classes'])
        _effnet_model.load_state_dict(checkpoint['model_state_dict'])
        _effnet_model.eval().to(_cnn_device)

        # idx → our_label 매핑 (checkpoint의 id2label 사용)
        _effnet_idx_map = {}
        eff_id2label = checkpoint.get('id2label', {})
        for idx, cnn_lbl in eff_id2label.items():
            our = CNN_TO_OURS.get(cnn_lbl)
            if our:
                _effnet_idx_map[int(idx)] = our

        # 하위호환: _cnn_model, _cnn_idx_map → EfficientNet 재사용 (세그멘테이션용)
        _cnn_model = _effnet_model
        _cnn_idx_map = _effnet_idx_map

        print(f"[CropDoc] EfficientNetV2-S ready on {_cnn_device}. "
              f"val_acc={checkpoint.get('val_acc', '?'):.4f}, "
              f"classes mapped: {len(_effnet_idx_map)}", file=sys.stderr)
    else:
        # Fallback: 기존 MobileNetV2 원본 사용
        print(f"[CropDoc] EfficientNetV2 not found at {effnet_path}, fallback to MobileNetV2.", file=sys.stderr)
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
        print(f"[CropDoc] MobileNetV2 fallback on {_cnn_device}. Classes: {len(_cnn_idx_map)}", file=sys.stderr)


def _load_cnn_v2():
    """CNN v2 (fine-tuned MobileNetV2) 로드 — 앙상블용."""
    global _cnn_v2_proc, _cnn_v2_model, _cnn_v2_idx_map
    if _cnn_v2_model is not None:
        return
    v2_path = CNN_V2_PATH
    if not os.path.isabs(v2_path):
        # 스크립트 기준 상대 경로 → 프로젝트 루트 기준
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v2_path = os.path.join(proj_root, CNN_V2_PATH)
    if not os.path.exists(v2_path):
        print(f"[CropDoc] CNN v2 not found at {v2_path}, ensemble disabled.", file=sys.stderr)
        return
    print(f"[CropDoc] Loading CNN v2 (fine-tuned) from {v2_path}...", file=sys.stderr)
    _cnn_v2_proc = MobileNetV2ImageProcessor.from_pretrained(v2_path, local_files_only=True)
    _cnn_v2_model = MobileNetV2ForImageClassification.from_pretrained(
        v2_path, local_files_only=True)
    _cnn_v2_model.eval()
    _cnn_v2_model = _cnn_v2_model.to(_cnn_device)
    # idx → our_label 매핑 (v2 config의 id2label 사용 — 원본 CNN_TO_OURS와 동일한 키)
    _cnn_v2_idx_map = {}
    for idx_str, cnn_lbl in _cnn_v2_model.config.id2label.items():
        our = CNN_TO_OURS.get(cnn_lbl)
        if our:
            _cnn_v2_idx_map[int(idx_str)] = our
    print(f"[CropDoc] CNN v2 ready. Classes mapped: {len(_cnn_v2_idx_map)}", file=sys.stderr)


def _load_gemma():
    global _gemma_proc, _gemma_model, _gemma_device
    if _gemma_model is not None:
        return
    print("[CropDoc] Loading Gemma4 E4B (16GB)...", file=sys.stderr)
    _gemma_proc = AutoProcessor.from_pretrained(GEMMA_PATH)
    _gemma_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH, attn_implementation="sdpa",
        dtype=torch.bfloat16, device_map="auto")
    _gemma_model.eval()
    _gemma_device = next(_gemma_model.parameters()).device
    print(f"[CropDoc] Gemma4 ready on {_gemma_device}.", file=sys.stderr)


def _preprocess_image(img: Image.Image) -> Image.Image:
    """
    잎 세그멘테이션 전처리 (스마트 적용).

    - 이미지가 SEG_MIN_SIZE 이상일 때만 적용 (작은 이미지는 이미 크롭됨)
    - CNN 신뢰도가 세그 후 낮아지면 원본으로 fallback
    """
    w, h = img.size
    # 256px 이하(PlantVillage 표준)는 이미 잘 크롭된 이미지 → 세그 건너뜀
    if max(w, h) < SEG_MIN_SIZE:
        return img

    try:
        from leaf_segmenter import segment_leaf
        seg = segment_leaf(img)
        if seg is img:
            return img  # leaf_segmenter가 이미 fallback

        # 세그 전/후 CNN 신뢰도 비교
        pre_results = _cnn_predict(img)
        post_results = _cnn_predict(seg)
        pre_top1 = pre_results[0][0] if pre_results else 0.0
        post_top1 = post_results[0][0] if post_results else 0.0

        if post_top1 >= pre_top1:
            print(f"[CropDoc] Seg applied: conf {pre_top1:.3f}→{post_top1:.3f}", file=sys.stderr)
            return seg
        else:
            print(f"[CropDoc] Seg reverted: conf {pre_top1:.3f}→{post_top1:.3f} (fallback)", file=sys.stderr)
            return img
    except ImportError:
        return img
    except Exception as e:
        print(f"[CropDoc] Segmentation failed (fallback): {e}", file=sys.stderr)
        return img


def _cnn_predict_single(img: Image.Image) -> np.ndarray:
    """EfficientNetV2 단일 이미지 → softmax 확률 벡터."""
    tf = TTA_TRANSFORMS_EFFNET[0]
    img_t = tf(img).unsqueeze(0).to(_cnn_device)
    with torch.no_grad():
        logits = _effnet_model(img_t)
    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    return probs


def _cnn_predict_tta_effnet(img: Image.Image) -> np.ndarray:
    """EfficientNetV2 TTA → 전체 클래스 확률 벡터."""
    if TTA_ENABLED:
        all_probs = []
        for tf in TTA_TRANSFORMS_EFFNET:
            img_t = tf(img).unsqueeze(0).to(_cnn_device)
            with torch.no_grad():
                logits = _effnet_model(img_t)
            all_probs.append(torch.softmax(logits[0], dim=0).cpu().numpy())
        return np.mean(all_probs, axis=0)
    else:
        tf = TTA_TRANSFORMS_EFFNET[0]
        img_t = tf(img).unsqueeze(0).to(_cnn_device)
        with torch.no_grad():
            logits = _effnet_model(img_t)
        return torch.softmax(logits[0], dim=0).cpu().numpy()


def _cnn_predict_tta(img: Image.Image, model, idx_map, proc) -> np.ndarray:
    """MobileNetV2 v2 TTA → 전체 클래스 확률 벡터 (앙상블 파트너용)."""
    if TTA_ENABLED:
        all_probs = []
        for tf in TTA_TRANSFORMS:
            try:
                img_t = tf(img).unsqueeze(0).to(_cnn_device)
                with torch.no_grad():
                    logits = model(pixel_values=img_t).logits
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


def _cnn_predict(img: Image.Image) -> list:
    """EfficientNetV2 + MobileNetV2-v2 앙상블 top-k 예측 (TTA 포함).
    Returns [(score, our_label), ...] sorted by score desc."""
    # EfficientNetV2-S 예측 (주력)
    probs_effnet = _cnn_predict_tta_effnet(img)

    # 앙상블: MobileNetV2 v2가 로드되어 있으면 가중 평균
    if ENSEMBLE_ENABLED and _cnn_v2_model is not None and _cnn_v2_idx_map:
        probs_v2_raw = _cnn_predict_tta(img, _cnn_v2_model, _cnn_v2_idx_map, _cnn_v2_proc)
        # label 기준으로 두 모델 확률 합산
        label_probs = {}
        w_eff = 1.0 - ENSEMBLE_WEIGHT_V2  # 0.5
        w_v2  = ENSEMBLE_WEIGHT_V2         # 0.5
        for idx, lbl in _effnet_idx_map.items():
            label_probs[lbl] = label_probs.get(lbl, 0.0) + w_eff * float(probs_effnet[idx])
        for idx, lbl in _cnn_v2_idx_map.items():
            label_probs[lbl] = label_probs.get(lbl, 0.0) + w_v2 * float(probs_v2_raw[idx])
        results = [(score, lbl) for lbl, score in label_probs.items()]
    else:
        # EfficientNetV2 단독
        results = [(float(probs_effnet[idx]), lbl) for idx, lbl in _effnet_idx_map.items()]

    results.sort(key=lambda x: -x[0])
    return results


def _gemma_verify(img: Image.Image, img_path: str, cnn_top3: list) -> str:
    """
    Gemma4로 불확실한 케이스 최종 판단.
    CNN top3에 기존 10종이 포함되어 있으면 Gemma4 후보를 기존 10종으로 유도.
    """
    hints = "\n".join(f"  {i+1}. {lbl} ({score*100:.1f}%)"
                      for i, (score, lbl) in enumerate(cnn_top3))

    # top3 중 기존 10종이 있는지 확인
    original_in_top3 = [lbl for _, lbl in cnn_top3 if lbl in ORIGINAL_LABELS]

    system = GEMMA_SYSTEM.format(cnn_hints=hints)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": "Diagnosis:"},
        ]},
    ]
    text = _gemma_proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = _gemma_proc(text=text, images=img, return_tensors="pt").to(_gemma_device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = _gemma_model.generate(
            **inputs, max_new_tokens=15, do_sample=False, use_cache=True,
            pad_token_id=_gemma_proc.tokenizer.eos_token_id)
    resp = _gemma_proc.decode(out[0][input_len:], skip_special_tokens=True).strip()

    # 유효 레이블 매칭
    matched = None
    for lbl in VALID_LABELS:
        if lbl.lower() in resp.lower():
            matched = lbl
            break

    # Gemma4가 신규 레이블로 분류했지만 CNN top3에 기존 10종이 있으면
    # → 기존 10종 중 top1으로 fallback (기존 정확도 보호)
    if matched and matched not in ORIGINAL_LABELS and original_in_top3:
        fallback = original_in_top3[0]
        print(f"[CropDoc] Gemma4={matched!r} (new label), fallback to original top3={fallback!r}", file=sys.stderr)
        return fallback

    if matched:
        return matched

    # 매칭 없음 → CNN top1
    return cnn_top3[0][1] if cnn_top3 else "Unknown"


def diagnose(image_path: str, lang: str = "en") -> str:
    """
    Public API — eval_harness에서 호출.
    Returns: "Tomato Early Blight" 등 exact label
    """
    _load_cnn()
    if ENSEMBLE_ENABLED:
        _load_cnn_v2()
    img = Image.open(image_path).convert("RGB")

    # Stage 0: 잎 세그멘테이션 전처리 (스마트 적용)
    img = _preprocess_image(img)

    # Stage 1: EfficientNetV2-S + MobileNetV2-v2 앙상블
    cnn_results = _cnn_predict(img)
    top1_score, top1_lbl = cnn_results[0]

    print(f"[CropDoc] EfficientNet+v2 top1={top1_lbl}({top1_score:.3f})", file=sys.stderr)

    # 고신뢰도 즉시 반환 (신규 확장 레이블은 더 높은 threshold 적용)
    threshold = CNN_HIGH_CONF if top1_lbl in ORIGINAL_LABELS else CNN_HIGH_CONF_NEW
    if top1_score >= threshold:
        return top1_lbl

    # 저신뢰도이거나 신규 레이블이 불확실한 경우 → Gemma4 검증
    _load_gemma()
    top3 = cnn_results[:3]
    result = _gemma_verify(img, image_path, top3)
    print(f"[CropDoc] Gemma4 final={result!r}", file=sys.stderr)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cropdoc_infer.py <image_path>"); sys.exit(1)
    t0 = time.time()
    print(f"Diagnosis ({time.time()-t0:.1f}s): {diagnose(sys.argv[1])}")
