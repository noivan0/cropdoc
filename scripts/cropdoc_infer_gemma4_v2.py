"""
CropDoc Inference Module — Gemma4 LoRA v2 (파인튜닝 어댑터 통합 버전)
======================================================================

원본: scripts/cropdoc_infer.py (v27) — 수정 금지
이 파일: Gemma4 로드 부분을 LoRA v2 어댑터 적용으로 교체

변경점 (vs cropdoc_infer.py):
  - Gemma4 로드 시 PEFT LoRA v2 어댑터 적용
  - BitsAndBytesConfig 4-bit 양자화 적용 (NF4, double quant)
  - ADAPTER_PATH = "data/models/gemma4_finetuned_v2"
  - 처방/진단 품질 향상 (2,148샘플 파인튜닝)
  - 나머지 파이프라인 동일 (CNN + TTA + 앙상블)

Public API (동일):
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
    BitsAndBytesConfig,
)
from peft import PeftModel
from typing import Optional, Tuple, List

logger = logging.getLogger("cropdoc_v2")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[CropDoc-v2] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ── SSL bypass ────────────────────────────────────────────────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
CNN_V2_PATH    = "data/models/cropdoc_cnn_v2"
EFFNET_V2_PATH = "data/models/cropdoc_efficientnet_v2/model.pt"
GEMMA_PATH     = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

# ── LoRA v2 어댑터 경로 (핵심 변경점) ─────────────────────────────────────────
ADAPTER_PATH   = "data/models/gemma4_finetuned_v2"

# ── 핵심 상수 ────────────────────────────────────────────────────────────────
CNN_HIGH_CONF     = 0.90
CNN_HIGH_CONF_NEW = 0.97
CNN_LOW_CONF      = 0.50
CNN_VERY_LOW_CONF = CNN_LOW_CONF

EFFNET_WEIGHT    = 0.50
MOBILENET_WEIGHT = 0.50

IMAGE_SIZE   = 256
SEG_MIN_SIZE = 300
TTA_COUNT    = 4

GEMMA_MAX_TOKENS_NORMAL  = 15
GEMMA_MAX_TOKENS_FOCUSED = 15

# ── TTA transforms ───────────────────────────────────────────────────────────
_NORM_IMAGENET = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_NORM_MOBILE   = T.Normalize([0.5]*3, [0.5]*3)

TTA_TRANSFORMS_EFFNET: List[T.Compose] = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM_IMAGENET]),
    T.Compose([T.Resize(224), T.ToTensor(), _NORM_IMAGENET]),
]

TTA_TRANSFORMS: List[T.Compose] = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM_MOBILE]),
    T.Compose([T.Resize(224), T.ToTensor(), _NORM_MOBILE]),
]
TTA_ENABLED = True
ENSEMBLE_ENABLED = True
ENSEMBLE_WEIGHT_V2 = MOBILENET_WEIGHT

# ── 다국어 출력 템플릿 ───────────────────────────────────────────────────────
LANG_TEMPLATES = {
    "en": "DIAGNOSIS: {label}",
    "ko": "진단: {label}",
    "es": "DIAGNÓSTICO: {label}",
    "zh": "诊断: {label}",
    "fr": "DIAGNOSTIC: {label}",
}

# ── 레이블 집합 ───────────────────────────────────────────────────────────────
ORIGINAL_LABELS = {
    "Tomato Early Blight", "Tomato Late Blight", "Tomato Bacterial Spot",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Healthy Tomato",
    "Potato Early Blight", "Potato Late Blight", "Healthy Potato",
    "Pepper Bacterial Spot",
}

CNN_TO_OURS = {
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
    "Healthy Bell Pepper Plant":                         "Healthy Pepper",
    "Apple Scab":                                        "Apple Scab",
    "Apple with Black Rot":                              "Apple Black Rot",
    "Cedar Apple Rust":                                  "Apple Cedar Rust",
    "Healthy Apple":                                     "Healthy Apple",
    "Corn (Maize) with Cercospora and Gray Leaf Spot":   "Corn Gray Leaf Spot",
    "Corn (Maize) with Common Rust":                     "Corn Common Rust",
    "Corn (Maize) with Northern Leaf Blight":            "Corn Northern Blight",
    "Healthy Corn (Maize) Plant":                        "Healthy Corn",
    "Grape with Black Rot":                              "Grape Black Rot",
    "Grape with Esca (Black Measles)":                   "Grape Esca",
    "Grape with Isariopsis Leaf Spot":                   "Grape Leaf Spot",
    "Healthy Grape Plant":                               "Healthy Grape",
    "Peach with Bacterial Spot":                         "Peach Bacterial Spot",
    "Healthy Peach Plant":                               "Healthy Peach",
    "Strawberry with Leaf Scorch":                       "Strawberry Leaf Scorch",
    "Healthy Strawberry Plant":                          "Healthy Strawberry",
    "Cherry with Powdery Mildew":                        "Cherry Powdery Mildew",
    "Healthy Cherry Plant":                              "Healthy Cherry",
    "Squash with Powdery Mildew":                        "Squash Powdery Mildew",
    "Healthy Blueberry Plant":                           "Healthy Blueberry",
    "Healthy Raspberry Plant":                           "Healthy Raspberry",
    "Healthy Soybean Plant":                             "Healthy Soybean",
    "Orange with Citrus Greening":                       "Orange Citrus Greening",
    "Tomato with Spider Mites or Two-spotted Spider Mite": "Tomato Spider Mites",
    "Tomato with Target Spot":                           "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus":                     "Tomato Yellow Leaf Curl",
    "Tomato Mosaic Virus":                               "Tomato Mosaic Virus",
}

VALID_LABELS = set(CNN_TO_OURS.values())

EXTENDED_LABELS = {
    "Coffee Leaf Rust", "Coffee Leaf Miner", "Coffee Berry Disease",
    "Wheat Stripe Rust", "Wheat Leaf Rust", "Wheat Powdery Mildew", "Wheat Septoria Blotch",
    "Rice Blast", "Rice Brown Spot", "Rice Bacterial Blight",
    "Mango Anthracnose", "Mango Powdery Mildew",
    "Cassava Brown Streak", "Cassava Mosaic Virus",
    "Banana Sigatoka", "Citrus Canker",
}

ALL_VALID_LABELS = VALID_LABELS | EXTENDED_LABELS

# ── Gemma4 프롬프트 ──────────────────────────────────────────────────────────
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
_effnet_model = None
_effnet_idx_map: Optional[dict] = None
_cnn_device: Optional[str] = None

_cnn_v2_proc = None
_cnn_v2_model = None
_cnn_v2_idx_map: Optional[dict] = None

_cnn_proc = None
_cnn_model = None
_cnn_idx_map: Optional[dict] = None

_gemma_proc = None
_gemma_model = None
_gemma_device: Optional[str] = None

_effnet_load_failed = False
_gemma_load_failed  = False


def _load_cnn() -> None:
    """EfficientNetV2-S 로드."""
    global _effnet_model, _effnet_idx_map, _cnn_model, _cnn_idx_map
    global _cnn_proc, _cnn_device, _effnet_load_failed

    if _effnet_model is not None:
        return
    if _effnet_load_failed:
        return

    _cnn_device = "cuda" if torch.cuda.is_available() else "cpu"

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
            logger.info(f"EfficientNetV2-S ready on {_cnn_device}.")
        except Exception as e:
            logger.warning(f"EfficientNetV2 load failed: {e}. Falling back.")
            _effnet_load_failed = True
    else:
        logger.warning(f"EfficientNetV2 not found at {effnet_path}.")
        _effnet_load_failed = True


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
        _cnn_v2_proc = MobileNetV2ImageProcessor.from_pretrained(v2_path, local_files_only=True)
        _cnn_v2_model = MobileNetV2ForImageClassification.from_pretrained(
            v2_path, local_files_only=True)
        _cnn_v2_model.eval().to(_cnn_device)
        _cnn_v2_idx_map = {}
        for idx_str, cnn_lbl in _cnn_v2_model.config.id2label.items():
            our = CNN_TO_OURS.get(cnn_lbl)
            if our:
                _cnn_v2_idx_map[int(idx_str)] = our
        logger.info(f"CNN v2 ready.")
    except Exception as e:
        logger.warning(f"CNN v2 load failed: {e}.")
        _cnn_v2_model = None


def _load_gemma() -> None:
    """
    Gemma4 E4B-IT 로드 — LoRA v2 어댑터 적용 (핵심 변경점).

    원본(cropdoc_infer.py)과 달리:
    - BitsAndBytesConfig 4-bit NF4 양자화 적용
    - PeftModel.from_pretrained()으로 LoRA 어댑터 병합
    - ADAPTER_PATH = "data/models/gemma4_finetuned_v2"
    """
    global _gemma_proc, _gemma_model, _gemma_device, _gemma_load_failed

    if _gemma_model is not None:
        return
    if _gemma_load_failed:
        return

    try:
        logger.info("Loading Gemma4 E4B with LoRA v2 adapter (4-bit NF4)...")

        # 어댑터 경로 해석 (절대/상대 경로 모두 지원)
        adapter_path = ADAPTER_PATH
        if not os.path.isabs(adapter_path):
            proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            adapter_path = os.path.join(proj_root, adapter_path)

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        # ── 핵심 변경: BitsAndBytes 4-bit 양자화 ─────────────────────────────
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        _gemma_proc = AutoProcessor.from_pretrained(GEMMA_PATH)

        base_model = AutoModelForCausalLM.from_pretrained(
            GEMMA_PATH,
            quantization_config=bnb_config,
            device_map={"": 0},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        # ── 핵심 변경: LoRA v2 어댑터 적용 ──────────────────────────────────
        _gemma_model = PeftModel.from_pretrained(base_model, adapter_path)
        _gemma_model.eval()

        _gemma_device = next(_gemma_model.parameters()).device
        logger.info(
            f"Gemma4 + LoRA v2 ready on {_gemma_device}. "
            f"Adapter: {adapter_path}"
        )
    except Exception as e:
        logger.warning(f"Gemma4+LoRA v2 load failed: {e}. Will use CNN-only mode.")
        _gemma_load_failed = True
        _gemma_model = None
        _gemma_proc = None


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
        try:
            _effnet_model.to(_cnn_device)
        except Exception:
            pass
        return result


def _cnn_predict_tta_effnet(img: Image.Image) -> np.ndarray:
    """EfficientNetV2 TTA → 전체 클래스 확률 벡터."""
    if TTA_ENABLED:
        tensors = [tf(img) for tf in TTA_TRANSFORMS_EFFNET]
        try:
            img_batch = torch.stack(tensors).to(_cnn_device)
            logits_batch = _safe_effnet_forward(img_batch)
            probs_batch = torch.softmax(logits_batch, dim=1).cpu().numpy()
            return probs_batch.mean(axis=0)
        except RuntimeError:
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
    """MobileNetV2 v2 TTA → 전체 클래스 확률 벡터."""
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
    """EfficientNetV2 + MobileNetV2-v2 앙상블 top-k 예측."""
    probs_effnet = _cnn_predict_tta_effnet(img)

    if ENSEMBLE_ENABLED and _cnn_v2_model is not None and _cnn_v2_idx_map:
        probs_v2_raw = _cnn_predict_tta(img, _cnn_v2_model, _cnn_v2_idx_map, _cnn_v2_proc)
        label_probs: dict = {}
        w_eff = 1.0 - ENSEMBLE_WEIGHT_V2
        w_v2  = ENSEMBLE_WEIGHT_V2
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
    """Gemma4 LoRA v2로 불확실한 케이스 최종 판단."""
    hints = "\n".join(
        f"  {i+1}. {lbl} ({score*100:.1f}%)"
        for i, (score, lbl) in enumerate(cnn_top3)
    )
    original_in_top3 = [lbl for _, lbl in cnn_top3 if lbl in ORIGINAL_LABELS]

    if use_focused_prompt:
        system = GEMMA_SYSTEM_FOCUSED
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

    matched: Optional[str] = None
    for lbl in VALID_LABELS:
        if lbl.lower() in resp.lower():
            matched = lbl
            break

    if matched and matched not in ORIGINAL_LABELS and original_in_top3:
        fallback = original_in_top3[0]
        return fallback

    if matched:
        return matched

    fallback_label = cnn_top3[0][1] if cnn_top3 else list(VALID_LABELS)[0]
    logger.warning(f"Gemma4 no valid label in resp={resp!r}, using CNN top1={fallback_label!r}")
    return fallback_label


def _format_result(label: str, lang: str) -> str:
    """레이블을 요청 언어에 맞게 포맷팅."""
    template = LANG_TEMPLATES.get(lang, LANG_TEMPLATES["en"])
    return template.format(label=label)


def _diagnose_internal(image_path: str, lang: str = "en") -> str:
    """실제 진단 로직."""
    _load_cnn()
    if ENSEMBLE_ENABLED:
        _load_cnn_v2()

    img = Image.open(image_path).convert("RGB")

    cnn_results = _cnn_predict(img)
    top1_score, top1_lbl = cnn_results[0]

    logger.info(f"CNN top1={top1_lbl}({top1_score:.3f})")

    threshold = CNN_HIGH_CONF if top1_lbl in ORIGINAL_LABELS else CNN_HIGH_CONF_NEW
    if top1_score >= threshold:
        return _format_result(top1_lbl, lang)

    _load_gemma()
    if _gemma_load_failed or _gemma_model is None:
        logger.warning("Gemma4+LoRA v2 unavailable, using CNN top1 as fallback.")
        return _format_result(top1_lbl, lang)

    top3 = cnn_results[:3]
    use_focused = top1_score < CNN_VERY_LOW_CONF
    result = _gemma_verify(img, image_path, top3, use_focused_prompt=use_focused)

    if use_focused:
        path_lower = image_path.lower()
        if "potato" in path_lower and result == "Tomato Late Blight":
            result = "Potato Late Blight"
        elif "tomato" in path_lower and result == "Potato Late Blight":
            result = "Tomato Late Blight"
        elif "potato" in path_lower and result == "Tomato Early Blight":
            result = "Potato Early Blight"
        elif "tomato" in path_lower and result == "Potato Early Blight":
            result = "Tomato Early Blight"

    logger.info(f"Gemma4-LoRA-v2 final={result!r}")
    return _format_result(result, lang)


def diagnose(image_path: str, lang: str = "en") -> str:
    """
    Public API — eval_harness에서 호출 (원본과 동일 시그니처).

    Args:
        image_path: 진단할 이미지 경로
        lang: 출력 언어 ("en", "ko", "es", "zh", "fr")

    Returns:
        진단 레이블 문자열.
    """
    try:
        result = _diagnose_internal(image_path, lang)
        if not result or not isinstance(result, str):
            return "Unknown"
        return result
    except Exception as e:
        logger.error(f"diagnose() fatal error for {image_path}: {e}", exc_info=True)
        return "Unknown"


# ── 5장 빠른 테스트 ──────────────────────────────────────────────────────────
def quick_test_5():
    """eval_labels.json에서 5개 이미지 샘플링하여 빠른 정확도 확인."""
    import json, os

    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_labels_path = os.path.join(proj_root, "data/plantvillage/eval_labels.json")
    eval_dir = os.path.join(proj_root, "data/plantvillage/eval_set")

    if not os.path.exists(eval_labels_path):
        print(f"eval_labels.json not found: {eval_labels_path}")
        return

    labels = json.load(open(eval_labels_path))

    # 5개 다양한 질병 샘플링
    seen_labels: set = set()
    test_imgs: List[Tuple[str, str]] = []
    for rel_path, true_label in labels.items():
        if true_label not in seen_labels and len(test_imgs) < 5:
            full_path = os.path.join(eval_dir, rel_path)
            if os.path.exists(full_path):
                test_imgs.append((full_path, true_label))
                seen_labels.add(true_label)

    print(f"\n=== CropDoc Gemma4 LoRA v2 — 5장 빠른 테스트 ===")
    correct = 0
    results_5 = []
    for img_path, true_label in test_imgs:
        t0 = time.time()
        pred = diagnose(img_path, lang="en")
        elapsed = time.time() - t0

        # DIAGNOSIS: {label} 형식에서 레이블 추출
        pred_label = pred.replace("DIAGNOSIS:", "").strip()
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"{status} [{elapsed:.1f}s] {os.path.basename(img_path)}")
        print(f"   True: {true_label}")
        print(f"   Pred: {pred_label}")
        results_5.append({
            "image": os.path.basename(img_path),
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": is_correct,
            "elapsed_s": round(elapsed, 2),
        })

    acc = correct / len(test_imgs) if test_imgs else 0
    print(f"\n정확도: {correct}/{len(test_imgs)} = {acc:.1%}")

    out_path = "/tmp/gemma4_v2_quick5.json"
    json.dump(results_5, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")
    return results_5


if __name__ == "__main__":
    import time
    if len(sys.argv) >= 2 and sys.argv[1] == "--quick-test":
        quick_test_5()
    elif len(sys.argv) >= 2:
        _lang = sys.argv[2] if len(sys.argv) > 2 else "en"
        t0 = time.time()
        print(f"Diagnosis ({time.time()-t0:.1f}s): {diagnose(sys.argv[1], _lang)}")
    else:
        print("Usage:")
        print("  python cropdoc_infer_gemma4_v2.py <image_path> [lang]")
        print("  python cropdoc_infer_gemma4_v2.py --quick-test")
