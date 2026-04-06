"""
CropDoc — Data Processing Pipeline
====================================
Handles:
  - PlantVillage dataset loading (HuggingFace datasets)
  - Image preprocessing for Gemma 4 E4B API
  - Audio preprocessing
  - Response parsing into structured results
  - Sample image download for offline/demo use
"""

import base64
import hashlib
import io
import json
import logging
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input Validation Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
MAX_IMAGE_SIZE_MB = 20


def _image_hash(image_path: str) -> str:
    """이미지 파일의 MD5 해시 반환."""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def validate_inputs(image_path: str, audio_path: Optional[str] = None) -> None:
    """
    이미지 및 오디오 입력 파일 검증.

    Args:
        image_path: 작물 이미지 경로
        audio_path: 오디오 파일 경로 (선택)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        ValueError: 지원하지 않는 형식이거나 파일 크기 초과 시
    """
    # 이미지 검증
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일 없음: {image_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"지원하지 않는 형식: {path.suffix}. 지원: {SUPPORTED_EXTENSIONS}"
        )
    if path.stat().st_size > MAX_IMAGE_SIZE_MB * 1e6:
        raise ValueError(
            f"이미지 크기 초과: {path.stat().st_size / 1e6:.1f}MB (최대 {MAX_IMAGE_SIZE_MB}MB)"
        )

    # 오디오 검증
    if audio_path:
        apath = Path(audio_path)
        if not apath.exists():
            raise FileNotFoundError(f"오디오 파일 없음: {audio_path}")
        if apath.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"지원하지 않는 오디오 형식: {apath.suffix}. 지원: {AUDIO_EXTENSIONS}"
            )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_IMAGE_SIZE = (512, 512)   # Resize target (preserves aspect ratio if not square)
MAX_AUDIO_DURATION_SEC = 30      # Clip audio to this length
SAMPLE_DIR = Path(__file__).parent.parent / "samples"

# PlantVillage disease classes (subset — 38 classes in full dataset)
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Sample images from GitHub (CC0 PlantVillage subset hosted publicly)
# Using a public CDN mirror for demo purposes
SAMPLE_IMAGE_URLS = {
    "tomato_late_blight": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/"
        "Phytophthora_infestans_on_tomato.jpg/320px-Phytophthora_infestans_on_tomato.jpg"
    ),
    "corn_rust": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/"
        "Common_rust_%28Puccinia_sorghi%29_on_maize.jpg/320px-Common_rust_%28Puccinia_sorghi%29_on_maize.jpg"
    ),
    "apple_scab": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/"
        "Apple_scab_2.jpg/320px-Apple_scab_2.jpg"
    ),
    "healthy_leaf": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/"
        "Simple_leaf_illustration_%28aka%29.jpg/320px-Simple_leaf_illustration_%28aka%29.jpg"
    ),
}


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, target_size: tuple = TARGET_IMAGE_SIZE) -> Optional[Image.Image]:
    """
    Load and preprocess an image for the Gemma API.

    Steps:
      1. Open with PIL
      2. Convert to RGB (handles RGBA, grayscale, etc.)
      3. Resize to target_size (LANCZOS for quality)
      4. Return PIL Image (Gemma API accepts PIL directly)

    Args:
        image_path: Path to the input image.
        target_size: (width, height) tuple.

    Returns:
        PIL.Image or None on failure.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(target_size, Image.LANCZOS)
        logger.debug("Image preprocessed: %s → %s", img.size, target_size)
        return img_resized
    except Exception as exc:  # noqa: BLE001
        logger.error("Image preprocessing failed for %s: %s", image_path, exc)
        return None


def image_to_base64(image_path: str, target_size: tuple = TARGET_IMAGE_SIZE) -> Optional[str]:
    """
    Convert image file to base64 string (for JSON serialisation / logging).

    Returns:
        Base64-encoded JPEG string, or None on failure.
    """
    img = preprocess_image(image_path, target_size)
    if img is None:
        return None
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL Image to base64 JPEG string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Audio Preprocessing
# ---------------------------------------------------------------------------

def preprocess_audio(audio_path: str, max_duration: int = MAX_AUDIO_DURATION_SEC) -> Optional[bytes]:
    """
    Load and preprocess an audio file.

    For WAV files, clips to max_duration seconds.
    For other formats, returns raw bytes (Gemma API handles them natively).

    Args:
        audio_path: Path to audio file.
        max_duration: Maximum duration in seconds.

    Returns:
        Audio bytes, or None on failure.
    """
    path = Path(audio_path)
    if not path.exists():
        logger.warning("Audio file not found: %s", audio_path)
        return None

    suffix = path.suffix.lower()

    if suffix == ".wav":
        return _clip_wav(audio_path, max_duration)

    # For MP3/OGG/FLAC etc., return raw bytes directly
    try:
        with open(audio_path, "rb") as f:
            return f.read()
    except OSError as exc:
        logger.error("Cannot read audio: %s", exc)
        return None


def _clip_wav(wav_path: str, max_duration: int) -> Optional[bytes]:
    """Clip a WAV file to max_duration seconds and return bytes."""
    try:
        with wave.open(wav_path, "rb") as wf:
            frame_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            max_frames = frame_rate * max_duration
            frames = wf.readframes(max_frames)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out_wf:
            out_wf.setnchannels(n_channels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(frame_rate)
            out_wf.writeframes(frames)

        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.error("WAV clipping failed: %s", exc)
        return None


def generate_silent_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a minimal silent WAV for testing audio pipeline without a real mic.

    Returns:
        WAV file as bytes.
    """
    n_samples = int(sample_rate * duration_sec)
    samples = np.zeros(n_samples, dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Result Parsing
# ---------------------------------------------------------------------------

def parse_diagnosis(raw_text: str) -> dict:
    """
    Parse the free-form model response into a structured result dict.

    Expected sections (best-effort extraction via heuristics):
      - disease_name
      - severity
      - treatment
      - prevention
      - cost_estimate

    Args:
        raw_text: Full markdown response from the model.

    Returns:
        Structured dict.
    """
    lines = raw_text.splitlines()
    sections = {
        "disease_name": "",
        "severity": "",
        "treatment": [],
        "prevention": [],
        "cost_estimate": "",
        "full_text": raw_text,
    }

    current_section = None

    # Section header keywords → field name mapping
    section_map = {
        # English
        "disease": "disease_name",
        "pest": "disease_name",
        "identification": "disease_name",
        "severity": "severity",
        "treatment": "treatment",
        "recommendation": "treatment",
        "prevention": "prevention",
        "tip": "prevention",
        "cost": "cost_estimate",
        # Swahili
        "utambuzi": "disease_name",
        "ukali": "severity",
        "matibabu": "treatment",
        "kuzuia": "prevention",
        "gharama": "cost_estimate",
        # Hindi
        "पहचान": "disease_name",
        "गंभीरता": "severity",
        "उपचार": "treatment",
        "रोकथाम": "prevention",
        "लागत": "cost_estimate",
        # Bengali
        "শনাক্ত": "disease_name",
        "তীব্রতা": "severity",
        "চিকিৎসা": "treatment",
        "প্রতিরোধ": "prevention",
        "খরচ": "cost_estimate",
    }

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect section headers (## or bold)
        if stripped.startswith("#") or stripped.startswith("**"):
            header_lower = stripped.lstrip("#").strip().lower().replace("**", "")
            for keyword, field in section_map.items():
                if keyword in header_lower:
                    current_section = field
                    break
            else:
                current_section = None
            continue

        if current_section:
            content = stripped.lstrip("- •*").strip()
            if not content:
                continue

            if current_section in ("treatment", "prevention"):
                sections[current_section].append(content)
            elif current_section == "disease_name" and not sections["disease_name"]:
                sections["disease_name"] = content
            elif current_section == "severity" and not sections["severity"]:
                # Extract keyword
                for kw in ("CRITICAL", "MODERATE", "MONITOR", "HATARI", "WASTANI",
                           "FUATILIA", "गंभीर", "मध्यम", "निगरानी", "জটিল", "মাঝারি", "পর্যবেক্ষণ"):
                    if kw in stripped.upper() or kw in stripped:
                        sections["severity"] = kw
                        break
                else:
                    sections["severity"] = content[:60]
            elif current_section == "cost_estimate" and not sections["cost_estimate"]:
                sections["cost_estimate"] = content

    # Flatten list fields for display
    sections["treatment_text"] = "\n".join(f"• {t}" for t in sections["treatment"])
    sections["prevention_text"] = "\n".join(f"• {p}" for p in sections["prevention"])

    return sections


# ---------------------------------------------------------------------------
# PlantVillage Dataset Loading
# ---------------------------------------------------------------------------

def load_plantvillage_sample(n_samples: int = 5, split: str = "train") -> list:
    """
    Load sample images from the PlantVillage dataset via HuggingFace Datasets.

    Args:
        n_samples: Number of samples to load.
        split:     Dataset split ('train' or 'test').

    Returns:
        List of dicts: [{"image": PIL.Image, "label": int, "label_name": str}, ...]

    Note:
        Requires `datasets` package and internet connection for first load.
        Subsequent loads use HuggingFace cache.
    """
    try:
        from datasets import load_dataset  # lazy import

        logger.info("Loading PlantVillage dataset (first run downloads ~1.4 GB)...")
        ds = load_dataset("AI-Lab-Makerere/beans", split=split, trust_remote_code=True)

        samples = []
        for i, item in enumerate(ds):
            if i >= n_samples:
                break
            samples.append({
                "image": item["image"],
                "label": item["labels"],
                "label_name": ds.features["labels"].names[item["labels"]],
            })
        logger.info("Loaded %d PlantVillage samples.", len(samples))
        return samples

    except Exception as exc:  # noqa: BLE001
        logger.warning("HuggingFace dataset load failed (%s). Using local samples.", exc)
        return []


def load_local_samples(sample_dir: Optional[str] = None) -> list:
    """
    Load sample images from the local samples/ directory.

    Returns:
        List of dicts: [{"image": PIL.Image, "path": str, "label_name": str}, ...]
    """
    base = Path(sample_dir) if sample_dir else SAMPLE_DIR
    samples = []
    if not base.exists():
        logger.warning("Sample directory not found: %s", base)
        return samples

    for img_file in sorted(base.glob("*.jpg")) + sorted(base.glob("*.png")):
        try:
            img = Image.open(img_file).convert("RGB")
            samples.append({
                "image": img,
                "path": str(img_file),
                "label_name": img_file.stem.replace("_", " ").title(),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot load %s: %s", img_file, exc)

    return samples


# ---------------------------------------------------------------------------
# Sample Image Download
# ---------------------------------------------------------------------------

def download_sample_images(output_dir: Optional[str] = None, timeout: int = 15) -> list:
    """
    Download a small set of sample crop disease images from public sources.

    These are used for offline demo / Kaggle notebook without internet.
    Images are CC-licensed or public domain.

    Args:
        output_dir: Where to save images. Defaults to samples/ directory.
        timeout:    HTTP request timeout in seconds.

    Returns:
        List of saved file paths.
    """
    out = Path(output_dir) if output_dir else SAMPLE_DIR
    out.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, url in SAMPLE_IMAGE_URLS.items():
        dest = out / f"{name}.jpg"
        if dest.exists():
            logger.info("Sample already exists: %s", dest)
            saved.append(str(dest))
            continue

        try:
            logger.info("Downloading sample image: %s", name)
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()

            # Decode and re-save as JPEG to standardise format
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img_resized = img.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
            img_resized.save(dest, format="JPEG", quality=85)
            saved.append(str(dest))
            logger.info("Saved: %s", dest)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download %s from %s: %s", name, url, exc)

    return saved


def create_dummy_sample(output_dir: Optional[str] = None) -> str:
    """
    Create a synthetic green-leaf dummy image for testing without downloads.
    Useful in fully offline / sandboxed environments.

    Returns:
        Path to the saved dummy image.
    """
    out = Path(output_dir) if output_dir else SAMPLE_DIR
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "dummy_healthy_leaf.jpg"

    # Simple 512x512 green gradient
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[:, :, 1] = np.linspace(80, 200, 512, dtype=np.uint8)   # Green channel gradient
    arr[:, :, 0] = 30
    arr[:, :, 2] = 20
    img = Image.fromarray(arr)
    img.save(dest, format="JPEG")
    logger.info("Dummy sample created: %s", dest)
    return str(dest)


# ---------------------------------------------------------------------------
# DiagnosisPipeline — 캐시 + 입력 검증 통합 래퍼
# ---------------------------------------------------------------------------

class DiagnosisPipeline:
    """
    CropDoctorModel 래퍼: 진단 결과 캐싱 및 입력 검증을 통합 관리.

    - 동일 이미지 + 언어 조합에 대한 재진단 방지 (인메모리 캐시)
    - 오디오 포함 시 캐시 미사용 (동적 컨텍스트)
    - 입력 파일 검증 (형식, 존재 여부, 크기)
    """

    def __init__(self, model):
        """
        Args:
            model: CropDoctorModel 인스턴스
        """
        self._model = model
        self._cache: dict = {}  # {(img_hash, lang): result}
        logger.info("DiagnosisPipeline 초기화 완료 (캐시 활성화)")

    def diagnose(
        self,
        image_path: str,
        language: str = "en",
        audio_path: Optional[str] = None,
    ) -> dict:
        """
        작물 이미지 진단 (캐시 히트 시 재추론 생략).

        Args:
            image_path:  작물 이미지 경로
            language:    ISO 639-1 언어 코드 (기본: "en")
            audio_path:  오디오 파일 경로 (선택, 캐시 무효화)

        Returns:
            dict: CropDoctorModel.analyze_image / analyze_with_audio 와 동일 구조
        """
        # 입력 검증
        try:
            validate_inputs(image_path, audio_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error("입력 검증 실패: %s", e)
            return {
                "language": language,
                "diagnosis": str(e),
                "disease": "Unknown",
                "severity": "UNKNOWN",
                "audio_included": False,
                "error": str(e),
            }

        # 캐시 확인 (오디오 없는 경우만)
        if audio_path is None:
            cache_key = (_image_hash(image_path), language)
            if cache_key in self._cache:
                logger.info("캐시 히트: %s [%s]", image_path, language)
                return self._cache[cache_key]

        # 실제 추론
        result = self._run_diagnosis(image_path, language, audio_path)

        # 캐시 저장 (오디오 없는 경우만)
        if audio_path is None:
            self._cache[cache_key] = result

        return result

    def _run_diagnosis(
        self,
        image_path: str,
        language: str,
        audio_path: Optional[str],
    ) -> dict:
        """모델 추론 실행."""
        if audio_path:
            return self._model.analyze_with_audio(image_path, audio_path, language=language)
        return self._model.analyze_image(image_path, language=language)

    def clear_cache(self) -> None:
        """캐시 초기화."""
        self._cache.clear()
        logger.info("DiagnosisPipeline 캐시 초기화 완료")

    @property
    def cache_size(self) -> int:
        """현재 캐시 항목 수 반환."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# Batch Evaluation Helpers
# ---------------------------------------------------------------------------

def evaluate_batch(model, samples: list, language: str = "en") -> list:
    """
    Run diagnosis on a list of sample images and collect results.

    Args:
        model:   CropDoctorModel instance.
        samples: List of dicts from load_local_samples / load_plantvillage_sample.
        language: Language code for responses.

    Returns:
        List of result dicts (model output + ground truth label).
    """
    results = []
    for i, sample in enumerate(samples):
        img = sample.get("image")
        path = sample.get("path", f"sample_{i}")
        label = sample.get("label_name", "unknown")

        if img is None:
            logger.warning("No image for sample %d, skipping.", i)
            continue

        # Save PIL image to a temp file so model.analyze_image can open it
        tmp_path = Path("/tmp") / f"cropdoc_eval_{i}.jpg"
        try:
            img.save(tmp_path, format="JPEG")
            result = model.analyze_image(str(tmp_path), language=language)
            result["ground_truth"] = label
            result["source_path"] = path
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Evaluation failed for sample %d: %s", i, exc)
        finally:
            tmp_path.unlink(missing_ok=True)

    return results


def save_results(results: list, output_path: str) -> None:
    """Save a list of result dicts as a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    paths = download_sample_images()
    if not paths:
        paths = [create_dummy_sample()]

    for p in paths:
        b64 = image_to_base64(p)
        print(f"{p}: base64 length = {len(b64) if b64 else 'FAILED'}")
