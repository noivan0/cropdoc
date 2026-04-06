"""
CropDoc — Dual-Backend Model Integration
=========================================
지원 백엔드:
  1. [PRIMARY]   Kaggle 네이티브 Transformers — google/gemma-4/transformers/gemma-4-e4b-it
                 kagglehub.model_download()로 직접 다운로드 → AutoModelForCausalLM 로드
  2. [FALLBACK]  Google AI Studio API     — gemini-1.5-pro / gemini-2.0-flash

Gemma 4 E4B는 Kaggle이 직접 호스팅하는 모델을 kagglehub로 다운로드 후
HuggingFace Transformers AutoModelForCausalLM 방식으로 추론합니다.
이미지 + 오디오 동시 입력을 지원합니다.
API 백엔드는 GPU가 없는 환경에서 Gemini를 통한 fallback으로 동작합니다.

Usage:
    # Transformers 백엔드 (Kaggle GPU 환경, GPU 필요)
    model = CropDoctorModel(backend="transformers")

    # API 백엔드 (API 키만 있는 환경, GPU 불필요)
    model = CropDoctorModel(backend="api", api_key="YOUR_KEY")

    # auto 모드 (GPU/transformers 가능하면 Kaggle 네이티브, 아니면 API)
    model = CropDoctorModel(backend="auto", api_key="YOUR_KEY")

    result = model.analyze_image("leaf.jpg", language="sw")
    result = model.analyze_with_audio("leaf.jpg", "voice.wav", language="hi")
"""

import os
import importlib
import base64
import time
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language Configuration
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "en": "English",
    "sw": "Swahili (Kiswahili)",
    "hi": "Hindi (हिंदी)",
    "bn": "Bengali (বাংলা)",
    "fr": "French (Français)",
    "pt": "Portuguese (Português)",
    "es": "Spanish (Español)",
    "ar": "Arabic (العربية)",
    "am": "Amharic (አማርኛ)",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "zu": "Zulu (isiZulu)",
    "tl": "Filipino (Tagalog)",
    "vi": "Vietnamese (Tiếng Việt)",
    "th": "Thai (ภาษาไทย)",
    "id": "Indonesian (Bahasa Indonesia)",
    "my": "Burmese (မြန်မာဘာသာ)",
    "km": "Khmer (ភាសាខ្មែរ)",
    "ne": "Nepali (नेपाली)",
}

# System prompts per language (localised greeting + instruction)
SYSTEM_PROMPTS = {
    "en": """You are CropDoc, an expert agricultural assistant helping smallholder farmers in developing countries.
Analyze the crop image (and audio description if provided) and respond ONLY in English.

Provide a structured diagnosis:

## 🌿 Disease / Pest Identification
- Name (common + scientific)
- Confidence level

## ⚠️ Severity Level
- **CRITICAL** / **MODERATE** / **MONITOR**
- Brief explanation of why

## 💊 Treatment Recommendations
- Specific product names (generic/affordable alternatives too)
- Dosage and application method
- Best timing for application

## 🛡️ Prevention Tips
- Actions for next season
- Cultural practices (crop rotation, spacing, etc.)

## 💰 Estimated Cost
- Approximate treatment cost in USD
- Local alternatives if available

Be concise, practical, and compassionate. These farmers have limited resources.""",

    "sw": """Wewe ni CropDoc, msaidizi wa kilimo wa kisasa anayesaidia wakulima wadogo katika nchi zinazoendelea.
Angalia picha ya mazao (na maelezo ya sauti ikiwa yametolewa) na jibu KWA KISWAHILI TU.

Toa uchunguzi uliopangwa:

## 🌿 Utambuzi wa Ugonjwa / Wadudu
- Jina (la kawaida + la kisayansi)
- Kiwango cha uhakika

## ⚠️ Kiwango cha Ukali
- **HATARI** / **WASTANI** / **FUATILIA**
- Maelezo mafupi ya sababu

## 💊 Mapendekezo ya Matibabu
- Majina ya bidhaa maalum (pamoja na mbadala za bei nafuu)
- Kipimo na njia ya kutumia
- Wakati bora wa kutumia

## 🛡️ Vidokezo vya Kuzuia
- Hatua za msimu ujao
- Mazoea ya kilimo (mzunguko wa mazao, umbali, n.k.)

## 💰 Gharama Inayokadiriwa
- Gharama ya takriban ya matibabu kwa USD / KES
- Mbadala za ndani ikiwa zinapatikana

Kuwa mfupi, wa vitendo, na wenye huruma. Wakulima hawa wana rasilimali chache.""",

    "hi": """आप CropDoc हैं, एक कृषि विशेषज्ञ सहायक जो विकासशील देशों में छोटे किसानों की मदद करता है।
फसल की छवि (और यदि उपलब्ध हो तो ऑडियो विवरण) का विश्लेषण करें और केवल हिंदी में जवाब दें।

संरचित निदान प्रदान करें:

## 🌿 रोग / कीट पहचान
- नाम (सामान्य + वैज्ञानिक)
- विश्वास स्तर

## ⚠️ गंभीरता स्तर
- **गंभीर** / **मध्यम** / **निगरानी**
- संक्षिप्त कारण स्पष्टीकरण

## 💊 उपचार की सिफारिशें
- विशिष्ट उत्पाद नाम (सस्ते विकल्प भी)
- खुराक और प्रयोग की विधि
- उपयोग का सबसे अच्छा समय

## 🛡️ रोकथाम के सुझाव
- अगले सीज़न के लिए कदम
- कृषि प्रथाएं (फसल चक्र, दूरी आदि)

## 💰 अनुमानित लागत
- USD / INR में उपचार की अनुमानित लागत
- यदि उपलब्ध हो तो स्थानीय विकल्प

संक्षिप्त, व्यावहारिक और सहानुभूतिपूर्ण रहें। इन किसानों के पास सीमित संसाधन हैं।""",

    "bn": """আপনি CropDoc, একজন কৃষি বিশেষজ্ঞ সহকারী যিনি উন্নয়নশীল দেশের ক্ষুদ্র কৃষকদের সাহায্য করেন।
ফসলের ছবি (এবং যদি পাওয়া যায় অডিও বিবরণ) বিশ্লেষণ করুন এবং শুধুমাত্র বাংলায় উত্তর দিন।

কাঠামোবদ্ধ নির্ণয় প্রদান করুন:

## 🌿 রোগ / কীটপতঙ্গ শনাক্তকরণ
- নাম (সাধারণ + বৈজ্ঞানিক)
- আস্থার স্তর

## ⚠️ তীব্রতার স্তর
- **জটিল** / **মাঝারি** / **পর্যবেক্ষণ**
- সংক্ষিপ্ত ব্যাখ্যা

## 💊 চিকিৎসার সুপারিশ
- নির্দিষ্ট পণ্যের নাম (সাশ্রয়ী বিকল্পও)
- মাত্রা এবং প্রয়োগ পদ্ধতি
- প্রয়োগের সেরা সময়

## 🛡️ প্রতিরোধ টিপস
- পরবর্তী মৌসুমের জন্য পদক্ষেপ
- কৃষি অনুশীলন (ফসল আবর্তন, ব্যবধান ইত্যাদি)

## 💰 আনুমানিক খরচ
- USD / BDT-তে আনুমানিক চিকিৎসা ব্যয়
- স্থানীয় বিকল্প যদি পাওয়া যায়

সংক্ষিপ্ত, ব্যবহারিক এবং সহানুভূতিশীল হোন। এই কৃষকদের সীমিত সম্পদ রয়েছে।""",
}


def _build_prompt(language_code: str, language_name: str) -> str:
    """Build a system prompt for the given language."""
    if language_code in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[language_code]
    # Generic template with target language injection
    return SYSTEM_PROMPTS["en"].replace(
        "respond ONLY in English",
        f"respond ONLY in {language_name}"
    ).replace(
        "Be concise",
        f"Respond in {language_name}. Be concise"
    )


# ===========================================================================
# Backend 1: HuggingFace Transformers (PRIMARY — Gemma 4 공식 방식)
# ===========================================================================

class _TransformersBackend:
    """
    [PRIMARY BACKEND] Kaggle 네이티브 방식으로 Gemma 4 E4B 추론.

    - 모델 다운로드: kagglehub.model_download("google/gemma-4/transformers/gemma-4-e4b-it")
    - 클래스: AutoModelForCausalLM (Kaggle 공식 — ImageTextToText 아님!)
    - 이미지: {"type": "image", "url": path} 형식
    - 오디오: {"type": "audio", "audio": path} 형식
    - apply_chat_template: tokenize=False → processor(text=text) 분리 호출
    - GPU 필요 (Kaggle T4/P100 GPU 환경)
    """

    KAGGLE_MODEL_ID = "google/gemma-4/transformers/gemma-4-e4b-it"  # Kaggle 네이티브
    HF_MODEL_ID = "google/gemma-4-E4B-it"                            # HF fallback

    def __init__(self, model_id: Optional[str] = None, use_kaggle: bool = True):
        self.torch = importlib.import_module("torch")
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "HuggingFace Transformers 백엔드를 사용하려면 transformers, torch, accelerate가 필요합니다.\n"
                "설치: pip install transformers>=4.50.0 torch accelerate kagglehub\n"
                f"원래 오류: {e}"
            )

        # 모델 경로 결정: Kaggle 네이티브 → HF fallback 순서
        if use_kaggle:
            try:
                import kagglehub
                MODEL_PATH = kagglehub.model_download(model_id or self.KAGGLE_MODEL_ID)
                logger.info("✅ Kaggle 모델 로드: %s", MODEL_PATH)
            except Exception as e:
                logger.warning("⚠️ kagglehub 실패 (%s), HF로 fallback", e)
                MODEL_PATH = self.HF_MODEL_ID
        else:
            MODEL_PATH = model_id or self.HF_MODEL_ID

        logger.info("Transformers 백엔드 로드 중: %s", MODEL_PATH)

        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

        # bfloat16 + device_map="auto" + Flash Attention 2 (graceful fallback)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=self.torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",  # PyTorch SDPA (flash_attn 패키지 없이도 동작)
            )
            logger.info("✅ SDPA (Scaled Dot Product Attention) 활성화")
        except Exception as fa_err:
            logger.info("Flash Attention 2 불가 (%s), 기본 attention 사용", fa_err)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=self.torch.bfloat16,
                device_map="auto",
            )
        self.model.eval()

        # torch.compile (PyTorch 2.0+) — 추론 오버헤드 감소
        if hasattr(self.torch, 'compile') and self.torch.__version__ >= '2.0':
            try:
                self.model = self.torch.compile(self.model, mode="reduce-overhead")
                logger.info("✅ torch.compile 활성화 (reduce-overhead)")
            except Exception as compile_err:
                logger.warning("torch.compile 실패: %s", compile_err)

        device = next(self.model.parameters()).device
        logger.info(
            "✅ Gemma 4 E4B 로드 완료 | Device: %s | 지원: 텍스트, 이미지, 오디오",
            device,
        )

    def _preprocess_image(self, image_path: str) -> str:
        """이미지 크기 최적화 (1024x1024 이내로 제한, LANCZOS 리샘플링)."""
        import tempfile
        img = Image.open(image_path)
        MAX_SIZE = 1024
        if max(img.size) > MAX_SIZE:
            ratio = MAX_SIZE / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp.name, 'JPEG', quality=90)
                logger.info("이미지 리사이즈: %s → %s", (img.width, img.height), new_size)
                return tmp.name
        return image_path

    def infer(
        self,
        system_prompt: str,
        image_path: str,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Kaggle 공식 방식으로 추론.
        apply_chat_template(tokenize=False) → processor(text=text) 분리 호출.

        메시지 구조:
          - system: 시스템 프롬프트 (native 지원)
          - user:   [audio?] + image + text
        """
        # 이미지 전처리 최적화 (1024x1024 이내로 제한)
        processed_image_path = self._preprocess_image(image_path)

        # 시스템 메시지
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # 유저 콘텐츠 빌드
        user_content = []

        # 오디오 (선택) — {"type": "audio", "audio": path}
        if audio_path and Path(audio_path).exists():
            user_content.append({"type": "audio", "audio": audio_path})
            logger.info("오디오 포함: %s", audio_path)
        elif audio_path:
            logger.warning("오디오 파일 없음 (%s), 이미지만 분석합니다.", audio_path)

        # 이미지 — {"type": "image", "url": path}
        user_content.append({"type": "image", "url": processed_image_path})

        # 텍스트 지시
        user_content.append({"type": "text", "text": "Please diagnose this crop."})

        messages.append({"role": "user", "content": user_content})

        # Kaggle 공식 방식: tokenize=False → 텍스트 문자열만 반환
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Gemma 4 신기능 — thinking 모드 비활성화
        )

        # processor()로 토크나이징 및 GPU 이동
        inputs = self.processor(text=text, return_tensors="pt").to(
            next(self.model.parameters()).device
        )

        input_len = inputs["input_ids"].shape[-1]

        # 추론 시간 측정 + 최적화된 generate() 파라미터
        t0 = time.perf_counter()
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy decoding (결정적, 빠름)
                temperature=1.0,
                use_cache=True,         # KV cache (명시적 활성화)
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        generated_tokens = outputs.shape[1] - input_len
        logger.info("추론 완료: %.2f초, %d 토큰 생성", elapsed, generated_tokens)

        # 입력 토큰 제외하고 디코딩 (skip_special_tokens=False: Gemma 4 공식)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # parse_response가 있으면 사용 (Gemma 4 구조화 출력 파싱)
        if hasattr(self.processor, "parse_response"):
            try:
                response = self.processor.parse_response(response)
            except Exception:
                pass  # 파싱 실패 시 raw 텍스트 사용

        return response.strip()


# ===========================================================================
# Backend 2: Google AI Studio API (FALLBACK — API 키 환경)
# ===========================================================================

class _APIBackend:
    """
    [FALLBACK BACKEND] Google AI Studio API를 사용한 추론.

    - SDK: google-genai >= 1.0.0 (google.generativeai 완전 대체)
    - 모델: gemini-2.0-flash (기본값) 또는 gemini-1.5-pro
    - Gemma 4 E4B는 AI Studio에서 직접 지원되지 않으므로 Gemini 사용
    - GPU 불필요, API 키만 있으면 동작
    - Kaggle 환경: kaggle-models accelerator를 통한 Gemma 4 접근도 가능
      (Add-ons > Accelerators > Gemma 4 선택 후 로컬 엔드포인트 사용)

    Note:
        google.generativeai 패키지는 2025-11-30 EOL로 완전 deprecated.
        반드시 google-genai (google.genai) SDK를 사용해야 합니다.
    """

    DEFAULT_MODEL = "gemini-2.0-flash"       # 빠른 추론, 무료 티어 지원
    FALLBACK_MODEL = "gemini-1.5-pro"        # 더 정확, 유료
    DEFAULT_TIMEOUT = 60
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self, api_key: str, model_name: Optional[str] = None) -> None:
        """
        _APIBackend 초기화.

        Args:
            api_key:    Google AI Studio API 키
            model_name: 사용할 Gemini 모델명 (기본: gemini-2.0-flash)

        Raises:
            ImportError: google-genai 패키지 미설치 시
            ValueError:  api_key가 빈 문자열일 시
        """
        if not api_key:
            raise ValueError(
                "API 키가 없습니다. export GOOGLE_API_KEY='your-key' 로 설정하거나 "
                "api_key= 인자로 전달하세요."
            )

        try:
            import google.genai as genai
            from google.genai import types as genai_types  # noqa: F401 (사전 검증용)
        except ImportError as e:
            raise ImportError(
                "API 백엔드를 사용하려면 google-genai가 필요합니다.\n"
                "설치: pip install google-genai>=1.0.0\n"
                "⚠️ google-generativeai는 EOL(2025-11-30). google-genai를 사용하세요.\n"
                f"원래 오류: {e}"
            )

        self._genai = genai
        self.model_name = model_name or self.DEFAULT_MODEL
        # google.genai 신 SDK: Client 인스턴스 기반
        self._client = genai.Client(api_key=api_key)
        logger.info("✅ API 백엔드 초기화 완료 | SDK: google-genai %s | 모델: %s",
                    getattr(genai, "__version__", "unknown"), self.model_name)

    def infer(
        self,
        system_prompt: str,
        image_path: str,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Google AI Studio API를 통한 추론 (google-genai 신 SDK).

        Args:
            system_prompt:  시스템 프롬프트 (언어별 진단 지침)
            image_path:     작물 이미지 경로
            audio_path:     오디오 파일 경로 (선택)
            max_new_tokens: 최대 생성 토큰 수 (API에서는 힌트로만 사용)

        Returns:
            str: 모델이 생성한 진단 텍스트

        Raises:
            FileNotFoundError: 이미지 파일이 없을 시
            RuntimeError:      모든 재시도 실패 시
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # contents 빌드: [system_prompt, image, (audio_part?)]
        contents: list = [system_prompt, image]

        if audio_path and Path(audio_path).exists():
            audio_part = self._build_audio_part(audio_path)
            if audio_part is not None:
                contents.append(audio_part)
                logger.info("오디오 포함: %s", audio_path)
            else:
                logger.warning("오디오 파트 생성 실패, 이미지만 분석합니다.")
        elif audio_path:
            logger.warning("오디오 파일 없음 (%s), 이미지만 분석합니다.", audio_path)

        return self._call_with_retry(contents)

    def _call_with_retry(self, contents: list) -> str:
        """
        지수 백오프 재시도 포함 API 호출.

        Args:
            contents: generate_content에 전달할 콘텐츠 리스트

        Returns:
            str: 모델 응답 텍스트

        Raises:
            RuntimeError: MAX_RETRIES 초과 시
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                result_text = response.text
                if not result_text:
                    raise ValueError("API 응답이 비어 있습니다.")
                return result_text
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "API 시도 %d/%d 실패: %s", attempt, self.MAX_RETRIES, exc
                )
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY * attempt
                    logger.info("%.1f초 후 재시도...", delay)
                    time.sleep(delay)
        raise RuntimeError(
            f"모든 API 시도({self.MAX_RETRIES}회) 실패. 마지막 오류: {last_exc}"
        )

    def _build_audio_part(self, audio_path: str):
        """
        오디오 파일을 google-genai API용 Part 객체로 변환.

        Args:
            audio_path: 오디오 파일 경로 (.wav/.mp3/.ogg/.flac/.m4a/.webm)

        Returns:
            google.genai.types.Part 또는 None (로드 실패 시)
        """
        from google.genai import types as genai_types

        suffix = Path(audio_path).suffix.lower()
        mime_map: dict[str, str] = {
            ".wav":  "audio/wav",
            ".mp3":  "audio/mpeg",
            ".ogg":  "audio/ogg",
            ".flac": "audio/flac",
            ".m4a":  "audio/mp4",
            ".webm": "audio/webm",
        }
        mime = mime_map.get(suffix, "audio/wav")

        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            return genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime)
        except OSError as exc:
            logger.error("오디오 파일 읽기 실패: %s", exc)
            return None


# ===========================================================================
# CropDoctorModel — 통합 인터페이스 (듀얼 백엔드)
# ===========================================================================

class CropDoctorModel:
    """
    CropDoc 진단 모델 — 듀얼 백엔드 지원.

    backends:
      "transformers" [PRIMARY]  → HuggingFace Transformers + google/gemma-4-E4B-it
                                   GPU 필요. Kaggle T4/P100/A100 환경.
      "api"          [FALLBACK] → Google AI Studio API + gemini-2.0-flash
                                   API 키 필요. GPU 불필요.
      "auto"         [DEFAULT]  → transformers 먼저 시도, 실패시 api로 fallback.

    동일한 공개 인터페이스:
      - analyze_image(image_path, language)
      - analyze_with_audio(image_path, audio_path, language)
      - get_supported_languages()
    """

    def __init__(
        self,
        backend: str = "auto",
        api_key: Optional[str] = None,
        kaggle_model_id: Optional[str] = None,
        api_model_name: Optional[str] = None,
        use_kaggle: bool = True,
    ):
        """
        Args:
            backend:         "auto" | "transformers" | "api"
            api_key:         Google AI Studio API 키 (api/auto 모드에서 필요)
            kaggle_model_id: Kaggle 모델 슬러그 (기본: google/gemma-4/transformers/gemma-4-e4b-it)
            api_model_name:  API 모델명 (기본: gemini-2.0-flash)
            use_kaggle:      True면 kagglehub로 다운로드, False면 HF에서 직접 로드
        """
        self._backend_name: str = ""
        self._backend = None

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

        if backend == "transformers":
            # [PRIMARY] Kaggle 네이티브 Transformers — GPU 필요
            logger.info("백엔드: Kaggle 네이티브 Transformers (PRIMARY)")
            self._backend = _TransformersBackend(
                model_id=kaggle_model_id, use_kaggle=use_kaggle
            )
            self._backend_name = "transformers"

        elif backend == "api":
            # [FALLBACK] API 전용 — GPU 불필요
            if not resolved_key:
                raise EnvironmentError(
                    "API 백엔드를 사용하려면 GOOGLE_API_KEY가 필요합니다.\n"
                    "export GOOGLE_API_KEY='your-key' 또는 api_key= 인자로 전달하세요."
                )
            logger.info("백엔드: Google AI Studio API (FALLBACK) | 모델: %s",
                        api_model_name or _APIBackend.DEFAULT_MODEL)
            self._backend = _APIBackend(api_key=resolved_key, model_name=api_model_name)
            self._backend_name = "api"

        elif backend == "auto":
            # [AUTO] Kaggle 네이티브 Transformers 먼저 시도, 실패 시 API fallback
            logger.info("백엔드: auto 모드 — Kaggle 네이티브 Transformers 먼저 시도합니다.")
            try:
                self._backend = _TransformersBackend(
                    model_id=kaggle_model_id, use_kaggle=use_kaggle
                )
                self._backend_name = "transformers"
                logger.info("✅ auto: Kaggle 네이티브 Transformers 백엔드 선택됨")
            except Exception as tf_err:
                logger.warning(
                    "Transformers 백엔드 초기화 실패 (%s). API 백엔드로 fallback합니다.", tf_err
                )
                if not resolved_key:
                    raise EnvironmentError(
                        f"Transformers 백엔드 실패 ({tf_err}).\n"
                        "API fallback을 위해 GOOGLE_API_KEY를 설정하세요."
                    )
                self._backend = _APIBackend(api_key=resolved_key, model_name=api_model_name)
                self._backend_name = "api"
                logger.info("✅ auto: Google AI Studio API 백엔드 선택됨 (fallback)")
        else:
            raise ValueError(f"알 수 없는 backend: '{backend}'. 'auto', 'transformers', 'api' 중 하나를 선택하세요.")

        logger.info("CropDoctorModel 준비 완료 | 백엔드: %s", self._backend_name)

    # ------------------------------------------------------------------
    # 공개 인터페이스 (두 백엔드 모두 동일한 시그니처)
    # ------------------------------------------------------------------

    def get_supported_languages(self) -> dict:
        """지원 언어 코드 → 이름 매핑 반환."""
        return dict(SUPPORTED_LANGUAGES)

    def get_backend_name(self) -> str:
        """현재 사용 중인 백엔드 이름 반환 ('transformers' 또는 'api')."""
        return self._backend_name

    def analyze_image(
        self,
        image_path: str,
        language: str = "en",
    ) -> dict:
        """
        이미지만으로 작물 질병 진단.

        Args:
            image_path: 작물 이미지 경로 (JPEG / PNG / WEBP)
            language:   ISO 639-1 언어 코드 (기본 "en")

        Returns:
            dict: language, diagnosis, disease, severity, audio_included, error
        """
        if not image_path or not Path(image_path).exists():
            return self._error_response(language, f"이미지를 찾을 수 없습니다: {image_path}")

        language_name = SUPPORTED_LANGUAGES.get(language, "English")
        system_prompt = _build_prompt(language, language_name)

        logger.info(
            "[%s 백엔드] 이미지 분석: %s | 언어: %s",
            self._backend_name, image_path, language,
        )

        try:
            response_text = self._backend.infer(
                system_prompt=system_prompt,
                image_path=image_path,
                audio_path=None,
            )
            return self._build_response(language, response_text)
        except FileNotFoundError:
            return self._error_response(language, f"이미지 파일 없음: {image_path}")
        except Exception as exc:
            logger.error("analyze_image 실패: %s", exc)
            return self._error_response(language, str(exc))

    def analyze_with_audio(
        self,
        image_path: str,
        audio_path: str,
        language: str = "en",
    ) -> dict:
        """
        이미지 + 농부 음성 설명으로 작물 질병 진단.
        Gemma 4 E4B의 멀티모달 독점 기능 (이미지 + 오디오 동시 입력).

        Args:
            image_path: 작물 이미지 경로
            audio_path: 오디오 파일 경로 (WAV / MP3 / OGG / FLAC)
            language:   ISO 639-1 언어 코드

        Returns:
            analyze_image와 동일한 구조, audio_included=True
        """
        if not image_path or not Path(image_path).exists():
            return self._error_response(language, f"이미지를 찾을 수 없습니다: {image_path}")

        language_name = SUPPORTED_LANGUAGES.get(language, "English")
        system_prompt = _build_prompt(language, language_name)

        logger.info(
            "[%s 백엔드] 이미지+오디오 분석: %s + %s | 언어: %s",
            self._backend_name, image_path, audio_path, language,
        )

        try:
            response_text = self._backend.infer(
                system_prompt=system_prompt,
                image_path=image_path,
                audio_path=audio_path,
            )
            result = self._build_response(language, response_text)
            result["audio_included"] = bool(audio_path and Path(audio_path).exists())
            return result
        except Exception as exc:
            logger.error("analyze_with_audio 실패: %s", exc)
            return self._error_response(language, str(exc))

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _build_response(self, language: str, diagnosis: str) -> dict:
        """모델 응답을 구조화된 dict로 변환."""
        return {
            "language": language,
            "language_name": SUPPORTED_LANGUAGES.get(language, language),
            "diagnosis": diagnosis,
            "disease": self._extract_disease(diagnosis),
            "severity": self._extract_severity(diagnosis),
            "audio_included": False,
            "backend": self._backend_name,
            "error": None,
        }

    def _error_response(self, language: str, message: str) -> dict:
        """오류 발생 시 안전한 fallback 응답."""
        fallback_messages = {
            "sw": "Samahani, kulikuwa na hitilafu. Tafadhali jaribu tena.",
            "hi": "क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।",
            "bn": "দুঃখিত, একটি ত্রুটি হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।",
        }
        friendly = fallback_messages.get(
            language,
            "Sorry, an error occurred. Please try again."
        )
        return {
            "language": language,
            "language_name": SUPPORTED_LANGUAGES.get(language, language),
            "diagnosis": friendly,
            "disease": "Unknown",
            "severity": "UNKNOWN",
            "audio_included": False,
            "backend": self._backend_name,
            "error": message,
        }

    @staticmethod
    def _extract_disease(text: str) -> str:
        """응답 텍스트에서 질병명 추출 (휴리스틱)."""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if len(line) > 5 and not line.startswith("-"):
                clean = line.replace("**", "").replace("*", "").strip()
                if clean:
                    return clean[:120]
        return "See full diagnosis"

    @staticmethod
    def _extract_severity(text: str) -> str:
        """응답 텍스트에서 심각도 추출."""
        text_upper = text.upper()
        if "CRITICAL" in text_upper or "HATARI" in text_upper or "गंभीर" in text or "জটিল" in text:
            return "CRITICAL"
        if "MODERATE" in text_upper or "WASTANI" in text_upper or "मध्यम" in text or "মাঝারি" in text:
            return "MODERATE"
        if "MONITOR" in text_upper or "FUATILIA" in text_upper or "निगरानी" in text or "পর্যবেক্ষণ" in text:
            return "MONITOR"
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# 직접 실행 테스트: python model.py <image_path> [language] [--backend transformers|api|auto]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json
    import argparse

    parser = argparse.ArgumentParser(description="CropDoc 진단 테스트")
    parser.add_argument("image", help="작물 이미지 경로")
    parser.add_argument("language", nargs="?", default="en", help="언어 코드 (기본: en)")
    parser.add_argument("--backend", choices=["auto", "transformers", "api"], default="auto")
    parser.add_argument("--audio", default=None, help="오디오 파일 경로 (선택)")
    args = parser.parse_args()

    m = CropDoctorModel(backend=args.backend)

    if args.audio:
        result = m.analyze_with_audio(args.image, args.audio, language=args.language)
    else:
        result = m.analyze_image(args.image, language=args.language)

    print(json.dumps(result, ensure_ascii=False, indent=2))
