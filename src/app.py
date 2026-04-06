"""
CropDoc — Gradio Web Demo (듀얼 백엔드 지원)
=============================================
백엔드:
  🤗 HuggingFace Transformers (GPU)  → google/gemma-4-E4B-it 로컬 추론
  ☁️ Google AI Studio API (API Key) → gemini-2.0-flash 원격 API

Features:
  - 백엔드 선택 라디오버튼 (상단)
  - Tab 1: 이미지 단독 진단
  - Tab 2: 이미지 + 농부 음성(오디오) 진단 — E4B 독점
  - 언어 선택기 (English, Swahili, Hindi, Bengali 등)
  - 샘플 이미지 원클릭 테스트
  - 심각도 뱃지 (색상 코딩)
  - HuggingFace Spaces 호환

Run locally:
    # Transformers 백엔드 (GPU 필요)
    python app.py

    # API 백엔드
    export GOOGLE_API_KEY="your-key"
    python app.py

Deploy to HuggingFace Spaces:
    1. 새 Space 생성 (Gradio SDK)
    2. app.py, requirements.txt 업로드
    3. GOOGLE_API_KEY를 Space Secrets에 추가
    4. GPU가 있는 Space: Transformers 백엔드 자동 선택
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

import gradio as gr

# src/ 를 Python 경로에 추가 (HuggingFace Spaces 호환)
sys.path.insert(0, str(Path(__file__).parent))

from model import CropDoctorModel, SUPPORTED_LANGUAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 파이프라인 유틸 (pipeline.py 의존성 완화를 위해 인라인 정의)
# ---------------------------------------------------------------------------
try:
    from pipeline import download_sample_images, create_dummy_sample, parse_diagnosis, preprocess_audio
except ImportError:
    def download_sample_images():
        return []
    def create_dummy_sample():
        import numpy as np
        from PIL import Image
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        arr[:, :, 1] = 120
        tmp = tempfile.mktemp(suffix=".jpg")
        Image.fromarray(arr).save(tmp)
        return tmp
    def parse_diagnosis(text):
        return {}
    def preprocess_audio(audio):
        return audio


# ---------------------------------------------------------------------------
# 모델 초기화 (지연 로딩 — 백엔드 선택 후 초기화)
# ---------------------------------------------------------------------------
# 백엔드별 모델 캐시 (한 번 로드 후 재사용)
_model_cache: dict[str, CropDoctorModel] = {}

def get_model(backend_choice: str) -> CropDoctorModel:
    """
    백엔드 선택에 따라 CropDoctorModel을 초기화하거나 캐시에서 반환.

    Args:
        backend_choice: "🤗 HuggingFace Transformers (GPU)" 또는
                        "☁️ Google AI Studio API (API Key)"
    """
    # UI 선택 → backend 코드 매핑
    if "HuggingFace" in backend_choice or "Transformers" in backend_choice:
        backend = "transformers"
    elif "API" in backend_choice or "Google" in backend_choice:
        backend = "api"
    else:
        backend = "auto"

    if backend not in _model_cache:
        logger.info("백엔드 초기화: %s", backend)
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        try:
            _model_cache[backend] = CropDoctorModel(backend=backend, api_key=api_key or None)
        except Exception as exc:
            logger.error("모델 초기화 실패 (%s): %s", backend, exc)
            raise
    return _model_cache[backend]


# ---------------------------------------------------------------------------
# 샘플 이미지
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).parent.parent / "samples"

def get_sample_paths() -> list[str]:
    paths = list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.png"))
    if not paths:
        try:
            downloaded = download_sample_images()
            if downloaded:
                return downloaded
        except Exception:
            pass
        return [create_dummy_sample()]
    return [str(p) for p in sorted(paths)]


# ---------------------------------------------------------------------------
# 심각도 뱃지 HTML
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "CRITICAL": ("#FF4B4B", "🔴 CRITICAL"),
    "MODERATE": ("#FF9F0A", "🟠 MODERATE"),
    "MONITOR":  ("#34C759", "🟢 MONITOR"),
    "HATARI":   ("#FF4B4B", "🔴 HATARI (Critical)"),
    "WASTANI":  ("#FF9F0A", "🟠 WASTANI (Moderate)"),
    "FUATILIA": ("#34C759", "🟢 FUATILIA (Monitor)"),
    "गंभीर":    ("#FF4B4B", "🔴 गंभीर (Critical)"),
    "मध्यम":    ("#FF9F0A", "🟠 मध्यम (Moderate)"),
    "निगरानी":  ("#34C759", "🟢 निगरानी (Monitor)"),
    "জটিল":     ("#FF4B4B", "🔴 জটিল (Critical)"),
    "মাঝারি":   ("#FF9F0A", "🟠 মাঝারি (Moderate)"),
    "পর্যবেক্ষণ": ("#34C759", "🟢 পর্যবেক্ষণ (Monitor)"),
}

def severity_html(severity: str) -> str:
    color, label = SEVERITY_COLORS.get(severity, ("#888888", f"❓ {severity}"))
    return (
        f'<div style="display:inline-block; padding:8px 20px; border-radius:20px; '
        f'background:{color}; color:white; font-size:1.1em; font-weight:bold;">'
        f'{label}</div>'
    )


# ---------------------------------------------------------------------------
# 언어 선택지
# ---------------------------------------------------------------------------
LANGUAGE_CHOICES = [
    ("English", "en"),
    ("Swahili (Kiswahili)", "sw"),
    ("Hindi (हिंदी)", "hi"),
    ("Bengali (বাংলা)", "bn"),
    ("French (Français)", "fr"),
    ("Spanish (Español)", "es"),
    ("Portuguese (Português)", "pt"),
    ("Arabic (العربية)", "ar"),
    ("Indonesian", "id"),
    ("Vietnamese", "vi"),
]

# 백엔드 선택지 (UI)
BACKEND_CHOICES = [
    "🤗 HuggingFace Transformers (GPU)",    # PRIMARY — Gemma 4 공식
    "☁️ Google AI Studio API (API Key)",    # FALLBACK — Gemini
]


# ---------------------------------------------------------------------------
# 진단 콜백 함수
# ---------------------------------------------------------------------------

def diagnose_image(backend_choice: str, image, language_code: str):
    """
    Tab 1 콜백: 이미지 단독 진단.

    Args:
        backend_choice: 선택된 백엔드 (라디오버튼 값)
        image:          PIL Image (gr.Image 컴포넌트)
        language_code:  ISO 639-1 언어 코드

    Returns:
        (severity_html_str, diagnosis_markdown, status_str)
    """
    if image is None:
        return "", "⚠️ 이미지를 업로드하세요.", "이미지 없음"

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_f:
            tmp_path = tmp_f.name
        image.save(tmp_path, format="JPEG")
    except Exception as exc:
        return "", f"❌ 이미지 처리 오류: {exc}", "오류"

    try:
        model = get_model(backend_choice)
        result = model.analyze_image(tmp_path, language=language_code)
    except EnvironmentError as exc:
        return (
            severity_html("UNKNOWN"),
            f"❌ **API Key 오류**\n\n{exc}\n\n`GOOGLE_API_KEY`를 설정하거나 Transformers 백엔드를 사용하세요.",
            "API 키 누락",
        )
    except Exception as exc:
        return severity_html("UNKNOWN"), f"❌ 오류: {exc}", "실패"

    if result.get("error"):
        return (
            severity_html("UNKNOWN"),
            f"❌ 진단 실패:\n\n{result['error']}",
            "오류",
        )

    backend_tag = f"[{result.get('backend', '?')} 백엔드]"
    badge = severity_html(result["severity"])
    return badge, result["diagnosis"], f"✅ 완료 {backend_tag} | {result['language_name']}"


def diagnose_image_audio(backend_choice: str, image, audio, language_code: str):
    """
    Tab 2 콜백: 이미지 + 오디오(농부 음성) 진단.
    Gemma 4 E4B 독점 멀티모달 기능.

    Args:
        backend_choice: 선택된 백엔드
        image:          PIL Image
        audio:          (sample_rate, numpy_array) 또는 파일 경로
        language_code:  ISO 639-1 언어 코드
    """
    if image is None:
        return "", "⚠️ 이미지를 업로드하세요.", "이미지 없음"

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_f:
            tmp_img = tmp_f.name
        image.save(tmp_img, format="JPEG")
    except Exception as exc:
        return "", f"❌ 이미지 오류: {exc}", "오류"

    audio_path = None
    if audio is not None:
        try:
            import numpy as np
            import soundfile as sf
            if isinstance(audio, tuple):
                sr, data = audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_a:
                    tmp_audio = tmp_a.name
                sf.write(tmp_audio, data, sr)
                audio_path = tmp_audio
            elif isinstance(audio, str):
                audio_path = audio
        except Exception as exc:
            logger.warning("오디오 저장 실패: %s. 이미지만 분석합니다.", exc)

    try:
        model = get_model(backend_choice)
        if audio_path:
            result = model.analyze_with_audio(tmp_img, audio_path, language=language_code)
        else:
            result = model.analyze_image(tmp_img, language=language_code)
    except EnvironmentError as exc:
        return (
            severity_html("UNKNOWN"),
            f"❌ **API Key 오류**\n\n{exc}",
            "API 키 누락",
        )
    except Exception as exc:
        return severity_html("UNKNOWN"), f"❌ 오류: {exc}", "실패"

    if result.get("error"):
        return (
            severity_html("UNKNOWN"),
            f"❌ {result['error']}",
            "오류",
        )

    audio_note = " + 🎙️ 음성" if result.get("audio_included") else " (이미지만)"
    backend_tag = f"[{result.get('backend', '?')} 백엔드]"
    badge = severity_html(result["severity"])
    return badge, result["diagnosis"], f"✅ 완료{audio_note} {backend_tag} | {result['language_name']}"


# ---------------------------------------------------------------------------
# Gradio UI 빌드
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    sample_paths = get_sample_paths()

    with gr.Blocks(
        title="CropDoc 🌾 — AI 작물 질병 진단",
        theme=gr.themes.Soft(primary_hue="green"),
        css="""
            .severity-badge { margin: 10px 0; }
            .hero { text-align: center; padding: 20px 0; }
            .backend-selector { border: 2px solid #e0e0e0; border-radius: 10px; padding: 12px; margin-bottom: 10px; }
            footer { visibility: hidden; }
        """,
    ) as demo:

        # ── 헤더 ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="hero">
            <h1>🌾 CropDoc</h1>
            <p style="font-size:1.1em; color:#666;">
                AI 기반 작물 질병 진단 — 5억 소규모 농부를 위해<br>
                Powered by <strong>Gemma 4 E4B</strong> (이미지 + 오디오 + 140개 언어)
            </p>
        </div>
        """)

        # ── 🆕 백엔드 선택 (상단 라디오버튼) ────────────────────────────
        with gr.Group(elem_classes="backend-selector"):
            gr.Markdown("### 🖥️ 백엔드 선택")
            gr.Markdown(
                "- **🤗 HuggingFace Transformers**: `google/gemma-4-E4B-it` 로컬 추론 — GPU 필요 (Kaggle T4/P100)\n"
                "- **☁️ Google AI Studio API**: `gemini-2.0-flash` 원격 API — GPU 불필요, API 키 필요"
            )
            backend_radio = gr.Radio(
                choices=BACKEND_CHOICES,
                value=BACKEND_CHOICES[0],   # 기본: HF Transformers (PRIMARY)
                label="백엔드 선택",
                interactive=True,
                info="HuggingFace Transformers가 공식 PRIMARY 방식입니다. GPU가 없으면 API를 선택하세요.",
            )

        # ── 언어 선택 (탭 공통) ───────────────────────────────────────────
        language_dd = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            value="en",
            label="🌍 언어 / Language / Lugha / भाषा / ভাষা",
            interactive=True,
        )

        with gr.Tabs():

            # ── Tab 1: 이미지 단독 진단 ──────────────────────────────────
            with gr.TabItem("📷 이미지 진단"):
                gr.Markdown(
                    "작물 사진을 업로드하면 즉시 진단합니다.\n\n"
                    "*팁: 감염된 잎을 가까이서 촬영한 선명한 사진을 사용하세요.*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            type="pil",
                            label="작물 사진",
                            height=300,
                        )
                        gr.Examples(
                            examples=sample_paths,
                            inputs=img_input,
                            label="📂 샘플 이미지 (클릭해서 로드)",
                        )
                        diagnose_btn = gr.Button(
                            "🔍 진단하기", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        severity_out = gr.HTML(label="심각도")
                        diagnosis_out = gr.Markdown(label="전체 진단 결과")
                        status_out = gr.Textbox(
                            label="상태", interactive=False, lines=1
                        )

                # backend_radio도 입력으로 포함
                diagnose_btn.click(
                    fn=diagnose_image,
                    inputs=[backend_radio, img_input, language_dd],
                    outputs=[severity_out, diagnosis_out, status_out],
                )

            # ── Tab 2: 이미지 + 오디오 진단 ──────────────────────────────
            with gr.TabItem("🎙️ 이미지 + 음성 (E4B 독점)"):
                gr.Markdown(
                    "**Gemma 4 E4B 독점 기능**: 작물 사진과 함께 농부의 음성 설명을 녹음하거나 업로드하세요. "
                    "AI가 두 입력을 결합해 더 정확한 진단을 제공합니다.\n\n"
                    "*예시 음성: '잎이 3일 전부터 노랗게 변하면서 갈색 반점이 생겼어요.'*\n\n"
                    "⚠️ **HuggingFace Transformers 백엔드에서만 진정한 audio+image 동시 처리가 지원됩니다.**\n"
                    "API 백엔드에서는 이미지만 분석됩니다."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input2 = gr.Image(
                            type="pil",
                            label="작물 사진",
                            height=250,
                        )
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            label="🎤 음성 설명 (선택, 권장)",
                            type="numpy",
                        )
                        gr.Examples(
                            examples=sample_paths,
                            inputs=img_input2,
                            label="📂 샘플 이미지",
                        )
                        diagnose_btn2 = gr.Button(
                            "🔍 음성+이미지 진단", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        severity_out2 = gr.HTML(label="심각도")
                        diagnosis_out2 = gr.Markdown(label="전체 진단 결과")
                        status_out2 = gr.Textbox(
                            label="상태", interactive=False, lines=1
                        )

                diagnose_btn2.click(
                    fn=diagnose_image_audio,
                    inputs=[backend_radio, img_input2, audio_input, language_dd],
                    outputs=[severity_out2, diagnosis_out2, status_out2],
                )

            # ── Tab 3: 소개 ───────────────────────────────────────────────
            with gr.TabItem("ℹ️ CropDoc 소개"):
                gr.Markdown("""
## CropDoc 소개 🌾

**CropDoc**은 개발도상국의 5억 소규모 농부를 위한 AI 기반 작물 질병 진단 시스템입니다.

### 문제
- 5억 소규모 농부가 전 세계 식량의 70%를 생산
- 예방 가능한 작물 질병으로 연간 **$2,200억** 손실
- 농촌 농부는 인터넷, 농업 전문가, 도움이 없음

### 솔루션
CropDoc은 **Gemma 4 E4B** — Google의 가장 효율적인 멀티모달 모델 — 을 사용합니다:
- 📷 **시각적 진단** — 단일 작물 사진으로
- 🎙️ **음성 + 이미지** 결합 진단 (E4B 독점)
- 🌍 **140개 언어** — 스와힐리어, 힌디어, 벵골어 포함
- 📱 **오프라인 가능** — 저사양 안드로이드 기기에서도 작동
- 💊 **실용적인 조언** — 저렴한 치료 옵션 포함

### 듀얼 백엔드 구조

| 백엔드 | 모델 | 환경 | GPU |
|--------|------|------|-----|
| 🤗 HuggingFace Transformers (PRIMARY) | google/gemma-4-E4B-it | Kaggle GPU 노트북 | 필요 |
| ☁️ Google AI Studio API (FALLBACK) | gemini-2.0-flash | HF Spaces / 로컬 | 불필요 |

### 기술 스택
| 컴포넌트 | 기술 |
|---------|------|
| AI 모델 (PRIMARY) | Gemma 4 E4B (HuggingFace Transformers) |
| AI 모델 (FALLBACK) | Gemini 2.0 Flash (Google AI Studio API) |
| 프론트엔드 | Gradio / HuggingFace Spaces |
| 데이터셋 | PlantVillage (CC0 라이선스) |

### 라이선스
Apache 2.0 — 누구나, 어디서나 무료로 사용 가능.

---
*Gemma 4 Good 해커톤 2026용으로 제작. Team: CropDoc*
                """)

    return demo


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY가 설정되지 않았습니다.")
        print("   API 백엔드를 사용하려면: export GOOGLE_API_KEY='your-key'")
        print("   GPU가 있으면 HuggingFace Transformers 백엔드를 사용하세요.\n")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
