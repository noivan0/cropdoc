#!/usr/bin/env python3
"""
CropDoc 임포트 및 기본 동작 테스트
===================================
- 핵심 라이브러리 임포트 검증
- CropDoc 모듈(model.py, pipeline.py) 임포트 검증
- 실제 모델 로딩은 하지 않음 (API 키 / GPU 불필요)

Usage:
    cd /root/.openclaw/workspace/companies/Hackathon/projects/gemma4good
    python3 scripts/test_imports.py
"""
import sys
import os

# src/ 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_all() -> None:
    """모든 임포트 테스트를 실행하고 결과를 출력."""
    errors: list[str] = []

    print("=" * 60)
    print("CropDoc 임포트 테스트")
    print("=" * 60)

    # ── 1. 핵심 외부 라이브러리 ──────────────────────────────────────────
    print("\n[1/2] 핵심 라이브러리 임포트 테스트")
    tests = [
        (
            "numpy",
            "import numpy as np; print(f'  ✅ numpy {np.__version__}')",
        ),
        (
            "PIL (Pillow)",
            "from PIL import Image; import PIL; print(f'  ✅ Pillow {PIL.__version__}')",
        ),
        (
            "gradio",
            "import gradio as gr; print(f'  ✅ gradio {gr.__version__}')",
        ),
        (
            "soundfile",
            "import soundfile; print(f'  ✅ soundfile {soundfile.__version__}')",
        ),
        (
            "transformers (AutoProcessor + AutoModelForImageTextToText)",
            (
                "from transformers import AutoProcessor, AutoModelForImageTextToText; "
                "import transformers; "
                "print(f'  ✅ transformers {transformers.__version__}')"
            ),
        ),
        (
            "google-genai",
            (
                "import google.genai as genai; "
                "print(f'  ✅ google-genai {genai.__version__}')"
            ),
        ),
        (
            "torch",
            "import torch; print(f'  ✅ torch {torch.__version__} | CUDA={torch.cuda.is_available()}')",
        ),
        (
            "pandas",
            "import pandas as pd; print(f'  ✅ pandas {pd.__version__}')",
        ),
    ]

    for name, code in tests:
        try:
            exec(code)  # noqa: S102
        except Exception as e:
            errors.append(f"❌ {name}: {e}")
            print(f"  ❌ {name}: {e}")

    # ── 2. CropDoc 모듈 임포트 ───────────────────────────────────────────
    print("\n[2/2] CropDoc 모듈 임포트 테스트")

    try:
        from model import CropDoctorModel, SUPPORTED_LANGUAGES  # noqa: F401
        print(f"  ✅ model.py — CropDoctorModel OK, {len(SUPPORTED_LANGUAGES)} languages")
    except Exception as e:
        errors.append(f"❌ model.py: {e}")
        print(f"  ❌ model.py: {e}")

    try:
        from model import _TransformersBackend, _APIBackend  # noqa: F401
        print("  ✅ model.py — _TransformersBackend, _APIBackend OK")
    except Exception as e:
        errors.append(f"❌ model.py (internal classes): {e}")
        print(f"  ❌ model.py (internal classes): {e}")

    try:
        from pipeline import preprocess_audio, parse_diagnosis  # noqa: F401
        print("  ✅ pipeline.py — preprocess_audio, parse_diagnosis OK")
    except ImportError as e:
        # pipeline.py가 없을 수도 있으므로 경고만 표시
        print(f"  ⚠️  pipeline.py — 선택적 모듈 없음 (무시 가능): {e}")
    except Exception as e:
        errors.append(f"❌ pipeline.py: {e}")
        print(f"  ❌ pipeline.py: {e}")

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if errors:
        print("=== 오류 목록 ===")
        for err in errors:
            print(err)
        print(f"\n❌ {len(errors)}개 오류 발생!")
        sys.exit(1)
    else:
        print("✅ 모든 임포트 성공!")
        print("=" * 60)


if __name__ == "__main__":
    test_all()
