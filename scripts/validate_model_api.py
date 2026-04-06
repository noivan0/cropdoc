#!/usr/bin/env python3
"""
CropDoc 모델 코드 구조 & 인터페이스 검증
==========================================
API 키 없이 실행 가능. 실제 모델 로딩은 하지 않고 코드 경로만 확인.

검증 항목:
  1. 모든 클래스 임포트 성공
  2. SUPPORTED_LANGUAGES 최소 10개, 필수 언어 포함
  3. CropDoctorModel 공개 인터페이스 확인
  4. _TransformersBackend — Kaggle 네이티브 방식 확인
     4a. KAGGLE_MODEL_ID 확인 (google/gemma-4/transformers/gemma-4-e4b-it)
     4b. AutoModelForCausalLM 사용 확인 (ImageTextToText 아님)
     4c. kagglehub.model_download 패턴 사용 확인
     4d. apply_chat_template(tokenize=False) 방식 확인
  5. _APIBackend가 deprecated SDK(google.generativeai) 미사용 확인
  6. _APIBackend가 신 SDK(google.genai) 사용 확인

Usage:
    cd /root/.openclaw/workspace/companies/Hackathon/projects/gemma4good
    python3 scripts/validate_model_api.py
"""
import sys
import inspect
import os

# src/ 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 60)
print("CropDoc 모델 구조 검증 (API 키 불필요)")
print("=" * 60)

# ── 1. 모듈 임포트 확인 ───────────────────────────────────────────────────
print("\n[1/6] 모듈 임포트 확인...")
try:
    from model import (  # noqa: E402
        CropDoctorModel,
        SUPPORTED_LANGUAGES,
        _TransformersBackend,
        _APIBackend,
    )
    print("  ✅ 모든 클래스 임포트 성공")
except Exception as e:
    print(f"  ❌ 임포트 실패: {e}")
    sys.exit(1)

# ── 2. SUPPORTED_LANGUAGES 확인 ──────────────────────────────────────────
print("\n[2/6] SUPPORTED_LANGUAGES 확인...")
try:
    assert len(SUPPORTED_LANGUAGES) >= 10, (
        f"최소 10개 언어 지원 필요. 현재: {len(SUPPORTED_LANGUAGES)}"
    )
    assert "sw" in SUPPORTED_LANGUAGES, "스와힐리어(sw) 필수"
    assert "hi" in SUPPORTED_LANGUAGES, "힌디어(hi) 필수"
    assert "en" in SUPPORTED_LANGUAGES, "영어(en) 필수"
    assert "bn" in SUPPORTED_LANGUAGES, "벵골어(bn) 필수"
    print(f"  ✅ {len(SUPPORTED_LANGUAGES)}개 언어 지원 확인")
    print(f"     포함 언어: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")
except AssertionError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# ── 3. CropDoctorModel 인터페이스 확인 ───────────────────────────────────
print("\n[3/6] CropDoctorModel 인터페이스 확인...")
try:
    model_methods = inspect.getmembers(CropDoctorModel, predicate=inspect.isfunction)
    method_names = [m[0] for m in model_methods]

    required_methods = ["analyze_image", "analyze_with_audio", "get_supported_languages"]
    for method in required_methods:
        assert method in method_names, f"필수 메서드 없음: {method}"
        print(f"  ✅ CropDoctorModel.{method}() 확인")
except AssertionError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# ── 4. _TransformersBackend — Kaggle 네이티브 방식 확인 ──────────────────
print("\n[4/6] _TransformersBackend Kaggle 네이티브 방식 확인...")

# 4a. KAGGLE_MODEL_ID 확인
try:
    assert hasattr(_TransformersBackend, "KAGGLE_MODEL_ID"), (
        "_TransformersBackend에 KAGGLE_MODEL_ID 속성이 없습니다."
    )
    assert _TransformersBackend.KAGGLE_MODEL_ID == "google/gemma-4/transformers/gemma-4-e4b-it", (
        f"예상 KAGGLE_MODEL_ID: google/gemma-4/transformers/gemma-4-e4b-it\n"
        f"  실제: {_TransformersBackend.KAGGLE_MODEL_ID}"
    )
    print(f"  ✅ KAGGLE_MODEL_ID: {_TransformersBackend.KAGGLE_MODEL_ID}")
except AssertionError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# 4b. AutoModelForCausalLM 사용 확인 (ImageTextToText 금지)
try:
    init_src = inspect.getsource(_TransformersBackend.__init__)
    assert "AutoModelForCausalLM" in init_src, (
        "❌ AutoModelForCausalLM 미사용! Kaggle 공식은 CausalLM을 사용합니다."
    )
    assert "AutoModelForImageTextToText" not in init_src, (
        "❌ 구버전 AutoModelForImageTextToText 감지! AutoModelForCausalLM으로 교체하세요."
    )
    print("  ✅ AutoModelForCausalLM 사용 확인 (ImageTextToText 미사용)")
except AssertionError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# 4c. kagglehub.model_download 패턴 확인
try:
    assert "kagglehub" in init_src, (
        "❌ kagglehub 임포트 없음! kagglehub.model_download()를 사용하세요."
    )
    assert "model_download" in init_src, (
        "❌ kagglehub.model_download() 호출 없음!"
    )
    print("  ✅ kagglehub.model_download() 패턴 사용 확인")
except AssertionError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# 4d. tokenize=False 방식 확인
try:
    infer_src = inspect.getsource(_TransformersBackend.infer)
    assert "tokenize=False" in infer_src, (
        "❌ tokenize=False 방식 미사용!\n"
        "  Kaggle 공식: apply_chat_template(tokenize=False) → processor(text=text) 분리 호출"
    )
    assert "tokenize=True" not in infer_src, (
        "❌ 구버전 tokenize=True 방식 감지! tokenize=False로 교체하세요."
    )
    assert "enable_thinking=False" in infer_src, (
        "❌ enable_thinking=False 미설정! Gemma 4 신기능입니다."
    )
    print("  ✅ tokenize=False + enable_thinking=False 방식 확인 (Kaggle 공식)")
except AssertionError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# ── 5. _APIBackend 신 SDK 사용 확인 ──────────────────────────────────────
print("\n[5/6] _APIBackend deprecated SDK 미사용 확인...")
try:
    api_init_src = inspect.getsource(_APIBackend.__init__)
    api_infer_src = inspect.getsource(_APIBackend.infer)
    full_src = api_init_src + api_infer_src

    assert "google.generativeai" not in full_src, (
        "❌ deprecated SDK 'google.generativeai' 사용 감지! "
        "google-genai (google.genai)로 교체하세요."
    )
    print("  ✅ deprecated 'google.generativeai' 미사용 확인")
except AssertionError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# ── 6. _APIBackend 신 SDK 사용 확인 ──────────────────────────────────────
print("\n[6/6] _APIBackend 신 SDK(google-genai) 사용 확인...")
try:
    assert "google.genai" in full_src or "genai.Client" in full_src, (
        "❌ 신 SDK 'google.genai' 또는 'genai.Client' 미사용. "
        "_APIBackend를 google-genai SDK로 교체하세요."
    )
    print("  ✅ 신 SDK(google-genai / genai.Client) 사용 확인")

    call_src = inspect.getsource(_APIBackend._call_with_retry)
    assert (
        "client.models.generate_content" in call_src
        or "_client.models.generate_content" in call_src
    ), "❌ client.models.generate_content 호출 방식 미확인."
    print("  ✅ client.models.generate_content() 호출 방식 확인")

except AssertionError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# ── 완료 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🎉 모든 검증 통과! (6/6 항목)")
print("")
print("✅ Kaggle 네이티브 방식 확인 완료:")
print(f"   - 모델 슬러그: {_TransformersBackend.KAGGLE_MODEL_ID}")
print(f"   - HF fallback: {_TransformersBackend.HF_MODEL_ID}")
print("   - AutoModelForCausalLM 사용 (ImageTextToText ❌)")
print("   - kagglehub.model_download() 사용")
print("   - tokenize=False → processor(text=text) 분리 호출")
print("   - enable_thinking=False (Gemma 4 신기능)")
print("=" * 60)
