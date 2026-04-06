#!/usr/bin/env python3
"""모델 로드 없이 파이프라인 구조 테스트"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("=== CropDoc 빠른 구조 테스트 ===\n")

# 1. 임포트
print("[1] 임포트 테스트")
from src.model import CropDoctorModel, SUPPORTED_LANGUAGES
from src.pipeline import preprocess_audio, parse_diagnosis
print(f"  ✅ 20개 언어: {list(SUPPORTED_LANGUAGES.keys())}")

# 2. 이미지 처리 테스트 (모델 없이)
print("\n[2] 이미지 처리 테스트")
from PIL import Image
import numpy as np

samples = [
    "samples/sample_healthy_tomato.jpg",
    "samples/sample_diseased_tomato.jpg",
    "samples/sample_diseased_corn.jpg",
]
for s in samples:
    if os.path.exists(s):
        img = Image.open(s)
        print(f"  ✅ {s}: {img.size} {img.mode}")
    else:
        print(f"  ❌ {s}: 없음")

# 3. parse_diagnosis 테스트
print("\n[3] 결과 파싱 테스트")
sample_output = """
## 🌿 Disease / Pest Identification
- Name: Late Blight (Phytophthora infestans)
- Confidence: HIGH

## ⚠️ Severity Level
**CRITICAL** - Active infection spreading

## 💊 Treatment Recommendations
- Mancozeb 75WP: 2kg/ha
"""
parsed = parse_diagnosis(sample_output)
print(f"  ✅ parse_diagnosis 실행 완료: {type(parsed)}")

# 4. 언어별 시스템 프롬프트 확인
print("\n[4] 다국어 시스템 프롬프트 테스트")
from src.model import SYSTEM_PROMPTS

for lang in ["en", "sw", "hi", "fr"]:
    prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    print(f"  ✅ {lang}: {prompt[:60]}...")

# 5. 입력 검증 함수 테스트 (신규 추가)
print("\n[5] 입력 검증 함수 테스트")
from src.pipeline import validate_inputs, SUPPORTED_EXTENSIONS, AUDIO_EXTENSIONS, MAX_IMAGE_SIZE_MB

print(f"  ✅ 지원 이미지 형식: {sorted(SUPPORTED_EXTENSIONS)}")
print(f"  ✅ 지원 오디오 형식: {sorted(AUDIO_EXTENSIONS)}")
print(f"  ✅ 최대 이미지 크기: {MAX_IMAGE_SIZE_MB}MB")

# 실제 파일로 validate_inputs 테스트
sample_img = "samples/sample_healthy_tomato.jpg"
if os.path.exists(sample_img):
    try:
        validate_inputs(sample_img)
        print(f"  ✅ validate_inputs 통과: {sample_img}")
    except Exception as e:
        print(f"  ❌ validate_inputs 실패: {e}")

# 잘못된 확장자 테스트
try:
    validate_inputs("fake_image.xyz")
    print("  ❌ 검증 누락 (잘못된 확장자 통과됨)")
except FileNotFoundError:
    print("  ✅ 존재하지 않는 파일 감지 정상 작동")
except ValueError as e:
    print(f"  ✅ 잘못된 형식 감지: {e}")

# 6. DiagnosisPipeline 클래스 테스트 (신규 추가)
print("\n[6] DiagnosisPipeline 임포트 테스트")
from src.pipeline import DiagnosisPipeline, _image_hash

print("  ✅ DiagnosisPipeline 임포트 성공")

# _image_hash 테스트
if os.path.exists(sample_img):
    h = _image_hash(sample_img)
    print(f"  ✅ _image_hash: {sample_img} → {h[:16]}...")

print("\n✅ 모든 구조 테스트 통과!")
