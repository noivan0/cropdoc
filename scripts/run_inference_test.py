#!/usr/bin/env python3
"""
CropDoc 실제 추론 테스트 — Gemma 4 E4B-IT
모델 다운로드 완료 후 실행하세요.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SSL 우회 (사내 네트워크)
import requests
from requests.adapters import HTTPAdapter
_old_send = HTTPAdapter.send
def _patched_send(self, request, **kwargs):
    kwargs['verify'] = False
    return _old_send(self, request, **kwargs)
HTTPAdapter.send = _patched_send

import urllib3
urllib3.disable_warnings()

os.environ['KAGGLE_API_TOKEN'] = 'os.environ.get('KAGGLE_KEY', 'YOUR_KEY_HERE')'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import torch

def print_gpu_info():
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        used = torch.cuda.memory_allocated() / 1e9
        total = dev.total_memory / 1e9
        print(f"  GPU: {dev.name}")
        print(f"  VRAM: {used:.1f}/{total:.1f}GB ({used/total*100:.0f}%)")
    else:
        print("  GPU: 없음 (CPU 모드)")

print("=" * 65)
print("🌾 CropDoc — Gemma 4 E4B-IT 실제 추론 테스트")
print("=" * 65)
print_gpu_info()

# ─── 모델 로드 ───────────────────────────────────────────────
print("\n[1/4] 모델 로드 중...")
t0 = time.perf_counter()

from src.model import CropDoctorModel
model = CropDoctorModel(backend="transformers", use_kaggle=True)

load_time = time.perf_counter() - t0
print(f"  ✅ 로드 완료: {load_time:.1f}초")
print_gpu_info()

# ─── 테스트 1: 건강한 토마토 (영어) ─────────────────────────
print("\n[2/4] 건강한 토마토 진단 (영어)")
t0 = time.perf_counter()
result1 = model.analyze_image("samples/sample_healthy_tomato.jpg", language="en")
t1 = time.perf_counter() - t0
print(f"  추론 시간: {t1:.1f}초")
print(f"  결과 ({len(result1)}자):")
print("  " + result1[:500].replace("\n", "\n  "))

# ─── 테스트 2: 병든 토마토 (스와힐리) ───────────────────────
print("\n[3/4] 병든 토마토 진단 (Swahili)")
t0 = time.perf_counter()
result2 = model.analyze_image("samples/sample_diseased_tomato.jpg", language="sw")
t2 = time.perf_counter() - t0
print(f"  추론 시간: {t2:.1f}초")
print(f"  결과 ({len(result2)}자):")
print("  " + result2[:500].replace("\n", "\n  "))

# ─── 테스트 3: 캐시 히트 테스트 ─────────────────────────────
print("\n[4/4] 캐시 히트 테스트 (동일 이미지 재진단)")
t0 = time.perf_counter()
result3 = model.analyze_image("samples/sample_healthy_tomato.jpg", language="en")
t3 = time.perf_counter() - t0
cache_hit = t3 < 1.0
print(f"  재진단 시간: {t3:.3f}초 {'← 캐시 히트 ✅' if cache_hit else '← 캐시 미사용'}")

# ─── 요약 ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("📊 성능 요약")
print("=" * 65)
print(f"  모델 로드:  {load_time:.1f}초")
print(f"  1차 추론:   {t1:.1f}초 (건강한 토마토, 영어)")
print(f"  2차 추론:   {t2:.1f}초 (병든 토마토, Swahili)")
print(f"  캐시 재사용: {t3:.3f}초 ({'✅ 즉시 반환' if cache_hit else '캐시 미작동'})")
if torch.cuda.is_available():
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM 사용:  {used:.1f}/{total:.1f}GB ({used/total*100:.0f}%)")
print("=" * 65)
