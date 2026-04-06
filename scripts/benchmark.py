#!/usr/bin/env python3
"""
CropDoc 성능 벤치마크
- 추론 시간 측정
- 메모리 사용량 측정
- 3개 샘플 이미지 진단 결과 출력
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from src.model import CropDoctorModel


def benchmark():
    print("=" * 60)
    print("CropDoc 성능 벤치마크")
    print("=" * 60)

    # GPU 정보
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("GPU: 미감지 (CPU 모드)")

    # 모델 로드 시간
    print("\n[1] 모델 로드...")
    t0 = time.perf_counter()
    model = CropDoctorModel(backend="auto")
    load_time = time.perf_counter() - t0
    print(f"  로드 시간: {load_time:.1f}초")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU 메모리: {used:.1f}/{total:.1f}GB ({used / total * 100:.0f}%)")

    # 샘플 이미지 진단
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
    samples = [
        ("sample_healthy_tomato.jpg", "en", "건강한 토마토"),
        ("sample_diseased_tomato.jpg", "en", "병든 토마토"),
        ("sample_diseased_corn.jpg", "sw", "병든 옥수수 (Swahili)"),
    ]

    print("\n[2] 진단 성능 테스트")
    times = []
    for fname, lang, desc in samples:
        fpath = os.path.join(sample_dir, fname)
        if not os.path.exists(fpath):
            print(f"  건너뜀: {fname} (파일 없음)")
            continue

        print(f"\n  [{desc}]")
        t0 = time.perf_counter()
        try:
            result = model.analyze_image(fpath, language=lang)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  추론 시간: {elapsed:.1f}초")
            diagnosis = result.get("diagnosis", "") if isinstance(result, dict) else str(result)
            print(f"  진단 결과 (첫 300자):")
            print(f"  {diagnosis[:300]}...")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  ❌ 오류 ({elapsed:.1f}초): {e}")

    if times:
        print(f"\n[3] 성능 요약")
        print(f"  평균 추론 시간: {sum(times) / len(times):.1f}초")
        print(f"  최소: {min(times):.1f}초, 최대: {max(times):.1f}초")

    print("\n✅ 벤치마크 완료")


if __name__ == "__main__":
    benchmark()
