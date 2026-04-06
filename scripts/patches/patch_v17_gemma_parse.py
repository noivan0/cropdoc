"""
CropDoc v17 패치: Gemma4 파싱 개선 + Late Blight 프롬프트 + 클래스별 Threshold
==========================================================================

적용 방법:
    python3 scripts/patches/patch_v17_gemma_parse.py

이 스크립트는 cropdoc_infer.py를 직접 수정하지 않고
개선된 함수들을 export합니다. cropdoc_infer.py에 import하거나
copy-paste해서 사용하세요.

변경 사항 (v16 → v17):
  1. _parse_gemma_response(): 줄바꿈/언더스코어 처리 개선
  2. GEMMA_SYSTEM: Late Blight 시각적 특징 힌트 추가
  3. CNN_CLASS_THRESHOLDS: 클래스별 threshold 차별화
  4. _gemma_verify(): 개선된 파싱 함수 사용
"""

import re

# ══════════════════════════════════════════════════════════════════════════════
# 1. 개선된 파싱 함수
# ══════════════════════════════════════════════════════════════════════════════

def _parse_gemma_response(resp: str, valid_labels: set) -> str | None:
    """
    개선된 Gemma4 응답 파싱.

    기존 버그:
      - 'Potato\\nLate Blight' → None (줄바꿈 삽입 시 실패)
      - 'Potato_Late_Blight'   → None (언더스코어 형식 실패)

    개선:
      - 1차: 정규화 후 substring 매칭 (긴 레이블 우선)
      - 2차: 퍼지 매칭 (80% 단어 겹침)
    """
    # 정규화: 줄바꿈→공백, 언더스코어→공백, 다중공백→단일
    norm = re.sub(r'[\n\r]+', ' ', resp)
    norm = re.sub(r'[_]+', ' ', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()

    # 1차: 정확한 substring 매칭 (긴 레이블 우선 — 모호성 방지)
    for lbl in sorted(valid_labels, key=len, reverse=True):
        if lbl.lower() in norm.lower():
            return lbl

    # 2차: 퍼지 매칭 (80% 단어 겹침)
    norm_words = set(norm.lower().split())
    best_lbl, best_score = None, 0
    for lbl in valid_labels:
        lbl_words = set(lbl.lower().split())
        if not lbl_words:
            continue
        overlap = len(lbl_words & norm_words)
        if overlap > best_score and overlap >= len(lbl_words) * 0.8:
            best_score = overlap
            best_lbl = lbl

    return best_lbl  # None이면 호출부에서 CNN top1 사용


# ══════════════════════════════════════════════════════════════════════════════
# 2. 개선된 GEMMA_SYSTEM 프롬프트
# ══════════════════════════════════════════════════════════════════════════════

GEMMA_SYSTEM_V17 = """You are CropDoc, an expert plant pathologist AI.
A CNN model analyzed this plant leaf image:
{cnn_hints}

Examine the image carefully and determine the most accurate disease label.

KEY VISUAL DISAMBIGUATION RULES:
• Potato Late Blight: Large DARK water-soaked lesions (green→brown), white fluffy mold on leaf UNDERSIDE, fast irregular spread, affects entire leaf sections
• Potato Early Blight: Brown spots with CONCENTRIC RINGS (bullseye/target pattern), yellow chlorotic HALO around spots, usually smaller discrete spots
• Tomato Late Blight: Large irregular dark-green to BROWN water-soaked lesions, gray mold on underside, oily appearance
• Tomato Early Blight: CONCENTRIC ring brown spots with yellow halo, angular/irregular shape, starts on lower leaves
• Note: Late Blight lesions are typically LARGER and more water-soaked; Early Blight shows distinct RING patterns

Output ONLY one exact label from the list below (no explanation, no punctuation):
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


# ══════════════════════════════════════════════════════════════════════════════
# 3. 클래스별 CNN Threshold
# ══════════════════════════════════════════════════════════════════════════════

# Late Blight / Early Blight는 혼동이 많으므로 낮은 threshold 적용
# → Gemma4 검증을 더 자주 사용하여 정확도 향상
CNN_CLASS_THRESHOLDS = {
    # Potato Blight 그룹: 가장 혼동 많음
    "Potato Late Blight":   0.80,   # 기본 0.90 → 0.80 (Gemma4 더 자주 검증)
    "Potato Early Blight":  0.80,
    # Tomato Blight 그룹
    "Tomato Late Blight":   0.82,
    "Tomato Early Blight":  0.82,
    # 나머지: 기존 threshold 유지 (아래 코드에서 fallback 처리)
}

CNN_HIGH_CONF_DEFAULT = 0.90     # 기존 10종 기본값
CNN_HIGH_CONF_NEW_DEFAULT = 0.97  # 신규 확장 레이블 기본값


def get_threshold(label: str, original_labels: set) -> float:
    """클래스별 최적화된 threshold 반환."""
    if label in CNN_CLASS_THRESHOLDS:
        return CNN_CLASS_THRESHOLDS[label]
    if label in original_labels:
        return CNN_HIGH_CONF_DEFAULT
    return CNN_HIGH_CONF_NEW_DEFAULT


# ══════════════════════════════════════════════════════════════════════════════
# 4. cropdoc_infer.py 적용 방법 (diff 형태)
# ══════════════════════════════════════════════════════════════════════════════

PATCH_INSTRUCTIONS = """
=== cropdoc_infer.py 수정 방법 ===

[수정 1] GEMMA_SYSTEM 교체:
  기존: GEMMA_SYSTEM = \"\"\"You are CropDoc. ...\"\"\"
  신규: GEMMA_SYSTEM_V17 (위 코드 참조)

[수정 2] _gemma_verify() 내 파싱 로직 교체:
  기존:
    matched = None
    for lbl in VALID_LABELS:
        if lbl.lower() in resp.lower():
            matched = lbl
            break
  
  신규:
    matched = _parse_gemma_response(resp, VALID_LABELS)

[수정 3] diagnose() 내 threshold 계산 교체:
  기존:
    threshold = CNN_HIGH_CONF if top1_lbl in ORIGINAL_LABELS else CNN_HIGH_CONF_NEW
  
  신규:
    threshold = get_threshold(top1_lbl, ORIGINAL_LABELS)
"""

if __name__ == "__main__":
    print("=== CropDoc v17 패치 내용 ===")
    print(PATCH_INSTRUCTIONS)
    print()
    print("테스트: 파싱 함수 검증")

    # 테스트 케이스
    from scripts.patches.patch_v17_gemma_parse import _parse_gemma_response

    VALID = {
        'Tomato Early Blight', 'Tomato Late Blight', 'Potato Early Blight',
        'Potato Late Blight', 'Healthy Potato', 'Healthy Tomato',
    }

    test_cases = [
        ('Potato Late Blight', 'Potato Late Blight'),
        ('Potato\nLate Blight', 'Potato Late Blight'),    # 줄바꿈 버그 수정
        ('Potato_Late_Blight', 'Potato Late Blight'),     # 언더스코어 버그 수정
        ('DIAGNOSIS: Tomato Late Blight', 'Tomato Late Blight'),
        ('POTATO LATE BLIGHT', 'Potato Late Blight'),
        ('potato early blight', 'Potato Early Blight'),
        ('Tomato\nEarly\nBlight', 'Tomato Early Blight'),  # 여러 줄바꿈
    ]

    passed = 0
    for resp, expected in test_cases:
        result = _parse_gemma_response(resp, VALID)
        status = "✅" if result == expected else "❌"
        print(f"{status} {repr(resp)[:40]:42} → {result} (expected: {expected})")
        if result == expected:
            passed += 1

    print(f"\n{passed}/{len(test_cases)} 테스트 통과")
