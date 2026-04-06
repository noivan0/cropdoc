"""
CropDoc Gemma4 파인튜닝 데이터셋 준비
======================================
PlantVillage 10개 클래스에 대한 Alpaca 형식 텍스트 Q&A 데이터셋 생성.
멀티모달 파인튜닝은 Phase 2에서 진행.

출력: data/gemma4_finetune_dataset.json
"""

import json, os
from pathlib import Path

os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')

# ── 38종 식물 질병 정보 (한/영) ────────────────────────────────────────────────
DISEASE_INFO: dict[str, dict[str, str]] = {

    # ── 토마토 ──────────────────────────────────────────────────────────────────
    "Tomato Late Blight": {
        "ko": (
            "토마토 역병 (Tomato Late Blight)\n\n"
            "원인균: Phytophthora infestans\n"
            "증상: 잎에 수침상(물에 젖은 듯한) 갈색 반점이 나타나며, "
            "습한 조건에서 잎 뒷면에 흰색 곰팡이 포자층이 형성됩니다. "
            "병반은 빠르게 확산되어 잎 전체가 갈색으로 변하며 고사합니다.\n"
            "처방:\n"
            "  1. 만코제브(Mancozeb) 또는 메탈락실(Metalaxyl) 살균제 즉시 적용\n"
            "  2. 감염된 잎·줄기 즉시 제거 및 소각\n"
            "  3. 관수 방법을 점적관수로 전환 (잎 표면 젖음 방지)\n"
            "  4. 재배 밀도 줄여 통풍 확보\n"
            "긴급도: 🚨 즉시 (48시간 내 주변 전체 전파 가능)"
        ),
        "en": (
            "Tomato Late Blight\n\n"
            "Cause: Phytophthora infestans (oomycete)\n"
            "Symptoms: Water-soaked dark lesions on leaves and stems; "
            "white fuzzy sporulation on leaf underside in humid conditions. "
            "Spreads rapidly and can destroy entire crops within days.\n"
            "Treatment:\n"
            "  1. Apply Mancozeb or Metalaxyl fungicide immediately\n"
            "  2. Remove and destroy all infected plant material\n"
            "  3. Avoid overhead irrigation; switch to drip irrigation\n"
            "  4. Improve air circulation by reducing planting density\n"
            "Urgency: CRITICAL — can spread to entire field within 48 hours"
        ),
    },

    "Tomato Early Blight": {
        "ko": (
            "토마토 조기역병 (Tomato Early Blight)\n\n"
            "원인균: Alternaria solani\n"
            "증상: 하부 잎에서 시작하는 동심원 무늬의 갈색 반점(과녁 모양). "
            "반점 주위에 노란 테두리가 형성되며 잎이 황화·낙엽됩니다.\n"
            "처방:\n"
            "  1. 클로로탈로닐(Chlorothalonil) 또는 코퍼 계열 살균제 살포\n"
            "  2. 감염된 하부 잎 제거 (지면 접촉 방지)\n"
            "  3. 지면 멀칭으로 토양 전파 차단\n"
            "  4. 질소 과다시비 지양 (무름 현상 방지)\n"
            "긴급도: ⚠️ 1주일 내 조치 권장"
        ),
        "en": (
            "Tomato Early Blight\n\n"
            "Cause: Alternaria solani (fungus)\n"
            "Symptoms: Concentric ring lesions (target board pattern) on lower leaves first. "
            "Yellow halo surrounds dark brown spots; leaves yellow and drop.\n"
            "Treatment:\n"
            "  1. Apply Chlorothalonil or copper-based fungicide\n"
            "  2. Remove infected lower leaves; avoid ground contact\n"
            "  3. Mulch soil surface to prevent splash dispersal\n"
            "  4. Avoid excessive nitrogen fertilization\n"
            "Urgency: WARNING — treat within 1 week"
        ),
    },

    "Tomato Bacterial Spot": {
        "ko": (
            "토마토 세균성 반점 (Tomato Bacterial Spot)\n\n"
            "원인균: Xanthomonas vesicatoria\n"
            "증상: 잎에 작고 갈색이며 물에 젖은 듯한 반점 다수 발생. "
            "반점이 합쳐져 큰 병반 형성. 열매에도 검은 반점·코르크화 발생.\n"
            "처방:\n"
            "  1. 구리 살균제(Copper bactericide) 7~10일 간격 살포\n"
            "  2. 감염 식물체 즉시 제거\n"
            "  3. 이병종자 사용 금지, 무병 종자 또는 씨앗 소독\n"
            "  4. 작업 도구·장화 소독 (교차 감염 방지)\n"
            "긴급도: ⚠️ 조기 발견 시 방제 가능"
        ),
        "en": (
            "Tomato Bacterial Spot\n\n"
            "Cause: Xanthomonas vesicatoria (bacterium)\n"
            "Symptoms: Small water-soaked dark spots on leaves and fruit. "
            "Spots coalesce; fruit develops raised scab-like lesions.\n"
            "Treatment:\n"
            "  1. Apply copper bactericide every 7–10 days\n"
            "  2. Remove heavily infected plants\n"
            "  3. Use certified disease-free seeds; seed treatment before planting\n"
            "  4. Sanitize tools and footwear to prevent spread\n"
            "Urgency: WARNING — manageable if caught early"
        ),
    },

    "Tomato Leaf Mold": {
        "ko": (
            "토마토 잎곰팡이 (Tomato Leaf Mold)\n\n"
            "원인균: Passalora fulva (구 Cladosporium fulvum)\n"
            "증상: 잎 윗면에 연두빛 노란 반점, 잎 뒷면에 올리브 녹색~갈색 곰팡이층 형성. "
            "심한 경우 잎 전체 황화 및 낙엽.\n"
            "처방:\n"
            "  1. 온실 환기 강화, 습도 85% 이하 유지\n"
            "  2. 만코제브 또는 클로로탈로닐 살균제 살포\n"
            "  3. 저항성 품종 선택 (Cf 저항성 유전자)\n"
            "  4. 이른 아침 관수로 잎 건조 시간 확보\n"
            "긴급도: ℹ️ 온실 재배 시 주의"
        ),
        "en": (
            "Tomato Leaf Mold\n\n"
            "Cause: Passalora fulva (fungus)\n"
            "Symptoms: Pale green/yellow spots on upper leaf surface; "
            "olive-green to brown fuzzy mold on leaf undersides. "
            "Primarily a greenhouse disease; high humidity favors spread.\n"
            "Treatment:\n"
            "  1. Improve greenhouse ventilation; keep humidity below 85%\n"
            "  2. Apply Mancozeb or Chlorothalonil fungicide\n"
            "  3. Plant resistant varieties (Cf resistance genes)\n"
            "  4. Water in early morning to allow leaf drying\n"
            "Urgency: INFO — mainly greenhouse concern"
        ),
    },

    "Tomato Septoria Leaf Spot": {
        "ko": (
            "토마토 셉토리아 반점 (Tomato Septoria Leaf Spot)\n\n"
            "원인균: Septoria lycopersici\n"
            "증상: 하부 잎에서 시작, 흰~회색 중앙부와 갈색 테두리의 작은 원형 반점. "
            "반점 중앙에 검은 소립점(분생포자기) 관찰 가능. 급속 낙엽 진행.\n"
            "처방:\n"
            "  1. 클로로탈로닐 또는 만코제브 살포 (7~10일 간격)\n"
            "  2. 감염 하부 잎 제거\n"
            "  3. 멀칭으로 토양 비산 방지\n"
            "  4. 작물 잔재물 제거 후 심경\n"
            "긴급도: ⚠️ 조기 방제로 확산 억제 가능"
        ),
        "en": (
            "Tomato Septoria Leaf Spot\n\n"
            "Cause: Septoria lycopersici (fungus)\n"
            "Symptoms: Small circular spots with white/gray centers and dark margins, "
            "starting on lower leaves. Tiny black pycnidia visible in spot centers. "
            "Rapid defoliation can occur.\n"
            "Treatment:\n"
            "  1. Apply Chlorothalonil or Mancozeb every 7–10 days\n"
            "  2. Remove infected lower leaves\n"
            "  3. Mulch to prevent soil splash\n"
            "  4. Remove crop debris and till after season\n"
            "Urgency: WARNING — early control prevents rapid spread"
        ),
    },

    "Healthy Tomato": {
        "ko": (
            "건강한 토마토\n\n"
            "현황: 이 토마토 식물은 건강합니다. 가시적인 병해 증상이 없습니다.\n"
            "권고사항:\n"
            "  1. 정기적인 모니터링 지속 (주 2회 이상)\n"
            "  2. 예방적 구리 살균제 1달 1회 살포 (강우 후)\n"
            "  3. 적절한 영양 관리 (N-P-K 균형)\n"
            "  4. 통풍 확보, 과도한 고밀도 재배 지양\n"
            "긴급도: ✅ 정상"
        ),
        "en": (
            "Healthy Tomato\n\n"
            "Status: This tomato plant appears healthy with no visible disease symptoms.\n"
            "Recommendations:\n"
            "  1. Continue regular monitoring (at least twice weekly)\n"
            "  2. Apply preventive copper fungicide monthly (especially after rain)\n"
            "  3. Maintain balanced nutrition (N-P-K balance)\n"
            "  4. Ensure adequate air circulation; avoid overcrowding\n"
            "Urgency: ✅ Normal — continue preventive care"
        ),
    },

    # ── 감자 ──────────────────────────────────────────────────────────────────
    "Potato Late Blight": {
        "ko": (
            "감자 역병 (Potato Late Blight)\n\n"
            "원인균: Phytophthora infestans\n"
            "증상: 잎에 수침상 갈색 병반, 습도 높을 때 잎 뒷면 흰색 포자층. "
            "줄기·덩이줄기로 급속 확산, 수확 후 저장 중에도 부패 진행.\n"
            "처방:\n"
            "  1. 만코제브 또는 아미스타(Azoxystrobin) 살균제 즉시 살포\n"
            "  2. 지상부 전체 제거 및 소각 (수확 전 2주)\n"
            "  3. 감염 덩이줄기 선별 폐기\n"
            "  4. 배수 개선, 과도한 잎 면 습윤 방지\n"
            "긴급도: 🚨 즉시 (역사적으로 대기근 원인)"
        ),
        "en": (
            "Potato Late Blight\n\n"
            "Cause: Phytophthora infestans (oomycete)\n"
            "Symptoms: Dark water-soaked lesions on leaves and stems; "
            "white sporulation on leaf undersides. Spreads to tubers causing rot. "
            "Historically responsible for the Irish Famine.\n"
            "Treatment:\n"
            "  1. Apply Mancozeb or Azoxystrobin fungicide immediately\n"
            "  2. Remove and destroy all above-ground plant material\n"
            "  3. Discard infected tubers; do not store with healthy ones\n"
            "  4. Improve field drainage and reduce leaf wetness\n"
            "Urgency: CRITICAL — can devastate entire crop rapidly"
        ),
    },

    "Potato Early Blight": {
        "ko": (
            "감자 조기역병 (Potato Early Blight)\n\n"
            "원인균: Alternaria solani\n"
            "증상: 하부 잎에서 동심원 갈색 반점(과녁 모양). "
            "잎 황화 후 낙엽, 주로 노쇠한 식물이나 스트레스 상태에서 발생.\n"
            "처방:\n"
            "  1. 클로로탈로닐 또는 만코제브 살포\n"
            "  2. 균형 시비 (특히 질소, 칼리)\n"
            "  3. 저항성 품종 재배\n"
            "  4. 작물 잔재물 관리\n"
            "긴급도: ⚠️ 수확량 감소 방지를 위해 1주 내 조치"
        ),
        "en": (
            "Potato Early Blight\n\n"
            "Cause: Alternaria solani (fungus)\n"
            "Symptoms: Concentric ring (target board) brown spots on lower/older leaves first. "
            "Yellowing and defoliation follow. Often affects stressed or aging plants.\n"
            "Treatment:\n"
            "  1. Apply Chlorothalonil or Mancozeb\n"
            "  2. Balanced fertilization (especially nitrogen and potassium)\n"
            "  3. Plant resistant varieties where available\n"
            "  4. Manage crop residues\n"
            "Urgency: WARNING — treat within 1 week to prevent yield loss"
        ),
    },

    "Healthy Potato": {
        "ko": (
            "건강한 감자\n\n"
            "현황: 이 감자 식물은 건강합니다. 병해 증상이 없습니다.\n"
            "권고사항:\n"
            "  1. 예방적 역병 방제 프로그램 유지\n"
            "  2. 정기적 모니터링 (특히 강우 후)\n"
            "  3. 저항성 품종 및 인증 씨감자 사용\n"
            "  4. 4~5년 윤작 실시\n"
            "긴급도: ✅ 정상"
        ),
        "en": (
            "Healthy Potato\n\n"
            "Status: This potato plant appears healthy with no visible disease.\n"
            "Recommendations:\n"
            "  1. Maintain preventive late blight spray program\n"
            "  2. Regular monitoring especially after rainfall\n"
            "  3. Use certified disease-free seed potatoes\n"
            "  4. Rotate crops every 4–5 years\n"
            "Urgency: ✅ Normal — continue preventive management"
        ),
    },

    # ── 고추 ──────────────────────────────────────────────────────────────────
    "Pepper Bacterial Spot": {
        "ko": (
            "고추 세균성 반점 (Pepper Bacterial Spot)\n\n"
            "원인균: Xanthomonas euvesicatoria\n"
            "증상: 잎에 작고 물에 젖은 듯한 갈색 반점 다수 발생. "
            "반점 주위 노란 테두리 형성. 열매에 융기된 갈색 반점·상처 발생. "
            "심할 경우 낙엽 및 낙과.\n"
            "처방:\n"
            "  1. 구리 살균제 (Copper hydroxide) 5~7일 간격 살포\n"
            "  2. 무병 씨앗 또는 씨앗 열탕 소독 (52°C, 30분)\n"
            "  3. 감염 식물체 제거 (흡연 후 손 세척 — 담배 모자이크 교차 주의)\n"
            "  4. 과습 방지, 적절한 이랑 배수\n"
            "긴급도: ⚠️ 수확기 직전 특히 주의"
        ),
        "en": (
            "Pepper Bacterial Spot\n\n"
            "Cause: Xanthomonas euvesicatoria (bacterium)\n"
            "Symptoms: Small dark water-soaked spots on leaves; "
            "yellow halo surrounds spots. Raised lesions on fruit. "
            "Severe cases cause defoliation and fruit drop.\n"
            "Treatment:\n"
            "  1. Apply copper hydroxide every 5–7 days\n"
            "  2. Use certified pathogen-free seeds; hot water seed treatment (52°C, 30 min)\n"
            "  3. Remove infected plants carefully (wash hands after handling tobacco)\n"
            "  4. Avoid waterlogging; ensure proper row drainage\n"
            "Urgency: WARNING — especially critical near harvest time"
        ),
    },
}


def build_dataset() -> list[dict]:
    """Alpaca 형식 파인튜닝 데이터셋 구성."""
    dataset = []

    for label, info in DISEASE_INFO.items():
        for lang, response in info.items():
            lang_name = "Korean" if lang == "ko" else "English"

            # Q&A 형식 1: 직접 진단 질문
            dataset.append({
                "instruction": f"A farmer shows you a plant image and asks about this condition: {label}",
                "input": f"Please provide diagnosis and treatment information in {lang_name}.",
                "output": response,
            })

            # Q&A 형식 2: 증상 기반 질문 (역방향 — 더 실용적)
            if lang == "en":
                dataset.append({
                    "instruction": "What plant disease causes these symptoms?",
                    "input": _get_symptom_description(label),
                    "output": response,
                })

            # Q&A 형식 3: 처방 요청
            if lang == "ko":
                dataset.append({
                    "instruction": f"농부가 {label} 진단을 받았습니다. 어떻게 해야 하나요?",
                    "input": "긴급도와 구체적인 처방을 알려주세요.",
                    "output": response,
                })

    return dataset


def _get_symptom_description(label: str) -> str:
    """레이블 → 증상 설명 (역방향 Q&A용)."""
    symptom_map = {
        "Tomato Late Blight":         "Dark water-soaked lesions on tomato leaves with white mold on undersides",
        "Tomato Early Blight":        "Concentric ring target-board brown spots on lower tomato leaves",
        "Tomato Bacterial Spot":      "Small dark water-soaked spots with yellow halos on tomato leaves and fruit",
        "Tomato Leaf Mold":           "Yellow spots on upper tomato leaf surface with olive-brown mold underneath",
        "Tomato Septoria Leaf Spot":  "Small white-centered circular spots with dark margins on tomato lower leaves",
        "Healthy Tomato":             "Green healthy tomato leaves with no visible spots or discoloration",
        "Potato Late Blight":         "Dark lesions on potato leaves and stems, white sporulation visible",
        "Potato Early Blight":        "Target-board concentric brown spots on lower/older potato leaves",
        "Healthy Potato":             "Healthy green potato plant foliage with no disease signs",
        "Pepper Bacterial Spot":      "Small dark spots with yellow halos on pepper leaves and fruit",
    }
    return symptom_map.get(label, f"Symptoms consistent with {label}")


def main():
    dataset = build_dataset()

    out_path = Path("data/gemma4_finetune_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"데이터셋 생성 완료: {len(dataset)}개 항목")
    print(f"저장 위치: {out_path}")

    # 통계
    by_lang = {"ko": 0, "en": 0, "other": 0}
    for item in dataset:
        inp = item.get("input", "")
        if "Korean" in inp or "농부" in item.get("instruction", ""):
            by_lang["ko"] += 1
        elif "English" in inp or "symptoms" in item.get("instruction", ""):
            by_lang["en"] += 1
        else:
            by_lang["other"] += 1

    print(f"  - 총 레이블: {len(DISEASE_INFO)}종")
    print(f"  - 한국어 항목: {by_lang['ko']}개")
    print(f"  - 영어 항목: {by_lang['en']}개")
    print(f"  - 기타: {by_lang['other']}개")

    # 샘플 출력
    print("\n[샘플 항목]")
    for item in dataset[:2]:
        print(f"  instruction: {item['instruction'][:60]}...")
        print(f"  output:      {item['output'][:80].strip()}...")
        print()

    return dataset


if __name__ == "__main__":
    main()
