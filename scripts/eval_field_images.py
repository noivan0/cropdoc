#!/usr/bin/env python3.10
"""
Field Image Evaluation Script
수집된 실제 현장 이미지로 v13 CropDoc 모델 정확도 측정
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

FIELD_IMG_DIR = Path(PROJECT_ROOT) / "data" / "field_images"
LABELS_FILE = FIELD_IMG_DIR / "field_labels.json"
OUTPUT_REPORT = Path(PROJECT_ROOT) / "docs" / "FIELD_TEST_REPORT.md"

# 카테고리 → 기대 레이블 매핑 (eval_harness와 동일한 형식)
CATEGORY_TO_EXPECTED = {
    "tomato/healthy": "Healthy Tomato",
    "tomato/early_blight": "Tomato Early Blight",
    "tomato/late_blight": "Tomato Late Blight",
    "tomato/bacterial_spot": "Tomato Bacterial Spot",
    "tomato/leaf_mold": "Tomato Leaf Mold",
    "tomato/septoria": "Tomato Septoria Leaf Spot",
    "potato/healthy": "Healthy Potato",
    "potato/early_blight": "Potato Early Blight",
    "potato/late_blight": "Potato Late Blight",
    "pepper/healthy": "Healthy Pepper",
    "pepper/bacterial_spot": "Pepper Bacterial Spot",
    "apple/healthy": "Healthy Apple",
    "apple/scab": "Apple Scab",
    "apple/rust": "Apple Cedar Rust",
    "apple/black_rot": "Apple Black Rot",
    "corn/healthy": "Healthy Corn",
    "corn/gray_leaf_spot": "Corn Gray Leaf Spot",
    "corn/common_rust": "Corn Common Rust",
    "corn/northern_blight": "Corn Northern Blight",
    "grape/healthy": "Healthy Grape",
    "grape/black_rot": "Grape Black Rot",
    "grape/leaf_blight": "Grape Leaf Spot",
    "wheat/healthy": "Healthy Wheat",
    "wheat/stripe_rust": "Wheat Stripe Rust",
    "wheat/leaf_rust": "Wheat Leaf Rust",
    "strawberry/healthy": "Healthy Strawberry",
    "strawberry/leaf_scorch": "Strawberry Leaf Scorch",
    "blueberry/healthy": "Healthy Blueberry",
}

# 간단한 결과 판정 함수 (eval_harness 방식)
def is_correct(diagnosis: str, expected: str) -> bool:
    """진단 결과가 기대값과 일치하는지 확인"""
    diag_lower = diagnosis.lower()
    exp_lower = expected.lower()
    
    # 식물종 추출
    plant_words = {
        "tomato", "potato", "pepper", "apple", "corn", "grape",
        "wheat", "strawberry", "blueberry", "cherry", "peach", "soybean"
    }
    
    exp_parts = set(exp_lower.split())
    diag_parts = set(diag_lower.split())
    
    # 식물종 확인
    exp_plant = exp_parts & plant_words
    diag_plant = diag_parts & plant_words
    
    if not exp_plant.intersection(diag_plant):
        return False  # 식물종 불일치
    
    # Healthy 케이스
    if "healthy" in exp_lower:
        disease_words = {"blight", "spot", "mold", "mildew", "rust", "scab",
                        "wilt", "rot", "mosaic", "virus", "bacterial", "septoria"}
        return "healthy" in diag_lower and not (diag_parts & disease_words)
    
    # Disease 케이스
    disease_keywords = {w for w in exp_parts if w not in plant_words and w != "leaf" and len(w) > 3}
    return all(kw in diag_lower for kw in disease_keywords) if disease_keywords else False


def collect_field_images():
    """field_images 디렉토리에서 이미지-레이블 쌍 수집"""
    image_tasks = []
    
    for category, expected_label in CATEGORY_TO_EXPECTED.items():
        cat_dir = FIELD_IMG_DIR / category
        if not cat_dir.exists():
            continue
        
        img_files = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.JPG")) + \
                    list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.png"))
        
        for img_path in img_files:
            image_tasks.append({
                'path': str(img_path),
                'category': category,
                'expected': expected_label,
            })
    
    return image_tasks


def run_evaluation(max_images=None):
    """모델 평가 실행"""
    print("=" * 60)
    print("🧪 CropDoc Field Image Evaluation")
    print(f"⏱️  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 이미지 수집
    tasks = collect_field_images()
    print(f"\n📁 총 {len(tasks)}개 이미지 발견")
    
    if max_images:
        tasks = tasks[:max_images]
        print(f"   (최대 {max_images}개로 제한)")
    
    if not tasks:
        print("❌ 평가할 이미지가 없습니다!")
        return None
    
    # 모델 로드
    print("\n🤖 v13 모델 로딩 중...")
    try:
        from cropdoc_infer import diagnose
        print("  ✅ 모델 로드 완료")
    except Exception as e:
        print(f"  ❌ 모델 로드 실패: {e}")
        traceback.print_exc()
        return None
    
    # 추론 실행
    results = []
    correct_count = 0
    total_time = 0
    
    by_category = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})
    by_plant = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    print(f"\n🔍 추론 시작 ({len(tasks)}개 이미지)")
    print("-" * 60)
    
    for i, task in enumerate(tasks, 1):
        img_path = task['path']
        expected = task['expected']
        category = task['category']
        plant = category.split('/')[0]
        
        print(f"[{i:3d}/{len(tasks)}] {Path(img_path).name[:40]}")
        print(f"         Expected: {expected}")
        
        start = time.time()
        try:
            diagnosis = diagnose(img_path, lang="en")
            elapsed = time.time() - start
            total_time += elapsed
            
            correct = is_correct(diagnosis, expected)
            if correct:
                correct_count += 1
                status = "✅"
            else:
                status = "❌"
                by_category[category]['errors'].append({
                    'image': Path(img_path).name,
                    'expected': expected,
                    'got': diagnosis[:100]
                })
            
            by_category[category]['total'] += 1
            by_plant[plant]['total'] += 1
            if correct:
                by_category[category]['correct'] += 1
                by_plant[plant]['correct'] += 1
            
            print(f"         Got:      {diagnosis[:80]}")
            print(f"         {status} ({elapsed:.1f}s)")
            
            results.append({
                'image': Path(img_path).name,
                'category': category,
                'expected': expected,
                'diagnosis': diagnosis,
                'correct': correct,
                'time_s': round(elapsed, 2)
            })
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"         ❌ ERROR: {str(e)[:80]} ({elapsed:.1f}s)")
            by_category[category]['total'] += 1
            by_plant[plant]['total'] += 1
            results.append({
                'image': Path(img_path).name,
                'category': category,
                'expected': expected,
                'diagnosis': f"ERROR: {e}",
                'correct': False,
                'time_s': round(elapsed, 2)
            })
        
        print()
    
    # 최종 통계
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    
    print("=" * 60)
    print(f"📊 전체 정확도: {accuracy:.1%} ({correct_count}/{total})")
    print(f"⏱️  평균 추론 시간: {avg_time:.1f}s")
    print("=" * 60)
    
    print("\n📊 카테고리별 정확도:")
    for cat in sorted(by_category.keys()):
        d = by_category[cat]
        acc = d['correct'] / d['total'] if d['total'] > 0 else 0
        bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
        print(f"  {cat:35s} {bar} {acc:.0%} ({d['correct']}/{d['total']})")
    
    print("\n🌿 식물별 정확도:")
    for plant in sorted(by_plant.keys()):
        d = by_plant[plant]
        acc = d['correct'] / d['total'] if d['total'] > 0 else 0
        print(f"  {plant:20s} {acc:.1%} ({d['correct']}/{d['total']})")
    
    return {
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total,
        'avg_time': avg_time,
        'by_category': dict(by_category),
        'by_plant': dict(by_plant),
        'results': results,
        'evaluated_at': datetime.now().isoformat()
    }


def write_report(eval_results):
    """Field Test 보고서 작성"""
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    
    if eval_results is None:
        # 모델 없이 수집 현황만 보고
        tasks = collect_field_images()
        by_cat = Counter(t['category'] for t in tasks)
        by_plant = Counter(t['category'].split('/')[0] for t in tasks)
        
        lines = [
            "# CropDoc Field Image Test Report",
            f"> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 📋 수집 현황",
            "",
            f"**총 수집 이미지**: {len(tasks)}장  ",
            f"**데이터 소스**: PlantDoc (GitHub), Wikimedia Commons, PlantVillage eval_set  ",
            f"**라이선스**: MIT (PlantDoc) / CC-BY 4.0 (PlantVillage) / CC (Wikimedia)",
            "",
            "## 📊 카테고리별 수집 현황",
            "",
            "| 카테고리 | 이미지 수 | 상태 |",
            "|---------|---------|------|",
        ]
        for cat, cnt in sorted(by_cat.items()):
            status = "✅" if cnt >= 3 else "⚠️"
            lines.append(f"| {cat} | {cnt} | {status} |")
        
        lines += [
            "",
            "## 🌿 식물별 수집 현황",
            "",
            "| 식물 | 이미지 수 | 상태 |",
            "|-----|---------|------|",
        ]
        for plant, cnt in sorted(by_plant.items()):
            status = "✅" if cnt >= 3 else "⚠️"
            lines.append(f"| {plant} | {cnt} | {status} |")
        
        lines += [
            "",
            "## ⚠️ 모델 추론 미실행",
            "",
            "> GPU 리소스 또는 모델 로드 실패로 추론을 실행하지 못했습니다.",
            "> 이미지 수집은 완료되었으며 field_labels.json에 메타데이터가 저장되어 있습니다.",
            "",
            "## 📁 저장 위치",
            "",
            "```",
            "data/field_images/",
            "├── tomato/   (healthy, early_blight, late_blight, bacterial_spot, leaf_mold, septoria)",
            "├── potato/   (healthy, early_blight, late_blight)",
            "├── pepper/   (healthy, bacterial_spot)",
            "├── apple/    (healthy, scab, rust)",
            "├── corn/     (healthy, gray_leaf_spot, common_rust, northern_blight)",
            "├── grape/    (healthy, black_rot)",
            "├── strawberry/ (healthy)",
            "├── blueberry/  (healthy)",
            "└── field_labels.json",
            "```",
            "",
            "## 🔍 주요 데이터셋",
            "",
            "### PlantDoc (주요 소스)",
            "- **저장소**: https://github.com/pratikkayal/PlantDoc-Dataset",
            "- **특징**: 실제 현장에서 촬영된 이미지 (field conditions)",
            "- **라이선스**: MIT",
            "- **종류**: 27개 카테고리, 2,569개 이미지",
            "",
            "### PlantVillage (보완 소스)",
            "- **특징**: 통제된 실험실 환경 이미지 (기준선 측정용)",
            "- **라이선스**: CC BY 4.0",
            "- **용도**: eval_set에서 누락된 카테고리 보완",
        ]
        
        report_text = "\n".join(lines)
        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n📄 보고서 저장: {OUTPUT_REPORT}")
        return
    
    # 모델 추론 결과 포함 보고서
    acc = eval_results['accuracy']
    by_cat = eval_results['by_category']
    by_plant = eval_results['by_plant']
    
    lines = [
        "# CropDoc Field Image Test Report",
        f"> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 📊 Executive Summary",
        "",
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| **전체 정확도** | **{acc:.1%}** ({eval_results['correct']}/{eval_results['total']}) |",
        f"| 평균 추론 시간 | {eval_results['avg_time']:.1f}s |",
        f"| 테스트 이미지 | {eval_results['total']}장 |",
        f"| 데이터 소스 | PlantDoc (field) + PlantVillage (lab) |",
        "",
        "## 📊 카테고리별 정확도",
        "",
        "| 카테고리 | 정확도 | 맞춤 | 전체 |",
        "|---------|--------|-----|------|",
    ]
    
    for cat in sorted(by_cat.keys()):
        d = by_cat[cat]
        cat_acc = d['correct'] / d['total'] if d['total'] > 0 else 0
        emoji = "✅" if cat_acc >= 0.7 else ("⚠️" if cat_acc >= 0.4 else "❌")
        lines.append(f"| {emoji} {cat} | {cat_acc:.0%} | {d['correct']} | {d['total']} |")
    
    lines += [
        "",
        "## 🌿 식물별 정확도",
        "",
        "| 식물 | 정확도 | 맞춤 | 전체 |",
        "|-----|--------|-----|------|",
    ]
    
    for plant in sorted(by_plant.keys()):
        d = by_plant[plant]
        p_acc = d['correct'] / d['total'] if d['total'] > 0 else 0
        emoji = "✅" if p_acc >= 0.7 else ("⚠️" if p_acc >= 0.4 else "❌")
        lines.append(f"| {emoji} {plant} | {p_acc:.0%} | {d['correct']} | {d['total']} |")
    
    # 오류 패턴 분석
    lines += [
        "",
        "## 🔍 오류 패턴 분석",
        "",
    ]
    
    error_cats = [(cat, d) for cat, d in by_cat.items() if d['errors']]
    if error_cats:
        for cat, d in sorted(error_cats, key=lambda x: -len(x[1]['errors'])):
            lines.append(f"### {cat}")
            for err in d['errors'][:3]:
                lines.append(f"- **{err['image']}**")
                lines.append(f"  - 기대: `{err['expected']}`")
                lines.append(f"  - 실제: `{err['got']}`")
            lines.append("")
    else:
        lines.append("> 오류 없음 (100% 정확도)")
    
    lines += [
        "## 📁 수집 데이터 현황",
        "",
        "```",
        "data/field_images/",
        "└── [각 카테고리별 폴더]",
        "    field_labels.json  ← 메타데이터",
        "```",
        "",
        "## 📌 결론 및 권고사항",
        "",
    ]
    
    if acc >= 0.8:
        lines.append(f"✅ **현장 이미지에서도 {acc:.0%} 정확도 달성** — 모델이 실제 환경에 잘 일반화됨")
    elif acc >= 0.6:
        lines.append(f"⚠️ **현장 이미지 정확도 {acc:.0%}** — 일부 개선 필요")
    else:
        lines.append(f"❌ **현장 이미지 정확도 {acc:.0%}** — 유의미한 도메인 갭 존재")
    
    # 개선 필요 카테고리
    weak_cats = [(cat, d['correct']/d['total']) for cat, d in by_cat.items() 
                 if d['total'] > 0 and d['correct']/d['total'] < 0.5]
    if weak_cats:
        lines.append("")
        lines.append("**개선 우선순위:**")
        for cat, cat_acc in sorted(weak_cats, key=lambda x: x[1]):
            lines.append(f"- `{cat}`: {cat_acc:.0%} — 추가 학습 데이터 필요")
    
    report_text = "\n".join(lines)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n📄 보고서 저장: {OUTPUT_REPORT}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-images', type=int, default=None, help='최대 평가 이미지 수')
    parser.add_argument('--no-inference', action='store_true', help='추론 없이 수집 현황만 보고')
    args = parser.parse_args()
    
    if args.no_inference:
        print("📋 수집 현황 보고서만 생성합니다 (추론 없음)")
        write_report(None)
    else:
        eval_results = run_evaluation(max_images=args.max_images)
        write_report(eval_results)
