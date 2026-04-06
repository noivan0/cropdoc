"""
CropDoc Gemma4 Enhanced — Gemma4 멀티모달 직접 이미지 분석 강화 버전
=======================================================================
기존 cropdoc_infer.py: CNN top-1 ≥ 0.90이면 Gemma4 skip
이 스크립트: Gemma4가 모든 이미지를 1차 분석 → CNN 앙상블이 검증

실행:
  python3.10 scripts/cropdoc_gemma4_enhanced.py
  python3.10 scripts/cropdoc_gemma4_enhanced.py --quick  # 5장만 빠른 테스트
  python3.10 scripts/cropdoc_gemma4_enhanced.py --full   # 전체 eval_set

GPU: A6000 (VRAM 48GB) — Gemma4 E4B bfloat16 + CNN 앙상블 동시 상주 가능
"""

import os, sys, ssl, json, time, argparse, re
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
import numpy as np

# ── SSL 우회 ──────────────────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context

# ── 작업 디렉토리 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
GEMMA_MODEL_PATH = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"
EVAL_SET_ROOT    = PROJECT_ROOT / "data/plantvillage/eval_set"
EVAL_LABELS_PATH = PROJECT_ROOT / "data/plantvillage/eval_labels.json"
RESULTS_DIR      = PROJECT_ROOT / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 레이블 정규화 맵 ───────────────────────────────────────────────────────────
# Gemma4 출력 → 표준 레이블
LABEL_ALIASES: dict[str, str] = {
    # Tomato Late Blight
    "tomato late blight":        "Tomato Late Blight",
    "late blight tomato":        "Tomato Late Blight",
    "tomato_late_blight":        "Tomato Late Blight",
    # Tomato Early Blight
    "tomato early blight":       "Tomato Early Blight",
    "early blight tomato":       "Tomato Early Blight",
    # Tomato Bacterial Spot
    "tomato bacterial spot":     "Tomato Bacterial Spot",
    "bacterial spot tomato":     "Tomato Bacterial Spot",
    # Tomato Leaf Mold
    "tomato leaf mold":          "Tomato Leaf Mold",
    "leaf mold":                 "Tomato Leaf Mold",
    # Tomato Septoria Leaf Spot
    "tomato septoria leaf spot": "Tomato Septoria Leaf Spot",
    "septoria leaf spot":        "Tomato Septoria Leaf Spot",
    "septoria":                  "Tomato Septoria Leaf Spot",
    # Healthy Tomato
    "healthy tomato":            "Healthy Tomato",
    "tomato healthy":            "Healthy Tomato",
    "tomato___healthy":          "Healthy Tomato",
    # Potato Late Blight
    "potato late blight":        "Potato Late Blight",
    "late blight potato":        "Potato Late Blight",
    # Potato Early Blight
    "potato early blight":       "Potato Early Blight",
    "early blight potato":       "Potato Early Blight",
    # Healthy Potato
    "healthy potato":            "Healthy Potato",
    "potato healthy":            "Healthy Potato",
    # Pepper Bacterial Spot
    "pepper bacterial spot":     "Pepper Bacterial Spot",
    "bacterial spot pepper":     "Pepper Bacterial Spot",
    "bell pepper bacterial spot": "Pepper Bacterial Spot",
}

VALID_LABELS = set(LABEL_ALIASES.values())

# ── Chain-of-Thought 강화 프롬프트 ────────────────────────────────────────────
GEMMA_COT_PROMPT = """You are an expert plant pathologist. Analyze this plant leaf image carefully.

Step 1: Identify the plant species (tomato, potato, pepper, apple, corn, grape, etc.)
Step 2: Examine the symptoms (color changes, spots, lesions, wilting, mold, etc.)
Step 3: Diagnose the disease or confirm healthy status.

Valid diagnoses:
- Tomato Late Blight
- Tomato Early Blight
- Tomato Bacterial Spot
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Healthy Tomato
- Potato Late Blight
- Potato Early Blight
- Healthy Potato
- Pepper Bacterial Spot

Respond in this EXACT format (no extra text):
PLANT: [plant name]
SYMPTOMS: [brief description]
DIAGNOSIS: [one of the valid diagnoses above]
CONFIDENCE: [high/medium/low]"""

# ── Gemma4 모델 싱글톤 ────────────────────────────────────────────────────────
_gemma_proc  = None
_gemma_model = None
_gemma_device = None
_gemma_load_failed = False


def load_gemma4() -> bool:
    """Gemma4 E4B-IT 로드. 이미 로드된 경우 재사용."""
    global _gemma_proc, _gemma_model, _gemma_device, _gemma_load_failed

    if _gemma_model is not None:
        return True
    if _gemma_load_failed:
        return False

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        print(f"[Gemma4] Loading from {GEMMA_MODEL_PATH} ...")
        t0 = time.time()

        _gemma_proc = AutoProcessor.from_pretrained(GEMMA_MODEL_PATH)
        _gemma_model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        _gemma_model.eval()
        _gemma_device = next(_gemma_model.parameters()).device

        elapsed = time.time() - t0
        vram = torch.cuda.memory_allocated(_gemma_device) / 1e9 if _gemma_device.type == 'cuda' else 0
        print(f"[Gemma4] Ready on {_gemma_device} in {elapsed:.1f}s | VRAM: {vram:.1f}GB")
        return True

    except Exception as e:
        print(f"[Gemma4] Load FAILED: {e}")
        _gemma_load_failed = True
        return False


def _parse_gemma_response(raw: str) -> dict:
    """Gemma4 응답 파싱 → 구조화된 결과."""
    result = {
        "plant": None,
        "symptoms": None,
        "diagnosis": None,
        "confidence": None,
        "raw": raw,
    }

    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith("PLANT:"):
            result["plant"] = line[6:].strip().strip("[]")
        elif line.startswith("SYMPTOMS:"):
            result["symptoms"] = line[9:].strip().strip("[]")
        elif line.startswith("DIAGNOSIS:"):
            result["diagnosis"] = line[10:].strip().strip("[]")
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line[11:].strip().strip("[]").lower()

    # 표준 레이블로 정규화
    if result["diagnosis"]:
        diag_lower = result["diagnosis"].lower()
        result["matched_label"] = LABEL_ALIASES.get(diag_lower, None)

        # partial match 시도
        if result["matched_label"] is None:
            for alias, label in LABEL_ALIASES.items():
                if alias in diag_lower or diag_lower in alias:
                    result["matched_label"] = label
                    break

    return result


def diagnose_with_gemma4_cot(image_path: str | Path) -> dict:
    """
    Chain-of-Thought Gemma4 직접 이미지 진단.

    Returns:
        {
            "plant": str,
            "symptoms": str,
            "diagnosis": str,  # 원문
            "matched_label": str | None,  # 표준 레이블
            "confidence": str,
            "raw": str,
            "latency_ms": float,
        }
    """
    if not load_gemma4():
        return {"error": "Gemma4 not available", "matched_label": None}

    image_path = Path(image_path)
    if not image_path.exists():
        return {"error": f"Image not found: {image_path}", "matched_label": None}

    img = Image.open(image_path).convert('RGB')

    # 멀티모달 메시지 구성 (image + CoT 프롬프트)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": GEMMA_COT_PROMPT}
    ]}]

    inputs = _gemma_proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_gemma_device)

    t0 = time.time()
    with torch.inference_mode():
        output = _gemma_model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=_gemma_proc.tokenizer.eos_token_id,
        )

    latency_ms = (time.time() - t0) * 1000
    input_len = inputs["input_ids"].shape[1]
    raw = _gemma_proc.decode(output[0][input_len:], skip_special_tokens=True).strip()

    result = _parse_gemma_response(raw)
    result["latency_ms"] = latency_ms
    return result


# ── 배치 평가 함수 ────────────────────────────────────────────────────────────

def run_evaluation(images: list[tuple[Path, str]], mode: str = "gemma4_primary") -> dict:
    """
    이미지 목록에 대해 Gemma4 직접 진단 실행.

    Args:
        images: [(image_path, true_label), ...]
        mode: 실행 모드 식별자

    Returns:
        {
            "accuracy": float,
            "results": [...],
            "per_class": {...},
            "latency_avg_ms": float,
        }
    """
    print(f"\n[Eval] {len(images)}장 Gemma4 직접 진단 시작 (mode={mode})")
    print("=" * 60)

    all_results = []
    correct = 0
    per_class: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "errors": []})
    latencies = []

    for i, (img_path, true_label) in enumerate(images, 1):
        result = diagnose_with_gemma4_cot(img_path)

        matched = result.get("matched_label")
        is_correct = (matched == true_label)
        if is_correct:
            correct += 1

        latency = result.get("latency_ms", 0)
        latencies.append(latency)

        per_class[true_label]["total"] += 1
        if is_correct:
            per_class[true_label]["correct"] += 1
        else:
            per_class[true_label]["errors"].append({
                "path": str(img_path),
                "predicted": matched,
                "raw": result.get("raw", "")[:100],
            })

        status = "✅" if is_correct else "❌"
        print(f"[{i:3d}/{len(images)}] {status} "
              f"true={true_label!r:30s} "
              f"pred={str(matched)!r:30s} "
              f"conf={result.get('confidence', '?')!r} "
              f"lat={latency:.0f}ms")

        all_results.append({
            "image": str(img_path),
            "true_label": true_label,
            "predicted": matched,
            "correct": is_correct,
            "confidence": result.get("confidence"),
            "symptoms": result.get("symptoms"),
            "latency_ms": latency,
            "raw": result.get("raw", ""),
        })

    accuracy = correct / len(images) if images else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print("\n" + "=" * 60)
    print(f"[Eval] 완료: {correct}/{len(images)} = {accuracy:.1%} | 평균 지연: {avg_lat:.0f}ms")

    # 클래스별 정확도
    print("\n[클래스별 정확도]")
    for label in sorted(per_class.keys()):
        info = per_class[label]
        cls_acc = info["correct"] / info["total"] if info["total"] else 0
        print(f"  {label:35s}: {info['correct']}/{info['total']} = {cls_acc:.1%}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(images),
        "results": all_results,
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "latency_avg_ms": avg_lat,
        "mode": mode,
    }


def quick_test() -> dict:
    """5장으로 빠른 동작 검증."""
    eval_labels = json.loads(EVAL_LABELS_PATH.read_text())

    # 클래스별 1장씩 최대 5종
    by_class: dict[str, list] = defaultdict(list)
    for rel_path, label in eval_labels.items():
        by_class[label].append(rel_path)

    test_images = []
    for label in sorted(by_class.keys())[:5]:
        rel = by_class[label][0]
        abs_path = EVAL_SET_ROOT / rel
        if abs_path.exists():
            test_images.append((abs_path, label))

    print(f"[QuickTest] {len(test_images)}장으로 빠른 검증")
    return run_evaluation(test_images, mode="quick_test")


def full_eval() -> dict:
    """전체 eval_set 실행."""
    eval_labels = json.loads(EVAL_LABELS_PATH.read_text())
    all_images = []
    for rel_path, label in eval_labels.items():
        abs_path = EVAL_SET_ROOT / rel_path
        if abs_path.exists():
            all_images.append((abs_path, label))

    print(f"[FullEval] 전체 {len(all_images)}장 평가")
    return run_evaluation(all_images, mode="gemma4_primary_full")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CropDoc Gemma4 Enhanced — 직접 이미지 진단")
    parser.add_argument("--quick",  action="store_true", help="5장 빠른 테스트 (기본)")
    parser.add_argument("--full",   action="store_true", help="전체 eval_set 실행")
    parser.add_argument("--image",  type=str,            help="단일 이미지 진단 경로")
    args = parser.parse_args()

    if args.image:
        # 단일 이미지 진단
        result = diagnose_with_gemma4_cot(args.image)
        print("\n[단일 진단 결과]")
        for k, v in result.items():
            if k != "raw":
                print(f"  {k}: {v}")
        print(f"  raw:\n{result.get('raw', '')}")
        return

    if args.full:
        results = full_eval()
    else:
        # 기본: quick test
        results = quick_test()

    # 결과 저장
    out_path = RESULTS_DIR / "gemma4_enhanced_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")

    return results


if __name__ == "__main__":
    main()
