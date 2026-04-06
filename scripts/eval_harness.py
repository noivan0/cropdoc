"""
CropDoc Evaluation Harness — v3.1 (Extended Labels)
========================================================
⚠️  DO NOT MODIFY THIS FILE  ⚠️
v3.1: 확장 레이블 LABEL_RULES 추가 (Apple/Corn/Grape/Peach/Strawberry/Cherry/Squash/기타)

판정 원칙 (명확하고 일관성 있는 기준):
  1. 식물 종(tomato / potato / pepper) 반드시 응답에 포함 — 없으면 오답
  2. Healthy 레이블: 'healthy' 포함 + 식물종 포함 + disease 단어 없어야 함
     ※ negation 제거 ("no spots" → spots 카운트 안 함)
  3. Disease 레이블: 식물종 포함 + 핵심 disease 키워드 ALL 포함
     ※ negation 제거 후 매칭
  4. "leaf"는 disease 키워드에서 제외 (너무 일반적)
  5. 식물종 혼동은 오답 (Tomato Early Blight ≠ Potato Early Blight)

Usage:
    python eval_harness.py [--max-images N]

Output (grep-friendly):
    diagnosis_accuracy: 0.734
    total_images: 300
    correct: 220
    inference_time_avg: 1.3
    peak_vram_mb: 2560.0
"""

import os, sys, re, json, time, argparse, traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

EVAL_SET_BASE = os.path.join(PROJECT_ROOT, "data", "plantvillage", "eval_set")
LABELS_PATH   = os.path.join(PROJECT_ROOT, "data", "plantvillage", "eval_labels.json")

# ── Disease word set (Healthy 판정 시 이 중 하나라도 있으면 오답) ─────────────
DISEASE_WORDS = {
    "blight", "spot", "mold", "mildew", "rust", "scab",
    "wilt", "rot", "mosaic", "virus", "bacterial", "septoria",
    "alternaria", "phytophthora", "fungal", "infection", "disease",
    "necrosis", "lesion", "lesions",
}

# ── 레이블별 판정 규칙: (식물종 집합, 필수 disease 키워드 집합, is_healthy) ───
LABEL_RULES = {
    "Tomato Early Blight":       ({"tomato"},  {"early", "blight"},   False),
    "Tomato Late Blight":        ({"tomato"},  {"late",  "blight"},   False),
    "Tomato Bacterial Spot":     ({"tomato"},  {"bacterial", "spot"}, False),
    "Tomato Leaf Mold":          ({"tomato"},  {"mold"},              False),
    "Tomato Septoria Leaf Spot": ({"tomato"},  {"septoria"},          False),
    "Healthy Tomato":            ({"tomato"},  set(),                 True),
    "Potato Early Blight":       ({"potato"},  {"early", "blight"},   False),
    "Potato Late Blight":        ({"potato"},  {"late",  "blight"},   False),
    "Healthy Potato":            ({"potato"},  set(),                 True),
    "Pepper Bacterial Spot":     ({"pepper"},  {"bacterial", "spot"}, False),
    "Healthy Pepper":            ({"pepper"},  set(),                 True),

    # ── Apple ──────────────────────────────────────────────────────────────
    "Apple Scab":                ({"apple"},   {"scab"},              False),
    "Apple Black Rot":           ({"apple"},   {"black", "rot"},      False),
    "Apple Cedar Rust":          ({"apple"},   {"rust"},              False),
    "Healthy Apple":             ({"apple"},   set(),                 True),

    # ── Corn (Maize) ───────────────────────────────────────────────────────
    "Corn Gray Leaf Spot":       ({"corn"},    {"gray", "spot"},      False),
    "Corn Common Rust":          ({"corn"},    {"rust"},              False),
    "Corn Northern Blight":      ({"corn"},    {"blight"},            False),
    "Healthy Corn":              ({"corn"},    set(),                 True),

    # ── Grape ──────────────────────────────────────────────────────────────
    "Grape Black Rot":           ({"grape"},   {"black", "rot"},      False),
    "Grape Esca":                ({"grape"},   {"esca"},              False),
    "Grape Leaf Spot":           ({"grape"},   {"spot"},              False),
    "Healthy Grape":             ({"grape"},   set(),                 True),

    # ── Peach ──────────────────────────────────────────────────────────────
    "Peach Bacterial Spot":      ({"peach"},   {"bacterial", "spot"}, False),
    "Healthy Peach":             ({"peach"},   set(),                 True),

    # ── Strawberry ─────────────────────────────────────────────────────────
    "Strawberry Leaf Scorch":    ({"strawberry"}, {"scorch"},         False),
    "Healthy Strawberry":        ({"strawberry"}, set(),              True),

    # ── Cherry ─────────────────────────────────────────────────────────────
    "Cherry Powdery Mildew":     ({"cherry"},  {"powdery", "mildew"}, False),
    "Healthy Cherry":            ({"cherry"},  set(),                 True),

    # ── Squash ─────────────────────────────────────────────────────────────
    "Squash Powdery Mildew":     ({"squash"},  {"powdery", "mildew"}, False),

    # ── Others ─────────────────────────────────────────────────────────────
    "Healthy Blueberry":         ({"blueberry"}, set(),              True),
    "Healthy Raspberry":         ({"raspberry"}, set(),              True),
    "Healthy Soybean":           ({"soybean"},   set(),              True),
    "Orange Citrus Greening":    ({"orange"},  {"citrus", "greening"}, False),

    # ── Additional Tomato diseases ──────────────────────────────────────────
    "Tomato Spider Mites":       ({"tomato"},  {"spider", "mites"},   False),
    "Tomato Target Spot":        ({"tomato"},  {"target", "spot"},    False),
    "Tomato Yellow Leaf Curl":   ({"tomato"},  {"yellow", "curl"},    False),
    "Tomato Mosaic Virus":       ({"tomato"},  {"mosaic"},            False),
}

_NEG_RE = re.compile(
    r'\b(no|not|without|never|absent|lack of|lacking|free of|free from)\b(\s+\w+){1,3}',
    re.IGNORECASE,
)


def _strip_negations(text: str) -> str:
    """'no blight', 'not infected' 같은 부정 표현을 제거한 텍스트 반환."""
    return _NEG_RE.sub(" [NEGATED] ", text)


def is_correct(prediction: str, label: str) -> bool:
    """
    명확하고 일관성 있는 판정:
      - 식물종 불일치 → 즉시 False
      - Healthy: healthy 있음 + disease 단어 없음 (negation 제거 후)
      - Disease: disease 키워드 전부 포함 (negation 제거 후)
    """
    if label not in LABEL_RULES:
        return label.lower() in prediction.lower()

    plant_genera, disease_kws, is_healthy = LABEL_RULES[label]
    pred_lower = prediction.lower()

    # 규칙 1: 식물종 확인
    if not any(g in pred_lower for g in plant_genera):
        return False

    cleaned = _strip_negations(pred_lower)

    if is_healthy:
        # 규칙 2: healthy 있고, disease 단어 없어야 함
        return "healthy" in pred_lower and not any(kw in cleaned for kw in DISEASE_WORDS)
    else:
        # 규칙 3: disease 키워드 전부 포함
        return all(kw in cleaned for kw in disease_kws)


# ── VRAM ─────────────────────────────────────────────────────────────────────
def _peak_vram_mb() -> float:
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0

def _reset_vram():
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────
def run_evaluation(labels_path: str, eval_set_base: str, max_images: int = None):
    with open(labels_path) as f:
        eval_labels = json.load(f)

    items = list(eval_labels.items())
    if max_images:
        items = items[:max_images]

    total = len(items)
    print(f"[eval] Evaluating {total} images ...", file=sys.stderr)

    from cropdoc_infer import diagnose
    _reset_vram()

    correct = 0
    errors  = 0
    times   = []
    per_label = {}

    for i, (rel_path, label) in enumerate(items):
        img_path = os.path.join(eval_set_base, rel_path)
        if not os.path.exists(img_path):
            print(f"[eval] SKIP: {img_path}", file=sys.stderr)
            errors += 1
            continue

        t0 = time.time()
        try:
            prediction = diagnose(img_path)
        except Exception as e:
            print(f"[eval] ERROR {rel_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            errors += 1
            continue
        times.append(time.time() - t0)

        ok = is_correct(prediction, label)
        correct += ok

        per_label.setdefault(label, {"ok": 0, "n": 0})
        per_label[label]["n"]  += 1
        per_label[label]["ok"] += ok

        sym = "✓" if ok else "✗"
        print(f"[eval] [{i+1}/{total}] {sym} | {label!r} | {prediction[:80]!r}", file=sys.stderr)

    evaluated = total - errors
    accuracy  = correct / evaluated if evaluated else 0.0
    avg_time  = sum(times) / len(times) if times else 0.0
    peak_vram = _peak_vram_mb()

    print("\n[eval] ── Per-label accuracy ──", file=sys.stderr)
    for lbl in sorted(per_label):
        r = per_label[lbl]
        pct = r["ok"] / r["n"] * 100 if r["n"] else 0
        bar = "✅" if pct >= 80 else ("⚠️ " if pct >= 50 else "❌")
        print(f"[eval] {bar} {lbl}: {r['ok']}/{r['n']} ({pct:.0f}%)", file=sys.stderr)

    # stdout (grep-friendly)
    print(f"diagnosis_accuracy: {accuracy:.3f}")
    print(f"total_images: {evaluated}")
    print(f"correct: {correct}")
    print(f"inference_time_avg: {avg_time:.1f}")
    print(f"peak_vram_mb: {peak_vram:.1f}")
    if errors:
        print(f"errors: {errors}")

    return {"diagnosis_accuracy": accuracy, "correct": correct,
            "total_images": evaluated, "per_label": per_label}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--labels",   default=LABELS_PATH)
    p.add_argument("--eval-set", default=EVAL_SET_BASE)
    args = p.parse_args()
    run_evaluation(args.labels, args.eval_set, args.max_images)


if __name__ == "__main__":
    main()
