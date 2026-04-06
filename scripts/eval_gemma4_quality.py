"""
Gemma4 파인튜닝 응답 품질 평가 스크립트
=========================================
파인튜닝된 LoRA 어댑터 vs 기본 모델 응답 품질 비교
eval_harness.py의 LABEL_RULES 기반으로 키워드 포함 여부 평가

Usage:
  python3.10 scripts/eval_gemma4_quality.py --adapter data/models/gemma4_finetuned_v2

출력:
  - 각 질병별 키워드 포함률
  - 전체 응답 품질 점수
  - 기본 vs 파인튜닝 비교
"""
import os, sys, json, argparse, time, torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)

GEMMA_PATH = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

# 평가용 테스트 케이스 (eval_harness LABEL_RULES 기반)
TEST_CASES = [
    # (disease_label, expected_keywords_en, lang)
    ("Tomato Late Blight",        ["tomato", "late", "blight"],           "en"),
    ("Tomato Early Blight",       ["tomato", "early", "blight"],          "en"),
    ("Tomato Bacterial Spot",     ["tomato", "bacterial", "spot"],        "en"),
    ("Tomato Leaf Mold",          ["tomato", "mold"],                     "en"),
    ("Tomato Septoria Leaf Spot", ["tomato", "septoria"],                 "en"),
    ("Healthy Tomato",            ["tomato", "healthy"],                  "en"),
    ("Potato Late Blight",        ["potato", "late", "blight"],           "en"),
    ("Potato Early Blight",       ["potato", "early", "blight"],          "en"),
    ("Healthy Potato",            ["potato", "healthy"],                  "en"),
    ("Pepper Bacterial Spot",     ["pepper", "bacterial", "spot"],        "en"),
    ("Healthy Pepper",            ["pepper", "healthy"],                  "en"),
    ("Apple Scab",                ["apple", "scab"],                      "en"),
    ("Apple Black Rot",           ["apple", "black", "rot"],              "en"),
    ("Apple Cedar Rust",          ["apple", "rust"],                      "en"),
    ("Healthy Apple",             ["apple", "healthy"],                   "en"),
    ("Corn Gray Leaf Spot",       ["corn", "gray", "spot"],               "en"),
    ("Corn Common Rust",          ["corn", "rust"],                       "en"),
    ("Corn Northern Blight",      ["corn", "blight"],                     "en"),
    ("Healthy Corn",              ["corn", "healthy"],                    "en"),
    ("Grape Black Rot",           ["grape", "black", "rot"],              "en"),
    ("Grape Esca",                ["grape", "esca"],                      "en"),
    ("Grape Leaf Spot",           ["grape", "spot"],                      "en"),
    ("Healthy Grape",             ["grape", "healthy"],                   "en"),
    ("Peach Bacterial Spot",      ["peach", "bacterial", "spot"],         "en"),
    ("Healthy Peach",             ["peach", "healthy"],                   "en"),
    ("Strawberry Leaf Scorch",    ["strawberry", "scorch"],               "en"),
    ("Healthy Strawberry",        ["strawberry", "healthy"],              "en"),
    ("Cherry Powdery Mildew",     ["cherry", "powdery", "mildew"],       "en"),
    ("Healthy Cherry",            ["cherry", "healthy"],                  "en"),
    ("Squash Powdery Mildew",     ["squash", "powdery", "mildew"],       "en"),
    ("Healthy Blueberry",         ["blueberry", "healthy"],               "en"),
    ("Healthy Raspberry",         ["raspberry", "healthy"],               "en"),
    ("Healthy Soybean",           ["soybean", "healthy"],                 "en"),
    ("Orange Citrus Greening",    ["orange", "citrus", "greening"],       "en"),
    ("Tomato Spider Mites",       ["tomato", "spider", "mites"],          "en"),
    ("Tomato Target Spot",        ["tomato", "target", "spot"],           "en"),
    ("Tomato Yellow Leaf Curl",   ["tomato", "yellow", "curl"],           "en"),
    ("Tomato Mosaic Virus",       ["tomato", "mosaic"],                   "en"),
]

# 한국어 추가 테스트 (주요 질병)
KO_TEST_CASES = [
    ("Tomato Late Blight",  ["토마토", "역병"],   "ko"),
    ("Potato Late Blight",  ["감자", "역병"],     "ko"),
    ("Corn Common Rust",    ["옥수수", "녹병"],   "ko"),
    ("Apple Scab",          ["사과", "부스럼"],   "ko"),
    ("Healthy Tomato",      ["토마토", "정상"],   "ko"),
]

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def load_model(adapter_path=None):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(GEMMA_PATH)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_PATH, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        print(f"  LoRA 어댑터 로드: {adapter_path}")

    return model, tokenizer

def generate_response(model, tokenizer, disease, lang="en", max_new_tokens=200):
    if lang == "ko":
        inp = "한국어로 진단 및 처방 정보를 제공해 주세요."
    elif lang == "es":
        inp = "Por favor proporcione información de diagnóstico en español."
    else:
        inp = "Please provide diagnosis and treatment information in English."

    prompt = ALPACA_TEMPLATE.format(
        instruction=f"A farmer shows you a plant image and asks about this condition: {disease}",
        input=inp,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp

def evaluate_response(response, expected_keywords, is_healthy=False):
    """응답에 필수 키워드가 모두 포함됐는지 확인"""
    resp_lower = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in resp_lower)
    score = hits / len(expected_keywords) if expected_keywords else 0
    return score, hits, len(expected_keywords)

def run_eval(model, tokenizer, test_cases, tag=""):
    correct = 0
    total = len(test_cases)
    results = []

    for disease, keywords, lang in test_cases:
        t0 = time.time()
        resp = generate_response(model, tokenizer, disease, lang)
        elapsed = time.time() - t0
        score, hits, total_kw = evaluate_response(resp, keywords)
        is_pass = score >= 1.0  # 모든 키워드 포함 시 pass
        if is_pass:
            correct += 1

        results.append({
            "disease": disease, "lang": lang,
            "score": score, "hits": hits, "total_kw": total_kw,
            "pass": is_pass, "elapsed": elapsed,
            "response_preview": resp[:200],
        })
        status = "✅" if is_pass else "❌"
        print(f"  {status} [{lang}] {disease}: {hits}/{total_kw} ({score:.0%}) | {elapsed:.1f}s")

    accuracy = correct / total
    print(f"\n{tag} 결과: {correct}/{total} = {accuracy:.1%}")
    return accuracy, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--compare-base", action="store_true", help="기본 모델과 비교")
    parser.add_argument("--ko-only", action="store_true", help="한국어 테스트만")
    args = parser.parse_args()

    adapter_path = args.adapter
    if adapter_path and not os.path.isabs(adapter_path):
        adapter_path = str(PROJECT_ROOT / adapter_path)

    print("=" * 60)
    print(f"Gemma4 응답 품질 평가")
    print(f"어댑터: {adapter_path or '기본 모델'}")
    print("=" * 60)

    # 파인튜닝 모델 평가
    print("\n[파인튜닝 모델 로드...]")
    model, tokenizer = load_model(adapter_path)

    test_cases = KO_TEST_CASES if args.ko_only else TEST_CASES + KO_TEST_CASES
    print(f"\n[평가 시작: {len(test_cases)}개 케이스]")
    ft_acc, ft_results = run_eval(model, tokenizer, test_cases, tag="파인튜닝")

    # 결과 저장
    output = {
        "adapter": adapter_path,
        "finetuned_accuracy": ft_acc,
        "total_cases": len(test_cases),
        "results": ft_results,
    }

    if args.compare_base:
        print("\n[기본 모델 평가...]")
        del model
        torch.cuda.empty_cache()
        base_model, base_tokenizer = load_model(None)
        base_acc, base_results = run_eval(base_model, base_tokenizer, test_cases, tag="기본 모델")
        output["base_accuracy"] = base_acc
        output["improvement"] = ft_acc - base_acc
        print(f"\n📊 개선: {base_acc:.1%} → {ft_acc:.1%} (+{(ft_acc-base_acc)*100:.1f}%p)")

    out_file = PROJECT_ROOT / "autoresearch/eval_quality_result.json"
    out_file.parent.mkdir(exist_ok=True)
    json.dump(output, open(out_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_file}")
    print(f"\n✅ 최종 응답 품질 점수: {ft_acc:.1%}")

if __name__ == "__main__":
    main()
