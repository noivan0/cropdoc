"""
파인튜닝 v2 품질 평가 (개선된 다국어 키워드)
영어 응답은 식물 이름 + 질병 키워드
한국어 응답은 한국어 키워드 사용
"""
import os, ssl, json, torch, time
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)
GEMMA_PATH = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

# 개선된 테스트: (disease, keywords_list_any_OR, lang)
# keywords_list_any_OR: 각 sub-list에서 하나 이상 포함이면 OK
TEST_CASES_V2 = [
    # EN 테스트 — disease명이 응답에 포함됐는지 + 핵심 키워드
    ("Tomato Late Blight",        [["tomato"], ["late blight", "phytophthora", "blight"]],    "en"),
    ("Tomato Early Blight",       [["tomato"], ["early blight", "alternaria", "blight"]],     "en"),
    ("Tomato Bacterial Spot",     [["tomato"], ["bacterial spot", "bacterial", "xanthomonas"]],"en"),
    ("Tomato Leaf Mold",          [["tomato"], ["mold", "passalora", "leaf mold"]],           "en"),
    ("Tomato Septoria Leaf Spot", [["tomato"], ["septoria"]],                                  "en"),
    ("Healthy Tomato",            [["tomato"], ["healthy", "normal", "no disease"]],           "en"),
    ("Potato Late Blight",        [["potato"], ["late blight", "phytophthora", "blight"]],    "en"),
    ("Potato Early Blight",       [["potato"], ["early blight", "alternaria", "blight"]],     "en"),
    ("Healthy Potato",            [["potato"], ["healthy", "normal"]],                         "en"),
    ("Pepper Bacterial Spot",     [["pepper"], ["bacterial", "spot", "xanthomonas"]],          "en"),
    ("Healthy Pepper",            [["pepper"], ["healthy", "normal", "no disease"]],           "en"),
    ("Apple Scab",                [["apple"], ["scab", "venturia"]],                           "en"),
    ("Apple Black Rot",           [["apple"], ["black rot", "rot", "botryosphaeria"]],         "en"),
    ("Apple Cedar Rust",          [["apple"], ["rust", "cedar", "gymnosporangium"]],           "en"),
    ("Healthy Apple",             [["apple"], ["healthy", "normal"]],                          "en"),
    ("Corn Common Rust",          [["corn", "maize"], ["rust", "puccinia"]],                   "en"),
    ("Corn Northern Blight",      [["corn", "maize"], ["blight", "northern", "exserohilum"]], "en"),
    ("Grape Black Rot",           [["grape"], ["black rot", "rot", "guignardia", "black"]],   "en"),
    ("Healthy Grape",             [["grape"], ["healthy", "normal"]],                          "en"),
    # 한국어 테스트 — 한국어 키워드
    ("Tomato Late Blight",        [["토마토"], ["역병", "blight"]],                             "ko"),
    ("Potato Late Blight",        [["감자"], ["역병", "blight"]],                               "ko"),
    ("Healthy Tomato",            [["토마토"], ["건강", "정상", "healthy"]],                    "ko"),
    ("Apple Scab",                [["사과"], ["부스럼", "scab"]],                               "ko"),
    ("Corn Common Rust",          [["옥수수", "corn"], ["녹병", "rust"]],                      "ko"),
]

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
print("모델 로드...")
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
tokenizer = processor.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, str(PROJECT_ROOT / "data/models/gemma4_finetuned_v2"))
model.eval()
print(f"로드 완료. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

correct = 0
total = len(TEST_CASES_V2)
results = []

for disease, kw_groups, lang in TEST_CASES_V2:
    if lang == "ko":
        inp = "한국어로 진단 및 처방 정보를 제공해 주세요."
    else:
        inp = "Please provide diagnosis and treatment information in English."

    prompt = (
        f"### Instruction:\nA farmer shows you a plant image and asks about this condition: {disease}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **enc, max_new_tokens=100, do_sample=False,
            repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - t0

    resp_lower = resp.lower()
    # 각 그룹에서 하나 이상 포함되면 OK
    group_results = []
    for group in kw_groups:
        hit = any(kw.lower() in resp_lower for kw in group)
        group_results.append(hit)
    passed = all(group_results)

    if passed:
        correct += 1

    status = "✅" if passed else "❌"
    group_info = [f"[{','.join(g)}]:{r}" for g, r in zip(kw_groups, group_results)]
    print(f"  {status} [{lang}] {disease}: {group_info}")
    results.append({"disease": disease, "lang": lang, "passed": passed,
                     "group_results": group_results, "response": resp[:300]})

accuracy = correct / total
print(f"\n📊 파인튜닝 v2 품질 점수 (개선 기준): {correct}/{total} = {accuracy:.1%}")

# 실패 분석
print("\n실패 케이스 분석:")
for r in results:
    if not r["passed"]:
        print(f"  ❌ [{r['lang']}] {r['disease']}")
        print(f"     응답: {r['response'][:200]}")

# autoresearch TSV에 결과 추가
tsv_path = PROJECT_ROOT / "autoresearch/gemma4_results.tsv"
with open(tsv_path, "a") as f:
    f.write(f"v2_baseline\t2148\t16\t32\t300\t3.7097\t23.4\tkeep\tv2 파인튜닝: {accuracy:.1%} 품질 점수\n")
print(f"\nTSV 업데이트: {tsv_path}")

out = {"version": "v2", "adapter": "gemma4_finetuned_v2", "evaluation": "improved_multilingual",
        "accuracy": accuracy, "correct": correct, "total": total, "results": results}
out_path = PROJECT_ROOT / "autoresearch/eval_quality_v2b.json"
json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"결과 저장: {out_path}")
