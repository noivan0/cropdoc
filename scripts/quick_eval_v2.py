"""
빠른 파인튜닝 v2 품질 평가 (20개 케이스)
"""
import os, ssl, json, torch, time
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)
GEMMA_PATH = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

TEST_CASES = [
    ("Tomato Late Blight",        ["tomato", "late", "blight"],           "en"),
    ("Tomato Early Blight",       ["tomato", "early", "blight"],          "en"),
    ("Tomato Bacterial Spot",     ["tomato", "bacterial", "spot"],        "en"),
    ("Tomato Septoria Leaf Spot", ["tomato", "septoria"],                 "en"),
    ("Healthy Tomato",            ["tomato", "healthy"],                  "en"),
    ("Potato Late Blight",        ["potato", "late", "blight"],           "en"),
    ("Healthy Potato",            ["potato", "healthy"],                  "en"),
    ("Pepper Bacterial Spot",     ["pepper", "bacterial", "spot"],        "en"),
    ("Healthy Pepper",            ["pepper", "healthy"],                  "en"),
    ("Apple Scab",                ["apple", "scab"],                      "en"),
    ("Apple Black Rot",           ["apple", "black", "rot"],              "en"),
    ("Corn Common Rust",          ["corn", "rust"],                       "en"),
    ("Corn Northern Blight",      ["corn", "blight"],                     "en"),
    ("Grape Black Rot",           ["grape", "black", "rot"],              "en"),
    ("Healthy Grape",             ["grape", "healthy"],                   "en"),
    # 한국어 테스트
    ("Tomato Late Blight",        ["tomato", "blight"],                   "ko"),
    ("Potato Late Blight",        ["potato", "blight"],                   "ko"),
    ("Healthy Tomato",            ["tomato", "healthy"],                  "ko"),
    ("Apple Scab",                ["apple", "scab"],                      "ko"),
    ("Corn Common Rust",          ["corn", "rust"],                       "ko"),
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
total = len(TEST_CASES)
results = []

for disease, keywords, lang in TEST_CASES:
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
    hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    passed = hits == len(keywords)
    if passed:
        correct += 1

    status = "✅" if passed else "❌"
    print(f"  {status} [{lang}] {disease}: {hits}/{len(keywords)} | {resp[:80]!r}")
    results.append({"disease": disease, "lang": lang, "passed": passed,
                     "hits": hits, "total_kw": len(keywords), "response": resp[:300]})

accuracy = correct / total
print(f"\n📊 파인튜닝 v2 품질 점수: {correct}/{total} = {accuracy:.1%}")

# 저장
out = {"version": "v2", "adapter": "gemma4_finetuned_v2",
        "accuracy": accuracy, "correct": correct, "total": total, "results": results}
out_path = PROJECT_ROOT / "autoresearch/eval_quality_v2.json"
out_path.parent.mkdir(exist_ok=True)
json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"결과 저장: {out_path}")
