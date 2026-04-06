"""
Gemma4 PEFT LoRA 파인튜닝 v2
================================
데이터셋: 2148개 (40개 → 2148개 대폭 확대)
LoRA: r=16, alpha=16 (baseline)
목표: 다양한 질병 진단 응답 품질 향상

실행:
  CUDA_VISIBLE_DEVICES=0 python3.10 scripts/finetune_gemma4_v2.py
"""

import os, sys, ssl, json, time
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)

GEMMA_PATH   = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"
DATASET_PATH = PROJECT_ROOT / "data/gemma4_finetune_v2.json"
OUTPUT_DIR   = PROJECT_ROOT / "data/models/gemma4_finetuned_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[Gemma4 LoRA 파인튜닝 v2]")
print(f"데이터: {DATASET_PATH}")
print(f"출력: {OUTPUT_DIR}")
print("=" * 60)

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# 4bit 양자화
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("\n[Step 1] 모델 로드...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
tokenizer = processor.tokenizer

model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
print(f"  로드 완료: {time.time()-t0:.1f}s | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

model = prepare_model_for_kbit_training(model)

# LoRA 설정
print("\n[Step 2] LoRA 설정 (r=16)...")
lm_linear_modules = set()
for name, module in model.named_modules():
    if "language_model" in name and isinstance(module, torch.nn.Linear):
        leaf = name.split(".")[-1]
        if leaf in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            lm_linear_modules.add(leaf)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # alpha=2r (v1보다 높임)
    target_modules=sorted(lm_linear_modules),
    exclude_modules=r".*\.(vision_tower|audio_tower|embed_vision|embed_audio|lm_head)\..*",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  학습 파라미터: {train_params/1e6:.1f}M / {total_params/1e9:.2f}B ({train_params/total_params*100:.3f}%)")

# 데이터셋
print("\n[Step 3] 데이터셋 로드...")
data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
print(f"  {len(data)}개 항목")

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_example(example):
    text = ALPACA_TEMPLATE.format(
        instruction=example["instruction"],
        input=example.get("input", ""),
        output=example["output"],
    ) + tokenizer.eos_token

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None,
    )
    labels = encoded["input_ids"].copy()
    labels = [-100 if t == tokenizer.pad_token_id else t for t in labels]
    encoded["labels"] = labels
    return encoded

from datasets import Dataset as HFDataset
hf_dataset = HFDataset.from_list(data)
tokenized = hf_dataset.map(
    tokenize_example,
    remove_columns=hf_dataset.column_names,
    num_proc=4,
)
print(f"  토크나이징 완료: {len(tokenized)}행")

# 학습 설정
print("\n[Step 4] 학습 설정...")

# 데이터 2148개 × 3 epoch = 6444 스텝 → max_steps=300으로 제한 (약 10분)
MAX_STEPS = 300

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch=8
    warmup_steps=10,
    max_steps=MAX_STEPS,
    learning_rate=2e-4,
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_steps=300,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    gradient_checkpointing=True,
)

from transformers import default_data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=default_data_collator,
)

print(f"  max_steps={MAX_STEPS} | lr={training_args.learning_rate} | eff_batch=8")
print(f"  예상 시간: ~{MAX_STEPS * 2 // 60}분")

# 학습
print("\n[Step 5] 파인튜닝 시작...")
print("=" * 60)
t_train = time.time()
train_result = trainer.train()
elapsed = time.time() - t_train

print("=" * 60)
print(f"✅ 학습 완료!")
print(f"   시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
print(f"   최종 loss: {train_result.training_loss:.4f}")
print(f"   스텝: {train_result.global_step}")

# 저장
print("\n[Step 6] 저장...")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))
print(f"  저장: {OUTPUT_DIR}")

# 추론 테스트
print("\n[Step 7] 추론 테스트 (3개 샘플)...")
model.eval()

test_cases = [
    ("Tomato Late Blight", "ko", "한국어로 진단 정보 제공"),
    ("Healthy Apple", "en", "English diagnosis"),
    ("Corn Common Rust", "es", "Diagnóstico en español"),
]

results = []
for disease, lang, inp in test_cases:
    prompt = (
        f"### Instruction:\nA farmer shows you a plant image and asks about this condition: {disease}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )
    inputs_t = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model.generate(
            **inputs_t,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(out[0][inputs_t["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[{disease} / {lang}]\n{resp[:400]}\n{'-'*40}")
    results.append({"disease": disease, "lang": lang, "response": resp[:400]})

# 결과 메타데이터 저장
meta = {
    "version": "v2",
    "dataset_size": len(data),
    "max_steps": MAX_STEPS,
    "lora_r": 16,
    "lora_alpha": 32,
    "training_loss": train_result.training_loss,
    "elapsed_seconds": elapsed,
    "test_results": results,
}
import json
json.dump(meta, open(OUTPUT_DIR / "training_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("✅ Gemma4 PEFT 파인튜닝 v2 완료")
print(f"   데이터: {len(data)}개 | Loss: {train_result.training_loss:.4f}")
print(f"   저장: {OUTPUT_DIR}")
