"""
CropDoc Gemma4 PEFT 파인튜닝 (Unsloth 대체)
=============================================
Gemma4는 멀티모달 모델 (Gemma4ForConditionalGeneration) → Unsloth FastLanguageModel 미지원
→ PEFT LoRA + BitsAndBytes 4bit quantization으로 대체

실행:
  python3.10 scripts/finetune_gemma4_peft.py
"""

import unsloth  # noqa: F401 — 맨 위에서 import (패치 우선 적용)

import os, sys, ssl, json, time
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

# 단일 GPU 사용 (DataParallel 충돌 방지)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)

GEMMA_PATH   = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"
DATASET_PATH = PROJECT_ROOT / "data/gemma4_finetune_dataset.json"
OUTPUT_DIR   = PROJECT_ROOT / "data/models/gemma4_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[Step 1] Gemma4 멀티모달 모델 4bit 로드")
print("=" * 60)

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

t0 = time.time()
print(f"  Loading from {GEMMA_PATH} ...")

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
tokenizer = processor.tokenizer

model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

print(f"  로드 완료: {time.time()-t0:.1f}s")
print(f"  GPU 메모리: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# kbit 훈련 준비
model = prepare_model_for_kbit_training(model)

# ── LoRA 적용 ─────────────────────────────────────────────────────────────────
print("\n[Step 2] LoRA 적용 (언어 모델 부분만)")

# Gemma4 텍스트 파트의 레이어 이름 확인
named_modules = [n for n, _ in model.named_modules()]
# q_proj, k_proj 등이 언어 모델에 있는지 확인
lm_qk = [n for n in named_modules if ('language_model' in n or 'text_model' in n) and ('q_proj' in n or 'k_proj' in n)]
print(f"  언어 모델 attention 레이어 예시: {lm_qk[:3]}")

# Gemma4 언어 모델 부분만 타겟: "model.language_model.layers.*.self_attn.*_proj"
# 비전 인코더(Gemma4ClippableLinear)는 제외
lm_linear_modules = set()
for name, module in model.named_modules():
    if "language_model" in name and isinstance(module, torch.nn.Linear):
        # 마지막 부분 이름만 추출
        leaf = name.split(".")[-1]
        if leaf in ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]:
            lm_linear_modules.add(leaf)

print(f"  타겟 모듈 (언어 모델 전용): {sorted(lm_linear_modules)}")

# 정규식으로 비전/오디오 타워 전체 제외
# PEFT는 exclude_modules가 str이면 re.fullmatch(pattern, key) 사용
exclude_pattern = r".*\.(vision_tower|audio_tower|embed_vision|embed_audio|lm_head)\..*"

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=sorted(lm_linear_modules),   # 언어 모델 레이어명
    exclude_modules=exclude_pattern,             # 정규식: vision/audio 타워 전체 제외
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  총 파라미터: {total_params/1e9:.2f}B")
print(f"  학습 파라미터: {train_params/1e6:.1f}M ({train_params/total_params*100:.3f}%)")

# ── 데이터셋 준비 ─────────────────────────────────────────────────────────────
print("\n[Step 3] 데이터셋 로드 및 토크나이징")

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
    # labels = input_ids (전체 시퀀스 언어 모델링)
    labels = encoded["input_ids"].copy()
    # pad token → -100 (loss 무시)
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

# ── 학습 설정 ─────────────────────────────────────────────────────────────────
print("\n[Step 4] TrainingArguments 설정")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 8
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_steps=60,
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

print(f"  max_steps: 60 | lr: {training_args.learning_rate} | eff_batch: 8")

# ── 학습 실행 ─────────────────────────────────────────────────────────────────
print("\n[Step 5] 파인튜닝 실행")
print("=" * 60)

t_train = time.time()
train_result = trainer.train()
elapsed = time.time() - t_train

print("=" * 60)
print(f"✅ 학습 완료: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
print(f"   최종 loss: {train_result.training_loss:.4f}")
print(f"   스텝: {train_result.global_step}")

# ── 저장 ──────────────────────────────────────────────────────────────────────
print("\n[Step 6] LoRA 어댑터 저장")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))

saved = [f.name for f in OUTPUT_DIR.glob("*")]
print(f"  저장: {OUTPUT_DIR}")
print(f"  파일: {saved}")

# ── 추론 테스트 ───────────────────────────────────────────────────────────────
print("\n[Step 7] 파인튜닝 모델 추론 테스트")
model.eval()

test_prompt = (
    "### Instruction:\n"
    "A farmer shows you a plant image and asks about this condition: Tomato Late Blight\n\n"
    "### Input:\nPlease provide diagnosis in Korean.\n\n"
    "### Response:\n"
)
inputs_t = tokenizer(test_prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.inference_mode():
    out = model.generate(
        **inputs_t,
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
resp = tokenizer.decode(out[0][inputs_t["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n[Gemma4 파인튜닝 응답]\n{resp[:600]}")

print("\n" + "=" * 60)
print("✅ Gemma4 PEFT 파인튜닝 전체 완료")
print(f"   방식: PEFT LoRA + BitsAndBytes 4bit NF4")
print(f"   학습 시간: {elapsed/60:.1f}분")
print(f"   Loss: {train_result.training_loss:.4f}")
print(f"   LoRA 저장: {OUTPUT_DIR}")
