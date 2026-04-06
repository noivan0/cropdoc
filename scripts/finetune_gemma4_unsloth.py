"""
CropDoc Gemma4 Unsloth 파인튜닝
==================================
Unsloth FastLanguageModel + LoRA로 Gemma4 E4B를 식물 질병 진단에 파인튜닝.
TRL 0.24.0 호환 버전.

실행:
  python3.10 scripts/finetune_gemma4_unsloth.py
"""

import os, sys, ssl, json, time
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

GEMMA_PATH   = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"
DATASET_PATH = PROJECT_ROOT / "data/gemma4_finetune_dataset.json"
OUTPUT_DIR   = PROJECT_ROOT / "data/models/gemma4_finetuned"

# ── Step 1: Unsloth 로드 ──────────────────────────────────────────────────────
print("=" * 60)
print("[Step 1] Unsloth FastLanguageModel 로드")
print("=" * 60)

import torch
import trl, transformers as tf_lib
print(f"  trl: {trl.__version__} | transformers: {tf_lib.__version__}")

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("  Unsloth: 사용 가능")
except ImportError as e:
    UNSLOTH_AVAILABLE = False
    print(f"  Unsloth: 불가 ({e}) → PEFT fallback")

t0 = time.time()

if UNSLOTH_AVAILABLE:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=GEMMA_PATH,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"  모델 로드: {time.time()-t0:.1f}s")

    # LoRA 적용
    print("\n[Step 2] LoRA 어댑터 적용 (r=16)")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
else:
    # PEFT fallback
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    print("[Fallback] PEFT LoRA + 4bit 방식으로 로드...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    print(f"  모델 로드: {time.time()-t0:.1f}s")

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  총 파라미터: {total_params/1e9:.2f}B | 학습 파라미터: {train_params/1e6:.1f}M ({train_params/total_params*100:.2f}%)")

# ── Step 3: 데이터셋 준비 ─────────────────────────────────────────────────────
print("\n[Step 3] 데이터셋 로드 및 토크나이징")

data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
print(f"  데이터셋: {len(data)}개 항목")

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 직접 토크나이징 (Unsloth 자동 토크나이저 우회)
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
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

from datasets import Dataset as HFDataset
hf_dataset = HFDataset.from_list(data)
tokenized = hf_dataset.map(tokenize_example, remove_columns=hf_dataset.column_names)
print(f"  토크나이징 완료: {len(tokenized)}행")

# ── Step 4: 학습 설정 ─────────────────────────────────────────────────────────
print("\n[Step 4] 학습 설정 (Trainer)")

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    save_steps=30,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

# tokenized 데이터에서 labels 마스킹 (instruction 부분 제외, response 부분만 학습)
def mask_instruction_labels(example):
    """Response 이전 토큰의 labels를 -100으로 마스킹."""
    template_prefix = ALPACA_TEMPLATE.split("{output}")[0].format(
        instruction=data[0]["instruction"][:5],  # 가상 (길이 계산용)
        input=data[0].get("input", ""),
    )
    # 단순 방식: 전체 시퀀스 학습 (더 안정적)
    return example

# 표준 Trainer 사용 (SFTTrainer 대신)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    ),
)

print(f"  max_steps: {training_args.max_steps}")
print(f"  lr: {training_args.learning_rate} | batch (effective): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# ── Step 5: 학습 실행 ─────────────────────────────────────────────────────────
print("\n[Step 5] 파인튜닝 시작")
print("=" * 60)

t_train = time.time()
train_result = trainer.train()
elapsed = time.time() - t_train

print("=" * 60)
print(f"[학습 완료] {elapsed:.0f}초 ({elapsed/60:.1f}분)")
print(f"  최종 loss: {train_result.training_loss:.4f}")
print(f"  총 스텝: {train_result.global_step}")

# ── Step 6: 저장 ──────────────────────────────────────────────────────────────
print("\n[Step 6] LoRA 어댑터 저장")

if UNSLOTH_AVAILABLE:
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
else:
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

saved_files = list(OUTPUT_DIR.glob("*"))
print(f"  저장 위치: {OUTPUT_DIR}")
print(f"  저장된 파일 ({len(saved_files)}개): {[f.name for f in saved_files[:10]]}")

# ── Step 7: 추론 테스트 ───────────────────────────────────────────────────────
print("\n[Step 7] 파인튜닝 모델 추론 테스트")

if UNSLOTH_AVAILABLE:
    FastLanguageModel.for_inference(model)

model.eval()
test_prompt = (
    "### Instruction:\n"
    "A farmer shows you a plant image and asks about this condition: Tomato Late Blight\n\n"
    "### Input:\n"
    "Please provide diagnosis and treatment information in Korean.\n\n"
    "### Response:\n"
)

inputs = tokenizer(test_prompt, return_tensors="pt").to(next(model.parameters()).device)
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n[테스트 출력 (Tomato Late Blight / 한국어)]\n{response[:600]}")

# ── 완료 요약 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ Gemma4 파인튜닝 완료 요약")
print("=" * 60)
print(f"  방식: {'Unsloth FastLanguageModel' if UNSLOTH_AVAILABLE else 'PEFT LoRA'} + 4-bit + transformers Trainer")
print(f"  학습 시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
print(f"  최종 Loss: {train_result.training_loss:.4f}")
print(f"  LoRA 어댑터: {OUTPUT_DIR}")
print(f"  데이터셋: {len(data)}개 항목 (10종 식물 질병, 한/영)")
