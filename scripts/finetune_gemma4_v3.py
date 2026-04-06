"""
Gemma4 PEFT LoRA 파인튜닝 v3 (autoresearch 실험 B)
=======================================================
변경사항:
  - LoRA r=16 → r=32 (학습 용량 증가)
  - 데이터셋: v2 + 문제 케이스 강화 데이터 추가
    * Corn Common Rust: "corn"/"maize" 명시 강화
    * Healthy Pepper: "healthy pepper" 강화
    * Tomato Septoria: "tomato" 명시 강화
    * Apple Cedar Rust: "apple" 명시 강화
  - max_steps=400 (더 많은 학습)

실행:
  CUDA_VISIBLE_DEVICES=0 python3.10 scripts/finetune_gemma4_v3.py
"""

import os, ssl, json, time, random
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
os.chdir(PROJECT_ROOT)

GEMMA_PATH   = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"
DATASET_V2   = PROJECT_ROOT / "data/gemma4_finetune_v2.json"
OUTPUT_DIR   = PROJECT_ROOT / "data/models/gemma4_finetuned_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 문제 케이스 보강 데이터 생성 ──────────────────────────────────────────────
BOOST_CASES = {
    # Corn Common Rust — "corn" 명시, "maize" 함께
    "Corn Common Rust": {
        "instructions": [
            "Diagnose this corn/maize crop disease: Corn Common Rust",
            "What disease affects this corn plant? Label: Corn Common Rust",
            "This corn (maize) plant has Corn Common Rust. Describe it.",
            "A farmer's corn field shows Corn Common Rust symptoms. Advise them.",
            "Corn Common Rust diagnosis and treatment for corn/maize crop.",
        ],
        "en": """PLANT: Corn / Maize (Zea mays)
DISEASE: Common Rust (Corn Common Rust)
CAUSE: Puccinia sorghi fungus. Wind-dispersed spores infect corn leaves.
SYMPTOMS: Oval to elongated rusty-orange pustules (uredia) on both surfaces of corn leaves. Pustules darken with age. Heavy corn infection causes premature yellowing and death of corn leaves.
TREATMENT: Apply Propiconazole or Azoxystrobin fungicide to corn. Use rust-resistant corn hybrids. Early detection critical for corn yield protection.
URGENCY: MODERATE — treat corn early to protect yield. Rust can reduce corn yield by 10-30%.""",
        "ko": """식물: 옥수수 / 마이즈 (Zea mays)
진단: 일반 녹병 (Corn Common Rust)
원인: Puccinia sorghi 진균, 바람에 의해 옥수수 잎에 포자 전파
증상: 옥수수 잎 양면에 타원형 적갈색 포자퇴, 포자퇴 노화 시 흑갈색화, 옥수수 심한 감염 시 조기 황화 및 고사
처방: 프로피코나졸 또는 아족시스트로빈 살균제 살포. 저항성 옥수수 품종 재배. 옥수수 수량 보호를 위한 조기 발견 필수.
긴급도: ⚠️ 보통 — 조기 처리로 옥수수 수량 보호. 녹병으로 옥수수 수량 10~30% 감소 가능""",
    },
    # Corn Northern Blight — "corn" 명시
    "Corn Northern Blight": {
        "instructions": [
            "Diagnose this corn crop disease: Corn Northern Blight",
            "What disease affects this corn leaf? Label: Corn Northern Blight",
            "This corn plant has Northern Corn Leaf Blight. Describe it.",
            "A corn farmer's field shows Northern Corn Leaf Blight. What to do?",
            "Northern Corn Leaf Blight in corn - diagnosis and treatment.",
        ],
        "en": """PLANT: Corn / Maize (Zea mays)
DISEASE: Northern Corn Leaf Blight (Corn Northern Blight)
CAUSE: Exserohilum turcicum fungus infects corn leaves.
SYMPTOMS: Long cigar-shaped grayish-green to tan lesions (2.5–15 cm) running length-wise on corn leaves. Sporulation visible on corn lesions in humid conditions. Severe corn blight above ear at silking causes major yield loss.
TREATMENT: Azoxystrobin or Mancozeb fungicide on corn. Plant resistant corn hybrids. Rotate corn with non-host crops.
URGENCY: HIGH — especially critical when corn blight occurs before or during corn silking.""",
        "ko": """식물: 옥수수 (Zea mays)
진단: 북부 잎마름병 (Corn Northern Blight)
원인: Exserohilum turcicum 진균이 옥수수 잎 감염
증상: 옥수수 잎에 긴 타원형 회녹색~황갈색 병반, 습한 조건에서 옥수수 병반에 포자 형성, 출사기 이삭 위 심한 옥수수 감염 시 수량 큰 손실
처방: 아족시스트로빈 또는 만코제브 살균제. 저항성 옥수수 품종 재배. 옥수수와 비기주 작물 윤작.
긴급도: 🔴 높음 — 출사기 전후 옥수수 잎마름병 발생 시 특히 중요""",
    },
    # Healthy Pepper — "healthy pepper" 강조
    "Healthy Pepper": {
        "instructions": [
            "Is this pepper plant healthy or diseased? Label: Healthy Pepper",
            "Diagnose this pepper plant condition: Healthy Pepper",
            "This is a healthy pepper plant. Describe its condition.",
            "A farmer wants to know if their pepper is healthy. Status: Healthy Pepper",
            "Pepper plant health assessment: Healthy Pepper",
        ],
        "en": """PLANT: Pepper (Capsicum annuum)
STATUS: Healthy Pepper
OBSERVATION: This pepper plant is healthy. Pepper leaves are uniformly bright green, glossy, with no spots, lesions, or yellowing. Pepper stems are firm and upright. No wilting or disease signs on this pepper plant.
RECOMMENDATION: Healthy pepper plant - continue current care. Monitor pepper regularly for bacterial spot and aphids during wet periods. Maintain consistent moisture for healthy pepper growth.
URGENCY: None — this pepper plant is completely healthy.""",
        "ko": """식물: 고추 (Capsicum annuum)
상태: 건강한 고추 (Healthy Pepper)
관찰: 이 고추 식물은 건강합니다. 고추 잎이 균일한 밝은 녹색, 광택 있음, 반점·병반·황변 없음. 줄기 견고하고 직립. 이 고추 식물에 시들음이나 병해 징후 없음.
권고: 건강한 고추 식물 — 현재 관리 유지. 습한 시기 세균성 반점병과 진딧물 정기 모니터링. 건강한 고추 생장을 위해 일정한 수분 유지.
긴급도: 없음 — 이 고추 식물은 완전히 건강함""",
    },
    # Tomato Septoria Leaf Spot — "tomato" 명시
    "Tomato Septoria Leaf Spot": {
        "instructions": [
            "Diagnose this tomato disease: Tomato Septoria Leaf Spot",
            "What disease affects this tomato plant? Label: Tomato Septoria Leaf Spot",
            "This tomato has Septoria Leaf Spot. Describe the condition.",
            "A tomato farmer sees Septoria Leaf Spot on tomato leaves. Advise.",
            "Tomato Septoria Leaf Spot - full tomato disease report.",
        ],
        "en": """PLANT: Tomato (Solanum lycopersicum)
DISEASE: Tomato Septoria Leaf Spot (Septoria lycopersici)
CAUSE: Septoria lycopersici fungus infects tomato leaves. Spreads via water splash on tomato.
SYMPTOMS: Small circular spots on tomato lower leaves with dark borders and light gray centers. Tiny dark pycnidia (fruiting bodies) visible inside tomato leaf spots. Severe tomato infection causes defoliation, exposing tomato fruit to sun scald.
TREATMENT: Apply Chlorothalonil or Mancozeb fungicide to tomato. Remove infected tomato lower leaves. Mulch around tomato base to prevent soil splash.
URGENCY: HIGH — tomato Septoria can cause complete tomato defoliation if untreated.""",
        "ko": """식물: 토마토 (Solanum lycopersicum)
진단: 토마토 셉토리아 잎반점병 (Tomato Septoria Leaf Spot)
원인: Septoria lycopersici 진균이 토마토 잎 감염. 토마토에서 물 비산으로 확산.
증상: 토마토 하부 잎에 어두운 테두리·밝은 회색 중심의 소형 원형 반점, 토마토 잎 반점 내 소흑점(분생포자각) 확인, 토마토 심한 감염 시 낙엽 → 과실 햇빛 데임
처방: 클로로탈로닐 또는 만코제브 살균제 토마토에 살포. 감염된 토마토 하부 잎 제거. 토마토 기부 멀칭으로 토양 비산 방지.
긴급도: 🔴 높음 — 방치 시 토마토 셉토리아로 완전 낙엽 가능""",
    },
    # Apple Cedar Rust — "apple" 명시
    "Apple Cedar Rust": {
        "instructions": [
            "Diagnose this apple tree disease: Apple Cedar Rust",
            "What disease affects this apple? Label: Apple Cedar Rust",
            "This apple tree has Cedar Apple Rust. Describe it.",
            "An apple grower has Cedar Rust on apple trees. What to do?",
            "Apple Cedar Rust (Cedar Apple Rust) - apple disease report.",
        ],
        "en": """PLANT: Apple (Malus domestica)
DISEASE: Apple Cedar Rust (Cedar Apple Rust — Gymnosporangium juniperi-virginianae)
CAUSE: Gymnosporangium juniperi-virginianae fungus requires both apple trees and cedar/juniper to complete life cycle.
SYMPTOMS: Bright orange-yellow spots on upper apple leaf surface. Tube-like orange spore structures (aecia) on apple leaf undersides. Infected apple fruit shows orange spots. Does NOT cause apple scab.
TREATMENT: Apply Myclobutanil or Propiconazole to apple trees. Remove nearby cedar/juniper trees if possible. Begin apple spray program at pink bud stage.
URGENCY: MODERATE — apple cedar rust primarily causes aesthetic apple damage but can weaken apple trees.""",
        "ko": """식물: 사과 (Malus domestica)
진단: 사과 붉은별무늬병 (Apple Cedar Rust)
원인: Gymnosporangium juniperi-virginianae — 사과나무와 향나무 모두 필요한 이종기생균
증상: 사과 잎 윗면 밝은 주황색 반점, 사과 잎 뒷면 관상 주황색 포자층, 감염 사과 과실 주황색 반점. 사과 부스럼병(Scab)과 다름.
처방: 마이클로부타닐 또는 프로피코나졸을 사과나무에 살포. 가능 시 주변 향나무 제거. 사과 분홍색 눈 단계부터 살포 시작.
긴급도: ⚠️ 보통 — 사과 붉은별무늬병은 주로 외관 피해이나 사과나무 약화 가능""",
    },
}

# 보강 데이터셋 생성
ALPACA_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"

boost_data = []
random.seed(42)

INPUTS = {
    "en": "Please provide diagnosis and treatment information in English.",
    "ko": "한국어로 진단 및 처방 정보를 제공해 주세요.",
}

for disease, case_data in BOOST_CASES.items():
    instructions = case_data["instructions"]
    for lang in ["en", "ko"]:
        response = case_data[lang]
        inp = INPUTS[lang]
        for inst in instructions:
            boost_data.append({
                "instruction": inst,
                "input": inp,
                "output": response,
                "lang": lang,
                "disease": disease,
                "boosted": True,
            })

# v2 데이터 + 보강 데이터 합치기 (보강 데이터 3배 oversampling)
v2_data = json.loads(DATASET_V2.read_text(encoding="utf-8"))
combined = v2_data + boost_data * 3  # 보강 3배
random.shuffle(combined)

v3_path = PROJECT_ROOT / "data/gemma4_finetune_v3.json"
json.dump(combined, open(v3_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"✅ v3 데이터셋: {len(combined)}개 (v2: {len(v2_data)} + 보강: {len(boost_data)}×3)")

# ── 파인튜닝 시작 ──────────────────────────────────────────────────────────────
import torch
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    BitsAndBytesConfig, TrainingArguments, Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

print("\n[Step 1] 모델 로드...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
tokenizer = processor.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)
print(f"  로드: {time.time()-t0:.1f}s")

print("\n[Step 2] LoRA 설정 (r=32)...")
lm_linear_modules = set()
for name, module in model.named_modules():
    if "language_model" in name and isinstance(module, torch.nn.Linear):
        leaf = name.split(".")[-1]
        if leaf in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            lm_linear_modules.add(leaf)

lora_config = LoraConfig(
    r=32,             # v2: 16 → v3: 32 (실험 B)
    lora_alpha=64,    # 2×r
    target_modules=sorted(lm_linear_modules),
    exclude_modules=r".*\.(vision_tower|audio_tower|embed_vision|embed_audio|lm_head)\..*",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"  학습 파라미터: {train_params/1e6:.1f}M / {total_params/1e9:.2f}B ({train_params/total_params*100:.3f}%)")

print("\n[Step 3] 데이터셋 토크나이징...")
from datasets import Dataset as HFDataset

ALPACA_TMPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

def tokenize_example(ex):
    text = ALPACA_TMPL.format(
        instruction=ex.get("instruction",""),
        input=ex.get("input",""),
        output=ex.get("output",""),
    ) + tokenizer.eos_token
    enc = tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors=None)
    labels = [-100 if t == tokenizer.pad_token_id else t for t in enc["input_ids"]]
    enc["labels"] = labels
    return enc

data = json.loads(v3_path.read_text(encoding="utf-8"))
hf_ds = HFDataset.from_list(data)
tokenized = hf_ds.map(tokenize_example, remove_columns=hf_ds.column_names, num_proc=4)
print(f"  {len(tokenized)}행 토크나이징 완료")

MAX_STEPS = 400  # v3: 300 → 400

print(f"\n[Step 4] 파인튜닝 시작 (max_steps={MAX_STEPS})...")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    max_steps=MAX_STEPS,
    learning_rate=1.5e-4,   # 약간 낮춤
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=40,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_steps=400,
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

print("=" * 60)
t_train = time.time()
result = trainer.train()
elapsed = time.time() - t_train

print("=" * 60)
print(f"✅ v3 학습 완료!")
print(f"   시간: {elapsed:.0f}초 ({elapsed/60:.1f}분)")
print(f"   최종 loss: {result.training_loss:.4f}")

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))
print(f"   저장: {OUTPUT_DIR}")

# 메타 저장
meta = {
    "version": "v3",
    "dataset_size": len(data),
    "lora_r": 32,
    "lora_alpha": 64,
    "max_steps": MAX_STEPS,
    "training_loss": result.training_loss,
    "elapsed_seconds": elapsed,
    "changes_from_v2": "r=16→32, boost data x3, max_steps=300→400",
}
json.dump(meta, open(OUTPUT_DIR / "training_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
