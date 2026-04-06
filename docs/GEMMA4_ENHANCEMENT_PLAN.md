# GEMMA4_ENHANCEMENT_PLAN.md — Gemma4 활용도 강화 계획
> Researcher Agent 작성 | 2026-04-05  
> 목표: Gemma4 활용도 극대화 + Unsloth Special Technology Track ($10K) 도전

---

## 📊 현황 분석 — 현재 Gemma4 사용 방식

### 현재 아키텍처 (cropdoc_infer.py 분석 기준)

```
CNN top1 신뢰도 ≥ 0.90 (혹은 0.97)  →  Gemma4 SKIP (즉시 반환)
CNN top1 신뢰도 < 0.90             →  Gemma4 표준 프롬프트 (CNN 힌트 포함)
CNN top1 신뢰도 < 0.50             →  Gemma4 FOCUSED 프롬프트 (집중 판단)
```

### 현재 Gemma4 역할의 한계
1. **레이블 분류기만**: max_new_tokens=15 → 한 단어(레이블) 출력만
2. **시각 분석 shallow**: 이미지를 받지만 CNN 힌트에 의존 → 독자적 시각 판단 약함
3. **Explanation 없음**: 농민에게 "왜 이 병인지" 설명 없음 (src/model.py는 있지만 cropdoc_infer.py는 미구현)
4. **Chain-of-thought 없음**: 한 번에 레이블만 출력 → 추론 품질 저하
5. **단방향 플로우**: 이미지 → CNN → Gemma4 순서만, 역방향 피드백 없음

---

## 🚀 방법 1: Gemma4 Chain-of-Thought (CoT) 추론 [즉시 구현 가능]

### 아이디어
현재 Gemma4는 `max_new_tokens=15`로 레이블 하나만 출력.  
CoT를 활성화하면 "왜 이 병인지" 추론 과정을 거쳐 더 정확한 레이블 도출.

### 구현 방법

**방법 A: enable_thinking=True 활성화**
```python
# 현재 코드 (cropdoc_infer.py line ~618)
text = _gemma_proc.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False  # ← 이 줄 변경
)

# 변경 후
text = _gemma_proc.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True   # ← Gemma4 thinking 모드 (E4B-IT 지원)
)
GEMMA_MAX_TOKENS_COT = 200  # 15 → 200으로 증가
```

**방법 B: CoT 프롬프트 엔지니어링 (thinking 없이)**
```python
GEMMA_SYSTEM_COT = """You are a plant pathologist expert.
A CNN model analyzed this plant leaf image:
{cnn_hints}

Analyze step by step:
1. What visual symptoms are visible in the image? (color, spots, texture)
2. Do the CNN predictions match the visual symptoms?
3. What is the most likely diagnosis?

Think through each step, then output ONLY the final label.
Final answer: [LABEL]"""
```

### 예상 효과
- CNN 오탐지 케이스에서 정확도 +3~5% (추론 기반 교정)
- 특히 0.50~0.90 구간 저신뢰도 케이스 개선
- **구현 시간: 1~2시간** (프롬프트 수정 + 토큰 수 조정)

---

## 🚀 방법 2: Gemma4 멀티모달 직접 이미지 분석 [중간 난이도]

### 아이디어
현재 Gemma4는 CNN 힌트(텍스트)에 의존해 판단. 이미지를 Gemma4가 직접 독립적으로 분석하고, CNN 결과와 다를 때 투표 로직 적용.

### 구현 방법

```python
def _gemma_direct_vision(img: Image.Image, img_path: str) -> Optional[str]:
    """
    Gemma4가 CNN 힌트 없이 이미지만 보고 독립 판단.
    CNN top1과 다르면 앙상블 투표에 활용.
    """
    VISION_PROMPT = """You are an expert plant pathologist.
Look at this plant leaf image carefully.
Based ONLY on the visual appearance (color, spots, texture, pattern):

Which disease label best describes this image?
Choose ONE from the valid labels below:
{valid_labels}

Output ONLY the label name, nothing else."""

    labels_str = "\n".join(f"- {lbl}" for lbl in VALID_LABELS)
    
    messages = [
        {"role": "system", "content": VISION_PROMPT.format(valid_labels=labels_str)},
        {"role": "user", "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": "Diagnosis:"},
        ]},
    ]
    
    text = _gemma_proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = _gemma_proc(text=text, images=img, return_tensors="pt").to(_gemma_device)
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        out = _gemma_model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            use_cache=True,
            pad_token_id=_gemma_proc.tokenizer.eos_token_id,
        )
    
    resp = _gemma_proc.decode(out[0][input_len:], skip_special_tokens=True).strip()
    
    for lbl in VALID_LABELS:
        if lbl.lower() in resp.lower():
            return lbl
    return None
```

### 앙상블 로직

```python
def _cnn_gemma_ensemble(cnn_results, gemma_cnn_hint_result, gemma_direct_result):
    """
    소프트 보팅: CNN (0.6) + Gemma4-힌트 (0.3) + Gemma4-직접 (0.1)
    모든 결과가 다를 때 CNN top1이 기준이 되나 가중치로 조정.
    """
    votes = {}
    
    # CNN: 신뢰도 × 0.6 가중치
    for score, lbl in cnn_results[:3]:
        votes[lbl] = votes.get(lbl, 0) + score * 0.6
    
    # Gemma4 힌트 기반: +0.3
    if gemma_cnn_hint_result in VALID_LABELS:
        votes[gemma_cnn_hint_result] = votes.get(gemma_cnn_hint_result, 0) + 0.3
    
    # Gemma4 직접 비전: +0.1
    if gemma_direct_result and gemma_direct_result in VALID_LABELS:
        votes[gemma_direct_result] = votes.get(gemma_direct_result, 0) + 0.1
    
    return max(votes, key=votes.get)
```

### 예상 효과
- CNN이 완전히 틀린 케이스에서 Gemma4 독립 시각 판단으로 교정
- 특히 신규 확장 레이블(16종) 정확도 향상 기대
- **구현 시간: 4~6시간**

---

## 🚀 방법 3: Gemma4 Rich Output (설명 + 처방 생성) [즉시 구현 가능]

### 아이디어
현재 cropdoc_infer.py는 레이블만 반환. src/model.py에는 이미 진단 설명 생성 로직이 구현되어 있음.  
두 파이프라인을 통합: **레이블 분류** 후 Gemma4로 **처방 설명 생성** 추가.

### 구현 방법

```python
GEMMA_EXPLANATION_PROMPT = """You are CropDoc, an agricultural expert for smallholder farmers.

Diagnosis: {final_label}
Crop image provided.

Provide a brief, practical treatment guide in Korean:

## 진단: {final_label}
## 원인
(1-2 sentences on pathogen/cause)

## 처방
• (immediate action with product name)
• (application method/dosage)

## 긴급도
🚨 즉시 조치 / ⚠️ 3일 내 / ℹ️ 모니터링 — choose one

Keep response under 150 words."""

def _gemma_explain(img: Image.Image, img_path: str, final_label: str) -> str:
    """레이블 확정 후 처방 설명 생성 (별도 호출)."""
    messages = [
        {"role": "system", "content": GEMMA_EXPLANATION_PROMPT.format(
            final_label=final_label)},
        {"role": "user", "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": "Provide diagnosis explanation:"},
        ]},
    ]
    
    text = _gemma_proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _gemma_proc(text=text, images=img, return_tensors="pt").to(_gemma_device)
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        out = _gemma_model.generate(
            **inputs, max_new_tokens=250, do_sample=False,
            use_cache=True,
            pad_token_id=_gemma_proc.tokenizer.eos_token_id,
        )
    
    return _gemma_proc.decode(out[0][input_len:], skip_special_tokens=True).strip()
```

### 예상 효과
- 심사위원 임팩트 ↑↑: 단순 레이블이 아닌 실제 처방 제공
- Gemma4 활용 깊이 증가 → Technical Implementation 점수 향상
- **구현 시간: 2~3시간**

---

## 🔬 Unsloth 파인튜닝 계획 — Special Technology Track ($10K)

### 수상 요건 분석

Unsloth Prize ($10,000) 수상 조건:
- **Unsloth 라이브러리를 사용하여 Gemma 4를 파인튜닝**한 결과물 제출
- 파인튜닝된 모델이 실제 작동하는 데모에 통합
- Kaggle Writeup에 Unsloth 사용 설명 명시

### 현재 환경 분석

| 항목 | 현황 |
|------|------|
| Python | 3.10 |
| transformers | 5.5.0 |
| torch | 2.11.0 |
| unsloth | ❌ 미설치 |
| GPU | 미확인 (Kaggle T4 필요) |

### Unsloth 지원 현황 (2026-04 기준)

Unsloth는 **Gemma 1~3** 공식 지원 확인됨 (HF: unsloth/gemma-3-4b-it-unsloth-bnb-4bit).  
Gemma 4 지원은 현재 **Unsloth Core(코드 기반)로 가능** — `unsloth` 패키지에서 FastLanguageModel 사용.

```python
# Unsloth Gemma4 파인튜닝 방식 (Kaggle T4 GPU 기준)
from unsloth import FastLanguageModel
import torch

# 1. 모델 로드 (4-bit 양자화로 메모리 절약)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-E4B-it",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,        # T4 16GB에서 필수
)

# 2. LoRA 설정
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                     # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # VRAM 절약 핵심
    random_state=42,
)
```

### 데이터셋 준비 전략 — VLM 파인튜닝 포맷

#### 현재 보유 데이터
- `data/plantvillage/`: 38종, ~175,767장
- `data/field_images/` (extended): 다수 필드 이미지

#### 파인튜닝용 변환 포맷 (ShareGPT 형식)

```python
import json
from pathlib import Path
from PIL import Image
import base64, io

def convert_to_sharegpt_format(image_path: str, label: str, lang: str = "ko") -> dict:
    """PlantVillage 이미지 → VLM 파인튜닝용 ShareGPT 포맷 변환."""
    
    # 레이블 → 한국어 처방 매핑
    LABEL_TO_KO = {
        "Tomato Late Blight": {
            "name": "토마토 역병 (Tomato Late Blight)",
            "cause": "Phytophthora infestans 난균류",
            "treatment": "만코제브 또는 클로로탈로닐 살균제 즉시 적용. 감염 잎 제거 후 소각.",
            "urgency": "🚨 즉시 조치 필요 (전파 속도 매우 빠름)"
        },
        "Tomato Early Blight": {
            "name": "토마토 조기역병 (Early Blight)",
            "cause": "Alternaria solani 진균",
            "treatment": "클로로탈로닐 살균제 7일 간격 적용. 하단 감염 잎 제거.",
            "urgency": "⚠️ 3일 내 조치 필요"
        },
        # ... (54종 전체 매핑)
    }
    
    info = LABEL_TO_KO.get(label, {
        "name": label,
        "cause": "병원균 분석 중",
        "treatment": "전문가 상담 권장",
        "urgency": "ℹ️ 모니터링"
    })
    
    response_text = (
        f"진단: {info['name']}\n\n"
        f"원인: {info['cause']}\n"
        f"처방: {info['treatment']}\n"
        f"긴급도: {info['urgency']}"
    )
    
    return {
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n이 식물 잎을 보고 병해충을 진단하세요. 원인균, 처방, 긴급도를 포함해 한국어로 답하세요."
            },
            {
                "from": "gpt",
                "value": response_text
            }
        ],
        "image": image_path
    }

# 데이터셋 생성 (클래스당 100장 샘플)
def build_finetuning_dataset(
    data_dir: str = "data/plantvillage",
    output_path: str = "data/finetune/cropdoc_vlm_train.jsonl",
    samples_per_class: int = 100,
) -> int:
    """PlantVillage 데이터를 VLM 파인튜닝용 JSONL로 변환."""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for class_dir in Path(data_dir).iterdir():
            label = class_dir.name.replace("___", " ").replace("_", " ")
            images = list(class_dir.glob("*.jpg"))[:samples_per_class]
            
            for img_path in images:
                sample = convert_to_sharegpt_format(str(img_path), label)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
    
    print(f"생성 완료: {count}개 샘플 → {output_path}")
    return count
```

### 파인튜닝 실행 계획

#### Phase 1: 미니 파인튜닝 (Kaggle T4, ~4시간)

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 학습 설정
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="data/models/gemma4-cropdoc-lora",
        num_train_epochs=1,           # 1 epoch (빠른 검증용)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # 효과적 배치 = 8
        warmup_steps=10,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,                    # bfloat16 (T4 지원)
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        max_seq_length=1024,
        report_to="none",
    ),
    train_dataset=train_ds,
)

trainer.train()

# LoRA 어댑터 저장
model.save_pretrained("data/models/gemma4-cropdoc-lora")
tokenizer.save_pretrained("data/models/gemma4-cropdoc-lora")
print("✅ LoRA 어댑터 저장 완료!")
```

#### 예상 결과

| 지표 | 예상값 | 근거 |
|------|--------|------|
| 학습 시간 | 3~4시간 | T4 16GB, 38종×100장=3,800샘플 |
| VRAM 사용 | ~12GB | 4-bit양자화 + Unsloth gradient checkpointing |
| 한국어 출력 품질 | 파인튜닝 전 대비 +40% | 농업 도메인 특화 |
| 처방 정확도 | 파인튜닝 전 대비 +20% | 레이블별 처방 학습 |

### Special Technology Track 수상 요건 체크리스트

```
✅ 필수 조건:
□ unsloth 패키지 실제 사용 (from unsloth import FastLanguageModel)
□ Gemma 4 모델 파인튜닝 (google/gemma-4-E4B-it)
□ 파인튜닝된 모델이 데모에 통합
□ Kaggle Writeup에 Unsloth 섹션 명시
□ LoRA 어댑터 공개 (HuggingFace Hub 업로드)

✅ 가점 조건:
□ 파인튜닝 전/후 정확도 비교표 제시
□ VRAM 사용량 절약 수치 기록
□ 학습 loss curve 그래프 포함
```

---

## 📋 구현 우선순위 로드맵

### 즉시 (오늘, 1~3시간)

1. **방법 1: Chain-of-Thought 프롬프트** 추가
   - `cropdoc_infer.py`: `enable_thinking=True` 또는 CoT 프롬프트 교체
   - `GEMMA_MAX_TOKENS_NORMAL`: 15 → 200
   - 테스트: 저신뢰도 케이스 5개 수동 검증

2. **방법 3: Rich Output** 추가
   - `cropdoc_infer.py`: `_gemma_explain()` 함수 추가
   - `diagnose()` 반환값에 explanation 필드 추가

### 단기 (내일, 4~6시간)

3. **방법 2: 독립 비전 분석** 구현
   - `_gemma_direct_vision()` 함수 추가
   - `_cnn_gemma_ensemble()` 소프트 보팅 로직
   - eval_harness로 정확도 측정

4. **Unsloth 파인튜닝 데이터셋** 준비
   - `build_finetuning_dataset()` 실행
   - 54종 × 100장 = 5,400개 샘플 JSONL 생성

### Kaggle 환경에서 (2~3일, GPU 필요)

5. **Unsloth 파인튜닝 실행**
   - Kaggle T4 GPU 노트북에서 실행
   - 1 epoch, ~4시간
   - LoRA 어댑터 저장 → HuggingFace Hub 업로드

6. **파인튜닝 전/후 비교 평가**
   - eval_harness 동일 조건으로 비교
   - 결과를 Kaggle Writeup에 포함

---

## 🎯 Gemma4 활용도 강화 — 심사 포인트 매핑

| 강화 방법 | 심사 항목 | 예상 점수 기여 |
|----------|----------|-------------|
| Chain-of-Thought | Technical Implementation | +10~15점 |
| Rich Output (처방 설명) | Impact Vision | +15~20점 |
| 독립 비전 앙상블 | Technical Implementation | +10~15점 |
| Unsloth 파인튜닝 | Special Technology Track | **$10,000** |

---

## ⚠️ 리스크 및 대응

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| Unsloth가 Gemma4 공식 미지원 | 중간 | HuggingFace PEFT + bitsandbytes로 대체 (동일 효과) |
| T4 VRAM 부족 | 높음 | per_device_batch=1 + gradient_accumulation=8 조정 |
| CoT 토큰 증가 → 추론 시간 초과 | 낮음 | max_new_tokens 단계적 조정 (50 → 100 → 200) |
| 파인튜닝 후 정확도 저하 | 낮음 | 기존 LoRA 없는 버전 병행 유지 (rollback 가능) |
| 파인튜닝 시간 부족 (5/18 마감) | 중간 | 2주 전부터 Kaggle GPU 세션 확보 |

---

## 🔗 참고 자료

- Unsloth 공식 비전 파인튜닝: https://unsloth.ai/docs/basics/vision-fine-tuning
- Unsloth Gemma3 지원 블로그: https://unsloth.ai/blog/gemma3 (Gemma4 유사 적용 가능)
- HuggingFace Unsloth: https://huggingface.co/unsloth (gemma-3-4b-it-unsloth-bnb-4bit 참조)
- PlantVillage 파인튜닝 포맷: ShareGPT (이미지+대화 쌍)
- Kaggle Unsloth 노트북: https://docs.unsloth.ai/get-started/unsloth-notebooks

---

_Researcher Agent | gemma4-research 세션 | 2026-04-05_
