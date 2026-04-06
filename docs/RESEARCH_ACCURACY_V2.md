# CropDoc 정확도 98% → 99%+ 달성 전략
> Researcher Agent | 2026-04-04 | 현재: 294/300 (98.0%)

---

## 현황 요약

| 지표 | 값 |
|------|-----|
| 현재 정확도 | **98.0%** (294/300) |
| 오답 수 | **6건** |
| 목표 | 99%+ = 297/300+ |
| 필요 개선 | **최소 3건 추가 정답** |

### 오답 6건 분류

| # | 파일 | 실제 레이블 | 원인 | 수정 가능? |
|---|------|------------|------|-----------|
| 1 | `inat_250907864.jpg` | Potato Late Blight | iNat 비표준 야외 이미지 | ❌ 어려움 |
| 2 | `inat_99652641.jpg` | Potato Late Blight | iNat 비표준 야외 이미지 | ❌ 어려움 |
| 3 | `cd38f533...2680.JPG` | Potato Late Blight | CNN 오분류 (256x256 표준 이미지) | ✅ 가능 |
| 4 | Tomato Late Blight #1 | Tomato Late Blight | CNN/앙상블 오류 | ✅ 가능 |
| 5 | Tomato Late Blight #2 | Tomato Late Blight | CNN/앙상블 오류 | ✅ 가능 |
| 6 | Potato Early Blight #1 | Potato Early Blight | CNN/앙상블 오류 | ✅ 가능 |

### 오답 이미지 특성 분석 (cd38f533...2680.JPG)
```
크기: 256x256 RGB (표준 PlantVillage 이미지)
색상 분포:
  - 평균 RGB: [130.7, 124.9, 111.5] → 회갈색 계열
  - 초록 픽셀 비율: 23% (잎이 있음)
  - 갈색 픽셀 비율: 2.5% (적음 → 병반이 불명확)
  - 어두운 픽셀 비율: 14.9% (Late Blight 특징적인 어두운 병반)
  - 명도 mean: 0.53, std: 0.19 (밝기 변화 큼)
진단: 초기 Late Blight 또는 Late/Early 경계 케이스 → CNN이 Early Blight로 오인
```

---

## 즉시 적용 가능 기법 (재학습 불필요)

### 🥇 기법 1: Gemma4 응답 파싱 개선
**예상 향상**: +0~2건 (파싱 오류로 인한 손실 방지)  
**구현 난이도**: ★☆☆  
**소요 시간**: 30분

#### 발견된 파싱 버그
```
현재 코드 버그:
  'Potato\nLate Blight'  → None  (줄바꿈 삽입 시 파싱 실패!)
  'Potato_Late_Blight'   → None  (언더스코어 형식 실패)
  'Late Blight (Potato)' → None  (역순 표현 실패)

정상 동작:
  'Potato Late Blight'   → OK
  'POTATO LATE BLIGHT'   → OK (대소문자 무시)
```

#### 개선 코드 (`_gemma_verify` 함수 내)
```python
def _parse_gemma_response(resp: str) -> str | None:
    """
    개선된 Gemma4 응답 파싱 — 줄바꿈/언더스코어/대소문자 정규화 후 매칭
    """
    import re
    
    # 정규화: 줄바꿈→공백, 언더스코어→공백, 다중공백→단일
    norm = re.sub(r'[\n\r]+', ' ', resp)
    norm = re.sub(r'[_]+', ' ', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    # 1차: 정확한 매칭 (긴 레이블 우선 — 'Potato Late Blight'가 'Late Blight' 보다 먼저)
    for lbl in sorted(VALID_LABELS, key=len, reverse=True):
        if lbl.lower() in norm.lower():
            return lbl
    
    # 2차: 퍼지 매칭 (80% 단어 겹침)
    norm_words = set(norm.lower().split())
    best_lbl, best_score = None, 0
    for lbl in VALID_LABELS:
        lbl_words = set(lbl.lower().split())
        overlap = len(lbl_words & norm_words)
        if overlap > best_score and overlap >= len(lbl_words) * 0.8:
            best_score = overlap
            best_lbl = lbl
    return best_lbl  # None이면 호출부에서 CNN top1 사용
```

**현재 `_gemma_verify` 마지막 부분 수정:**
```python
# 기존 코드 (버그 있음):
matched = None
for lbl in VALID_LABELS:
    if lbl.lower() in resp.lower():
        matched = lbl
        break

# 개선 코드:
matched = _parse_gemma_response(resp)
```

---

### 🥈 기법 2: Gemma4 프롬프트 개선 (Late Blight 특화)
**예상 향상**: +1~2건  
**구현 난이도**: ★★☆  
**소요 시간**: 1~2시간

현재 GEMMA_SYSTEM 프롬프트가 단순 레이블 목록만 제공하는 것이 문제.
Late Blight (Potato/Tomato) 구별 힌트를 추가:

```python
GEMMA_SYSTEM = """You are CropDoc, an expert plant pathologist AI.
A CNN model analyzed this plant leaf image:
{cnn_hints}

Examine the image carefully and pick the MOST ACCURATE label.

DISAMBIGUATION RULES for similar diseases:
- Potato Late Blight: Water-soaked dark lesions, white mold on leaf underside, fast-spreading necrosis
- Potato Early Blight: Concentric ring 'bullseye' brown spots, yellow halo, target-board pattern
- Tomato Late Blight: Large irregular dark-green to brown water-soaked lesions, gray mold
- Tomato Early Blight: Concentric rings, yellow chlorotic halo around lesions

Output ONLY the exact label name (no explanation):
Tomato Early Blight | Tomato Late Blight | Tomato Bacterial Spot
...
"""
```

---

### 🥉 기법 3: CNN Threshold 최적화 (Late Blight 클래스)
**예상 향상**: +1건  
**구현 난이도**: ★★☆  
**소요 시간**: 1시간

```python
# 현재 threshold (단일):
CNN_HIGH_CONF = 0.90

# 개선: 클래스별 threshold 차별화
CNN_CLASS_THRESHOLDS = {
    # Late Blight는 Early Blight와 혼동 많음 → 더 낮은 threshold로 Gemma4 검증 강제
    "Potato Late Blight":  0.80,  # 기존 0.90 → 0.80 (Gemma4 더 자주 사용)
    "Potato Early Blight": 0.80,
    "Tomato Late Blight":  0.82,
    "Tomato Early Blight": 0.82,
    # 기타 클래스는 기존 threshold 유지
}

# diagnose() 함수 수정:
threshold = CNN_CLASS_THRESHOLDS.get(top1_lbl, 
    CNN_HIGH_CONF if top1_lbl in ORIGINAL_LABELS else CNN_HIGH_CONF_NEW)
```

---

## 재학습 필요 기법 (EfficientNetV2 앙상블)

### 기법 4: EfficientNetV2-S Fine-tuning + 3-모델 앙상블
**예상 향상**: +1~3건 (약 +0.3~1.0%p)  
**구현 난이도**: ★★★  
**소요 시간**: 학습 4~8시간 + 코드 2시간

EfficientNetV2-S (22.2M params, 288x288 입력)는 MobileNetV2보다 훨씬 강력.

#### 학습 스크립트 핵심
```python
import timm
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# 모델 생성
model = timm.create_model(
    'tf_efficientnetv2_s.in21k_ft_in1k',  # ImageNet21k 사전학습 → 더 강력
    pretrained=True,
    num_classes=38
)

# Mixup + CutMix augmentation
mixup_fn = Mixup(
    mixup_alpha=0.2,     # Mixup
    cutmix_alpha=1.0,    # CutMix
    prob=0.5,            # 50% 확률 적용
    switch_prob=0.5,     # Mixup:CutMix = 50:50
    label_smoothing=0.1,
    num_classes=38
)

# Soft Target Loss (Mixup과 함께 사용)
criterion = SoftTargetCrossEntropy()

# 학습 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
```

#### Mixup 효과 (문헌 근거)
| 기법 | 데이터셋 | 향상 |
|------|---------|------|
| Mixup α=0.2 | CIFAR-100 | +1.3%p |
| CutMix α=1.0 | ImageNet ResNet-50 | +1.7%p vs Mixup |
| Label Smoothing 0.1 | ImageNet ResNet-50 | +0.3%p |
| Mixup+CutMix | PlantVillage (추정) | +0.5~1.0%p |

**중요**: 현재 `train_cnn_v2.py`에 이미 `label_smoothing=0.1` 적용됨.  
Mixup/CutMix는 미적용 → 추가 효과 기대 가능.

---

## CLIP/DINOv2 Zero-shot 가능성 분석

### Task 1 결론: 현재 98% 시스템에서 비효율적

| 방법 | PlantVillage 예상 정확도 | 우리 시스템 기여 | 추천 |
|------|------------------------|-----------------|------|
| CLIP ViT-L/14 zero-shot | ~72-80% | 저하 위험 | ❌ |
| DINOv2-L kNN (훈련 데이터) | ~92-95% | 저하 위험 | ❌ |
| DINOv2-L + few-shot head | ~96-97% | 미미한 효과 | △ |
| EfficientNetV2-S fine-tuned | ~98.5-99.5% | **+0.5~1.5%p** | ✅ |

**결론**: CLIP/DINOv2는 98% 시스템 개선에 적합하지 않음.  
이미 CNN+Gemma4 앙상블이 이들보다 강력함.

#### CLIP 설치 및 테스트 (참고용)
```bash
pip install git+https://github.com/openai/CLIP.git

python3 -c "
import clip, torch
from PIL import Image

model, preprocess = clip.load('ViT-L/14', device='cuda')

# PlantVillage 38개 레이블로 zero-shot
labels = ['Potato Late Blight', 'Potato Early Blight', ...]
text_inputs = clip.tokenize([f'a photo of {l}' for l in labels]).to('cuda')

img = preprocess(Image.open('test.jpg')).unsqueeze(0).to('cuda')
with torch.no_grad():
    logits, _ = model(img, text_inputs)
probs = logits.softmax(dim=-1)
print(labels[probs.argmax()])
"
```

---

## Gemma4 Fine-tuning 가능성 (Unsloth Prize $10K)

### Task 5 결론: 기술적으로 가능하나 시간 집중 필요

**현재 환경**: Unsloth 미설치, PEFT/TRL 미설치
**설치 가능**: `pip install unsloth peft trl` (약 5~10분)
**학습 데이터**: PlantVillage 38종 x (이미지+레이블) → conversation 포맷 변환 필요

#### 설치 및 데이터 준비
```bash
pip install unsloth peft trl datasets

# 데이터 준비: 이미지 → base64 인코딩 후 QA 포맷
python3 scripts/prepare_unsloth_dataset.py
```

#### 학습 데이터 포맷
```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "image", "url": "data:image/jpeg;base64,..."},
        {"type": "text", "text": "What plant disease does this leaf have? Answer with the exact disease name."}
      ]
    },
    {
      "role": "assistant",
      "content": "Potato Late Blight"
    }
  ]
}
```

#### Unsloth Vision Fine-tuning 코드 골격
```python
from unsloth import FastVisionModel
import torch

# Gemma4 로드 (4-bit 양자화)
model, tokenizer = FastVisionModel.from_pretrained(
    "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# LoRA 설정
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=42,
)

# SFT 학습
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True,
        output_dir="data/models/gemma4_finetuned",
    ),
)
trainer.train()
```

#### 예상 효과
- Gemma4가 PlantVillage 도메인에 특화 → 파싱 오류 0%
- Late Blight vs Early Blight 구별 정확도 향상
- 예상 시스템 정확도: **+0.5~1.5%p** (구현 완성 시)
- **Unsloth Prize 조건 충족**: Unsloth로 모델 파인튜닝 + 성능 향상 증명

#### Unsloth Prize 전략
1. Gemma4 E4B를 Unsloth로 4-bit 양자화 fine-tuning
2. PlantVillage 38종 분류 데이터셋으로 500~1000 스텝 학습
3. eval set에서 비교: fine-tuning 전/후 정확도
4. Hackathon 제출물에 Unsloth 활용 사례로 포함

---

## 적용 우선순위 로드맵

### Phase 1: 즉시 적용 (0~2시간)
1. **파싱 개선** (`_parse_gemma_response` 함수 추가) → 파싱 오류 → 정답 전환
2. **Gemma4 프롬프트 Late Blight 힌트 추가** → Late Blight 오분류 감소
3. **CNN threshold 클래스별 차별화** → 불확실한 Late Blight를 Gemma4로 검증

### Phase 2: 재학습 (4~8시간)
4. **EfficientNetV2-S Mixup/CutMix 학습** → 3모델 앙상블
5. **앙상블 가중치 최적화** (원본:v2:effv2 = 0.3:0.3:0.4)

### Phase 3: 고급 (12~24시간, Unsloth Prize 목표)
6. **Gemma4 LoRA fine-tuning** (Unsloth 사용)
7. **전체 시스템 재평가** → 99%+ 달성 검증

---

## 예상 최종 정확도

| 적용 기법 | 예상 추가 정답 | 누적 정확도 |
|----------|--------------|------------|
| 현재 v16 | - | 294/300 = **98.0%** |
| + 파싱 개선 | +0~1 | ~98.0~98.3% |
| + Gemma4 프롬프트 | +1~2 | ~98.3~98.7% |
| + CNN threshold 조정 | +0~1 | ~98.3~99.0% |
| + EfficientNetV2 앙상블 | +1~2 | ~98.7~99.3% |
| + Gemma4 fine-tuning | +1~2 | ~99.0~99.7% |

**보수적 예측**: Phase 1+2 완성 시 **99.0~99.3%** (297~298/300)  
**낙관적 예측**: Phase 1+2+3 완성 시 **99.3~99.7%** (298~299/300)

---

## 결론

98% → 99%+ 달성을 위한 핵심 3가지:

1. 🔧 **즉시**: Gemma4 파싱 버그 수정 (`\n` 포함 응답 → 파싱 실패)
2. 📝 **즉시**: Gemma4 프롬프트에 Late Blight 시각적 특징 설명 추가
3. 🏋️ **중기**: EfficientNetV2-S Mixup+CutMix 학습 후 3모델 앙상블

**Unsloth Prize**: 시간이 허락한다면 Gemma4 LoRA fine-tuning으로 $10K 추가 도전 가치 있음.
현재 환경에서 `pip install unsloth peft trl`로 5분 내 준비 가능.

_생성: 2026-04-04 | Researcher Agent_
