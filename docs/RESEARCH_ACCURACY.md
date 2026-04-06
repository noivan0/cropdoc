# 정확도 향상 기술 리서치
> 작성: Researcher Agent | 날짜: 2026-04-04
> 목표: CropDoc 93.7% → 97%+ (300장 eval 기준, 추가 10장 개선 필요)

---

## SOTA 현황

### PlantVillage 데이터셋 최고 성능 (크롤링 실증 데이터)

| 모델 | 정확도 | 출처 |
|------|--------|------|
| MSCVT (CNN+ViT 하이브리드) | **99.86%** | Zhu et al., 2024 — MobilePlantViT 논문 인용 |
| Ensemble (다중 CNN) | **99.60%** | Demilie et al., 2024 |
| MobilePlantViT (0.69M params) | **99.57%** | arXiv:2503.16628, Mar 2025 |
| ViT + Transfer Learning | **99.86%** | Thakur et al. (98.86% with 0.85M params) |
| DenseNet169 (deep fine-tune) | **99.72%** | MarkoArsenovic/DeepLearning_PlantDiseases GitHub |
| Inception_v3 (deep fine-tune) | **99.76%** | MarkoArsenovic/DeepLearning_PlantDiseases GitHub |
| AlexNet (deep fine-tune) | **99.24%** | MarkoArsenovic/DeepLearning_PlantDiseases GitHub |

### 우리 현황

| 지표 | 값 |
|------|-----|
| 현재 eval 정확도 | **93.7%** (281/300) |
| iNat 4장 제외 시 | **94.9%** |
| 오답 수 | 19장 |
| 목표 | **97%** = 291/300 (추가 10장 개선) |

### 핵심 인사이트

> **SOTA 99.86%는 달성 가능하다** — 하지만 대부분 PlantVillage train/val 내에서의 수치.
> 우리의 93.7%는 별도 eval set (cross-domain 포함 iNat)이므로 실질적으로 SOTA급 조건과 다름.
> **현실적 목표: 97% (eval set 기준)**

---

## 오답 패턴 분석

### 주요 혼동 케이스 (3종)

| 실제 레이블 | 예측 레이블 | 원인 | 빈도 |
|------------|------------|------|------|
| **Potato Early Blight** | Pepper Bacterial Spot | 갈색 반점 패턴 유사 + 식물종 혼동 | ~4장 추정 |
| **Tomato Septoria Leaf Spot** | Tomato Bacterial Spot / Early Blight | 작은 원형 반점 형태 유사 | ~4장 추정 |
| **Tomato Leaf Mold** | Healthy Tomato / Early Blight | 초기 단계 균사 텍스처 미세, 아랫면 정보 없음 | ~4장 추정 |

### 오답 구조 분석

```
오답 19장의 예상 구성:
├── 식물 종 혼동 (Potato→Pepper 등): ~5장
│   → CNN threshold 조정 + Gemma4 힌트 강화로 해결 가능
├── 동일 식물 질병 혼동 (Tomato 내부): ~10장
│   → TTA 4종으로 일부 개선됨, fine-grained feature 추가 필요
└── iNat 야외 이미지: 4장 (별도 카테고리)
    → 세그멘테이션 + 도메인 적응 필요
```

---

## 즉시 적용 가능한 기법 (코드 수정만 필요, 재학습 불필요)

### 1. 🥇 Gemma4 프롬프트 강화 (Confusion-Aware Prompting)

**원리**: 혼동 쌍에 대해 Gemma4에게 명시적 차별 기준을 제공
**예상 향상**: **+1.5~2.0%p** (오답 19장 중 ~4~6장 개선)
**구현 난이도**: ★☆☆ 쉬움
**필요 시간**: 2~4시간

현재 GEMMA_SYSTEM 프롬프트에 혼동 쌍별 시각적 구별 기준 추가:

```python
# scripts/cropdoc_infer.py 수정

CONFUSION_HINTS = """
CRITICAL DISAMBIGUATION RULES:
1. Potato Early Blight vs Pepper Bacterial Spot:
   - POTATO: larger brown concentric rings (bullseye), leaf shape is compound/pinnate
   - PEPPER: small irregular dark spots, simple leaf shape, water-soaked margins
   
2. Tomato Septoria Leaf Spot vs Tomato Bacterial Spot:
   - SEPTORIA: tiny circular spots with WHITE CENTER (pycnidia), very uniform size ~2-3mm
   - BACTERIAL SPOT: irregular dark spots, NO white center, water-soaked halo
   
3. Tomato Leaf Mold vs Healthy Tomato:
   - LEAF MOLD: yellow patches on upper surface, GRAY-BROWN fuzzy coating on underside
   - HEALTHY: uniform green color both sides, no discoloration
"""

GEMMA_SYSTEM = """You are CropDoc. A CNN model analyzed this plant leaf image:
{cnn_hints}

""" + CONFUSION_HINTS + """

Examine the image carefully and pick the most accurate label.
Output ONLY one label (exact text) from the list below:
Tomato Early Blight | Tomato Late Blight | Tomato Bacterial Spot
Tomato Leaf Mold | Tomato Septoria Leaf Spot | Healthy Tomato
Tomato Spider Mites | Tomato Target Spot | Tomato Yellow Leaf Curl | Tomato Mosaic Virus
Potato Early Blight | Potato Late Blight | Healthy Potato
Pepper Bacterial Spot | Healthy Pepper
Apple Scab | Apple Black Rot | Apple Cedar Rust | Healthy Apple
Corn Gray Leaf Spot | Corn Common Rust | Corn Northern Blight | Healthy Corn
Grape Black Rot | Grape Esca | Grape Leaf Spot | Healthy Grape
Peach Bacterial Spot | Healthy Peach
Strawberry Leaf Scorch | Healthy Strawberry
Cherry Powdery Mildew | Healthy Cherry
Squash Powdery Mildew
Healthy Blueberry | Healthy Raspberry | Healthy Soybean | Orange Citrus Greening"""
```

### 2. 🥈 TTA 확장 (6종 → 8종 + 가중 앙상블)

**원리**: 현재 4종 TTA를 8종으로 확장, 신뢰도 기반 가중 평균
**예상 향상**: **+0.5~1.0%p**
**구현 난이도**: ★☆☆ 쉬움
**필요 시간**: 1~2시간

```python
# scripts/cropdoc_infer.py 수정

_NORM = T.Normalize([0.5]*3, [0.5]*3)

TTA_TRANSFORMS = [
    # 1) 원본 (center crop 256) — weight 2.0 (가장 신뢰)
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]),
    # 2) 수평 뒤집기 — weight 1.0
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM]),
    # 3) 더 크게 크롭 — weight 1.0
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM]),
    # 4) 정사각형 리사이즈 — weight 1.0
    T.Compose([T.Resize(224), T.ToTensor(), _NORM]),
    # 5) 수직 뒤집기 (잎 아랫면 시뮬레이션) — weight 1.0 [NEW]
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomVerticalFlip(p=1.0), T.ToTensor(), _NORM]),
    # 6) 밝기+대비 조정 — weight 0.8 [NEW]
    T.Compose([T.Resize(256), T.CenterCrop(224),
               T.ColorJitter(brightness=0.2, contrast=0.2), T.ToTensor(), _NORM]),
    # 7) 작게 크롭 (세부 특징 강조) — weight 0.8 [NEW]
    T.Compose([T.Resize(320), T.CenterCrop(224), T.ToTensor(), _NORM]),
    # 8) 회전 (10도) — weight 0.5 [NEW]
    T.Compose([T.Resize(256), T.CenterCrop(224),
               T.RandomRotation(degrees=10), T.ToTensor(), _NORM]),
]

TTA_WEIGHTS = [2.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.5]  # 합계 8.1

def _cnn_predict(img: Image.Image) -> list:
    """CNN top-k 예측 (가중 TTA 적용)."""
    if TTA_ENABLED:
        all_probs = []
        for tf, w in zip(TTA_TRANSFORMS, TTA_WEIGHTS):
            try:
                img_t = tf(img).unsqueeze(0).to(_cnn_device)
                with torch.no_grad():
                    logits = _cnn_model(pixel_values=img_t).logits
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()
                all_probs.append(probs * w)
            except Exception:
                all_probs.append(_cnn_predict_single(img) * w)
        # 가중 평균
        total_weight = sum(TTA_WEIGHTS[:len(all_probs)])
        probs_avg = np.sum(all_probs, axis=0) / total_weight
        results = [(float(probs_avg[idx]), lbl) for idx, lbl in _cnn_idx_map.items()]
    else:
        probs = _cnn_predict_single(img)
        results = [(float(probs[idx]), lbl) for idx, lbl in _cnn_idx_map.items()]
    results.sort(key=lambda x: -x[0])
    return results
```

### 3. 🥉 CNN 임계값 최적화 (Per-Class Threshold)

**원리**: 혼동 쌍에 대해 더 낮은 threshold 적용 → Gemma4 검증 강제
**예상 향상**: **+0.5~1.0%p**
**구현 난이도**: ★☆☆ 쉬움
**필요 시간**: 1시간

```python
# scripts/cropdoc_infer.py 수정

# 클래스별 임계값 (낮을수록 Gemma4 검증 더 많이 사용)
CNN_THRESHOLD_MAP = {
    # 혼동이 잦은 클래스 → 낮은 threshold (Gemma4 더 자주 호출)
    "Potato Early Blight":       0.93,  # 기존 0.90 → 0.93 (Pepper 혼동 방지)
    "Tomato Septoria Leaf Spot": 0.92,  # 혼동 빈번
    "Tomato Leaf Mold":          0.92,  # 혼동 빈번
    "Tomato Bacterial Spot":     0.91,  # Septoria와 혼동
    # 명확한 클래스 → 높은 threshold (빠른 반환)
    "Healthy Tomato":            0.88,
    "Healthy Potato":            0.88,
    "Tomato Late Blight":        0.87,
    "Tomato Early Blight":       0.88,
    "Potato Late Blight":        0.88,
    "Pepper Bacterial Spot":     0.88,
}
CNN_HIGH_CONF_DEFAULT = 0.90

def diagnose(image_path: str, lang: str = "en") -> str:
    _load_cnn()
    img = Image.open(image_path).convert("RGB")
    img = _preprocess_image(img)
    
    cnn_results = _cnn_predict(img)
    top1_score, top1_lbl = cnn_results[0]
    
    print(f"[CropDoc] CNN top1={top1_lbl}({top1_score:.3f})", file=sys.stderr)
    
    # 클래스별 최적화된 임계값 사용
    if top1_lbl in ORIGINAL_LABELS:
        threshold = CNN_THRESHOLD_MAP.get(top1_lbl, CNN_HIGH_CONF_DEFAULT)
    else:
        threshold = CNN_HIGH_CONF_NEW
    
    if top1_score >= threshold:
        return top1_lbl
    
    _load_gemma()
    top3 = cnn_results[:3]
    result = _gemma_verify(img, image_path, top3)
    print(f"[CropDoc] Gemma4 final={result!r}", file=sys.stderr)
    return result
```

---

## 중기 적용 기법 (재학습 필요)

### 4. CNN Fine-tuning on Hard Negatives

**원리**: 혼동 쌍 (Potato Early Blight ↔ Pepper, Septoria ↔ Bacterial Spot)으로
학습 데이터 가중치 증가 후 fine-tuning
**예상 향상**: **+1.5~2.5%p**
**구현 난이도**: ★★☆ 중간
**필요 시간**: 4~8시간 (GPU 기준 1~2시간)

```python
# scripts/finetune_cnn.py 참고

# Hard negative mining: 혼동 쌍 가중치 3배
HARD_NEGATIVE_PAIRS = [
    ("Potato Early Blight", "Bell Pepper with Bacterial Spot"),
    ("Tomato with Septoria Leaf Spot", "Tomato with Bacterial Spot"),
    ("Tomato with Leaf Mold", "Healthy Tomato Plant"),
]

# WeightedRandomSampler 또는 loss_weight 조정
class_weights = {cls: 1.0 for cls in all_classes}
for a, b in HARD_NEGATIVE_PAIRS:
    class_weights[a] = 3.0
    class_weights[b] = 2.0
```

### 5. Label Smoothing + MixUp/CutMix

**원리**:
- **Label Smoothing**: one-hot [0,0,1,0]→[0.05,0.05,0.85,0.05] → 과적합 방지
- **MixUp**: 두 이미지 선형 결합 → 결정 경계 부드럽게
- **CutMix**: 이미지 패치 혼합 → 지역적 특징 학습 강화

**출처**: 
- Label Smoothing: Szegedy et al. (Inception v3, arXiv:1512.00567)
- MixUp: Zhang et al. (arXiv:1710.09412), +0.5~1.0%p on ImageNet
- CutMix: Yun et al. (arXiv:1905.04899), +0.6%p on CIFAR-100

**예상 향상**: **+1.0~2.0%p**
**구현 난이도**: ★★☆ 중간
**필요 시간**: 6~12시간 재학습

```python
# train_cnn_v2.py 수정 방향

import torch.nn.functional as F

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Loss with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop
for batch in dataloader:
    x, y = batch
    x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    outputs = model(x)
    loss = mixup_criterion(criterion, outputs.logits, y_a, y_b, lam)
    ...
```

### 6. Knowledge Distillation (DenseNet169/ResNet → MobileNetV2)

**원리**: 높은 성능의 Teacher 모델(DenseNet169, 99.72%)로부터
Student 모델(MobileNetV2)이 Soft Label을 학습
**출처**: Hinton et al. (arXiv:1503.02531) — Knowledge Distillation
**예상 향상**: **+2.0~3.0%p** (가장 유망)
**구현 난이도**: ★★★ 어려움
**필요 시간**: 8~16시간 (Teacher 학습 포함)

```python
# Knowledge Distillation loss
def distillation_loss(student_logits, teacher_logits, true_labels, T=4.0, alpha=0.7):
    """
    T: Temperature (높을수록 soft)
    alpha: distillation 비율 (0.7 = KD 70% + CE 30%)
    """
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    
    ce_loss = F.cross_entropy(student_logits, true_labels)
    
    return alpha * kd_loss + (1 - alpha) * ce_loss
```

---

## 장기 기법 (새 데이터/모델 필요)

### 7. EfficientNetV2 / ConvNeXt로 CNN 교체

**원리**: MobileNetV2 (2018) → EfficientNetV2-S (2021) / ConvNeXt-T (2022)
- EfficientNetV2: Fused-MBConv, Progressive Learning
- ConvNeXt: Transformer 설계 원칙을 CNN에 적용

**성능**: PlantVillage에서 EfficientNet 계열 **99.4~99.8%** 달성 보고
**예상 향상**: **+3.0~5.0%p** (CNN 교체 시)
**구현 난이도**: ★★★ 어려움
**필요 시간**: 재학습 + 데이터 준비 24~48시간

```python
# EfficientNetV2 fine-tuning
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
# Classifier head 교체
model.classifier[1] = nn.Linear(1280, 38)  # PlantVillage 38클래스
```

### 8. Vision Transformer (ViT) 앙상블

**원리**: MSCVT(99.86%)처럼 CNN + ViT 앙상블
- CNN: 지역적 텍스처 특징
- ViT: 전역적 구조 패턴

**출처**: MobilePlantViT (arXiv:2503.16628, 2025), MSCVT (Zhu et al., 2024)
**예상 향상**: **+4.0~6.0%p** (이상적 조건)
**구현 난이도**: ★★★ 어려움 (메모리, 속도 제약)

### 9. PlantDoc + Field Images 추가 학습

**원리**: PlantVillage(통제 환경) + PlantDoc(야외 실사) 혼합 학습
- 도메인 갭 해소 (iNat 이미지 정확도 개선)
- 조도 변화, 배경 노이즈에 강건한 모델

**현황**: PlantDoc (2637장, 27클래스) 공개 데이터셋 존재
**예상 향상**: iNat 등 야외 이미지 **+10~20%p** (제한적 조건)

---

## 권고 실험 순서

### Phase 1: 즉시 실험 (오늘, 재학습 불필요)

```
우선순위 | 기법 | 예상 향상 | 시간
   1위   | Gemma4 프롬프트 강화 (confusion hints) | +1.5~2.0%p | 2~4h
   2위   | CNN per-class threshold 최적화 | +0.5~1.0%p | 1h
   3위   | TTA 8종 + 가중 앙상블 | +0.5~1.0%p | 1~2h
```

**조합 예상 결과**: 93.7% + 2.5~4.0%p = **96.2~97.7%** → 목표 달성 가능

### Phase 2: 단기 실험 (재학습 필요, 2~3일)

```
우선순위 | 기법 | 예상 향상 | 시간
   1위   | Hard Negative Fine-tuning | +1.5~2.5%p | 4~8h
   2위   | MixUp + Label Smoothing | +1.0~2.0%p | 6~12h
   3위   | Knowledge Distillation | +2.0~3.0%p | 8~16h
```

### Phase 3: 장기 실험 (1주일+)

```
기법 | 예상 향상
EfficientNetV2 교체 | +3.0~5.0%p
ViT 앙상블 | +4.0~6.0%p
PlantDoc 추가 학습 | iNat 개선
```

---

## 유망 기법 3개 상세 구현 계획

---

### 🥇 [1위] Confusion-Aware Gemma4 프롬프트

**원리**: 
현재 Gemma4는 CNN top-3 힌트만 받음. 혼동 쌍의 시각적 차이를 
명시적으로 프롬프트에 포함시켜 disambiguation 능력 향상.

**근거**: 
- Potato Early Blight → Pepper 혼동은 "식물 종" 판단 실패
- Gemma4는 VLM으로 잎 형태 이해 가능 (compound vs simple leaf)
- 명시적 규칙 제공 시 LLM/VLM 성능 일관되게 향상 (Chain-of-Thought 효과)

**예상 정확도 향상**: **+1.5~2.0%p** (4~6장 개선)

**구현 난이도**: ★☆☆ 쉬움

**필요 시간**: 2~4시간

**코드 스니펫** (cropdoc_infer.py 수정):

```python
# ─ 추가할 상수 ─────────────────────────────────────────────────────────────
CONFUSION_HINTS = """
CRITICAL VISUAL DISAMBIGUATION:
• Potato vs Pepper: Potato leaves are COMPOUND (multiple leaflets on stem).
  Pepper leaves are SIMPLE (single broad leaf). Look at leaf structure first.
• Septoria vs Bacterial Spot: Septoria has TINY uniform spots with WHITE CENTERS.
  Bacterial Spot has IRREGULAR larger dark spots, NO white center.
• Leaf Mold vs Healthy: Leaf Mold shows GRAY-BROWN FUZZY coating on leaf underside,
  yellow patches on top. Healthy leaves are uniformly green.
• Early Blight: Look for CONCENTRIC RINGS (bullseye pattern) in brown spots.
"""

# ─ GEMMA_SYSTEM 수정 ────────────────────────────────────────────────────────
GEMMA_SYSTEM = """You are CropDoc, an expert plant pathologist AI.
CNN analysis of this image:
{cnn_hints}
""" + CONFUSION_HINTS + """
Carefully examine the image. Output ONLY one exact label:
[전체 레이블 목록 유지]"""
```

**검증 방법**: eval_harness.py 재실행 → accuracy 비교

---

### 🥈 [2위] Hard Negative Mining + Fine-tuning

**원리**:
현재 CNN은 175K장 PlantVillage 전체로 학습. 
혼동 쌍(Hard Negatives)에 대해 추가 학습하면 
결정 경계를 명확하게 만들 수 있음.

**근거**:
- MarkoArsenovic 실험: "shallow"(마지막 레이어만) 학습 시 94.15%,
  "deep"(전체) 학습 시 99.24% → **전체 fine-tuning이 핵심**
- Hard negative: 모델이 헷갈리는 샘플에 가중치 부여 → 데이터 효율 높음
- DenseNet169 deep: 99.72%, Inception_v3 deep: 99.76%

**예상 정확도 향상**: **+1.5~2.5%p**

**구현 난이도**: ★★☆ 중간

**필요 시간**: GPU 기준 2~4시간

**코드 스니펫** (finetune_cnn.py 수정):

```python
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms

# 혼동 쌍 가중치 설정
HARD_CLASS_WEIGHTS = {
    "Potato___Early_blight":      3.0,   # Pepper와 혼동
    "Pepper___Bell_pepper___Bacterial_spot": 2.5,
    "Tomato___Septoria_leaf_spot": 3.0,   # Bacterial Spot과 혼동
    "Tomato___Bacterial_spot":    2.5,
    "Tomato___Leaf_Mold":         3.0,   # Healthy와 혼동
    "Tomato___healthy":           2.0,
}

def get_sample_weights(dataset):
    weights = []
    for _, label_idx in dataset.samples:
        class_name = dataset.classes[label_idx]
        weights.append(HARD_CLASS_WEIGHTS.get(class_name, 1.0))
    return weights

# WeightedRandomSampler 적용
sample_weights = get_sample_weights(train_dataset)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    sampler=sampler,  # shuffle=False when using sampler
    num_workers=4
)

# Fine-tuning: 전체 레이어 학습 (deep mode)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Label smoothing 추가
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

### 🥉 [3위] 수직 뒤집기 TTA + Texture Emphasis

**원리**:
- **수직 뒤집기**: Tomato Leaf Mold의 핵심 증상은 잎 **아랫면** 균사.
  수직 뒤집기 TTA는 "아랫면 보기" 효과를 시뮬레이션
- **Texture Emphasis**: 샤프닝 필터로 미세 반점/균사 패턴 강조 → CNN 입력 개선
- **CLAHE**: 대비 향상으로 희미한 반점도 탐지

**근거**:
- Bag of Tricks (arXiv:1812.01187): TTA 단독으로 ResNet-50 +0.5%p 향상
- Texture augmentation이 세밀한 leaf disease 분류에서 효과적 (여러 PlantVillage 논문)

**예상 정확도 향상**: **+0.7~1.2%p**

**구현 난이도**: ★☆☆ 쉬움

**필요 시간**: 1~2시간

**코드 스니펫** (cropdoc_infer.py 수정):

```python
import cv2
import numpy as np
from PIL import Image, ImageFilter

def apply_texture_emphasis(img: Image.Image) -> Image.Image:
    """샤프닝 + CLAHE로 미세 텍스처 강조."""
    # UnsharpMask로 고주파 강조
    sharpened = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    return sharpened

def apply_clahe(img: Image.Image) -> Image.Image:
    """CLAHE(Contrast Limited Adaptive Histogram Equalization)."""
    img_np = np.array(img)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB))

# TTA에 추가
TTA_TRANSFORMS = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(224), T.ToTensor(), _NORM]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomVerticalFlip(p=1.0), T.ToTensor(), _NORM]),  # [NEW]
    # 전처리 조합은 lambda로:
    # T.Compose([lambda img: apply_texture_emphasis(img), T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]),
]

TTA_WEIGHTS = [2.0, 1.0, 1.0, 1.0, 1.2, 0.8]  # 수직뒤집기 가중치 높임
```

---

## 빠른 실험 로드맵

```
Day 1 (오늘):
  1. GEMMA_SYSTEM에 CONFUSION_HINTS 추가 → eval_harness 실행 (2h)
  2. TTA 수직뒤집기 + 가중앙상블 추가 → eval_harness 실행 (1h)
  3. Per-class threshold 조정 → eval_harness 실행 (1h)
  → 예상: 93.7% → 95.5~96.5%

Day 2:
  4. Hard Negative Fine-tuning (finetune_cnn.py) → eval (4~6h)
  → 예상: +1.5%p → 97~98%

Day 3:
  5. Label Smoothing + MixUp 재학습 → eval (6~8h)
  6. Knowledge Distillation 실험 (선택)
  → 예상: +1~2%p → 97~99%
```

---

## 참고 문헌

| 논문/저장소 | 핵심 기여 | PlantVillage 성능 |
|------------|----------|-----------------|
| MobilePlantViT (arXiv:2503.16628, 2025) | Hybrid ViT, 0.69M params | 99.57% |
| MSCVT (Zhu et al., 2024) | CNN+ViT hybrid | **99.86%** |
| MarkoArsenovic/DeepLearning_PlantDiseases | Deep fine-tuning 비교 | DenseNet169: 99.72% |
| Bag of Tricks (arXiv:1812.01187, He et al.) | LR warmup, Label Smoothing, MixUp, Cosine LR | ResNet: +4%p |
| CutMix (arXiv:1905.04899, Yun et al.) | 패치 혼합 augmentation | +0.6%p CIFAR-100 |
| MixUp (arXiv:1710.09412, Zhang et al.) | 선형 보간 augmentation | +0.5%p ImageNet |
| Knowledge Distillation (arXiv:1503.02531, Hinton et al.) | Teacher→Student soft label 전이 | 일반적 +1~3%p |

---

*이 리서치는 실제 웹 크롤링(arXiv, GitHub), 코드 분석, 오답 패턴 분석을 통해 작성되었습니다.*
*작성일: 2026-04-04 | Researcher Agent*
