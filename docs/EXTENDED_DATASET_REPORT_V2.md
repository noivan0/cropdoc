# 확장 데이터셋 수집 보고서 V2

> 작성일: 2026-04-04  
> 작성: Researcher Agent (data-augment-collect 세션)  
> 목적: Corn Common Rust / Cassava 정확도 개선을 위한 추가 데이터 수집 및 보강

---

## 요약

| 항목 | 이전 | 이후 | 변화 |
|------|------|------|------|
| 총 이미지 수 | 16,404장 | 19,145장 | **+2,741장 (+16.7%)** |
| Corn Common Rust | 1,306장 | **2,261장** | +955장 (+73%) |
| Cassava Mosaic Disease | 240장 | **426장** | +186장 (+78%) |
| Cassava Brown Streak | 372장 | **558장** | +186장 (+50%) |
| Cassava Bacterial Blight | 306장 | **468장** | +162장 (+53%) |
| Cassava Green Mottle | 285장 | **471장** | +186장 (+65%) |
| 200장 미만 클래스 | 12개 | **0개** | 모두 보강 완료 |

---

## Task 1: 데이터 품질 분석 결과

### 문제 클래스 초기 현황

| 클래스 | 이미지 수 | 평균 크기 | 문제점 |
|--------|---------|---------|--------|
| Corn_Common_Rust | 1,306장 | 256x256 | 데이터 소스 단일 (smaranjitghose만) |
| Cassava Mosaic Disease | 240장 | 256x256 | **수량 부족** (가장 낮음) |
| Cassava Brown Streak Disease | 372장 | 256x256 | 수량 적음 |
| Coffee Leaf Rust | 380장 | 2048x1024 | 해상도 불균일 |

### 핵심 진단

Corn Common Rust가 1,306장으로 많음에도 불구하고 낮은 정확도를 보이는 이유:
1. **단일 소스 의존**: smaranjitghose 데이터셋 하나에만 의존 → 다양성 부족
2. **학습 전략 문제**: Stage 1에서 backbone 동결 → feature extractor가 신규 클래스에 미적응
3. **Cassava 클래스 수량**: 200~370장 수준으로 타 클래스(500장) 대비 부족

---

## Task 2: 추가 데이터셋 탐색 결과

### 성공한 소스

| 소스 | 대상 클래스 | 추가 이미지 | 비고 |
|------|------------|-----------|------|
| vipoooool/new-plant-diseases-dataset (PlantVillage augmented) | Corn Common Rust | +955장 | 원본(flip 제외) |
| asheniranga/leaf-disease-dataset-combination (test+val) | Cassava 4종 | +720장 | train에 없던 test/validation |

### 실패한 소스 (접근 권한/SSL 문제)

| 데이터셋 | 실패 원인 |
|---------|---------|
| nirmalsankalana/cassava-leaf-disease-dataset | SSL 인증서 오류 |
| oluwafemiebuka/cassava-leaf-disease | 403 권한 없음 |
| rajkumarl/corn-disease-dataset | 403 권한 없음 |
| noulam/cassava | 403 권한 없음 |
| usmanalibutt9393/cassava-leaf-disease | 403 권한 없음 |
| c/cassava-leaf-disease-classification (대회 데이터) | API 미지원 |

### PlantVillage 기존 캐시 활용

이미 로컬에 다운로드된 데이터셋에서 미활용 데이터 발굴:

```
vipoooool/new-plant-diseases-dataset:
  Corn_(maize)___Common_rust_: 1,907장 → 원본(flip 미포함) 1,192장 중 955장 추가

abdallahalidev/plantvillage-dataset:
  (Cassava 클래스 없음 — PlantVillage 원본에는 Cassava 미포함)

asheniranga/leaf-disease-dataset-combination:
  test + validation 분할에서 미활용 이미지 720장 발굴
```

---

## Task 3: 기존 PlantVillage Corn 데이터 확인

```
PlantVillage (vipoooool, augmented):
  Corn_(maize)___Common_rust_: 1,907장 (원본+flip)
  Corn_(maize)___Northern_Leaf_Blight: 1,908장
  Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot: 1,642장
  Corn_(maize)___healthy: 1,859장

PlantVillage (abdallahalidev, color original):
  Corn_(maize)___Common_rust_: 1,192장
  Corn_(maize)___Northern_Leaf_Blight: 985장
```

**활용 결정**: vipoooool 원본(flip 파일 제외)을 Corn_Common_Rust에 추가  
→ 1,306장 + 955장 = **2,261장** (73% 증가)

---

## Task 4: 데이터 보강 (Augmentation)

### Augmentation 기법

```python
augmentations = [
    FLIP_LEFT_RIGHT,        # 좌우 반전
    FLIP_TOP_BOTTOM,        # 상하 반전  
    rotate([90, 180, 270]), # 90도 단위 회전
    brightness(0.7~1.3),    # 밝기 조절
    contrast(0.8~1.2),      # 대비 조절
    color(0.8~1.2),         # 색조 조절
]
```

### 보강 결과 (200장 미만 → 200장 목표)

| 클래스 | 원본 | Augmented | 최종 |
|--------|------|-----------|------|
| Citrus Melanose | 13장 | +187장 | **200장** |
| Bacterial leaf blight | 40장 | +160장 | **200장** |
| Brown spot | 40장 | +160장 | **200장** |
| Leaf smut | 40장 | +160장 | **200장** |
| Banana Bract Mosaic Virus | 50장 | +150장 | **200장** |
| Banana Moko Disease | 55장 | +147장 | **202장** |
| Banana Panama Disease | 100장 | +102장 | **202장** |

---

## 최종 클래스별 이미지 현황

### 🌽 Corn (4종)

| 레이블 | 이미지 수 | 변화 |
|--------|---------|------|
| Corn_Common_Rust | **2,261장** | +955장 ↑ (vipoooool 추가) |
| Corn_Blight | 1,146장 | 유지 |
| Corn_Healthy | 1,162장 | 유지 |
| Corn_Gray_Leaf_Spot | 574장 | 유지 |

### 🌿 Cassava (4종)

| 레이블 | 이미지 수 | 변화 |
|--------|---------|------|
| Cassava Brown Streak Disease | **558장** | +186장 ↑ |
| Cassava Green Mottle | **471장** | +186장 ↑ |
| Cassava Bacterial Blight | **468장** | +162장 ↑ |
| Cassava Mosaic Disease | **426장** | +186장 ↑ |

### ☕ Coffee (3종)

| 레이블 | 이미지 수 |
|--------|---------|
| Coffee Phoma | 484장 |
| Coffee Leaf Miner | 460장 |
| Coffee Leaf Rust | 380장 |

### 🌾 Rice (5종)

| 레이블 | 이미지 수 |
|--------|---------|
| Rice Bacterial Blight | 500장 |
| Rice Blast | 500장 |
| Rice Brown Spot | 500장 |
| Rice Leaf Smut | 500장 |
| Rice Hispa | 408장 |

### 🌾 Wheat (4종)

| 레이블 | 이미지 수 |
|--------|---------|
| Wheat Leaf Rust | 500장 |
| Wheat Stripe Rust | 500장 |
| Wheat Stem Rust | 401장 |
| Wheat Loose Smut | 354장 |

### 🥭 Mango (6종)

| 레이블 | 이미지 수 |
|--------|---------|
| Mango Anthracnose | 500장 |
| Mango Bacterial Canker | 500장 |
| Mango Die Back | 500장 |
| Mango Gall Midge | 500장 |
| Mango Powdery Mildew | 500장 |
| Mango Sooty Mould | 500장 |

### 🍌 Banana (5종)

| 레이블 | 이미지 수 |
|--------|---------|
| Banana Black Sigatoka | 488장 |
| Banana Yellow Sigatoka | 496장 |
| Banana Panama Disease | 202장 |
| Banana Moko Disease | 202장 |
| Banana Bract Mosaic Virus | 200장 |

### 🍊 Citrus (4종)

| 레이블 | 이미지 수 |
|--------|---------|
| Citrus Black Spot | 500장 |
| Citrus Canker | 500장 |
| Citrus Greening | 204장 |
| Citrus Melanose | 200장 |

### 기타 (소량)

| 레이블 | 이미지 수 | 비고 |
|--------|---------|------|
| Bacterial leaf blight | 200장 | aug 포함 |
| Brown spot | 200장 | aug 포함 |
| Leaf smut | 200장 | aug 포함 |

---

## 재학습 권고사항

### 🔴 최우선: Stage 2 Fine-tuning (backbone unfreeze)

현재 확장 모델은 **feature extractor를 완전 동결한 채** classifier만 학습했습니다.  
Corn Common Rust / Cassava의 낮은 정확도 원인은 **데이터 부족보다 학습 전략 문제**입니다.

**권고 설정:**
```python
# train_ext_stage2.py 실행
# features.6, features.7 + classifier unfreeze
LR = 3e-5
EPOCHS = 8
SCHEDULER = CosineAnnealingLR

# 데이터 현황으로 기대 정확도:
# Corn Common Rust: 2,261장 → 95%+ 예상
# Cassava Mosaic: 426장 → 85%+ 예상 (Cassava 이미지 특성상 변동성 큼)
```

### 🟡 보완: 추가 Cassava 데이터 필요

현재 Cassava 4종 평균 480장으로 타 클래스(500장) 대비 다소 부족.  
**다음 소스 우선순위:**
1. `c/cassava-leaf-disease-classification` — Kaggle 대회 21,397장 (수락 필요)
2. iNaturalist Cassava 이미지 (실외 환경 다양성)
3. IITA (국제열대농업연구소) 공개 데이터

### 🟢 완료: 소량 클래스 보강

모든 클래스 200장 이상 확보 완료.  
Citrus Melanose (13→200), Bacterial leaf blight (40→200) 등 극소량 클래스 augmentation 처리.

---

## 디스크 사용량

| 항목 | 용량 |
|------|------|
| data/extended_datasets/ | ~1.5GB |
| kagglehub 캐시 (전체) | ~25GB |
| 여유 공간 목표 | 5GB 제한 내 유지 |

---

## 다음 액션 아이템

1. **즉시**: `python3 scripts/train_ext_stage2.py` 실행 (backbone unfreeze Stage 2)
2. **검증**: 재학습 후 `python3 scripts/eval_harness.py` 로 Corn/Cassava 정확도 측정
3. **목표 정확도**: Corn Common Rust ≥ 90%, Cassava 전 클래스 ≥ 80%
4. **장기**: Kaggle 대회 Cassava 데이터셋 접근 권한 취득 후 재보강

---

_보고서 생성: Researcher Agent | data-augment-collect | 2026-04-04_
