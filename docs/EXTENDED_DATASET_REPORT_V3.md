# EXTENDED_DATASET_REPORT_V3.md
> 생성일: 2026-04-04  
> 작업: data-quality-boost (Researcher Agent)

---

## 📋 Executive Summary

- **전체 클래스**: 38개 (신규 34종 포함)
- **총 이미지**: 21,719장
- **400장 미달 클래스**: 0개 (38/38 완전 달성 ✅)
- **핵심 개선**: Rice Blast 500→990장, Coffee Leaf Miner 460→860장

---

## 🔍 Task 1: 문제 이미지 분석 결과

| 항목 | 결과 |
|------|------|
| Rice Blast 파일명 공백 이미지 | 328장 (asheniranga LeafBlast 소스, 정상 이미지) |
| 손상 이미지 (Coffee Leaf Miner) | 0장 |
| 손상 이미지 (Rice Blast) | 0장 |
| 최소 크기 미달 이미지 | 0장 |

> **주목**: 파일명에 공백이 있는 이미지(예: `shape 600 .jpg`)는 손상이 아닌 정상 이미지. 모두 256×256 RGB.

---

## 🗑️ Task 2: 손상/저품질 이미지 제거

4개 주요 폴더 검사 결과: **제거 0장** (모두 정상)

---

## 🗃️ Task 3: Kagglehub 캐시 탐색 결과

| 소스 | 관련 클래스 | 이미지 수 |
|------|------------|---------|
| `musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd` | Rice_leaf_blast | 440장 |
| `thegoanpanda/rice-crop-diseases` | Blast Disease | 50장 |
| `asheniranga/leaf-disease-dataset-combination` | Rice LeafBlast | 779장 |
| `badasstechie/coffee-leaf-diseases` | Coffee (분류 미확인) | 1,664장 |

> Coffee Leaf Miner 전용 캐시 데이터셋 없음 → Augmentation으로 대체

---

## 📦 Task 4: 추가 이미지 복사 (Rice Blast)

**musfiqurtuhin** 소스에서 440장 추가 (bcdd_ 접두사):
- train/Rice_leaf_blast: 200장
- val/Rice_leaf_blast: 80장  
- test/Rice_leaf_blast: 160장

**thegoanpanda** Blast Disease: 50장 추가 (goan_ 접두사)

→ Rice Blast: **500 → 990장**

---

## 🔄 Task 5: Augmentation 결과

### Coffee Leaf Miner
- 원본: 460장
- 증강: 400장 (원본 200장 × 2가지 aug)
- **최종: 860장** ✅

### 부족 클래스 일괄 보강 (→ 400장)

| 클래스 | 원본 | 최종 | 증강 |
|--------|------|------|------|
| Bacterial leaf blight | 40 | 403 | +363 |
| Banana Bract Mosaic Virus | 50 | 402 | +352 |
| Banana Moko Disease | 55 | 407 | +352 |
| Banana Panama Disease | 100 | 408 | +308 |
| Brown spot | 40 | 403 | +363 |
| **Citrus Melanose** | 13 | 400 | +387 (24배!) |
| Leaf smut | 40 | 403 | +363 |
| Citrus Greening | 204 | 400 | +196 |
| Coffee Leaf Rust | 380 | 400 | +20 |
| Wheat Loose Smut | 354 | 400 | +46 |

**Augmentation 기법**: 총 11종 (hflip, vflip, rot90/180/270, bright_hi/lo, contrast_hi, saturation, blur, sharpen) + 조합 aug

---

## 📊 Task 6: 최종 전체 현황

| 클래스 | 이미지 수 | 상태 |
|--------|---------|------|
| Bacterial leaf blight | 403 | ✅ |
| Banana Black Sigatoka | 488 | ✅ |
| Banana Bract Mosaic Virus | 402 | ✅ |
| Banana Moko Disease | 407 | ✅ |
| Banana Panama Disease | 408 | ✅ |
| Banana Yellow Sigatoka | 496 | ✅ |
| Brown spot | 403 | ✅ |
| Cassava Bacterial Blight | 468 | ✅ |
| Cassava Brown Streak Disease | 558 | ✅ |
| Cassava Green Mottle | 471 | ✅ |
| Cassava Mosaic Disease | 426 | ✅ |
| Citrus Black Spot | 500 | ✅ |
| Citrus Canker | 500 | ✅ |
| Citrus Greening | 400 | ✅ |
| Citrus Melanose | 400 | ✅ |
| **Coffee Leaf Miner** | **860** | ✅ |
| Coffee Leaf Rust | 400 | ✅ |
| Coffee Phoma | 484 | ✅ |
| Corn_Blight | 1,146 | ✅ |
| Corn_Common_Rust | 2,261 | ✅ |
| Corn_Gray_Leaf_Spot | 574 | ✅ |
| Corn_Healthy | 1,162 | ✅ |
| Leaf smut | 403 | ✅ |
| Mango Anthracnose | 500 | ✅ |
| Mango Bacterial Canker | 500 | ✅ |
| Mango Die Back | 500 | ✅ |
| Mango Gall Midge | 500 | ✅ |
| Mango Powdery Mildew | 500 | ✅ |
| Mango Sooty Mould | 500 | ✅ |
| Rice Bacterial Blight | 500 | ✅ |
| **Rice Blast** | **990** | ✅ |
| Rice Brown Spot | 500 | ✅ |
| Rice Hispa | 408 | ✅ |
| Rice Leaf Smut | 500 | ✅ |
| Wheat Leaf Rust | 500 | ✅ |
| Wheat Loose Smut | 400 | ✅ |
| Wheat Stem Rust | 401 | ✅ |
| Wheat Stripe Rust | 500 | ✅ |

**총합: 21,719장 | 38개 클래스 | 400장 미달: 0개**

---

## ⚠️ 주의사항

1. **Citrus Melanose**: 원본 13장 → 400장 (29배 증강). 다양성 낮을 수 있어 모델 과적합 주의
2. **Rice Blast 공백 파일명**: `shape NNN .jpg` 형태의 파일 정상이지만, 파일 경로 처리 시 공백 이스케이프 필요
3. **Aug 이미지 식별**: `_hflip`, `_vflip`, `_rot90` 등 접미사로 원본 구분 가능
4. **Rice Blast 다양성**: musfiqurtuhin(BCDD 스타일)과 기존 소스가 혼합되어 다양성 확보됨

---

_보고서 생성: Researcher Agent | data-quality-boost session_
