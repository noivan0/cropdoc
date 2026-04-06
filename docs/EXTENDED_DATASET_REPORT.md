# 확장 데이터셋 수집 결과

> 수집일: 2026-04-04  
> 작성: Researcher Agent  
> 목적: CropDoc 신규 7개 작물(Coffee/Wheat/Rice/Mango/Cassava/Banana/Citrus) 31종 질병 데이터 수집

---

## 수집된 데이터셋 (소스)

| 소스 | 작물 | 주요 클래스 | 이미지 수 | 크기 |
|------|------|------------|---------|------|
| badasstechie/coffee-leaf-diseases | Coffee | Leaf Rust, Leaf Miner, Phoma | ~3,328장 | 195MB |
| vbookshelf/rice-leaf-diseases | Rice | Brown Spot, Bacterial Blight, Leaf Smut | 120장 | 36MB |
| jay7080dev/rice-plant-diseases-dataset | Rice | Brown Spot, Bacterial Blight, Leaf Smut | 4,684장 | 176MB |
| thegoanpanda/rice-crop-diseases | Rice | Blast, Bacterial Blight, Brown Spot | 200장 | 4MB |
| asheniranga/leaf-disease-dataset-combination | Rice + Cassava | Rice Blast/Hispa, Cassava (4종) | 53,303장 | 761MB |
| aryashah2k/mango-leaf-disease-dataset | Mango | 8종 (Anthracnose, Canker 등) | 4,000장 | 103MB |
| shuvokumarbasak4004/the-mango-leaf-disease-classification-dataset | Mango | Anthracnose, Rust | 8,631장 | 58MB |
| ahmadzargar/mango-leaf-diseases-dataset | Mango | 8종 (Anthracnose, Canker 등) | 4,000장 | 34MB |
| shifatearman/bananalsd | Banana | Sigatoka, Cordana, Pestalotiopsis | 2,537장 | 36MB |
| harshitbana/banana-leaf-disease | Banana | 일반 | 29장 | 24MB |
| sujaykapadnis/banana-disease-recognition-dataset | Banana | 7종 (Black Sigatoka, Panama 등) | 3,264장 | 121MB |
| myprojectdictionary/citrus-leaf-disease-image | Citrus | Greening, Canker, Black Spot, Melanose | 607장 | 41MB |
| jonathansilva2020/dataset-for-classification-of-citrus-diseases | Citrus | Black Spot, Citrus Canker | 2,439장 | 350MB |
| sabaunnisa/wheat-rust-disease | Wheat | Leaf Rust, Stripe Rust, Stem Rust | 1,331장 | 114MB |
| engrirf/wheat-plants-disease-dataset-for-kp-pakistan | Wheat | (PASCAL VOC) | 25장 | 61MB |
| musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd | Rice + Wheat | Rice (3종), Wheat (3종) | 8,992장 | 37MB |

**총 다운로드: 16개 데이터셋, ~12GB 원본 데이터**

---

## 클래스별 이미지 수 (정리 후)

### ☕ Coffee (3종)
| 레이블 | 이미지 수 | 소스 | 상태 |
|--------|---------|------|------|
| Coffee Leaf Rust | 380 | badasstechie/coffee-leaf-diseases | ✅ |
| Coffee Leaf Miner | 460 | badasstechie/coffee-leaf-diseases | ✅ |
| Coffee Phoma | 484 | badasstechie/coffee-leaf-diseases | ✅ |

### 🌾 Rice (5종)
| 레이블 | 이미지 수 | 소스 | 상태 |
|--------|---------|------|------|
| Rice Blast | 500 | thegoanpanda + asheniranga (combination) | ✅ |
| Rice Brown Spot | 500 | vbookshelf + jay7080dev + thegoanpanda + BCDD | ✅ |
| Rice Bacterial Blight | 500 | vbookshelf + jay7080dev + thegoanpanda + BCDD | ✅ |
| Rice Leaf Smut | 500 | vbookshelf + jay7080dev | ✅ |
| Rice Hispa | 408 | asheniranga/leaf-disease-dataset-combination | ✅ |

### 🌾 Wheat (4종)
| 레이블 | 이미지 수 | 소스 | 상태 |
|--------|---------|------|------|
| Wheat Leaf Rust | 500 | sabaunnisa + BCDD | ✅ |
| Wheat Stripe Rust | 500 | sabaunnisa + BCDD | ✅ |
| Wheat Stem Rust | 401 | sabaunnisa/wheat-rust-disease | ✅ |
| Wheat Loose Smut | 354 | musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd | ✅ |

### 🥭 Mango (6종)
| 레이블 | 이미지 수 | 소스 | 상태 |
|--------|---------|------|------|
| Mango Anthracnose | 500 | aryashah2k + shuvokumarbasak4004 + ahmadzargar | ✅ |
| Mango Bacterial Canker | 500 | aryashah2k + ahmadzargar | ✅ |
| Mango Die Back | 500 | aryashah2k + ahmadzargar | ✅ |
| Mango Powdery Mildew | 500 | aryashah2k + ahmadzargar + shuvokumarbasak4004 | ✅ |
| Mango Sooty Mould | 500 | aryashah2k + ahmadzargar | ✅ |
| Mango Gall Midge | 500 | aryashah2k + ahmadzargar | ✅ |

### 🌿 Cassava (4종)
| 레이블 | 이미지 수 | 소스 | 상태 |
|--------|---------|------|------|
| Cassava Bacterial Blight | 306 | asheniranga/leaf-disease-dataset-combination | ✅ |
| Cassava Mosaic Disease | 240 | asheniranga/leaf-disease-dataset-combination | ✅ |
| Cassava Brown Streak Disease | 372 | asheniranga/leaf-disease-dataset-combination | ✅ |
| Cassava Green Mottle | 285 | asheniranga/leaf-disease-dataset-combination | ✅ |

### 🍌 Banana (5종)
| 레이블 | 이미지 수 | 소스 | 상태 |
|--------|---------|------|------|
| Banana Black Sigatoka | 488 | shifatearman/bananalsd + sujaykapadnis | ✅ |
| Banana Yellow Sigatoka | 496 | shifatearman/bananalsd + sujaykapadnis | ✅ |
| Banana Panama Disease | 100 | sujaykapadnis (Original+Augmented) | ✅ |
| Banana Moko Disease | 55 | sujaykapadnis/banana-disease-recognition-dataset | ✅ |
| Banana Bract Mosaic Virus | 50 | sujaykapadnis/banana-disease-recognition-dataset | ✅ |

### 🍊 Citrus (4종)
| 레이블 | 이미지 수 | 소스 | 상태 |
|--------|---------|------|------|
| Citrus Greening | 204 | myprojectdictionary/citrus-leaf-disease-image | ✅ |
| Citrus Canker | 500 | myprojectdictionary + jonathansilva2020 | ✅ |
| Citrus Black Spot | 500 | myprojectdictionary + jonathansilva2020 | ✅ |
| Citrus Melanose | 13 | myprojectdictionary/citrus-leaf-disease-image | ⚠️ 부족 |

---

## 다운로드 실패

| 데이터셋 | 이유 |
|---------|------|
| engrirf/wheat-plants-disease-dataset-for-kp-pakistan | PASCAL VOC YOLO 포맷, 이미지 25장만 포함 (실용성 낮음) |
| usmanalibutt9393/ccl-20-citrus-leaf-disease-dataset | 클래스명 익명화(Disease-A/B/C)로 Melanose 식별 불가 |
| harshitbana/banana-leaf-disease | 29장 수집 후 별도 Banana Panama Disease 보완에 사용 |

---

## 학습 준비 상태

- **재학습 가능: 30종** (각 50장 이상)
- **데이터 부족: 1종** — Citrus Melanose (13장, 소스 데이터셋 자체가 13장으로 제한)

### 권고사항

1. **Citrus Melanose** (13장): Melanose 특화 데이터셋 추가 탐색 필요. 또는 해당 클래스 제외 후 30종으로 학습 진행
2. **Banana Panama Disease** (100장, augmented 포함): 원본 41장 + augmented 59장 혼합. 순수 원본만 원할 경우 41장
3. **Rice Blast** (500장): thegoanpanda(50장) + asheniranga-combination에서 확보. 충분
4. **Coffee 3종**: CSV 레이블 기반 다중 레이블 이미지 포함 (한 이미지에 복수 질병 가능)

---

## 데이터 경로

```
data/extended_datasets/
├── Coffee Leaf Rust/      (380장)
├── Coffee Leaf Miner/     (460장)
├── Coffee Phoma/          (484장)
├── Rice Blast/            (500장)
├── Rice Brown Spot/       (500장)
├── Rice Bacterial Blight/ (500장)
├── Rice Leaf Smut/        (500장)
├── Rice Hispa/            (408장)
├── Wheat Leaf Rust/       (500장)
├── Wheat Stripe Rust/     (500장)
├── Wheat Stem Rust/       (401장)
├── Wheat Loose Smut/      (354장)
├── Mango Anthracnose/     (500장)
├── Mango Bacterial Canker/(500장)
├── Mango Die Back/        (500장)
├── Mango Powdery Mildew/  (500장)
├── Mango Sooty Mould/     (500장)
├── Mango Gall Midge/      (500장)
├── Cassava Bacterial Blight/(306장)
├── Cassava Mosaic Disease/  (240장)
├── Cassava Brown Streak Disease/(372장)
├── Cassava Green Mottle/  (285장)
├── Banana Black Sigatoka/ (488장)
├── Banana Yellow Sigatoka/(496장)
├── Banana Panama Disease/ (100장)
├── Banana Moko Disease/   (55장)
├── Banana Bract Mosaic Virus/(50장)
├── Citrus Greening/       (204장)
├── Citrus Canker/         (500장)
├── Citrus Black Spot/     (500장)
└── Citrus Melanose/       (13장) ⚠️
```

**총 정리 이미지: 12,096장 | 31종 중 30종 학습 준비 완료**

---

## 원본 데이터 캐시 위치

```
/root/.cache/kagglehub/datasets/
├── badasstechie/coffee-leaf-diseases/versions/3/
├── vbookshelf/rice-leaf-diseases/versions/1/
├── jay7080dev/rice-plant-diseases-dataset/versions/1/
├── thegoanpanda/rice-crop-diseases/versions/1/
├── asheniranga/leaf-disease-dataset-combination/versions/1/  (761MB)
├── aryashah2k/mango-leaf-disease-dataset/versions/1/
├── shuvokumarbasak4004/the-mango-leaf-disease-classification-dataset/versions/1/
├── ahmadzargar/mango-leaf-diseases-dataset/versions/1/
├── shifatearman/bananalsd/versions/1/
├── sujaykapadnis/banana-disease-recognition-dataset/versions/1/
├── myprojectdictionary/citrus-leaf-disease-image/versions/1/
├── jonathansilva2020/dataset-for-classification-of-citrus-diseases/versions/2/
├── sabaunnisa/wheat-rust-disease/versions/1/
└── musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd/versions/1/
```

_생성: 2026-04-04 | Researcher Agent_
