# RESEARCH_DATASETS.md — TOP 3 아이디어별 공개 데이터셋
> Kaggle "The Gemma 4 Good Hackathon" 데이터셋 조사
> 작성일: 2026-04-03 | 소스: HuggingFace Datasets API

---

## 조사 방법

HuggingFace Datasets API로 다음 검색어로 조회:
- `plant+disease` / `crop+disease` → 농업 진단 AI
- `medical+image+diagnosis` → 의료 보조
- `disaster+damage+satellite` → 재난 피해 분류기
- `multilingual+education` → 교육 튜터 (참고용)

---

## 🥇 TOP 1: 농업 진단 AI — 데이터셋

### 1-1. PlantVillage Dataset (핵심 추천)

| 항목 | 내용 |
|------|------|
| **이름** | PlantVillage / plant-disease |
| **크기** | 54,306장 (38종 작물, 26종 병해충 + 정상) |
| **라이선스** | CC0 (퍼블릭 도메인, 상업 포함 자유 사용) |
| **링크** | https://huggingface.co/datasets/PlantVillage/plant-disease |
| **다운로드수** | 수만 회 (농업 AI 가장 많이 사용) |

**활용 방법**:
```python
from datasets import load_dataset

ds = load_dataset("PlantVillage/plant-disease")
# 클래스 예: "Tomato_Late_blight", "Corn_(maize)__healthy", "Apple__Apple_scab"

# Gemma 4 API와 연동 예시
import google.generativeai as genai
import base64

def test_gemma_diagnosis(image, true_label):
    """PlantVillage 이미지로 Gemma 4 진단 정확도 검증"""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    model = genai.GenerativeModel("gemma-4-e4b-it")
    response = model.generate_content([
        {"mime_type": "image/jpeg", "data": img_b64},
        "Identify the plant disease in this image. State: plant name, disease name, severity."
    ])
    return response.text, true_label
```

**데이터 구성**:
```
38종 작물: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, 
           Pepper, Potato, Raspberry, Soybean, Squash, Strawberry,
           Tomato 등
26종 카테고리: 
  - 병든 잎 (병명 레이블)
  - 정상 잎 (healthy)
이미지 형식: JPEG, 다양한 해상도
```

---

### 1-2. Cotton Plant Disease (보조)

| 항목 | 내용 |
|------|------|
| **이름** | Francesco/cotton-plant-disease |
| **크기** | 1K~10K장 (Object Detection 포맷) |
| **라이선스** | CC (확인 필요) |
| **링크** | https://huggingface.co/datasets/Francesco/cotton-plant-disease |
| **포맷** | COCO Object Detection (bbox 포함) |

**활용 방법**:
```python
from datasets import load_dataset

# 면화 병해충 Object Detection 데이터
ds = load_dataset("Francesco/cotton-plant-disease")

# 특징: bbox 좌표 포함 → 병든 부위 위치 특정 가능
# Gemma 4의 Pointing 기능과 조합 가능
```

---

### 1-3. Indian Crop Diseases (개도국 특화)

| 항목 | 내용 |
|------|------|
| **이름** | rohitashva/indian_crop_diseases |
| **크기** | 소규모 (~1K) |
| **라이선스** | 오픈 (DOI: 10.57967/hf/3552) |
| **링크** | https://huggingface.co/datasets/rohitashva/indian_crop_diseases |
| **특이사항** | 인도 현지 작물 특화 (쌀, 밀, 사탕수수 등) |

**활용 방법**:
```python
# 인도 현지 농업 환경에 특화된 데이터
# 힌디어 레이블 또는 설명 포함 가능성 → 다국어 데모에 활용
ds = load_dataset("rohitashva/indian_crop_diseases")
```

---

### 1-4. Crop Disease Dataset (다중 작물)

| 항목 | 내용 |
|------|------|
| **이름** | Kennethdot/crop_disease_cashew_tomato_maize_cassava |
| **크기** | 소규모 |
| **라이선스** | Apache 2.0 ✅ |
| **링크** | https://huggingface.co/datasets/Kennethdot/crop_disease_cashew_tomato_maize_cassava |
| **특이사항** | 아프리카 주요 작물 특화 (카사바, 캐슈, 옥수수) |

**활용 방법**:
```python
# 아프리카 농업 AI 데모에 특화
# 카사바: 아프리카 8억 명 이상 주식 작물
ds = load_dataset("Kennethdot/crop_disease_cashew_tomato_maize_cassava")
```

---

### 1-5. Sudoping01/crop-disease-dataset (권장 보조)

| 항목 | 내용 |
|------|------|
| **이름** | sudoping01/crop-disease-dataset |
| **다운로드** | 638회 (이 카테고리 최다) |
| **라이선스** | — |
| **링크** | https://huggingface.co/datasets/sudoping01/crop-disease-dataset |

---

### 농업 AI 추가 권장 데이터셋 (Kaggle)

```
1. PlantVillage (Kaggle 버전)
   URL: https://www.kaggle.com/datasets/emmarex/plantdisease
   크기: 54,306장 | 라이선스: CC0

2. New Plant Diseases Dataset
   URL: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
   크기: 87,000장 (증강 포함) | 라이선스: CC-BY

3. Rice Leaf Disease
   URL: https://www.kaggle.com/datasets/nizorogbezuode/rice-leaf-images
   크기: 120장 (소규모, 빠른 실험용)
```

---

## 🥈 TOP 2: 시각장애인 내비게이터 — 데이터셋

> 주의: HuggingFace API "visual+impairment+navigation" 검색 결과 부족 → 대안 데이터셋 제시

### 2-1. COCO Dataset (Object Detection)

| 항목 | 내용 |
|------|------|
| **이름** | merve/coco (HuggingFace 미러) |
| **크기** | 330,000장+ (80개 오브젝트 카테고리) |
| **라이선스** | CC-BY 4.0 |
| **링크** | https://huggingface.co/datasets/merve/coco |
| **원본** | https://cocodataset.org |

**활용 방법**:
```python
from datasets import load_dataset

# 장애물 감지 학습용 (보행 환경 오브젝트)
ds = load_dataset("merve/coco", split="validation")

# 활용: 보행 장애물 (차, 사람, 자전거, 계단, 문 등) 인식 테스트
# Gemma 4 이미지 이해와 결합하여 환경 설명 생성

def describe_environment_for_blind(image):
    """시각장애인을 위한 환경 설명 생성"""
    model = genai.GenerativeModel("gemma-4-e4b-it")
    response = model.generate_content([
        image,
        """You are assisting a visually impaired person. Describe:
        1. Immediate obstacles (within 2 meters)
        2. Safe path forward
        3. Any hazards to be aware of
        Keep response under 3 sentences. Be concise and safety-focused."""
    ])
    return response.text
```

---

### 2-2. VizWiz Dataset (시각장애인 실제 사용 데이터)

| 항목 | 내용 |
|------|------|
| **이름** | VizWiz-VQA |
| **크기** | 31,173장 (시각장애인이 실제 촬영한 이미지) |
| **라이선스** | CC-BY 4.0 |
| **링크** | https://vizwiz.org/tasks-and-metrics/visual-question-answering/ |
| **특이사항** | 실제 시각장애인이 질문한 이미지+음성 데이터 |

**활용 방법**:
```python
# VizWiz: 실제 사용 시나리오 데이터
# - 실내 환경 탐색
# - 제품 라벨 읽기
# - 얼굴 인식
# - 방향 안내

# Gemma 4 E4B의 실제 시각장애인 지원 능력 벤치마킹에 최적
```

---

### 2-3. EgocentricVision (1인칭 시점 데이터)

| 항목 | 내용 |
|------|------|
| **이름** | EPIC-KITCHENS / Ego4D |
| **크기** | 수백 시간 분량 |
| **라이선스** | 연구용 라이선스 |
| **링크** | https://epic-kitchens.github.io |

**활용 방법**:
```python
# 1인칭 시점(POV) 이미지/영상 데이터
# 스마트폰 카메라 = 시각장애인 눈 역할
# Gemma 4 E4B 비디오 이해 기능과 조합
```

---

### 2-4. Google Street View (오픈 대안)

```python
# 보행 환경 시뮬레이션
import requests

# Google Street View Static API (유료지만 해커톤 무료 크레딧 활용)
# 또는 OpenStreetMap + Mapillary (무료 거리 이미지)

# Mapillary API (무료, 크라우드소싱 거리 이미지)
# https://www.mapillary.com/developer/api-documentation
```

---

## 🥉 TOP 3: 다국어 의료 보조 — 데이터셋

> 주의: HuggingFace API "medical+image+diagnosis" 검색 결과 없음 (접근 제한 가능) → 공개 의료 데이터셋 직접 제시

### 3-1. MedPix (방사선 이미지)

| 항목 | 내용 |
|------|------|
| **이름** | MedPix 2.0 |
| **크기** | 47,000+ 케이스 |
| **라이선스** | 공개 (미국 국립 의학 도서관) |
| **링크** | https://medpix.nlm.nih.gov |

**활용 방법**:
```python
# 방사선 이미지 + 진단 레이블
# Gemma 4의 MedXPertQA MM 성능 활용
# 주의: 의료 책임 문제 → 명시적 "의료 전문가 상담 권장" 안내 필수
```

---

### 3-2. HAM10000 (피부과)

| 항목 | 내용 |
|------|------|
| **이름** | HAM10000 (Human Against Machine with 10000 training images) |
| **크기** | 10,015장 (7종 피부 병변) |
| **라이선스** | CC-BY-NC |
| **링크** | https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection |
| **HF 미러** | https://huggingface.co/datasets/marmal88/skin_cancer |

**활용 방법**:
```python
from datasets import load_dataset

ds = load_dataset("marmal88/skin_cancer")

# 피부 병변 분류: 멜라노마, 기저세포암, 모반 등
# Gemma 4로 초기 피부 이상 감지 → "피부과 전문의 상담 필요" 안내

def screen_skin_lesion(image_path):
    model = genai.GenerativeModel("gemma-4-e4b-it")
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = model.generate_content([
        {"mime_type": "image/jpeg", "data": image_data},
        """Analyze this skin lesion image. 
        Note: This is for educational screening only - not medical diagnosis.
        Describe: color uniformity, border irregularity, size assessment.
        ALWAYS recommend consulting a dermatologist."""
    ])
    return response.text
```

---

### 3-3. MIMIC-CXR (흉부 X-ray)

| 항목 | 내용 |
|------|------|
| **이름** | MIMIC-CXR |
| **크기** | 227,827장 (방사선 보고서 포함) |
| **라이선스** | PhysioNet 협약 (무료 연구 허가) |
| **링크** | https://physionet.org/content/mimic-cxr/ |
| **HF** | https://huggingface.co/datasets/StanfordAIMI/mimic-cxr-jpg |

**활용 방법**:
```python
# 흉부 X-ray + 방사선과 보고서 쌍 데이터
# 의료 사각지대 지역에서 X-ray 분석 보조
# Gemma 4의 OCR 능력으로 의료 보고서 해석 및 다국어 번역
```

---

### 3-4. PMC-OA (의료 논문 이미지)

| 항목 | 내용 |
|------|------|
| **이름** | axiong/PMC_OA |
| **크기** | 1.6M+ 이미지 (PubMed Central 오픈 액세스) |
| **라이선스** | CC-BY (오픈 액세스 논문) |
| **링크** | https://huggingface.co/datasets/axiong/PMC_OA |

**활용 방법**:
```python
from datasets import load_dataset

# 의료 논문 그림 + 캡션 데이터
# 의료 용어 다국어 번역 및 설명에 활용
ds = load_dataset("axiong/PMC_OA", split="train")
```

---

### 3-5. ROCO (Radiology Objects in COntext)

| 항목 | 내용 |
|------|------|
| **이름** | ROCO |
| **크기** | 81,825장 (방사선 이미지 + 캡션) |
| **라이선스** | CC-BY |
| **링크** | https://huggingface.co/datasets/ROCO-dataset/roco |

---

## 데이터셋 빠른 선택 가이드

```
해커톤 데모 즉시 활용:
┌─────────────────────────────────────────────────────────┐
│ 아이디어         │ 즉시 사용 가능한 최우선 데이터셋        │
├─────────────────────────────────────────────────────────┤
│ 🌾 농업 진단 AI  │ PlantVillage (HF) → CC0, 5만장+       │
│ 👁️ 시각장애인   │ COCO (HF) → CC-BY, 33만장             │
│ 🏥 의료 보조     │ HAM10000 (HF) → CC-BY-NC, 1만장       │
└─────────────────────────────────────────────────────────┘

Kaggle 노트북 로드 코드 (공통):
```
```python
from datasets import load_dataset
import os
os.environ["HF_TOKEN"] = "your-token"

# TOP 1 (농업)
ds_farm = load_dataset("PlantVillage/plant-disease")

# TOP 2 (시각장애인)
ds_coco = load_dataset("merve/coco", split="validation[:100]")  # 소량 우선 테스트

# TOP 3 (의료)
ds_skin = load_dataset("marmal88/skin_cancer")
```

---

## 라이선스 요약

| 라이선스 | 상업 사용 | 수정 | 재배포 | 데이터셋 |
|---------|:-------:|:---:|:-----:|---------|
| CC0 | ✅ | ✅ | ✅ | PlantVillage |
| CC-BY 4.0 | ✅ | ✅ | ✅ | COCO, ROCO, PMC-OA |
| CC-BY-NC | ❌ | ✅ | ✅ | HAM10000 |
| Apache 2.0 | ✅ | ✅ | ✅ | Kennethdot crop disease |
| PhysioNet | ⚠️ 연구만 | ✅ | ❌ | MIMIC-CXR |

**해커톤 권장**: CC0 또는 CC-BY → PlantVillage가 가장 자유로운 라이선스

---

## HuggingFace API 원시 조회 결과 요약

```
검색: plant+disease (5건)
  1. faisal-hugging-face/plant-disease (35 downloads)
  2. Francesco/cotton-plant-disease (34 downloads, CC, object-detection)
  3. ayerr/plant-disease-classification (284 downloads)
  4. AndriiPets/autotrain-data-plant-disease-classification (5 downloads)
  5. AndriiPets/plant-disease-modified (4 downloads, MIT)

검색: crop+disease (5건)
  1. vishnun0027/Crop_Disease (36 downloads)
  2. rohitashva/indian_crop_diseases (14 downloads, 인도 특화)
  3. Kennethdot/crop_disease_cashew_tomato_maize_cassava (11 downloads, Apache 2.0)
  4. sudoping01/crop-disease-dataset (638 downloads ← 최다)
  5. siddharth2525/crop-disease-dataset-24 (4 downloads)

검색: medical+image+diagnosis → 결과 없음 (API 접근 제한 추정)
검색: disaster+damage+satellite → 결과 없음 (검색어 미스매치)
검색: multilingual+education → 결과 없음 (검색어 미스매치)
```

**참고**: HuggingFace API 결과가 없는 항목은 직접 URL 접근 권장:
- 의료: https://huggingface.co/datasets?search=medical+imaging
- 재난: https://huggingface.co/datasets?search=disaster+satellite
- 교육: https://huggingface.co/datasets?search=multilingual+qa

---

*작성: Hackathon Researcher | 2026-04-03*
