# Gemma4 활용도 강화 보고서

**작성일**: 2026-04-05  
**담당**: Engineer Agent (gemma4-enhancement subagent)  
**GPU**: NVIDIA RTX A6000 × 2 (48GB VRAM each)

---

## 1. 현재 Gemma4 사용 현황 분석

### 기존 `scripts/cropdoc_infer.py` 방식

```
CNN top-1 신뢰도 ≥ 0.90  →  Gemma4 skip (즉시 반환, 속도 우선)
CNN top-1 신뢰도 < 0.90  →  Gemma4로 최종 검증
CNN top-1 신뢰도 < 0.50  →  집중 프롬프트(GEMMA_SYSTEM_FOCUSED) 사용
```

**문제점**:
- Gemma4의 멀티모달 시각 능력을 전체 이미지의 소수에만 활용
- CNN 고신뢰도 케이스에서 Gemma4의 독립적 검증 기회 없음
- Chain-of-Thought 추론 없이 단순 라벨 출력만 수행

---

## 2. Task 1: Gemma4 직접 이미지 진단 강화

### 생성 파일: `scripts/cropdoc_gemma4_enhanced.py`

**핵심 변경**:
1. **모든 이미지에 Gemma4 1차 분석** (CNN skip 없음)
2. **Chain-of-Thought 프롬프트** 도입:
   ```
   Step 1: Identify plant species
   Step 2: Examine symptoms  
   Step 3: Diagnose disease
   → PLANT / SYMPTOMS / DIAGNOSIS / CONFIDENCE 구조화 출력
   ```
3. **표준 레이블 정규화** (별칭 맵 포함)
4. **클래스별 정확도 리포트** 자동 생성

### Quick Test 결과 (5장)

| 이미지 | 실제 레이블 | Gemma4 예측 | 정확 | 지연 |
|--------|------------|-------------|------|------|
| Healthy Potato | Healthy Potato | **Healthy Tomato** | ❌ | 5553ms |
| Healthy Tomato | Healthy Tomato | Healthy Tomato | ✅ | 5066ms |
| Pepper Bacterial Spot | Pepper Bacterial Spot | **Tomato Early Blight** | ❌ | 5021ms |
| Potato Early Blight | Potato Early Blight | **Tomato Septoria Leaf Spot** | ❌ | 5292ms |
| Potato Late Blight | Potato Late Blight | **Tomato Septoria Leaf Spot** | ❌ | 4951ms |

**정확도**: 1/5 = 20.0% | **평균 지연**: 5,177ms

**분석**: Gemma4 E4B-IT 기본 모델은 토마토 편향이 강함. 감자·고추 이미지를 주로 토마토 질병으로 분류. 파인튜닝 후 개선 예상.

---

## 3. Task 2: Unsloth 설치 결과

```
설치: unsloth 2026.4.2 (pip install unsloth)
문제: Flash Attention 2 미지원 → Xformers 0.0.35 대체 사용
결과: Gemma4는 Gemma4ForConditionalGeneration (멀티모달)
     → Unsloth FastLanguageModel 미지원 (텍스트 전용 아키텍처)
     → PEFT LoRA 방식으로 대체
```

**설치된 패키지**:
- `unsloth-2026.4.2`
- `unsloth_zoo-2026.4.2`
- `peft-0.18.1`
- `trl-0.24.0`
- `xformers-0.0.35`

---

## 4. Task 3: 파인튜닝 데이터셋

### 생성 파일: `data/gemma4_finetune_dataset.json`

**구성**:
- **10개 식물 질병 클래스** (PlantVillage 기준)
- **40개 Q&A 항목** (클래스당 4개: EN 직접진단 + EN 증상기반 + KO 처방 + KO 직접)
- **형식**: Alpaca (instruction / input / output)
- **언어**: 한국어 20개 + 영어 20개

**포함 질병**:
| 번호 | 레이블 | 원인균 |
|------|--------|--------|
| 1 | Tomato Late Blight | Phytophthora infestans |
| 2 | Tomato Early Blight | Alternaria solani |
| 3 | Tomato Bacterial Spot | Xanthomonas vesicatoria |
| 4 | Tomato Leaf Mold | Passalora fulva |
| 5 | Tomato Septoria Leaf Spot | Septoria lycopersici |
| 6 | Healthy Tomato | — |
| 7 | Potato Late Blight | Phytophthora infestans |
| 8 | Potato Early Blight | Alternaria solani |
| 9 | Healthy Potato | — |
| 10 | Pepper Bacterial Spot | Xanthomonas euvesicatoria |

---

## 5. Task 4: Gemma4 파인튜닝 실행 결과

### 방식: PEFT LoRA + BitsAndBytes 4-bit NF4

**Unsloth 대신 PEFT를 사용한 이유**:
- Gemma4는 `Gemma4ForConditionalGeneration` (비전+언어 멀티모달)
- Unsloth `FastLanguageModel`은 텍스트 전용 모델만 지원
- PEFT `get_peft_model`은 멀티모달 모델에도 적용 가능 (비전 타워 제외)

**핵심 설정**:
```python
# 비전 타워 제외 정규식
exclude_modules = r".*\.(vision_tower|audio_tower|embed_vision|embed_audio|lm_head)\..*"

LoraConfig:
  r = 16
  lora_alpha = 16
  target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  (언어 모델 레이어만, 비전 인코더 제외)

TrainingArguments:
  max_steps = 60
  learning_rate = 2e-4
  per_device_train_batch_size = 2 (effective batch = 8)
  optim = paged_adamw_8bit
  bf16 = True (A6000 지원)
  CUDA_VISIBLE_DEVICES = 0 (단일 GPU, DataParallel 충돌 방지)
```

**학습 결과**:

| Step | Loss | Epoch |
|------|------|-------|
| 10 | 44.14 | 2 |
| 20 | 17.79 | 4 |
| 30 | 9.657 | 6 |
| 40 | 5.138 | 8 |
| 50 | 2.922 | 10 |
| 60 | 2.137 | 12 |

- **총 학습 시간**: 5.6분 (335초)
- **최종 Loss**: 2.137 (시작 44.14 대비 95% 감소)
- **학습 파라미터**: 36.7M / 5.79B (0.634%)
- **LoRA 어댑터 크기**: 141MB (`adapter_model.safetensors`)

**저장 위치**: `data/models/gemma4_finetuned/`

---

## 6. Task 5: 파이프라인 강화 포인트

### 현재 문제점 및 개선 방향

| 현황 | 문제 | 개선안 |
|------|------|--------|
| CNN ≥ 0.90 → Gemma4 skip | Gemma4 시각 능력 미활용 | Gemma4 1차 분석 → CNN 검증 |
| 단순 라벨 출력 | 추론 과정 불투명 | Chain-of-Thought 추론 |
| 기본 모델 사용 | 토마토 편향, 일반 지식 부족 | 파인튜닝 모델 사용 |
| 불확실 케이스만 Gemma4 | 신뢰도 의존적 | 전수 Gemma4 적용 |

### 추천 파이프라인 (차기 버전)

```
이미지 입력
    ↓
[Stage 1] Gemma4 멀티모달 직접 분석 (CoT)
    → PLANT, SYMPTOMS, DIAGNOSIS, CONFIDENCE 추출
    ↓
[Stage 2] CNN 앙상블 검증
    → top-3 예측 + 신뢰도
    ↓
[Stage 3] 앙상블 결합
    → Gemma4 high conf + CNN top-1 일치 → 즉시 반환
    → 불일치 → Gemma4 focused prompt 재분석
    ↓
최종 진단 + 한/영 처방 정보
```

---

## 7. 생성된 파일 목록

| 파일 | 설명 |
|------|------|
| `scripts/cropdoc_gemma4_enhanced.py` | Gemma4 직접 이미지 진단 강화 버전 |
| `scripts/prepare_finetune_dataset.py` | 파인튜닝 데이터셋 생성 스크립트 |
| `scripts/finetune_gemma4_peft.py` | PEFT LoRA 파인튜닝 실행 스크립트 |
| `scripts/finetune_gemma4_unsloth.py` | Unsloth 파인튜닝 시도 스크립트 (참고용) |
| `data/gemma4_finetune_dataset.json` | 40개 항목 Alpaca 형식 데이터셋 |
| `data/gemma4_enhanced_results.json` | Quick test 결과 (5장) |
| `data/models/gemma4_finetuned/` | LoRA 어댑터 가중치 (141MB) |
| `docs/GEMMA4_USAGE_REPORT.md` | 이 보고서 |

---

## 8. 다음 단계 (Phase 2)

1. **파인튜닝 모델 로드 후 eval_harness 평가**
   - 기본 모델 vs 파인튜닝 모델 정확도 비교
   
2. **멀티모달 파인튜닝 (이미지+텍스트)**
   - 현재 텍스트 전용 → 이미지를 함께 학습
   - `Gemma4ForConditionalGeneration` 이미지 입력 활용

3. **더 많은 학습 데이터**
   - 현재 40개 → PlantVillage 이미지 + 캡션 기반 1,000+ 데이터
   
4. **Gemma4 강화 파이프라인 실전 평가**
   - `cropdoc_gemma4_enhanced.py --full` 실행 (300장 전체)
   - eval_harness와 비교

---

## 요약

✅ **Unsloth 설치**: 완료 (2026.4.2)  
✅ **Gemma4 직접 진단 스크립트**: 완료 (cropdoc_gemma4_enhanced.py)  
✅ **파인튜닝 데이터셋**: 완료 (40개 항목, 10종 한/영)  
✅ **파인튜닝 실행**: 완료 (PEFT LoRA, 5.6분, loss 44→2.1)  
✅ **LoRA 어댑터 저장**: 완료 (141MB, data/models/gemma4_finetuned/)  
⚠️ **Quick Test 정확도**: 20% (기본 모델 토마토 편향 — 파인튜닝으로 개선 예정)  
