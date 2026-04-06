# PROGRAM.md — CropDoc autoresearch 실험 지시서

> **영감**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
> "에이전트에게 task를 주는 게 아니라, 어떤 연구자로 살아갈지(research identity)를 코딩하라."

---

## 🎯 실험 목표

**`diagnosis_accuracy` 최대화** — 식물 질병 이미지에 대한 정확한 진단 비율을 높여라.

- **metric**: `diagnosis_accuracy` (0.0 ~ 1.0, 높을수록 좋음)
- **측정 방법**: `scripts/eval_harness.py` 실행 → `diagnosis_accuracy:` 줄 grep
- **시간 예산**: 실험 1회당 최대 **3분** (A6000 GPU 기준)
- **목표 정확도**: ≥ 0.80 (stretch goal: ≥ 0.90)

---

## 📁 파일 역할

| 파일 | 역할 | 수정 여부 |
|------|------|-----------|
| `scripts/cropdoc_infer.py` | 추론 모듈 (실험 대상) | ✅ **수정 가능** |
| `scripts/eval_harness.py` | 평가 하네스 (고정) | ❌ **수정 금지** |
| `autoresearch/PROGRAM.md` | 이 파일 (실험 지시서) | 참고용 |
| `autoresearch/results.tsv` | 실험 결과 기록 | 실험마다 추가 |

---

## 🔄 자율 실험 루프

```
LOOP:
  1. 현재 diagnosis_accuracy 기록 (baseline)
  2. cropdoc_infer.py를 수정 (아이디어 1가지씩 적용)
  3. git commit -m "experiment: <description>"
  4. python scripts/eval_harness.py --max-images 30 → accuracy 측정
     (빠른 테스트: --max-images 10 으로 방향 확인 후 전체 실행)
  5. 개선됐으면 → KEEP (커밋 유지, results.tsv에 기록)
     나빠졌거나 같으면 → REVERT (git checkout scripts/cropdoc_infer.py)
  반복
```

### 명령어 예시

```bash
# 빠른 smoke test (5장)
cd /root/.openclaw/workspace/companies/Hackathon/projects/gemma4good
python scripts/eval_harness.py --max-images 5 2>&1 | grep -E "diagnosis_accuracy|correct|total"

# 중간 테스트 (30장)
python scripts/eval_harness.py --max-images 30 2>&1 | grep "diagnosis_accuracy:"

# 전체 테스트 (300장)
python scripts/eval_harness.py 2>&1 | grep -E "diagnosis_accuracy:|total_images:|correct:|inference_time"

# 결과 기록
echo -e "$(git rev-parse --short HEAD)\t0.734\tkeep\tPrompt engineering: disease names in system prompt" >> autoresearch/results.tsv
```

---

## 💡 실험 아이디어 10가지

### Idea 1: 프롬프트 엔지니어링 (구체적 system prompt)

현재 system prompt에 진단 가능한 질병 목록을 명시적으로 추가.

```python
SYSTEM_PROMPT = """You are CropDoc...
Known diseases to diagnose:
- Tomato: Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, or Healthy
- Potato: Early Blight, Late Blight, or Healthy
- Pepper: Bacterial Spot, or Healthy
Output format: "<Plant> <Condition>" (e.g., "Tomato Early Blight")
"""
```

**예상 효과**: +5~15% accuracy (모델이 정해진 레이블로 응답하도록 유도)

---

### Idea 2: 이미지 전처리 (크기 조정 + 대비 강화)

PIL/OpenCV로 이미지를 전처리 후 모델에 전달.

```python
from PIL import ImageEnhance
image = Image.open(path).convert("RGB").resize((448, 448))
image = ImageEnhance.Contrast(image).enhance(1.5)
image = ImageEnhance.Sharpness(image).enhance(1.3)
```

**예상 효과**: +2~8% (병반 특징이 더 뚜렷해짐)

---

### Idea 3: 다단계 진단 (2-pass)

1차: "이 식물이 건강한가 아니면 병에 걸렸는가?"
2차: (병인 경우) "어떤 병인가?"

```python
# Pass 1: Health check
is_healthy = "healthy" in diagnose_pass1(image)
if is_healthy:
    return "Healthy {plant}"
# Pass 2: Disease identification
return diagnose_disease(image)
```

**예상 효과**: +5~10% (Healthy vs Disease 경계 오류 감소)

---

### Idea 4: Few-shot 예시 추가

프롬프트에 진단 예시를 포함.

```python
FEW_SHOT = """
Examples:
- Dark brown circular spots with concentric rings on tomato → Early Blight
- Water-soaked brown lesions with white mold on edges → Late Blight
- Small dark spots with yellow halo on leaves → Bacterial Spot
"""
```

**예상 효과**: +5~12% (모델이 시각적 패턴과 레이블을 연결)

---

### Idea 5: Temperature / top_p 조정

낮은 temperature로 더 결정적인 출력 생성.

```python
TEMPERATURE = 0.1  # 기본값 0.3에서 낮춤
DO_SAMPLE = False  # Greedy decoding
```

또는 높은 temperature로 다양성 증가 후 majority voting.

**예상 효과**: ±3~7% (질병에 따라 방향이 다를 수 있음)

---

### Idea 6: 이미지 crop 앙상블 (패치 평균)

원본 이미지를 여러 crop으로 분할해서 각각 추론 후 투표.

```python
crops = [
    image.crop((0, 0, W//2, H//2)),    # top-left
    image.crop((W//2, 0, W, H//2)),    # top-right
    image.crop((0, H//2, W//2, H)),    # bottom-left
    image.crop((W//2, H//2, W, H)),    # bottom-right
    image,                              # full image
]
results = [diagnose_single(c) for c in crops]
# majority vote
```

**예상 효과**: +3~8% (병반이 한쪽에 집중된 경우 개선)

---

### Idea 7: 배경 제거 (GrabCut)

OpenCV GrabCut으로 배경을 제거하고 식물 부위만 모델에 전달.

```python
import cv2, numpy as np
mask = np.zeros(img.shape[:2], np.uint8)
bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
result = img * mask2[:,:,np.newaxis]
```

**예상 효과**: +2~5% (배경 노이즈 제거)

---

### Idea 8: Chain-of-Thought 추론

모델이 단계별로 사고하도록 프롬프트 수정.

```python
COT_PROMPT = """Analyze this plant image step by step:
1. What plant species is visible? (tomato, potato, pepper?)
2. What visual symptoms do you observe? (spots, color, pattern, texture?)
3. Based on the symptoms, what is your diagnosis?

Final answer format: "<Plant> <Condition>"
"""
```

**예상 효과**: +5~15% (추론 과정이 명확해져서 정확도 향상)

---

### Idea 9: 신뢰도 점수 + 재시도

모델 출력에서 확신도를 파악하고, 불확실하면 다른 프롬프트로 재시도.

```python
def diagnose_with_retry(image_path, max_retries=2):
    result = diagnose_single(image_path)
    # 불확실 키워드 감지
    uncertain_keywords = ["unsure", "unclear", "cannot determine", "hard to tell", "might be"]
    if any(kw in result.lower() for kw in uncertain_keywords):
        result = diagnose_single(image_path, prompt=DETAILED_PROMPT)
    return result
```

**예상 효과**: +2~5% (낮은 신뢰도 케이스 개선)

---

### Idea 10: 이미지 패치 4분할 투표

이미지를 4등분해서 각 패치에 대해 독립적으로 추론 후 다수결.

```python
from collections import Counter

def diagnose_patch_vote(image_path):
    img = Image.open(image_path)
    W, H = img.size
    patches = [
        img.crop((0, 0, W//2, H//2)),
        img.crop((W//2, 0, W, H//2)),
        img.crop((0, H//2, W//2, H)),
        img.crop((W//2, H//2, W, H)),
    ]
    votes = [extract_label(diagnose_patch(p)) for p in patches]
    winner = Counter(votes).most_common(1)[0][0]
    return winner
```

**예상 효과**: +3~7% (지역적 병변 패턴에 강건)

---

## 📊 results.tsv 형식

파일: `autoresearch/results.tsv`

```
commit	accuracy	status	description
baseline	0.000	baseline	Initial setup, no run yet
a1b2c3d	0.650	keep	Idea 1: Explicit disease list in system prompt
b2c3d4e	0.620	discard	Idea 5: temperature=0.1, worse than 0.3
c3d4e5f	0.680	keep	Idea 8: Chain-of-thought prompting
d4e5f60	0.000	crash	Idea 6: OOM on patch ensemble, reduce batch
e5f607a	0.710	keep	Idea 3: Two-pass healthy/disease split
```

**컬럼 설명:**
- `commit`: git commit hash (short) 또는 "baseline"
- `accuracy`: `diagnosis_accuracy` 값 (0.0~1.0)
- `status`: `keep` | `discard` | `crash` | `baseline`
- `description`: 실험 설명 (무엇을 바꿨는지, 결과 한 줄 요약)

---

## ✅ keep / revert 판단 기준

| 상황 | 판단 | 행동 |
|------|------|------|
| accuracy ≥ 이전 + 0.01 (1%p 이상 향상) | **KEEP** | 커밋 유지, results.tsv에 `keep` 기록 |
| accuracy < 이전 또는 개선 < 1%p | **DISCARD** | `git checkout scripts/cropdoc_infer.py`, `discard` 기록 |
| accuracy 동일 + 코드 단순화 | **KEEP** | simplicity win으로 유지 |
| 크래시 (OOM, 오류) | **CRASH** | 원인 파악 → 수정 시도 최대 3회 → 포기 시 다음 아이디어 |

---

## 🚨 제약 사항

1. **`scripts/eval_harness.py` 절대 수정 금지** — metric 기준이 변하면 실험 비교 불가
2. **`diagnose(image_path, lang)` 시그니처 유지** — 내부 구현만 바꿀 것
3. **시간 예산 3분 엄수** — 실험이 timeout되면 batch size/max_new_tokens 축소
4. **GPU VRAM 초과 주의** — A6000 48GB 한도, patch ensemble 시 주의

---

## 📌 Quick Start

```bash
# 0. 프로젝트 경로로 이동
cd /root/.openclaw/workspace/companies/Hackathon/projects/gemma4good

# 1. 베이스라인 확인 (5장)
python scripts/eval_harness.py --max-images 5

# 2. 실험: cropdoc_infer.py 수정

# 3. 빠른 검증 (10장)
python scripts/eval_harness.py --max-images 10

# 4. 개선되면 커밋, 아니면 revert
git add scripts/cropdoc_infer.py && git commit -m "experiment: <idea>"
# OR
git checkout scripts/cropdoc_infer.py

# 5. 결과 기록
echo -e "$(git rev-parse --short HEAD)\t0.72\tkeep\tIdea 1: prompt engineering" >> autoresearch/results.tsv
```

---

_이 파일은 CropDoc autoresearch 루프의 핵심 지시서입니다._
_실험 결과에 따라 `what_to_try` 목록을 에이전트 스스로 확장할 수 있습니다._
