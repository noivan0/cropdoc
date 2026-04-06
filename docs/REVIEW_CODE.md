# 🏆 CropDoc MVP 코드 심사 리뷰 (REVIEW_CODE.md)

> **Hackathon Reviewer — 전방위 코드 심사 리포트**  
> 심사 기준: REVIEW_FRAMEWORK.md (100점 만점 루브릭)  
> 심사일: 2026-04-03 | 대상: Gemma 4 Good Hackathon

---

## 총점 요약

| 심사 항목 | 배점 | 획득 | 비율 |
|-----------|------|------|------|
| 1. 보안/안전성 | 10 | **8** | 80% |
| 2. 재현 가능성 | 10 | **7** | 70% |
| 3. Gemma 4 차별화 활용도 | 15 | **11** | 73% |
| 4. 데모 품질 (Gradio) | 10 | **8** | 80% |
| 5. 노트북 품질 | 10 | **8.5** | 85% |
| 6. README 품질 | 10 | **9** | 90% |
| **총계** | **65** | **51.5** | **79%** |

> ⚠️ 본 리뷰는 코드 품질 + 재현성 + Gemma 4 활용도 중심으로 평가합니다.  
> 임팩트/스토리텔링/배포 계획 등은 REVIEW_FRAMEWORK.md 기준 별도 심사 대상입니다.

---

## 📊 합격/불합격 판정

```
✅ 합격 (PASS) — 제출 가능 수준

현재 상태: MVP 완성도 높음. Critical 이슈 없음.
단, 2개 항목 보완 시 수상 가능성 대폭 상승.
```

---

## 항목별 상세 평가

---

### 1. 보안/안전성 — 8/10

#### ✅ 잘 된 점

- **API 키 하드코딩 없음**: `os.environ.get("GOOGLE_API_KEY", "")` 패턴 일관 사용
- **Kaggle Secrets 지원**: 노트북에 `UserSecretsClient` fallback 구현
  ```python
  try:
      from kaggle_secrets import UserSecretsClient
      secrets = UserSecretsClient()
      GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY")
  except Exception:
      GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_KEY_HERE")
  ```
- **에러 핸들링**: 모든 API 호출에 `try/except` + Retry 로직 (3회, exponential backoff)
- **_error_response() 패턴**: 에러 시 다국어 친화적 fallback 메시지 반환
- **오디오 파일 안전 처리**: MIME 타입 검증, OSError 핸들링

#### ⚠️ 개선 필요

- **노트북의 "YOUR_KEY_HERE" 플레이스홀더**:
  ```python
  GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_KEY_HERE")
  ```
  → 심사위원이 플레이스홀더 문자열을 실수로 API 키로 전달할 수 있음.
  `""` (빈 문자열)로 변경하고 이후 `if not GOOGLE_API_KEY:` 체크 추가 권장.

- **app.py의 임시 파일 처리**: `/tmp/cropdoc_upload.jpg`에 고정 경로 사용. 동시 요청 시 충돌 가능.
  ```python
  tmp_path = "/tmp/cropdoc_upload.jpg"  # 단일 고정 경로
  ```
  → `tempfile.NamedTemporaryFile()` 사용 권장.

---

### 2. 재현 가능성 — 7/10

#### ✅ 잘 된 점

- **requirements.txt 존재**: 핵심 패키지 모두 포함
- **더미 이미지 fallback**: `create_dummy_sample()` 함수로 인터넷 없이도 UI 기동 가능
- **Kaggle 메타데이터 포함**: `isInternetEnabled: true`, `accelerator: none` 정확히 설정
- **샘플 이미지 자동 다운로드**: Wikimedia 공개 이미지 URL 사용 (CC 라이선스)
- **오프라인 fallback**: 이미지 다운로드 실패 시 합성 초록 이미지로 대체

#### ⚠️ 개선 필요

**requirements.txt 버전 고정 없음 (Critical에 가까운 Suggested)**:
```
google-generativeai>=0.8.0
gradio>=4.0.0
```
→ `>=` 사용으로 미래 breaking change 취약. 예:
```
google-generativeai==0.8.3
gradio==4.44.0
Pillow==10.4.0
requests==2.32.3
numpy==1.26.4
pandas==2.2.2
datasets==2.21.0
soundfile==0.12.1
```

**`soundfile` 의존성 설치 불안정**:
- `soundfile`은 `libsndfile` OS 라이브러리 필요. Kaggle에서는 기본 설치되어 있으나 일부 환경에서 실패할 수 있음.
- 노트북 설치 셀에 `!apt-get install -y libsndfile1 2>/dev/null` 추가 권장.

**`datasets` 패키지의 beans 데이터셋 불일치**:
```python
ds = load_dataset("AI-Lab-Makerere/beans", split=split, trust_remote_code=True)
```
→ README에서는 "PlantVillage"를 언급하지만 실제 코드는 "beans" 데이터셋 로딩. 이름 불일치로 심사위원 혼란 가능. (PlantVillage 전체 데이터셋은 별도 로딩 필요)

---

### 3. Gemma 4 차별화 활용도 — 11/15

#### ✅ 잘 된 점

**이미지 입력**: ✅ 완전 구현
```python
image = Image.open(image_path)
response = self.model.generate_content([system_prompt, image])
```
→ PIL Image 직접 API에 전달. 올바른 구현.

**오디오 입력**: ✅ 구조적으로 구현됨
```python
contents = [
    prompt,
    image,
    {"mime_type": mime, "data": audio_bytes}
]
```
→ 멀티모달 콘텐츠 구성 구현. 단, 실제 API 검증 흔적 없음 (실행 결과 없음).

**다국어 지원**: ✅ 우수 구현
- 20개 언어 코드 정의 (`SUPPORTED_LANGUAGES`)
- 영어/스와힐리/힌디/벵갈리 4개 언어 Native 시스템 프롬프트
- 나머지 언어는 영어 템플릿 치환 방식 (기능적)
- `_extract_severity()`에 스와힐리/힌디/벵갈리 키워드 하드코딩으로 다국어 파싱 지원

**E4B 온디바이스 언급**: ✅ 문서/UI에 언급됨
- README: "Offline-Capable — runs on low-end Android devices"
- About 탭: "runs on low-end Android devices"
- 하지만 실제 온디바이스 배포 코드나 LoRA fine-tuning은 없음

#### ⚠️ 차감 요인

**E4B 온디바이스 실제 구현 없음** (-2점):
- "gemma-4-e4b-it"는 API 호출 방식. 진정한 온디바이스 구현(Android APK, TFLite 변환 등)은 없음.
- REVIEW_FRAMEWORK 기준 "온디바이스 배포"는 구현됐다기보다 "언급" 수준.

**오디오 파이프라인 실제 검증 없음** (-1점):
- 노트북에서 silent WAV를 생성해 테스트하는 방식은 오디오 입력의 실질적 효용을 증명하지 못함.
- "The E4B-exclusive feature"라고 강조하지만 실제 음성으로 테스트된 결과가 없음.

**Function Calling / 구조화 출력 미활용** (-1점):
- 모델 응답이 자유형식 마크다운 → 정규식/휴리스틱 파싱에 의존.
- Gemma 4의 구조화 출력 기능을 활용하면 파싱 신뢰도 대폭 향상 가능.

---

### 4. 데모 품질 (Gradio UI) — 8/10

#### ✅ 잘 된 점

- **3탭 구조**: Image Diagnosis / Image+Voice / About — 직관적
- **Soft 테마 + 그린 컬러**: 농업 주제에 적합
- **심각도 색상 배지 구현**: CRITICAL(빨강) / MODERATE(주황) / MONITOR(초록) HTML 뱃지
  ```python
  SEVERITY_COLORS = {
      "CRITICAL": ("#FF4B4B", "🔴 CRITICAL"),
      "MODERATE": ("#FF9F0A", "🟠 MODERATE"),
      "MONITOR":  ("#34C759", "🟢 MONITOR"),
  }
  ```
- **언어 드롭다운**: 10개 언어 선택 (탭 간 공유)
- **샘플 이미지 원클릭 로딩**: `gr.Examples` 사용
- **API 키 없을 때 경고**: 앱 시작 시 경고 출력 (but 앱 크래시하지 않음)
- **E4B 전용 탭 강조**: "E4B Exclusive" 표시

#### ⚠️ 개선 필요

**API 키 없을 때 UI 동작 미흡** (-1점):
- 진단 버튼 클릭 시 `EnvironmentError`를 Gradio 내에서 처리하긴 하나, 사용자 경험 어색.
- Demo 모드 (mock response)를 추가하면 API 키 없는 심사위원도 UI 흐름 확인 가능.

**샘플 이미지 경로 의존성** (-1점):
- `get_sample_paths()`가 실행 환경에 따라 실패 가능. 다운로드 실패 → 더미 이미지 fallback은 있으나, 더미 이미지(단색 초록)는 시각적으로 uninspiring.
- 적어도 2-3개 샘플 이미지를 `samples/` 디렉토리에 repo에 포함시켜야 함.

---

### 5. 노트북 품질 — 8.5/10

#### ✅ 잘 된 점

**논리적 8섹션 구성**:
1. 인트로 + 문제 정의 → 스토리텔링 훅 명확
2. 설치 & 환경 설정 → pip install + API 키 설정
3. Gemma 4 API 통합 (CropDoctorModel 클래스)
4. 이미지 진단 데모 (샘플 다운로드 + 시각화)
5. 오디오+이미지 통합 진단
6. 다국어 테스트 (EN/SW/HI)
7. 정확도 평가 + 시각화 (bar chart + pie chart)
8. 임팩트 분석 + 로드맵 + 결과 저장

**시각화 포함** ✅:
- `matplotlib` 차트 (정확도 막대 + 심각도 파이차트)
- `plt.savefig("cropdoc_accuracy.png")` 저장
- 샘플 이미지 갤러리 출력

**재현 가능한 fallback** ✅:
- 이미지 다운로드 실패 시 합성 이미지 생성
- 오디오 테스트에 silent WAV 생성 함수 포함

**심사위원 친화적 안내** ✅:
- 각 섹션 마크다운 설명 충실
- 예외 상황 `print()` 출력으로 진행 상황 추적 가능

#### ⚠️ 개선 필요

**datasets/PlantVillage 불일치** (-0.5점):
- 노트북은 `beans` 데이터셋 3개 이미지만 사용하면서 "PlantVillage 54,306 images" 주장.
- 실제 PlantVillage 데이터셋을 최소 일부라도 로딩하거나, 또는 사용 데이터셋 이름을 정확히 기재.

**실행 결과(Output) 없음** (-1점):
- 모든 셀이 `execution_count: null`. 심사위원이 직접 실행해야 하는 부담.
- 최소한 샘플 실행 결과(진단 텍스트, 차트 이미지)를 pre-rendered로 포함해야 함.

---

### 6. README 품질 — 9/10

#### ✅ 잘 된 점

- **5분 이내 이해 가능**: 뱃지 → 문제 → 솔루션 → 퀵스타트 구조
- **임팩트 스토리**: Tanzania 농부 사례로 감정적 Hook
- **ASCII 아키텍처 다이어그램**: 파이프라인 흐름 시각화
- **다국어 테이블**: 언어별 사용자 도달 범위 수치화
- **임팩트 메트릭 테이블**: $220B, 500M farmers, 140 languages
- **배포 가이드**: HuggingFace Spaces 5단계 가이드 포함
- **기여 가이드**: 언어 추가, 병해 확장, Android 래퍼 등 구체적
- **라이선스 명시**: Apache 2.0 (코드), CC0 (데이터셋)

#### ⚠️ 개선 필요

**HuggingFace Spaces 배포 링크 미완성** (-0.5점):
```markdown
[![Open in HuggingFace Spaces](https://...)](https://huggingface.co/spaces)
```
→ 실제 Space URL이 없음. `https://huggingface.co/spaces` 는 플레이스홀더.

**GitHub URL 플레이스홀더** (-0.5점):
```bash
git clone https://github.com/your-org/cropdoc
```
→ 실제 GitHub 저장소 주소로 교체 필요.

---

## 🚨 Critical 이슈 목록 (즉시 수정 필요)

> 제출 전 반드시 수정해야 하는 항목. 미수정 시 심사에서 즉각적인 감점 발생.

### Critical-1: HuggingFace Spaces 실제 URL 없음
```markdown
# 현재
[![Open in HuggingFace Spaces](https://...)](https://huggingface.co/spaces)

# 수정 필요
[![Open in HuggingFace Spaces](https://...)](https://huggingface.co/spaces/your-team/cropdoc)
```
**이유**: REVIEW_FRAMEWORK 루브릭 "지금 바로 테스트 가능한 공개 데모 URL" 체크항목. 없으면 데모 완성도 -4~6점.

### Critical-2: 샘플 이미지가 repo에 포함되지 않음
```
gemma4good/
├── samples/    ← 비어 있음 (또는 미존재)
```
**이유**: 심사위원이 노트북 실행 전 샘플 다운로드에 실패하면 테스트 불가. `samples/` 디렉토리에 최소 3개 이미지를 commit으로 포함시켜야 함.

### Critical-3: 노트북 실행 결과(Output) 없음
- 현재 모든 셀 `execution_count: null`, 출력 결과 없음.
- Kaggle 제출 기준: **실행된 노트북**으로 제출해야 함.
- API 키 확보 후 전체 실행 → 결과 포함된 상태로 저장 필수.

---

## 💡 Suggested 개선사항 목록 (있으면 더 좋은 것)

### Suggested-1: requirements.txt 버전 고정 (Medium Priority)
```
# 현재
google-generativeai>=0.8.0

# 권장
google-generativeai==0.8.3
```
재현성 보증을 위해 `pip freeze > requirements.txt` 결과 사용 권장.

### Suggested-2: Demo 모드 (API 키 없이 작동하는 Mock) 추가
```python
DEMO_MODE = not os.environ.get("GOOGLE_API_KEY")

def diagnose_image(image, language_code):
    if DEMO_MODE:
        # Return pre-computed example result
        return mock_severity_html(), MOCK_DIAGNOSIS_EN, "Demo mode (no API key)"
```
API 키 없는 심사위원이 UI 흐름을 경험할 수 있게 됨.

### Suggested-3: 임시 파일 안전 처리
```python
# 현재
tmp_path = "/tmp/cropdoc_upload.jpg"

# 권장
import tempfile
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    tmp_path = f.name
    image.save(tmp_path, format="JPEG")
try:
    result = model.analyze_image(tmp_path, language=language_code)
finally:
    Path(tmp_path).unlink(missing_ok=True)
```

### Suggested-4: 구조화 출력 (JSON Mode) 활용
```python
# 현재: 자유형식 마크다운 → 정규식 파싱
# 권장: response_mime_type="application/json" 또는 response_schema 활용
response = model.generate_content(
    contents,
    generation_config=genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=DiagnosisSchema
    )
)
```
파싱 신뢰도 및 다국어 severity 추출 안정성 대폭 향상.

### Suggested-5: PlantVillage 데이터셋 사용 일관성
- `pipeline.py`의 `load_plantvillage_sample()`이 `AI-Lab-Makerere/beans` 로딩 → "PlantVillage" 아님
- README에서 PlantVillage 54,306 images 언급과 불일치
- 수정: 데이터셋 이름을 "PlantVillage-compatible (beans subset)"으로 정확히 표기, 또는 실제 PlantVillage HF 데이터셋(`plantvillage/plantdoc`) 사용

### Suggested-6: 오디오 테스트에 실제 의미 있는 WAV 포함
- 현재 silent WAV로 E4B 오디오 기능 "테스트"는 심사위원에게 설득력 부족
- `samples/farmer_voice_demo.wav`: "The leaves have yellow spots" 실제 녹음 파일 포함 권장
- 또는 TTS를 이용한 합성 음성 생성 코드 추가

### Suggested-7: E4B 온디바이스 배포 예시 추가
- 현재 "온디바이스 가능" 주장만 있고 실제 코드 없음
- 최소한 `exportfor_mobile.py` 또는 `README_mobile.md` 추가:
  ```
  # GGUF quantization example
  llama.cpp compatible export for Android deployment
  ```

### Suggested-8: 다국어 severity 추출 개선
```python
# 현재: text.upper()에서 "गंभीर" 같은 유니코드 찾기
if "गंभीर" in text:

# 권장: 별도 다국어 severity_keywords 딕셔너리
SEVERITY_KEYWORDS = {
    "CRITICAL": ["CRITICAL", "HATARI", "गंभीर", "জটিল"],
    "MODERATE": ["MODERATE", "WASTANI", "मध्यम", "মাঝারি"],
    ...
}
```

### Suggested-9: 노트북에 아키텍처 다이어그램 이미지 포함
- 현재 ASCII art만 있음
- Mermaid 또는 matplotlib으로 그린 실제 다이어그램 이미지 포함 시 가산점

---

## 🏆 수상 가능성 최종 평가

### 강점 (심사위원 호감 요인)

| 요인 | 평가 |
|------|------|
| 문제 선택 | ⭐⭐⭐⭐⭐ 탁월. 500M 농부, $220B 손실 — SDG 2/13 직결 |
| Gemma 4 E4B 활용 | ⭐⭐⭐⭐ 이미지+오디오 구현, 다국어 네이티브 프롬프트 |
| 코드 구조 | ⭐⭐⭐⭐ 모듈화, docstring, retry 로직 |
| 다국어 지원 | ⭐⭐⭐⭐ 20개 언어, 4개 네이티브 시스템 프롬프트 |
| README | ⭐⭐⭐⭐⭐ 5분 이내 이해 가능, 임팩트 수치화 |
| 노트북 구성 | ⭐⭐⭐⭐ 8섹션, 시각화, fallback 내장 |

### 약점 (수상 방해 요인)

| 요인 | 영향도 |
|------|--------|
| 공개 데모 URL 없음 | 🔴 High — 심사위원이 직접 테스트 불가 |
| 노트북 실행 결과 없음 | 🔴 High — Kaggle 심사 기준 미충족 |
| E4B 온디바이스 실제 구현 없음 | 🟠 Medium — 주장과 구현 괴리 |
| 오디오 실제 효용 미증명 | 🟠 Medium — Silent WAV는 데모 불충분 |
| 샘플 이미지 미포함 | 🟠 Medium — 첫 실행 마찰 |

### 종합 판정

```
📊 현재 예상 점수 (REVIEW_FRAMEWORK 100점 기준):
   임팩트      (30점): ~23점 — 문제 우수, 효과 측정 미흡
   기술 혁신   (30점): ~20점 — E4B 활용 있으나 온디바이스/FT 없음
   실현가능성  (20점): ~12점 — 공개 URL 없음이 치명적
   발표력      (20점): ~16점 — README 우수, 노트북 출력 없음

   예상 총점: ~71/100점
   예상 수상 등급: Honorable Mention ~ Runner-up 구간

✅ Critical 3개 수정 후 예상:
   실현가능성 +5점 (공개 URL + 노트북 실행)
   발표력 +2점 (노트북 결과 포함)
   예상 총점: ~78/100점 → Runner-up 또는 Category Prize 가능
```

### 최종 한 줄 평가

> **"뼈대는 최상급. 마지막 3개 Critical 이슈(공개 URL, 노트북 실행 결과, 샘플 이미지 commit)만 해결하면 수상권 진입 충분히 가능."**

---

## 액션 아이템 (우선순위 순)

```
🔴 [즉시] HuggingFace Spaces 배포 → 공개 URL 확보
🔴 [즉시] API 키 설정 후 노트북 전체 실행 → 결과 포함 저장
🔴 [즉시] samples/ 디렉토리에 이미지 3개 commit
🟠 [오늘] requirements.txt 버전 고정
🟠 [오늘] beans 데이터셋 불일치 수정
🟠 [내일] Demo 모드(mock response) 추가
🟡 [여유] 실제 음성 오디오 샘플 파일 포함
🟡 [여유] JSON 구조화 출력 모드 구현
```

---

*생성: Hackathon Reviewer Subagent | 2026-04-03*  
*기준: REVIEW_FRAMEWORK.md v1.0 심사 루브릭*
