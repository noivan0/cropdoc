# 실제 대회 규정 vs 현재 프로젝트 갭 분석
_scrapling + Kaggle API 직접 수집 — 2026-04-03_

---

## 📋 대회 기본 정보 (실제 확인)

| 항목 | 내용 |
|------|------|
| 주최 | Google LLC (Google DeepMind) |
| 상금 총액 | **$200,000** |
| 마감 | **2026-05-18 23:59 UTC** (한국시간 2026-05-19 08:59) — 약 45일 남음 |
| 팀 최대 인원 | **5명** |
| 제출 횟수 | **팀당 1회** (Writeup 1개, 재제출 가능) |
| 공식 데이터셋 | **없음** (NOTE.md 56B 파일 하나, 외부 데이터 사용 가능) |
| Winner 라이선스 | **CC-BY 4.0** |

---

## 💰 상금 구조 (실제 확인)

### Main Track ($100,000) — 전체 최우수
| 순위 | 상금 |
|------|------|
| 1st | $50,000 |
| 2nd | $25,000 |
| 3rd | $15,000 |
| 4th | $10,000 |

### Impact Track ($50,000) — 분야별
| 트랙 | 상금 |
|------|------|
| Health Sciences | $10,000 |
| Global Resilience | $10,000 |
| Future of Education | $10,000 |
| Digital Equity & Inclusivity | $10,000 |
| Safety & Trust | $10,000 |

### Special Technology Track ($50,000) — 기술별 ⭐ CropDoc 관련
| 상 | 내용 | 상금 |
|-----|------|------|
| **Cactus** | 모델 간 라우팅하는 로컬-퍼스트 모바일/웨어러블 앱 | **$10,000** |
| **LiteRT** | Google AI Edge LiteRT 구현 | $10,000 |
| **llama.cpp** | 리소스 제약 하드웨어 구현 | $10,000 |
| **Ollama** | Ollama로 Gemma 4 로컬 실행 | $10,000 |
| **Unsloth** | Unsloth로 파인튜닝한 Gemma 4 | $10,000 |

> **CropDoc은 Main Track + Impact Track(Global Resilience/Digital Equity) + Cactus(온디바이스) 동시 지원 가능!**

---

## 🏆 평가 기준 (실제 확인 — 현재 코드와 갭)

| 평가 항목 | 배점 | 설명 | 현재 상태 |
|---------|------|------|----------|
| **Impact Vision** | 40점 | 실제 사회 문제 해결, 임팩트 명확성 | ✅ EVIDENCE.md 근거 확보 |
| **Video Pitch & Storytelling** | 30점 | 비디오 품질, 스토리텔링 | ❌ **비디오 없음** |
| **Technical Implementation** | ? | Gemma 4 실제 구현, 작동 여부 | ⚠️ API 방식 수정 중 |
| (나머지 배점) | ? | 스크래핑 한계로 일부 미수집 | - |

> **핵심: 심사는 비디오가 주(primary)**. 코드/Writeup은 검증 용도.

---

## 📥 제출 필수 요건 (실제 확인)

```
✅ 필수 4가지:
1. Kaggle Writeup (1,500단어 이하, Track 선택 필수)
2. 공개 YouTube 비디오 (3분 이하, 로그인 불필요)
3. 공개 코드 저장소 (GitHub 또는 Kaggle Notebook)
4. 라이브 데모 (URL, 로그인 불필요)
```

---

## 🔴 현재 프로젝트 갭 (규정 기준)

### Critical — 없으면 실격
| # | 필요한 것 | 현재 상태 | 조치 필요 |
|---|----------|----------|---------|
| 1 | **YouTube 비디오 (3분 이하)** | ❌ 없음 | 제작 필요 |
| 2 | **Kaggle Writeup** | ❌ 없음 | 작성 필요 (1,500자) |
| 3 | **공개 GitHub 저장소** | ❌ 없음 (로컬만) | 업로드 필요 |
| 4 | **라이브 데모 URL** | ❌ 없음 | HF Spaces 배포 필요 |

### High — 점수에 직접 영향
| # | 필요한 것 | 현재 상태 | 조치 필요 |
|---|----------|----------|---------|
| 5 | **Track 선택** | ❌ 미선택 | Global Resilience 또는 Digital Equity 선택 |
| 6 | **실제 Gemma 4 동작 증명** | ⚠️ API 방식 수정 중 | HF Transformers 방식 검증 |
| 7 | **Winner License CC-BY 4.0** | ❌ 현재 Apache 2.0 | CC-BY 4.0으로 변경 |

### Medium — 경쟁력 영향
| # | 필요한 것 | 현재 상태 | 조치 필요 |
|---|----------|----------|---------|
| 8 | **Cactus Prize 도전** | ❌ 미구현 | Ollama/LiteRT 연동 추가 |
| 9 | **Unsloth 파인튜닝** | ❌ 미구현 | PlantVillage 데이터로 파인튜닝 시도 |
| 10 | **Video storyboard/스크립트** | ❌ 없음 | STRATEGY.md 기반 스크립트 작성 |

---

## ✅ 규정 준수 확인된 항목

| 항목 | 확인 내용 |
|------|---------|
| 외부 데이터 사용 | ✅ 허용 (공개·무료 데이터) — PlantVillage CC0 적합 |
| 외부 모델 사용 | ✅ 허용 (공개·무료 조건 충족) |
| 팀 구성 | ✅ 5명 이하 |
| 제출 횟수 | ✅ 1회 (재제출 가능) |
| 코드 공개 | ✅ 공개 저장소 필요 (GitHub 업로드 필요) |
| 데모 공개 | ✅ 로그인 불필요 조건 (HF Spaces 적합) |

---

## 🎯 Track 선택 권고

CropDoc 최적 Track:
1. **Impact Track: Global Resilience** — 오프라인 농업 진단, 식량 안보
2. **Impact Track: Digital Equity & Inclusivity** — 다국어, 인터넷 없는 지역
3. **Special: Cactus Prize** — 로컬-퍼스트 온디바이스 앱 (E4B)

→ **Writeup에서 Global Resilience 선택 + Cactus Prize 병행 도전 권장**

---

## 📅 즉시 실행 액션 아이템

1. **[운영자 필요]** Kaggle 가입 + 대회 참가 등록
2. **[운영자 필요]** GitHub 저장소 생성 + 코드 push
3. **[운영자 필요]** Google AI Studio API 키 발급
4. **[운영자 필요]** HuggingFace Spaces 배포 → 라이브 데모 URL 확보
5. **[운영자 필요]** 비디오 촬영 (3분, YouTube 업로드)
6. **[자동화 가능]** Kaggle Writeup 초안 작성 (Engineer)
7. **[자동화 가능]** 비디오 스크립트 작성 (Lead)
8. **[코드 수정]** 라이선스를 CC-BY 4.0으로 변경 (현재 Apache 2.0)

---

_이 분석은 2026-04-03 scrapling으로 Kaggle 5개 탭 직접 수집한 결과입니다._
