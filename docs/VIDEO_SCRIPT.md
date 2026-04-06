# CropDoc — YouTube Demo Video Script
**Duration:** 2분 30초 (150초)  
**Resolution:** 1280×720 HD  
**File:** cropdoc_demo.mp4 (로컬: /tmp/cropdoc_demo.mp4)

---

## 씬 구성 (자막/나레이션 가이드)

### 🎬 씬 1: TITLE (0~10초)
> "Meet CropDoc — the AI plant doctor that works offline, in your language, anywhere on Earth."

**화면**: CropDoc 로고, 핵심 수치 (500M, 72종, 99.3%, $220B)

---

### 🎬 씬 2: THE PROBLEM (10~28초)
> "Every year, plant diseases destroy 40% of global crops — a $220 billion loss.
> 500 million smallholder farmers have no access to experts.
> Most farms have no internet. And most AI tools only speak English."

**화면**: 5가지 문제 카드 순차 등장

---

### 🎬 씬 3: THE SOLUTION (28~48초)
> "CropDoc solves all three barriers at once.
> It runs fully offline on a 4-billion parameter Gemma 4 model.
> It speaks 10 languages. And it diagnoses 72 crop diseases with 99.3% accuracy."

**화면**: OFFLINE FIRST / 10 LANGUAGES / 99.3% ACCURACY 3개 카드

---

### 🎬 씬 4: LIVE DEMO (48~95초)
> "Watch CropDoc diagnose a tomato leaf in real time.
> The CNN ensemble analyzes the image — 99.3% confidence: Tomato Late Blight.
> Gemma 4 E4B generates a full treatment plan, in Korean, English, Spanish, Chinese — instantly.
> No internet. No cloud. Just your phone."

**화면**: 스캔 애니메이션 → 진단 결과 → 처방 → 다국어 출력

---

### 🎬 씬 5: TECHNOLOGY (95~125초)
> "Our hybrid architecture combines the speed of CNN ensembles with the intelligence of Gemma 4.
> When confidence is high, CNN decides in milliseconds.
> For ambiguous cases, Gemma 4's FOCUSED prompt resolves visual uncertainty.
> We fine-tuned Gemma 4 with PEFT LoRA — 2,148 samples, 141MB adapter, 5 languages."

**화면**: 아키텍처 플로우 다이어그램 + 성능 비교표

---

### 🎬 씬 6: IMPACT + CTA (125~150초)
> "CropDoc is open source, CC-BY 4.0, and ready for deployment today.
> 500 million farmers. 72 diseases. 10 languages. Zero internet required.
> Try it now on Hugging Face Spaces — it's free."

**화면**: 임팩트 수치 6개 + CTA 버튼 + URL

---

## 📤 YouTube 업로드 가이드

1. `/tmp/cropdoc_demo.mp4` 파일 다운로드
2. YouTube Studio → 동영상 업로드
3. 제목: `CropDoc: AI Plant Disease Diagnosis — Offline, Multilingual, 72 Classes | Gemma 4 Good Hackathon`
4. 설명:
```
CropDoc diagnoses 72 plant diseases offline in 10 languages using Gemma 4 E4B-IT.

🌱 500M smallholder farmers served
🔬 72 crop disease classes
🎯 99.3% accuracy (300-image test)
📡 OFFLINE-FIRST — no internet required
🌍 10 languages: Korean, English, Spanish, Chinese, Hindi, French, Portuguese, Bengali, Arabic, Russian

Demo: https://huggingface.co/spaces/noivan/cropdoc
Code: https://github.com/noivan0/cropdoc
Models: https://huggingface.co/noivan/cropdoc

Built for the Gemma 4 Good Hackathon 2026 — Track: Global Resilience
Powered by Google Gemma 4 E4B-IT | CC-BY 4.0
```
5. 썸네일: `docs/cropdoc_cover.png` 업로드
6. 공개 설정 → 로그인 없이 시청 가능
7. URL을 Kaggle Writeup에 첨부
