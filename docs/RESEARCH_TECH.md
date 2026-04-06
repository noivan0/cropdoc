# RESEARCH_TECH.md — Gemma 4 기술 레퍼런스
> Kaggle "The Gemma 4 Good Hackathon" 기술 구현 가이드
> 작성일: 2026-04-03 | 소스: HuggingFace google/gemma-4-E4B-it README

---

## 0. Gemma 4 모델 라인업 요약

| 모델 | 파라미터 | 컨텍스트 | 지원 모달리티 | 온디바이스 |
|------|---------|---------|------------|---------|
| **E2B** | 2.3B 유효 (5.1B 임베딩 포함) | 128K | 텍스트 + 이미지 + **오디오** | ✅ 모바일/랩탑 |
| **E4B** | 4.5B 유효 (8B 임베딩 포함) | 128K | 텍스트 + 이미지 + **오디오** | ✅ 랩탑/엣지 |
| **26B A4B (MoE)** | 25.2B (활성 3.8B) | 256K | 텍스트 + 이미지 | ❌ 서버 |
| **31B Dense** | 30.7B | 256K | 텍스트 + 이미지 | ❌ 서버 |

**핵심**: E2B/E4B만 오디오 입력 지원 → 해커톤 멀티모달 최고 점수 조합의 핵심

---

## 1. Google AI Studio API 사용법

### 1-1. 설치 및 인증

```bash
pip install google-generativeai
```

```python
import google.generativeai as genai
import os

# API 키 설정 (Google AI Studio에서 무료 발급)
# https://aistudio.google.com/app/apikey
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
```

### 1-2. 기본 텍스트 생성

```python
import google.generativeai as genai

model = genai.GenerativeModel("gemma-4-e4b-it")  # 또는 gemma-4-31b-it

response = model.generate_content(
    "세계 식량 안보 문제를 해결하는 방법을 3가지 제안하라.",
    generation_config=genai.GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_output_tokens=1024,
    )
)
print(response.text)
```

### 1-3. 멀티턴 채팅

```python
import google.generativeai as genai

model = genai.GenerativeModel(
    "gemma-4-e4b-it",
    system_instruction="You are a helpful agricultural advisor for farmers in developing countries."
)

chat = model.start_chat(history=[])

# 첫 번째 턴
response = chat.send_message("My crop leaves are turning yellow. What could be wrong?")
print(response.text)

# 두 번째 턴 (컨텍스트 유지)
response = chat.send_message("The yellowing started from the bottom leaves. Is it nitrogen deficiency?")
print(response.text)
```

### 1-4. Kaggle에서 Secrets 설정

```python
# Kaggle 노트북에서 Secrets 사용법
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
```

---

## 2. HuggingFace Transformers로 Gemma 4 로드

### 2-1. 의존성 설치

```bash
# 기본 (텍스트 전용)
pip install -U transformers torch accelerate

# 멀티모달 (이미지 + 오디오 포함)
pip install -U transformers torch torchvision librosa accelerate
```

### 2-2. 텍스트 전용 모델 로드

```python
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E4B-it"

# 프로세서 & 모델 로드
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",        # 자동 dtype 선택 (bfloat16 권장)
    device_map="auto"    # GPU/CPU 자동 배치
)

# 추론
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain crop rotation benefits in simple terms."},
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # True로 설정하면 추론 모드 활성화
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512,
                         temperature=1.0, top_p=0.95, top_k=64)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
result = processor.parse_response(response)
print(result)
```

### 2-3. 멀티모달 모델 로드 (이미지 + 오디오)

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)
```

### 2-4. Kaggle/Colab 메모리 최적화 로드

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
import torch

MODEL_ID = "google/gemma-4-E4B-it"

# 메모리 최적화: 4비트 양자화
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

## 3. 멀티모달 이미지 입력 코드 예제

### 3-1. URL에서 이미지 로드

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

# 이미지 URL을 직접 참조 (이미지는 텍스트 앞에 배치 — 최적 성능)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://example.com/crop_disease_image.jpg"},
            {"type": "text", "text": "이 작물 이미지를 분석하고 병해충 여부를 진단하라. 한국어로 답하라."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(processor.parse_response(response))
```

### 3-2. 로컬 이미지 파일 로드

```python
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

# 로컬 이미지 로드
image = Image.open("/path/to/crop_leaf.jpg").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},  # PIL Image 객체 직접 전달
            {"type": "text", "text": "Diagnose the plant disease visible in this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True,
    return_tensors="pt", add_generation_prompt=True
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
print(processor.parse_response(response))
```

### 3-3. 이미지 해상도 토큰 버짓 조정

```python
# 해상도별 토큰 버짓 가이드:
# - 70, 140: 빠른 분류, 캡션 생성 (저해상도)
# - 280, 560: 일반적 이미지 이해
# - 1120: OCR, 문서 파싱, 세부 텍스트 읽기 (최고 해상도)

processor = AutoProcessor.from_pretrained(MODEL_ID)
# image_seq_length 파라미터로 제어 (라이브러리 버전별 방법 다름)
# 최신 Transformers에서는 processor.image_processor.image_seq_length = 560
```

### 3-4. 인터리브 멀티 이미지 입력

```python
# 여러 이미지를 텍스트와 섞어서 입력 가능
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "다음 두 작물 이미지를 비교하라:"},
            {"type": "image", "url": "https://example.com/healthy_crop.jpg"},
            {"type": "text", "text": "vs"},
            {"type": "image", "url": "https://example.com/diseased_crop.jpg"},
            {"type": "text", "text": "어떤 차이가 있는가? 병든 작물의 진단 결과를 제시하라."}
        ]
    }
]
```

---

## 4. 오디오 입력 코드 예제 (E4B)

> ⚠️ 오디오는 E2B / E4B 모델에서만 지원됨. 최대 30초 오디오 처리 가능.

### 4-1. 오디오 URL에서 직접 처리 (ASR — 음성 인식)

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

# 오디오를 텍스트 앞에 배치 (최적 성능)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/journal1.wav"
            },
            {
                "type": "text",
                "text": """Transcribe the following speech segment in its original language.
Follow these specific instructions:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits (e.g., 1.7 not 'one point seven')."""
            },
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True,
    return_tensors="pt", add_generation_prompt=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
print(processor.parse_response(response))
```

### 4-2. 로컬 오디오 파일 처리

```python
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

# librosa로 오디오 로드 (최대 30초)
audio_path = "/path/to/farmer_voice.wav"
audio, sr = librosa.load(audio_path, sr=16000, duration=30)  # 16kHz, 최대 30초

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio},  # numpy array 직접 전달
            {"type": "text", "text": "농민이 설명하는 작물 문제를 듣고 한국어로 요약하라."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True,
    return_tensors="pt", add_generation_prompt=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
print(processor.parse_response(response))
```

### 4-3. 음성 번역 (AST — 다국어 지원)

```python
# 예: 힌디어 음성 → 영어 번역
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "path_or_url_to_hindi_audio.wav"},
            {
                "type": "text",
                "text": """Transcribe the following speech segment in Hindi, then translate it into English.
Format: First output transcription in Hindi, then newline, then 'English: ', then the English translation."""
            },
        ]
    }
]
```

### 4-4. 오디오 + 이미지 동시 처리 (E4B 전용 킬러 기능)

```python
# 농부가 작물 사진을 찍고 음성으로 문제를 설명하는 시나리오
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://example.com/diseased_leaf.jpg"},
            {"type": "audio", "audio": "farmer_complaint.wav"},
            {
                "type": "text",
                "text": "Based on the crop image and the farmer's voice description, diagnose the disease and provide treatment recommendations."
            }
        ]
    }
]
```

---

## 5. Function Calling 예제

Gemma 4는 네이티브 function calling을 지원하여 agentic workflow 구현 가능.

### 5-1. Google AI Studio API로 Function Calling

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# 도구 정의
tools = [
    genai.Tool(
        function_declarations=[
            genai.FunctionDeclaration(
                name="get_weather",
                description="Get current weather for a farming region",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or region name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            ),
            genai.FunctionDeclaration(
                name="get_crop_disease_info",
                description="Get information about a crop disease from agricultural database",
                parameters={
                    "type": "object",
                    "properties": {
                        "disease_name": {"type": "string"},
                        "crop_type": {"type": "string"},
                        "language": {"type": "string", "description": "ISO 639-1 language code"}
                    },
                    "required": ["disease_name", "crop_type"]
                }
            )
        ]
    )
]

model = genai.GenerativeModel("gemma-4-e4b-it", tools=tools)

# 실제 함수 구현
def get_weather(location, unit="celsius"):
    # 실제 날씨 API 호출
    return {"temperature": 28, "humidity": 75, "condition": "sunny", "location": location}

def get_crop_disease_info(disease_name, crop_type, language="en"):
    # 병해충 DB 조회
    return {
        "disease": disease_name,
        "crop": crop_type,
        "treatment": "Apply copper-based fungicide",
        "severity": "moderate"
    }

# Agentic 루프
def run_agent(user_query):
    response = model.generate_content(user_query)
    
    # Function call 처리
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'function_call'):
            fc = part.function_call
            
            if fc.name == "get_weather":
                result = get_weather(**dict(fc.args))
            elif fc.name == "get_crop_disease_info":
                result = get_crop_disease_info(**dict(fc.args))
            
            # 결과를 모델에 반환
            response = model.generate_content([
                user_query,
                response.candidates[0].content,
                genai.Part.from_function_response(
                    name=fc.name,
                    response=result
                )
            ])
    
    return response.text

# 사용 예
result = run_agent("I'm a farmer in Maharashtra, India. My tomato leaves show brown spots. What should I do?")
print(result)
```

### 5-2. HuggingFace Transformers로 Function Calling

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import json

MODEL_ID = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

# 시스템 프롬프트에 도구 정의 포함
tools_definition = json.dumps([
    {
        "name": "search_disease_database",
        "description": "Search agricultural disease database for treatment",
        "parameters": {
            "type": "object",
            "properties": {
                "disease_name": {"type": "string"},
                "crop_type": {"type": "string"}
            }
        }
    }
])

messages = [
    {"role": "system", "content": f"You have access to these tools:\n{tools_definition}"},
    {"role": "user", "content": "My rice crop shows signs of blast disease. Search for treatment options."}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

---

## 6. Kaggle 노트북 추천 셋업

### 6-1. Kaggle 노트북 초기 설정

```python
# Cell 1: 패키지 설치
!pip install -q -U transformers torch torchvision librosa accelerate google-generativeai

# Cell 2: GPU 확인
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 3: HuggingFace 토큰 설정 (Kaggle Secrets 사용)
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

import os
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["GOOGLE_API_KEY"] = secrets.get_secret("GOOGLE_API_KEY")

# Hugging Face 로그인
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
```

### 6-2. Kaggle GPU 환경별 권장 모델

```python
# Kaggle T4 (16GB VRAM) → E4B 4비트 양자화 권장
# Kaggle P100 (16GB VRAM) → E4B 4비트 양자화 권장  
# Kaggle 2xT4 (32GB VRAM) → E4B full precision 가능

from transformers import BitsAndBytesConfig, AutoModelForMultimodalLM, AutoProcessor
import torch

MODEL_ID = "google/gemma-4-E4B-it"

# T4 환경 최적화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, token=os.environ["HF_TOKEN"])
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=os.environ["HF_TOKEN"]
)

print(f"Model loaded! Memory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

### 6-3. Google AI Studio API 방식 (GPU 불필요)

```python
# GPU 없이 API로 바로 사용 — 해커톤 데모에 가장 빠른 방법

import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
import base64
from PIL import Image
import io

secrets = UserSecretsClient()
genai.configure(api_key=secrets.get_secret("GOOGLE_API_KEY"))

# 이미지 업로드 및 분석
def analyze_crop_image(image_path, query, language="ko"):
    model = genai.GenerativeModel("gemma-4-e4b-it")
    
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    image_part = {"mime_type": "image/jpeg", "data": base64.b64encode(image_data).decode()}
    
    response = model.generate_content([
        image_part,
        f"[Language: {language}] {query}"
    ])
    return response.text

# 사용 예
result = analyze_crop_image(
    "diseased_tomato.jpg",
    "이 토마토 잎의 병을 진단하고 치료 방법을 알려줘.",
    language="ko"
)
print(result)
```

### 6-4. Kaggle Dataset 추가 방법

```
Kaggle 노트북 우측 패널:
1. "Add data" 클릭
2. HuggingFace datasets: "Import Dataset" 탭
3. 또는 직접 datasets 라이브러리 사용:

from datasets import load_dataset
ds = load_dataset("PlantVillage/plant-disease")
```

---

## 7. HuggingFace Spaces 배포 방법

### 7-1. Gradio로 빠른 데모 배포

```python
# app.py — HuggingFace Spaces에 올릴 파일

import gradio as gr
import google.generativeai as genai
import os
import base64

# Spaces Secrets에서 API 키 로드
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemma-4-e4b-it")

def diagnose_crop(image, audio, language, query):
    """작물 진단 메인 함수"""
    parts = []
    
    if image is not None:
        # PIL Image → bytes
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        parts.append({
            "mime_type": "image/jpeg",
            "data": base64.b64encode(buf.getvalue()).decode()
        })
    
    prompt = f"[Respond in: {language}]\n{query}"
    if not query:
        prompt = f"[Respond in: {language}]\nDiagnose the crop disease shown in this image and provide treatment recommendations."
    
    parts.append(prompt)
    
    response = model.generate_content(parts)
    return response.text

# Gradio UI 구성
with gr.Blocks(title="🌾 CropDoc — AI Agricultural Assistant") as demo:
    gr.Markdown("# 🌾 CropDoc: Gemma 4 기반 작물 병해충 진단 AI")
    gr.Markdown("작물 사진을 업로드하면 Gemma 4 AI가 병해충을 진단합니다.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="작물 사진 업로드")
            audio_input = gr.Audio(type="filepath", label="음성 설명 (선택)")
            language = gr.Dropdown(
                ["Korean", "English", "Hindi", "Swahili", "Spanish", "French"],
                label="응답 언어", value="Korean"
            )
            query = gr.Textbox(
                label="추가 질문 (선택)",
                placeholder="예: 이 병의 예방 방법도 알려주세요."
            )
            submit_btn = gr.Button("진단하기 🔍", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(label="진단 결과", lines=15)
            gr.Examples(
                examples=[["examples/healthy_leaf.jpg", None, "Korean", ""]],
                inputs=[image_input, audio_input, language, query]
            )
    
    submit_btn.click(
        fn=diagnose_crop,
        inputs=[image_input, audio_input, language, query],
        outputs=output
    )

demo.launch()
```

### 7-2. HuggingFace Spaces 배포 절차

```
1. huggingface.co → New Space 클릭
2. Space 이름: "gemma4-crop-doctor" (예시)
3. License: Apache 2.0
4. SDK: Gradio 선택
5. Hardware: 
   - CPU Basic (무료) — Google AI Studio API 사용 시 충분
   - T4 GPU (유료) — HuggingFace 모델 직접 로드 시 필요
6. Visibility: Public (해커톤 심사용)

파일 구조:
├── app.py          # 메인 Gradio 앱
├── requirements.txt
└── examples/       # 예시 이미지 폴더
    ├── diseased_leaf.jpg
    └── healthy_crop.jpg
```

### 7-3. requirements.txt

```
gradio>=4.0.0
google-generativeai>=0.8.0
Pillow>=10.0.0
librosa>=0.10.0
numpy>=1.24.0
```

### 7-4. Spaces Secrets 설정

```
HuggingFace Space 설정 → Secrets 탭:
- GOOGLE_API_KEY: [Google AI Studio에서 발급한 키]
- HF_TOKEN: [HuggingFace 토큰]

앱에서 접근:
import os
api_key = os.environ.get("GOOGLE_API_KEY")
```

### 7-5. README.md (Space 카드)

```yaml
---
title: CropDoc — Gemma 4 Agricultural AI
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI-powered crop disease diagnosis using Gemma 4 multimodal
tags:
  - gemma4
  - agriculture
  - multimodal
  - plant-disease
  - developing-countries
---
```

---

## 8. 모범 사례 요약

### 최적 샘플링 파라미터
```python
generation_config = {
    "temperature": 1.0,   # Gemma 4 권장값
    "top_p": 0.95,
    "top_k": 64,
    "max_new_tokens": 1024
}
```

### 입력 순서 규칙
1. **이미지/오디오 → 텍스트** 순서로 배치 (최적 성능)
2. 멀티턴 대화: 이전 thinking 내용을 히스토리에 포함하지 말 것
3. 시스템 프롬프트: `{"role": "system", "content": "..."}` 형식 사용

### 오디오 제한
- **최대 길이**: 30초
- **권장 샘플링**: 16kHz
- **지원 형식**: WAV, MP3, FLAC
- **지원 모델**: E2B, E4B만 (26B/31B 불가)

---

*소스: https://huggingface.co/google/gemma-4-E4B-it | 업데이트: 2026-04-03*
