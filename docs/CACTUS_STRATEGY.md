# CropDoc — Cactus Prize Strategy
_Special Technology Track: Local-first mobile/wearable app_

## What is the Cactus Prize?
$10,000 for apps that route between models locally (on-device) — prioritizing edge deployment over cloud dependency.

## Why CropDoc Qualifies

### Current Implementation (On-Device Capable)
- **Gemma 4 E4B-IT**: 4B parameters — designed for on-device use
- **MobileNetV2 CNN**: 9MB — runs on any device, CPU-only
- **2-stage routing**: CNN first (fast, local) → Gemma4 only when needed
  - This IS local-first model routing: lightweight model → heavy model

### Quantized Deployment Path
```
Gemma 4 E4B-IT (bfloat16, 16GB) 
  → 4-bit quantized: ~4GB (fits iPhone 15 Pro / mid-range Android)
  → llama.cpp / LiteRT: further optimization possible
```

### Architecture Diagram
```
[User's Smartphone]
    │
    ├─► MobileNetV2 (9MB, CPU, instant)
    │       │ confidence ≥ 0.90: DONE (~0.01s)
    │       │ confidence < 0.90: ↓
    │       ▼
    └─► Gemma 4 E4B-IT (4GB quantized, on-device GPU)
            │ Final diagnosis + explanation
            ▼
        [No internet required]
```

### Roadmap for Full Cactus Prize Compliance
| Step | Status | Description |
|------|--------|-------------|
| 2-stage CNN → LLM routing | ✅ Done | Local model routing implemented |
| Gemma4 E4B-IT selection | ✅ Done | Smallest capable multimodal Gemma |
| 4-bit quantization | 🔄 Planned | bitsandbytes / GPTQ quantization |
| LiteRT conversion | 📋 Future | Google AI Edge LiteRT for CNN |
| Mobile wrapper | 📋 Future | Flutter/React Native demo |

### Why E4B and Not a Larger Model
The deliberate choice of E4B (Efficient 4B) over E2B or 26B reflects:
1. **Accuracy**: E4B outperforms E2B on plant disease visual tasks
2. **Efficiency**: 4B params → 4-8GB RAM (accessible on flagship phones)
3. **Latency**: ~5s inference on RTX-class GPU → ~30s on mobile GPU (acceptable)
4. **Battery**: Single inference per photo vs continuous cloud calls

## Competitive Differentiation
Unlike cloud-based agricultural AI (Plantix, etc.), CropDoc works where farmers actually are — in fields with no connectivity. The Gemma 4 E4B + MobileNetV2 combination is specifically engineered for this constraint.
