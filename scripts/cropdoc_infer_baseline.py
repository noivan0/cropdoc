"""
CropDoc Inference Module
========================
Public API (DO NOT CHANGE THE SIGNATURE):
    diagnose(image_path: str, lang: str = "en") -> str

This file is the autoresearch experiment target.
Internal implementation may be freely modified.
The diagnose() function signature must remain stable.

Model: Gemma 4 E4B Instruct (multimodal)
Backend: HuggingFace Transformers — Kaggle official approach:
  apply_chat_template(tokenize=False) → processor(text=text, images=img)
"""

import os
import sys
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# ─── SSL bypass (corporate network) ──────────────────────────────────────────
import urllib3
import requests
from requests.adapters import HTTPAdapter
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
_orig_send = HTTPAdapter.send
def _patched_send(self, request, **kwargs):
    kwargs['verify'] = False
    return _orig_send(self, request, **kwargs)
HTTPAdapter.send = _patched_send

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "/root/.cache/kagglehub/models/google/gemma-4/transformers/gemma-4-e4b-it/1"

# Generation settings (experiment-tunable)
MAX_NEW_TOKENS = 200
DO_SAMPLE = False          # greedy decoding for reproducibility
TEMPERATURE = 1.0          # unused when DO_SAMPLE=False
IMAGE_MAX_SIZE = 896       # resize longer side to this value

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CropDoc, an expert agricultural plant disease diagnostic AI.
Analyze the plant leaf image and identify the exact disease or confirm if it is healthy.

You must diagnose from ONLY these categories:
- Tomato Early Blight
- Tomato Late Blight
- Tomato Bacterial Spot
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Healthy Tomato
- Potato Early Blight
- Potato Late Blight
- Healthy Potato
- Pepper Bacterial Spot

Respond in 1-2 sentences. Begin your response with the exact diagnosis name from the list above.
Example: "Tomato Early Blight – dark concentric rings with yellow halo visible on the leaf surface."
"""

USER_PROMPT = "Diagnose this plant image. What disease or condition does it show?"

# ─── Model Loading (singleton) ────────────────────────────────────────────────
_processor = None
_model = None
_device = None


def _load_model():
    """Load model and processor once (lazy singleton)."""
    global _processor, _model, _device
    if _model is not None:
        return
    print(f"[CropDoc] Loading model from {MODEL_PATH} ...", file=sys.stderr)
    _processor = AutoProcessor.from_pretrained(MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    _device = next(_model.parameters()).device
    print(f"[CropDoc] Model loaded on {_device}.", file=sys.stderr)


def _load_image(image_path: str) -> Image.Image:
    """Load and resize image, keeping aspect ratio."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > IMAGE_MAX_SIZE:
        scale = IMAGE_MAX_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


# ─── Main API ─────────────────────────────────────────────────────────────────

def diagnose(image_path: str, lang: str = "en") -> str:
    """
    Diagnose plant disease from an image.

    Args:
        image_path: Path to the plant image file.
        lang: Language code (default: "en"). Currently "en" only.

    Returns:
        A string describing the diagnosed disease or health status.
    """
    _load_model()

    # Load image
    image = _load_image(image_path)

    # Build messages — use {"type": "image", "url": path} for template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},  # placeholder for <|image|> token
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    # Step 1: get text prompt string (no tokenization)
    # This inserts the <|image|> token placeholder in the right position
    text = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Step 2: tokenize with processor, passing the actual image
    # processor(text=text, images=image) handles image encoding
    inputs = _processor(
        text=text,
        images=image,
        return_tensors="pt",
    ).to(_device)

    input_len = inputs["input_ids"].shape[-1]

    # Generate
    with torch.inference_mode():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            use_cache=True,
            pad_token_id=_processor.tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][input_len:]
    response = _processor.decode(new_tokens, skip_special_tokens=True).strip()

    return response


# ─── CLI Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cropdoc_infer.py <image_path>")
        sys.exit(1)
    img = sys.argv[1]
    t0 = time.time()
    result = diagnose(img)
    elapsed = time.time() - t0
    print(f"Diagnosis ({elapsed:.1f}s): {result}")
