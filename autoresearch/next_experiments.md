# autoresearch 다음 실험 우선순위 (리서치 결과 반영)
_업데이트: 2026-04-04_

## SOTA 현황 (실제 크롤링 확인)
- PlantVillage SOTA: 99.86% (MSCVT, arXiv:2503.16628)
- DenseNet169: 99.72%, Inception_v3: 99.76%
- EfficientNet 계열: 99.4~99.8%
- **우리 현황: 93.7% (v13) → 목표: 97%+**

## 즉시 실험 순서

### 🥇 Idea A: Gemma4 CONFUSION_HINTS 프롬프트 (+1.5~2.0%p 예상)

GEMMA_VERIFY_CONFUSABLE_PROMPT에 아래 힌트 추가:

```python
CONFUSION_HINTS = """
CRITICAL DISAMBIGUATION RULES:
1. Potato Early Blight vs Pepper Bacterial Spot:
   - POTATO leaf: compound/pinnate (multiple leaflets), larger brown CONCENTRIC rings (bullseye pattern)
   - PEPPER leaf: simple oval leaf, SMALL water-soaked spots with yellow halo

2. Tomato Septoria Leaf Spot vs Tomato Bacterial Spot:
   - SEPTORIA: tiny circular spots, dark border + LIGHTER center (bull's-eye), NO yellow halo
   - BACTERIAL SPOT: irregular spots, YELLOW halo present, wet/greasy appearance

3. Tomato Leaf Mold vs Tomato Late Blight:
   - LEAF MOLD: yellow patches on TOP surface, olive-gray FUZZY mold on BOTTOM (check underside)
   - LATE BLIGHT: dark water-soaked lesions with pale green halo, white mold at lesion edge

4. Tomato Early Blight vs Tomato Leaf Mold:
   - EARLY BLIGHT: clear CONCENTRIC ring (bullseye) pattern, yellow halo, starts on OLDER leaves
   - LEAF MOLD: diffuse yellow patches without ring pattern, olive mold on leaf underside
"""
```

프롬프트 수정 위치: scripts/cropdoc_infer.py의 GEMMA_VERIFY_CONFUSABLE_PROMPT 변수
힌트를 프롬프트 시작 부분에 삽입.

### 🥈 Idea B: 가중 TTA (+0.5~1.0%p 예상)

```python
import torchvision.transforms as T
import numpy as np

_NORM = T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
TTA_TRANSFORMS = [
    (T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM]), 2.0),   # 원본 (가중치 2)
    (T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(p=1), T.ToTensor(), _NORM]), 1.0),  # 수평
    (T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomVerticalFlip(p=1), T.ToTensor(), _NORM]), 1.0),   # 수직 (Leaf Mold 유효)
    (T.Compose([T.Resize(280), T.CenterCrop(224), T.ToTensor(), _NORM]), 0.8),   # 약간 줌아웃
]

def cnn_predict_tta(img):
    all_probs, total_w = [], 0
    for tf, w in TTA_TRANSFORMS:
        img_t = tf(img).unsqueeze(0).to(_cnn_device)
        with torch.no_grad():
            logits = _cnn_model(pixel_values=img_t).logits
        all_probs.append(torch.softmax(logits, dim=-1)[0].cpu().numpy() * w)
        total_w += w
    return np.sum(all_probs, axis=0) / total_w
```

### 🥉 Idea C: CNN 앙상블 (+0.5~1.0%p 예상)

```python
CNN_ORIG = "/root/.cache/huggingface/hub/cropdoc_cnn/models--linkanjarad--mobilenet_v2_1.0_224-plant-disease-identification/snapshots/c1861579a670fb6232258805b801cd4137cb7176"
CNN_V2 = "data/models/cropdoc_cnn_v2"
# 두 모델 동시 로드 → softmax 평균
probs = (softmax(model_orig(img)) + softmax(model_v2(img))) / 2
```

### 🏆 Idea D: EfficientNetV2 재학습 (+3~5%p 예상, 시간 많이 소요)

```bash
# torchvision EfficientNetV2 사용
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(1280, 38)
# 데이터: /root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2
# 학습 스크립트: scripts/train_cnn_v2.py 참조하여 EfficientNetV2용으로 수정
```

## 평가 (반드시 nohup)
```bash
VERSION="v19"  # v20, v21...으로 증가
nohup python3.10 -c "
import sys, time, json, os
os.chdir('/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good')
sys.path.insert(0, 'scripts')
import eval_harness as eh
t0 = time.time()
result = eh.run_evaluation(
    labels_path='data/plantvillage/eval_labels.json',
    eval_set_base='data/plantvillage/eval_set'
)
result['elapsed_s'] = time.time() - t0
with open(f'/tmp/eval_${VERSION}_result.json', 'w') as f:
    json.dump(result, f, indent=2)
with open(f'/tmp/eval_${VERSION}_result.txt', 'w') as f:
    acc = result['diagnosis_accuracy']
    f.write(f'${VERSION}: {acc:.4f} ({result[\"correct\"]}/{result[\"total_images\"]})\n')
    for label, res in sorted(result['per_label'].items()):
        ok=res['ok']; n=res['n']
        f.write(f'  {label}: {ok}/{n} = {ok/n:.1%}\n')
print('DONE', result['diagnosis_accuracy'])
" > /tmp/eval_${VERSION}.log 2>&1 &
echo PID: $!
```

## keep 기준
- baseline: 0.937
- keep: >= 0.947 (+1.0%p)
- 목표: >= 0.970
