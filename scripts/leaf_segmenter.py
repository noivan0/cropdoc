"""
CropDoc Leaf Segmenter — v1
============================
GrabCut 기반 잎 영역 세그멘테이션.
중앙 영역을 foreground seed로 초기화하여 잎/식물 부위만 추출.
초록색 픽셀 < 10% 이거나 GrabCut 실패 시 원본 이미지 반환(fallback).

Public API:
    segment_leaf(image: PIL.Image.Image) -> PIL.Image.Image
"""

import numpy as np
from PIL import Image

# ── GrabCut 파라미터 ───────────────────────────────────────────────────────────
GRABCUT_ITERS = 5           # GrabCut 반복 횟수
CENTER_RATIO  = 0.50        # 중앙 foreground seed 비율 (이미지 크기 대비)
GREEN_THRESH  = 0.10        # 초록색 픽셀 비율 최소값 (이 이하면 fallback)


def _is_greenish(rgb_pixel) -> bool:
    """RGB 픽셀이 '초록색 계열'인지 판단."""
    r, g, b = int(rgb_pixel[0]), int(rgb_pixel[1]), int(rgb_pixel[2])
    # 녹색: G가 R, B보다 두드러지게 높고, 일정 밝기 이상
    return g > r * 0.85 and g > b * 0.85 and g > 40


def segment_leaf(image: Image.Image) -> Image.Image:
    """
    잎(leaf) 영역만 추출. 실패 시 원본 반환.

    Args:
        image: PIL RGB 이미지 (어떤 크기든 OK)

    Returns:
        잎 영역만 남긴 PIL RGB 이미지 (배경 → 흰색으로 처리),
        또는 실패 시 원본 이미지
    """
    try:
        import cv2
    except ImportError:
        # OpenCV 없으면 항상 원본 반환
        return image

    # PIL → OpenCV BGR
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    img_bgr = img_np[:, :, ::-1].copy()  # RGB→BGR, contiguous

    # ── 초록색 픽셀 사전 확인 ────────────────────────────────────────────────
    # 이미지 리샘플로 빠른 추정
    sample = img_np.reshape(-1, 3)
    green_count = sum(1 for px in sample[::max(1, len(sample)//2000)] if _is_greenish(px))
    sample_count = len(sample[::max(1, len(sample)//2000)])
    green_ratio_pre = green_count / max(sample_count, 1)
    if green_ratio_pre < 0.05:
        # 초록색이 거의 없으면 식물 이미지가 아닐 가능성 높음 → 원본
        return image

    # ── GrabCut 세그멘테이션 ─────────────────────────────────────────────────
    # 중앙 rect를 foreground 시드로
    margin_x = int(w * (1 - CENTER_RATIO) / 2)
    margin_y = int(h * (1 - CENTER_RATIO) / 2)
    # rect: (x, y, width, height)
    rect = (
        max(1, margin_x),
        max(1, margin_y),
        max(1, w - 2 * margin_x),
        max(1, h - 2 * margin_y),
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        import cv2
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return image

    # GrabCut 결과: 0=BG, 1=FG, 2=PR_BG, 3=PR_FG
    # foreground = GC_FGD(1) | GC_PR_FGD(3)
    fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # ── Fallback: 초록색 픽셀 비율 확인 ────────────────────────────────────
    # fg 영역 내 초록색 픽셀이 10% 미만이면 원본 반환
    fg_pixels = img_np[fg_mask == 255]
    if len(fg_pixels) == 0:
        return image

    green_in_fg = sum(1 for px in fg_pixels[::max(1, len(fg_pixels)//2000)]
                      if _is_greenish(px))
    sample_fg = len(fg_pixels[::max(1, len(fg_pixels)//2000)])
    green_ratio = green_in_fg / max(sample_fg, 1)

    if green_ratio < GREEN_THRESH:
        return image

    # ── 마스크 적용: 배경 → 흰색 ────────────────────────────────────────────
    result = img_np.copy()
    result[fg_mask == 0] = [255, 255, 255]  # 배경 흰색

    return Image.fromarray(result)


# ── 단독 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: python leaf_segmenter.py <image_path> [output_path]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "segmented_output.jpg"

    img = Image.open(in_path).convert("RGB")
    print(f"Input size: {img.size}")

    seg = segment_leaf(img)
    print(f"Output size: {seg.size}")

    if seg is img:
        print("⚠️  Fallback: returned original (green ratio too low or GrabCut failed)")
    else:
        print("✅ Segmentation applied")

    seg.save(out_path)
    print(f"Saved: {out_path}")
