#!/usr/bin/env python3
"""
CropDoc 확장 데이터셋 정리 스크립트
다운로드된 데이터셋을 CropDoc 레이블 기준으로 분류하고 정리
"""
import os, glob, json, shutil, csv
from pathlib import Path
import random

PROJECT_DIR = Path("/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good")
OUTPUT_DIR = PROJECT_DIR / "data" / "extended_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_BASE = Path("/root/.cache/kagglehub/datasets")

MAX_PER_CLASS = 500
MIN_PER_CLASS = 50

# ── CropDoc 타겟 레이블 (신규 16종) ──────────────────────────────────────
TARGET_LABELS = {
    # Coffee
    "Coffee Leaf Rust": [],
    "Coffee Leaf Miner": [],
    "Coffee Phoma": [],
    # Rice
    "Rice Blast": [],
    "Rice Brown Spot": [],
    "Rice Bacterial Blight": [],
    "Rice Leaf Smut": [],
    "Rice Hispa": [],
    # Wheat
    "Wheat Leaf Rust": [],
    "Wheat Stripe Rust": [],
    "Wheat Stem Rust": [],
    "Wheat Loose Smut": [],
    # Mango
    "Mango Anthracnose": [],
    "Mango Bacterial Canker": [],
    "Mango Die Back": [],
    "Mango Powdery Mildew": [],
    "Mango Sooty Mould": [],
    "Mango Gall Midge": [],
    # Cassava
    "Cassava Bacterial Blight": [],
    "Cassava Mosaic Disease": [],
    "Cassava Brown Streak Disease": [],
    "Cassava Green Mottle": [],
    # Banana
    "Banana Black Sigatoka": [],
    "Banana Yellow Sigatoka": [],
    "Banana Panama Disease": [],
    "Banana Moko Disease": [],
    "Banana Bract Mosaic Virus": [],
    # Citrus
    "Citrus Greening": [],
    "Citrus Canker": [],
    "Citrus Black Spot": [],
    "Citrus Melanose": [],
}

def find_images(folder):
    """폴더에서 이미지 파일 목록 반환"""
    exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(str(folder / '**' / ext), recursive=True))
    return imgs

def copy_images(src_list, dst_dir, max_count=500):
    """이미지 복사 (심볼링크 대신 복사, 최대 max_count장)"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    selected = src_list[:max_count] if len(src_list) > max_count else src_list
    # 랜덤 선택 (다양성 확보)
    if len(src_list) > max_count:
        selected = random.sample(src_list, max_count)
    copied = 0
    for src in selected:
        src_path = Path(src)
        dst_path = dst_dir / src_path.name
        # 중복 파일명 처리
        if dst_path.exists():
            dst_path = dst_dir / f"{src_path.stem}_{copied}{src_path.suffix}"
        try:
            shutil.copy2(src, dst_path)
            copied += 1
        except Exception as e:
            print(f"  ⚠️ 복사 실패: {src} → {e}")
    return copied

# ── 데이터소스별 매핑 처리 ────────────────────────────────────────────────

collected = {label: [] for label in TARGET_LABELS}
source_log = []

print("=" * 60)
print("CropDoc 확장 데이터셋 정리 시작")
print("=" * 60)

# ── 1. Coffee (badasstechie) ─────────────────────────────────────────────
print("\n[1] Coffee - badasstechie/coffee-leaf-diseases")
coffee_path = CACHE_BASE / "badasstechie/coffee-leaf-diseases/versions/3/coffee-leaf-diseases"

# CSV 기반 레이블 파악 + 이미지 매핑
for split in ['train', 'test']:
    csv_path = CACHE_BASE / "badasstechie/coffee-leaf-diseases/versions/3" / f"{split}_classes.csv"
    img_dir = coffee_path / split / "images"
    if not csv_path.exists() or not img_dir.exists():
        continue
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['id']
            # 이미지 파일 찾기
            for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
                img_file = img_dir / f"{img_id}{ext}"
                if img_file.exists():
                    if row.get('rust', '0') == '1':
                        collected["Coffee Leaf Rust"].append(str(img_file))
                    if row.get('miner', '0') == '1':
                        collected["Coffee Leaf Miner"].append(str(img_file))
                    if row.get('phoma', '0') == '1':
                        collected["Coffee Phoma"].append(str(img_file))
                    break

for label in ["Coffee Leaf Rust", "Coffee Leaf Miner", "Coffee Phoma"]:
    n = len(collected[label])
    print(f"  {label}: {n}장 수집")
    source_log.append({"source": "badasstechie/coffee-leaf-diseases", "label": label, "count": n})

# ── 2. Rice (vbookshelf) ─────────────────────────────────────────────────
print("\n[2] Rice - vbookshelf/rice-leaf-diseases")
rice_vbook = CACHE_BASE / "vbookshelf/rice-leaf-diseases/versions/1/rice_leaf_diseases"
rice_mapping = {
    "Brown spot": "Rice Brown Spot",
    "Bacterial leaf blight": "Rice Bacterial Blight",
    "Leaf smut": "Rice Leaf Smut",
}
for folder_name, label in rice_mapping.items():
    folder = rice_vbook / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "vbookshelf/rice-leaf-diseases", "label": label, "count": len(imgs)})

# ── 3. Rice (jay7080dev) ─────────────────────────────────────────────────
print("\n[3] Rice - jay7080dev/rice-plant-diseases-dataset")
rice_jay_base = CACHE_BASE / "jay7080dev/rice-plant-diseases-dataset/versions/1/rice leaf diseases dataset"
jay_mapping = {
    "Brownspot": "Rice Brown Spot",
    "Bacterialblight": "Rice Bacterial Blight",
    "Leafsmut": "Rice Leaf Smut",
}
for folder_name, label in jay_mapping.items():
    folder = rice_jay_base / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "jay7080dev/rice-plant-diseases-dataset", "label": label, "count": len(imgs)})

# ── 4. Rice (thegoanpanda) ───────────────────────────────────────────────
print("\n[4] Rice - thegoanpanda/rice-crop-diseases")
rice_goan_base = CACHE_BASE / "thegoanpanda/rice-crop-diseases/versions/1/Rice_Diseases"
goan_mapping = {
    "Blast Disease": "Rice Blast",
    "Brown Spot Disease": "Rice Brown Spot",
    "Bacterial Blight Disease": "Rice Bacterial Blight",
}
for folder_name, label in goan_mapping.items():
    folder = rice_goan_base / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "thegoanpanda/rice-crop-diseases", "label": label, "count": len(imgs)})

# ── 5. Rice + Cassava (leaf-disease-combination) ─────────────────────────
print("\n[5] Rice + Cassava - asheniranga/leaf-disease-dataset-combination")
combo_train = CACHE_BASE / "asheniranga/leaf-disease-dataset-combination/versions/1/image data/train"
cassava_mapping = {
    "Bacterial Blight (CBB)": "Cassava Bacterial Blight",
    "Mosaic Disease (CMD)": "Cassava Mosaic Disease",
    "Brown Streak Disease (CBSD)": "Cassava Brown Streak Disease",
    "Green Mottle (CGM)": "Cassava Green Mottle",
}
rice_combo_mapping = {
    "LeafBlast": "Rice Blast",
    "BrownSpot": "Rice Brown Spot",
    "Hispa": "Rice Hispa",
}
if (combo_train / "Cassava").exists():
    for folder_name, label in cassava_mapping.items():
        folder = combo_train / "Cassava" / folder_name
        if folder.exists():
            imgs = find_images(folder)
            collected[label].extend(imgs)
            print(f"  {label}: +{len(imgs)}장")
            source_log.append({"source": "asheniranga/leaf-disease-dataset-combination (Cassava)", "label": label, "count": len(imgs)})

if (combo_train / "Rice").exists():
    for folder_name, label in rice_combo_mapping.items():
        folder = combo_train / "Rice" / folder_name
        if folder.exists():
            imgs = find_images(folder)
            collected[label].extend(imgs)
            print(f"  {label}: +{len(imgs)}장")
            source_log.append({"source": "asheniranga/leaf-disease-dataset-combination (Rice)", "label": label, "count": len(imgs)})

# ── 6. Mango (aryashah2k) ────────────────────────────────────────────────
print("\n[6] Mango - aryashah2k/mango-leaf-disease-dataset")
mango_arya = CACHE_BASE / "aryashah2k/mango-leaf-disease-dataset/versions/1"
mango_arya_mapping = {
    "Anthracnose": "Mango Anthracnose",
    "Bacterial Canker": "Mango Bacterial Canker",
    "Die Back": "Mango Die Back",
    "Powdery Mildew": "Mango Powdery Mildew",
    "Sooty Mould": "Mango Sooty Mould",
    "Gall Midge": "Mango Gall Midge",
}
for folder_name, label in mango_arya_mapping.items():
    folder = mango_arya / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "aryashah2k/mango-leaf-disease-dataset", "label": label, "count": len(imgs)})

# ── 7. Mango (shuvokumarbasak4004) ──────────────────────────────────────
print("\n[7] Mango - shuvokumarbasak4004/the-mango-leaf-disease-classification-dataset")
mango_shuv_train = CACHE_BASE / "shuvokumarbasak4004/the-mango-leaf-disease-classification-dataset/versions/1/Mango/train"
shuv_mapping = {
    "Anthracnose_Fungal_Leaf_Disease_Mango": "Mango Anthracnose",
    "Rust_Leaf_Disease_Mango": "Mango Powdery Mildew",  # closest match
}
for folder_name, label in shuv_mapping.items():
    folder = mango_shuv_train / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "shuvokumarbasak4004/the-mango-leaf-disease-classification-dataset", "label": label, "count": len(imgs)})

# ── 8. Mango (ahmadzargar) ───────────────────────────────────────────────
print("\n[8] Mango - ahmadzargar/mango-leaf-diseases-dataset")
mango_ahmad = CACHE_BASE / "ahmadzargar/mango-leaf-diseases-dataset/versions/1/val"
ahmad_mapping = {
    "Anthracnose": "Mango Anthracnose",
    "Bacterial Canker": "Mango Bacterial Canker",
    "Die Back": "Mango Die Back",
    "Sooty Mould": "Mango Sooty Mould",
    "Gall Midge": "Mango Gall Midge",
    "Cutting Weevil": "Mango Gall Midge",  # closest
    "Powdery Mildew": "Mango Powdery Mildew",
}
for folder_name, label in ahmad_mapping.items():
    folder = mango_ahmad / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "ahmadzargar/mango-leaf-diseases-dataset", "label": label, "count": len(imgs)})

# ── 9. Banana (shifatearman/bananalsd) ──────────────────────────────────
print("\n[9] Banana - shifatearman/bananalsd")
banana_shifat = CACHE_BASE / "shifatearman/bananalsd/versions/1/BananaLSD/OriginalSet"
banana_shifat_mapping = {
    "sigatoka": "Banana Yellow Sigatoka",
    "cordana": "Banana Black Sigatoka",
    "pestalotiopsis": "Banana Black Sigatoka",
}
for folder_name, label in banana_shifat_mapping.items():
    folder = banana_shifat / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장 ({folder_name})")
        source_log.append({"source": "shifatearman/bananalsd", "label": label, "count": len(imgs)})

# ── 10. Banana (sujaykapadnis) ──────────────────────────────────────────
print("\n[10] Banana - sujaykapadnis/banana-disease-recognition-dataset")
banana_sujay = CACHE_BASE / "sujaykapadnis/banana-disease-recognition-dataset/versions/1/Banana Disease Recognition Dataset/Original Images/Original Images"
sujay_mapping = {
    "Banana Black Sigatoka Disease": "Banana Black Sigatoka",
    "Banana Yellow Sigatoka Disease": "Banana Yellow Sigatoka",
    "Banana Panama Disease": "Banana Panama Disease",
    "Banana Moko Disease": "Banana Moko Disease",
    "Banana Bract Mosaic Virus Disease": "Banana Bract Mosaic Virus",
    "Banana Insect Pest Disease": "Banana Black Sigatoka",  # no perfect match
}
for folder_name, label in sujay_mapping.items():
    folder = banana_sujay / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장 ({folder_name})")
        source_log.append({"source": "sujaykapadnis/banana-disease-recognition-dataset", "label": label, "count": len(imgs)})

# ── 11. Citrus (myprojectdictionary) ────────────────────────────────────
print("\n[11] Citrus - myprojectdictionary/citrus-leaf-disease-image")
citrus_my = CACHE_BASE / "myprojectdictionary/citrus-leaf-disease-image/versions/1/Citrus Leaf Disease Image"
citrus_my_mapping = {
    "Greening": "Citrus Greening",
    "Canker": "Citrus Canker",
    "Black spot": "Citrus Black Spot",
    "Melanose": "Citrus Melanose",
}
for folder_name, label in citrus_my_mapping.items():
    folder = citrus_my / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "myprojectdictionary/citrus-leaf-disease-image", "label": label, "count": len(imgs)})

# ── 12. Citrus (jonathansilva2020) ──────────────────────────────────────
print("\n[12] Citrus - jonathansilva2020/dataset-for-classification-of-citrus-diseases")
citrus_j = CACHE_BASE / "jonathansilva2020/dataset-for-classification-of-citrus-diseases/versions/2/dataset/dataset/train"
jonathan_mapping = {
    "black-spot": "Citrus Black Spot",
    "citrus-canker": "Citrus Canker",
}
for folder_name, label in jonathan_mapping.items():
    folder = citrus_j / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "jonathansilva2020/dataset-for-classification-of-citrus-diseases", "label": label, "count": len(imgs)})

# ── 13. Wheat (sabaunnisa/wheat-rust-disease) ───────────────────────────
print("\n[13] Wheat - sabaunnisa/wheat-rust-disease")
wheat_rust = CACHE_BASE / "sabaunnisa/wheat-rust-disease/versions/1"
wheat_mapping = {
    "leaf rust": "Wheat Leaf Rust",
    "stripe rust": "Wheat Stripe Rust",
    "stem rust": "Wheat Stem Rust",
}
for folder_name, label in wheat_mapping.items():
    folder = wheat_rust / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "sabaunnisa/wheat-rust-disease", "label": label, "count": len(imgs)})

# ── 14. BCDD (mixed - wheat included) ───────────────────────────────────
print("\n[14] BCDD Mixed - musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd")
bcdd_train = CACHE_BASE / "musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd/versions/1/Bangladeshi_Crops_Disease_Dataset/train"
bcdd_mapping = {
    "Rice_leaf_blast": "Rice Blast",
    "Rice_brown_spot": "Rice Brown Spot",
    "Rice_bacterial_leaf_blight": "Rice Bacterial Blight",
    "Wheat Brown rust": "Wheat Leaf Rust",
    "Wheat Yellow rust": "Wheat Stripe Rust",
    "Wheat Loose Smut": "Wheat Loose Smut",
}
for folder_name, label in bcdd_mapping.items():
    folder = bcdd_train / folder_name
    if folder.exists():
        imgs = find_images(folder)
        collected[label].extend(imgs)
        print(f"  {label}: +{len(imgs)}장")
        source_log.append({"source": "musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd", "label": label, "count": len(imgs)})

# ── 결과 요약 ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("수집 결과 요약")
print("=" * 60)
ready_labels = []
insufficient_labels = []
total_collected = 0

for label, imgs in collected.items():
    # 중복 제거
    imgs_unique = list(set(imgs))
    collected[label] = imgs_unique
    n = len(imgs_unique)
    total_collected += n
    if n >= MIN_PER_CLASS:
        ready_labels.append(label)
        print(f"  ✅ {label}: {n}장")
    elif n > 0:
        insufficient_labels.append(label)
        print(f"  ⚠️  {label}: {n}장 (부족, {MIN_PER_CLASS}장 미만)")
    else:
        insufficient_labels.append(label)
        print(f"  ❌ {label}: 0장")

print(f"\n총 수집: {total_collected}장")
print(f"학습 준비: {len(ready_labels)}종 ({MIN_PER_CLASS}장+)")
print(f"데이터 부족: {len(insufficient_labels)}종")

# ── 실제 파일 복사 ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("data/extended_datasets/ 에 파일 복사 중...")
print("=" * 60)

copy_results = {}
for label, imgs in collected.items():
    if not imgs:
        copy_results[label] = 0
        continue
    dst_dir = OUTPUT_DIR / label
    n_copied = copy_images(imgs, dst_dir, MAX_PER_CLASS)
    copy_results[label] = n_copied
    if n_copied > 0:
        print(f"  ✅ {label}: {n_copied}장 복사 완료")

# ── JSON 결과 저장 ───────────────────────────────────────────────────────
result = {
    "summary": {
        "total_labels": len(TARGET_LABELS),
        "ready_labels": len(ready_labels),
        "insufficient_labels": len(insufficient_labels),
        "total_images": sum(copy_results.values()),
    },
    "per_label": copy_results,
    "source_log": source_log,
    "ready": ready_labels,
    "insufficient": insufficient_labels,
}

with open(PROJECT_DIR / "data" / "extended_dataset_summary.json", 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✅ 완료! data/extended_dataset_summary.json 저장됨")
