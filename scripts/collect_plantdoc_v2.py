#!/usr/bin/env python3.10
"""
PlantDoc Dataset Collector v2 - 수정된 경로 사용
"""

import os
import json
import ssl
import time
import urllib.request
import urllib.parse
import hashlib
from pathlib import Path
from datetime import datetime

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE_DIR = Path("/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good/data/field_images")
LABELS_FILE = BASE_DIR / "field_labels.json"

# 기존 labels 로드
if LABELS_FILE.exists():
    with open(LABELS_FILE) as f:
        existing_data = json.load(f)
    labels = existing_data.get('images', [])
    downloaded_count = existing_data.get('total_images', 0)
else:
    labels = []
    downloaded_count = 0

new_downloads = 0
failed_count = 0

# PlantDoc GitHub 정확한 경로 (dir 접미사 없음)
PLANTDOC_MAPPING = {
    # train set - field images
    "train/Tomato leaf": ("tomato/healthy", "tomato", "healthy"),
    "train/Tomato Early blight leaf": ("tomato/early_blight", "tomato", "early_blight"),
    "train/Tomato leaf late blight": ("tomato/late_blight", "tomato", "late_blight"),
    "train/Tomato leaf bacterial spot": ("tomato/bacterial_spot", "tomato", "bacterial_spot"),
    "train/Tomato mold leaf": ("tomato/leaf_mold", "tomato", "leaf_mold"),
    "train/Tomato Septoria leaf spot": ("tomato/septoria", "tomato", "septoria"),
    "train/Potato leaf early blight": ("potato/early_blight", "potato", "early_blight"),
    "train/Potato leaf late blight": ("potato/late_blight", "potato", "late_blight"),
    "train/Corn Gray leaf spot": ("corn/gray_leaf_spot", "corn", "gray_leaf_spot"),
    "train/Corn rust leaf": ("corn/common_rust", "corn", "common_rust"),
    "train/Corn leaf blight": ("corn/northern_blight", "corn", "northern_blight"),
    "train/Apple leaf": ("apple/healthy", "apple", "healthy"),
    "train/Apple Scab Leaf": ("apple/scab", "apple", "scab"),
    "train/Apple rust leaf": ("apple/rust", "apple", "rust"),
    "train/grape leaf": ("grape/healthy", "grape", "healthy"),
    "train/grape leaf black rot": ("grape/black_rot", "grape", "black_rot"),
    "train/Strawberry leaf": ("strawberry/healthy", "strawberry", "healthy"),
    "train/Bell_pepper leaf": ("pepper/healthy", "pepper", "healthy"),
    "train/Bell_pepper leaf spot": ("pepper/bacterial_spot", "pepper", "bacterial_spot"),
    "train/Blueberry leaf": ("blueberry/healthy", "blueberry", "healthy"),
}

GITHUB_API = "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/contents"
MAX_PER_CATEGORY = 8


def safe_get(url, timeout=25):
    headers = {"User-Agent": "CropDoc/1.0 (Research)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"  ⚠️  {str(e)[:80]}")
        return None


def get_dir_files(path):
    encoded = urllib.parse.quote(path)
    url = f"{GITHUB_API}/{encoded}"
    data = safe_get(url)
    if not data:
        return []
    try:
        result = json.loads(data)
        if isinstance(result, list):
            return result
        return []
    except:
        return []


def download_image(url, save_path, meta):
    global downloaded_count, new_downloads, failed_count
    
    if save_path.exists():
        return True
    
    data = safe_get(url, timeout=30)
    if not data or len(data) < 3000 or len(data) > 5*1024*1024:
        failed_count += 1
        return False
    
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(data)
        
        meta['file_size'] = len(data)
        meta['local_path'] = str(save_path.relative_to(BASE_DIR.parent))
        meta['downloaded_at'] = datetime.now().isoformat()
        labels.append(meta)
        
        downloaded_count += 1
        new_downloads += 1
        print(f"  ✅ {save_path.name} ({len(data)//1024}KB)")
        return True
    except Exception as e:
        print(f"  ❌ {e}")
        failed_count += 1
        return False


def main():
    print("=" * 60)
    print("🌿 PlantDoc v2 Collector")
    print(f"📁 {BASE_DIR}")
    print(f"🔢 기존: {downloaded_count}장")
    print("=" * 60)
    
    # 추가 필요한 카테고리
    extra_dirs_needed = {}
    for gh_path, (save_cat, plant, cond) in PLANTDOC_MAPPING.items():
        save_dir = BASE_DIR / save_cat
        save_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(save_dir.glob("*.*")))
        if existing < MAX_PER_CATEGORY:
            extra_dirs_needed[gh_path] = (save_cat, plant, cond, MAX_PER_CATEGORY - existing)
    
    print(f"\n🎯 {len(extra_dirs_needed)}개 카테고리 추가 수집 필요")
    
    for gh_path, (save_cat, plant, cond, needed) in extra_dirs_needed.items():
        save_dir = BASE_DIR / save_cat
        print(f"\n📂 {gh_path} → {save_cat} ({needed}장 필요)")
        
        files = get_dir_files(gh_path)
        img_files = [f for f in files if f.get('type') == 'file' and 
                     any(f['name'].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.JPG'])]
        
        print(f"  총 {len(img_files)}개 이미지 발견")
        
        collected = 0
        for img_file in img_files:
            if collected >= needed:
                break
            
            dl_url = img_file.get('download_url', '')
            if not dl_url:
                continue
            
            fname = img_file['name']
            url_hash = hashlib.md5(dl_url.encode()).hexdigest()[:8]
            ext = '.jpg' if any(fname.lower().endswith(e) for e in ['.jpg', '.jpeg']) else '.png'
            save_path = save_dir / f"pd_{url_hash}{ext}"
            
            meta = {
                'source': 'plantdoc_github',
                'original_filename': fname,
                'github_path': gh_path,
                'image_url': dl_url,
                'license': 'MIT',
                'license_url': 'https://github.com/pratikkayal/PlantDoc-Dataset',
                'category': save_cat,
                'label': save_cat,
                'plant': plant,
                'condition': cond,
                'dataset': 'PlantDoc',
                'is_field_image': True,
                'note': 'Real-world field image from PlantDoc dataset'
            }
            
            if download_image(dl_url, save_path, meta):
                collected += 1
            
            time.sleep(0.6)
        
        print(f"  → {collected}장 수집 완료")
        time.sleep(0.5)
    
    # 결과 저장
    print(f"\n{'=' * 60}")
    print(f"✅ 신규: {new_downloads}장, 총: {downloaded_count}장, 실패: {failed_count}건")
    
    from collections import Counter
    labels_data = {
        'collected_at': datetime.now().isoformat(),
        'total_images': downloaded_count,
        'new_this_run': new_downloads,
        'sources': ['plantdoc_github', 'wikimedia_commons', 'plantvillage_evalset'],
        'license': 'MIT (PlantDoc) / CC-BY (PlantVillage)',
        'note': 'PlantDoc: real-world field images. PlantVillage: lab-controlled baseline.',
        'images': labels
    }
    
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 Labels → {LABELS_FILE}")
    print("\n📊 카테고리별:")
    cat_counts = Counter(item.get('category', 'unknown') for item in labels)
    for cat, cnt in sorted(cat_counts.items()):
        ok = "✅" if cnt >= 3 else "⚠️ "
        print(f"  {ok} {cat}: {cnt}장")
    
    print("\n🌿 식물별:")
    plant_counts = Counter(item.get('plant', 'unknown') for item in labels)
    for plant, cnt in sorted(plant_counts.items()):
        ok = "✅" if cnt >= 3 else "⚠️ "
        print(f"  {ok} {plant}: {cnt}장")
    
    return downloaded_count


if __name__ == '__main__':
    main()
