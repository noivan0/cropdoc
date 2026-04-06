#!/usr/bin/env python3.10
"""
PlantDoc Dataset Collector - GitHub API 경유
실제 현장(field) 이미지 수집
License: MIT (PlantDoc)
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

# SSL 우회
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

failed_count = 0
new_downloads = 0

# PlantDoc GitHub → field_images 매핑
PLANTDOC_MAPPING = {
    "train/Tomato leaf dir": "tomato/healthy",
    "train/Tomato Early blight leaf dir": "tomato/early_blight",
    "train/Tomato leaf late blight dir": "tomato/late_blight",
    "train/Tomato leaf bacterial spot dir": "tomato/bacterial_spot",
    "train/Tomato mold leaf dir": "tomato/leaf_mold",
    "train/Tomato Septoria leaf spot dir": "tomato/septoria",
    "train/Potato leaf early blight dir": "potato/early_blight",
    "train/Potato leaf late blight dir": "potato/late_blight",
    "train/Corn Gray leaf spot dir": "corn/gray_leaf_spot",
    "train/Corn rust leaf dir": "corn/common_rust",
    "train/Corn leaf blight dir": "corn/northern_blight",
    "train/Apple leaf dir": "apple/healthy",
    "train/Apple Scab Leaf dir": "apple/scab",
    "train/grape leaf dir": "grape/healthy",
    "train/grape leaf black rot dir": "grape/black_rot",
    "train/Strawberry leaf dir": "strawberry/healthy",
    "train/Bell_pepper leaf dir": "pepper/healthy",
    "train/Bell_pepper leaf spot dir": "pepper/bacterial_spot",
    # test set도 추가
    "test/Tomato leaf dir": "tomato/healthy",
    "test/Tomato Early blight leaf dir": "tomato/early_blight",
    "test/Tomato leaf late blight dir": "tomato/late_blight",
    "test/Tomato leaf bacterial spot dir": "tomato/bacterial_spot",
    "test/Potato leaf early blight dir": "potato/early_blight",
    "test/Potato leaf late blight dir": "potato/late_blight",
    "test/Corn Gray leaf spot dir": "corn/gray_leaf_spot",
    "test/Corn rust leaf dir": "corn/common_rust",
    "test/Apple leaf dir": "apple/healthy",
    "test/Apple Scab Leaf dir": "apple/scab",
    "test/grape leaf dir": "grape/healthy",
    "test/Strawberry leaf dir": "strawberry/healthy",
    "test/Bell_pepper leaf dir": "pepper/healthy",
    "test/Bell_pepper leaf spot dir": "pepper/bacterial_spot",
}

GITHUB_API = "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/contents"
GITHUB_RAW = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master"
MAX_PER_CATEGORY = 8  # 카테고리별 최대 다운로드 수


def safe_request(url, headers=None, timeout=30):
    """SSL 우회 HTTP 요청"""
    default_headers = {
        "User-Agent": "CropDoc/1.0 (Research Bot)",
        "Accept": "application/vnd.github.v3+json"
    }
    if headers:
        default_headers.update(headers)
    
    try:
        req = urllib.request.Request(url, headers=default_headers)
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"  ⚠️  Request failed: {str(e)[:80]}")
        return None


def get_github_files(path):
    """GitHub API로 디렉토리의 파일 목록 가져오기"""
    encoded = urllib.parse.quote(path)
    url = f"{GITHUB_API}/{encoded}"
    data = safe_request(url)
    if not data:
        return []
    try:
        return json.loads(data)
    except:
        return []


def download_plantdoc_image(raw_url, save_path, metadata):
    """PlantDoc GitHub raw URL에서 이미지 다운로드"""
    global downloaded_count, new_downloads, failed_count
    
    if save_path.exists():
        return True
    
    data = safe_request(raw_url)
    if not data:
        failed_count += 1
        return False
    
    if len(data) < 5000:
        print(f"  ⚠️  Too small ({len(data)} bytes)")
        failed_count += 1
        return False
    
    if len(data) > 5 * 1024 * 1024:
        print(f"  ⚠️  Too large ({len(data)//1024}KB)")
        failed_count += 1
        return False
    
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(data)
        
        metadata['file_size'] = len(data)
        metadata['local_path'] = str(save_path.relative_to(BASE_DIR.parent))
        metadata['downloaded_at'] = datetime.now().isoformat()
        labels.append(metadata)
        
        downloaded_count += 1
        new_downloads += 1
        print(f"  ✅ {save_path.name} ({len(data)//1024}KB)")
        return True
    except Exception as e:
        print(f"  ❌ Save failed: {e}")
        failed_count += 1
        return False


def collect_from_plantdoc():
    """PlantDoc 데이터셋 수집"""
    print("\n🌱 PlantDoc Dataset (GitHub) 수집 시작")
    print("=" * 50)
    
    for github_path, save_category in PLANTDOC_MAPPING.items():
        save_dir = BASE_DIR / save_category
        existing = len(list(save_dir.glob("*.jpg"))) + len(list(save_dir.glob("*.JPG"))) + \
                   len(list(save_dir.glob("*.jpeg"))) + len(list(save_dir.glob("*.png")))
        
        if existing >= MAX_PER_CATEGORY:
            print(f"✓ {save_category}: 이미 {existing}장 있음, skip")
            continue
        
        needed = MAX_PER_CATEGORY - existing
        print(f"\n📂 {github_path} → {save_category} (현재 {existing}장, {needed}장 추가 필요)")
        
        files = get_github_files(github_path)
        if not files:
            print("  ⚠️  파일 목록 가져오기 실패")
            time.sleep(1)
            continue
        
        # 이미지 파일만 필터링
        img_files = [f for f in files if f.get('type') == 'file' and 
                     any(f['name'].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
        
        print(f"  총 {len(img_files)}개 이미지 발견")
        
        collected_here = 0
        for img_file in img_files[:needed + 3]:  # 약간 여유 있게
            if collected_here >= needed:
                break
            
            filename = img_file['name']
            raw_url = f"{GITHUB_RAW}/{github_path.replace(' dir', ' dir')}/{urllib.parse.quote(filename)}"
            
            # 간소화: download_url 직접 사용
            download_url = img_file.get('download_url', '')
            if not download_url:
                download_url = raw_url
            
            # 저장 파일명 (해시로 중복 방지)
            url_hash = hashlib.md5(download_url.encode()).hexdigest()[:8]
            ext = '.jpg' if filename.lower().endswith(('.jpg', '.jpeg')) else '.png'
            save_filename = f"plantdoc_{url_hash}{ext}"
            save_path = save_dir / save_filename
            
            plant_name = save_category.split('/')[0]
            condition = save_category.split('/')[1] if '/' in save_category else 'unknown'
            
            metadata = {
                'source': 'plantdoc_github',
                'original_filename': filename,
                'github_path': github_path,
                'image_url': download_url,
                'license': 'MIT (PlantDoc Dataset - pratikkayal)',
                'license_url': 'https://github.com/pratikkayal/PlantDoc-Dataset',
                'category': save_category,
                'label': save_category,
                'plant': plant_name,
                'condition': condition,
                'dataset': 'PlantDoc',
                'is_field_image': True
            }
            
            if download_plantdoc_image(download_url, save_path, metadata):
                collected_here += 1
            
            time.sleep(0.8)  # GitHub API 레이트 리밋 방지
        
        print(f"  → {collected_here}장 새로 수집")
        time.sleep(1)


def copy_from_plantvillage():
    """기존 PlantVillage eval_set에서 추가 카테고리 복사"""
    import shutil
    
    EVAL_DIR = Path("/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good/data/plantvillage/eval_set")
    
    COPY_MAPPING = {
        "Pepper__bell___Bacterial_spot": "pepper/bacterial_spot",
        "Potato___Early_blight": "potato/early_blight",
        "Potato___Late_blight": "potato/late_blight",
        "Potato___healthy": "potato/healthy",
        "Tomato_Bacterial_spot": "tomato/bacterial_spot",
        "Tomato_Early_blight": "tomato/early_blight",
        "Tomato_Late_blight": "tomato/late_blight",
        "Tomato_Leaf_Mold": "tomato/leaf_mold",
        "Tomato_Septoria_leaf_spot": "tomato/septoria",
        "Tomato_healthy": "tomato/healthy",
    }
    
    print("\n\n📋 PlantVillage eval_set → field_images 보완 복사")
    print("=" * 50)
    
    for src_name, dest_cat in COPY_MAPPING.items():
        src_dir = EVAL_DIR / src_name
        dest_dir = BASE_DIR / dest_cat
        
        if not src_dir.exists():
            continue
        
        existing = len(list(dest_dir.glob("*.*")))
        if existing >= MAX_PER_CATEGORY:
            print(f"✓ {dest_cat}: 이미 {existing}장, skip")
            continue
        
        needed = MAX_PER_CATEGORY - existing
        src_files = list(src_dir.glob("*.JPG")) + list(src_dir.glob("*.jpg")) + \
                    list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.png"))
        
        copied = 0
        for src_file in src_files[:needed]:
            dest_file = dest_dir / f"pv_{src_file.stem[:20]}{src_file.suffix.lower()}"
            if dest_file.exists():
                continue
            
            try:
                shutil.copy2(src_file, dest_file)
                
                plant_name = dest_cat.split('/')[0]
                condition = dest_cat.split('/')[1] if '/' in dest_cat else 'unknown'
                
                metadata = {
                    'source': 'plantvillage_evalset',
                    'original_filename': src_file.name,
                    'image_url': str(src_file),
                    'license': 'CC BY 4.0 (PlantVillage)',
                    'license_url': 'https://github.com/spMohanty/PlantVillage-Dataset',
                    'category': dest_cat,
                    'label': dest_cat,
                    'plant': plant_name,
                    'condition': condition,
                    'dataset': 'PlantVillage',
                    'is_field_image': False,
                    'note': 'Lab-controlled image from PlantVillage'
                }
                
                file_size = src_file.stat().st_size
                metadata['file_size'] = file_size
                metadata['local_path'] = str(dest_file.relative_to(BASE_DIR.parent))
                metadata['downloaded_at'] = datetime.now().isoformat()
                labels.append(metadata)
                
                global downloaded_count, new_downloads
                downloaded_count += 1
                new_downloads += 1
                copied += 1
            except Exception as e:
                print(f"  ⚠️  Copy failed: {e}")
        
        print(f"  {src_name} → {dest_cat}: {copied}장 복사")


def save_labels():
    """labels JSON 저장"""
    from collections import Counter
    
    labels_data = {
        'collected_at': datetime.now().isoformat(),
        'total_images': downloaded_count,
        'sources': ['plantdoc_github', 'wikimedia_commons', 'plantvillage_evalset'],
        'license': 'MIT/CC-BY/CC0',
        'note': 'PlantDoc: field images. PlantVillage: lab images used as baseline.',
        'images': labels
    }
    
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 Labels saved: {LABELS_FILE}")
    print(f"   총 이미지: {downloaded_count}장")
    
    # 카테고리별 통계
    print("\n📊 Category Statistics:")
    cat_counts = Counter(item.get('category', 'unknown') for item in labels)
    for cat, cnt in sorted(cat_counts.items()):
        status = "✅" if cnt >= 3 else "⚠️"
        print(f"  {status} {cat}: {cnt}장")
    
    # 식물별 통계
    print("\n🌿 Plant Statistics:")
    plant_counts = Counter(item.get('plant', 'unknown') for item in labels)
    for plant, cnt in sorted(plant_counts.items()):
        status = "✅" if cnt >= 3 else "⚠️"
        print(f"  {status} {plant}: {cnt}장")
    
    return labels_data


if __name__ == '__main__':
    print("=" * 60)
    print("🌿 PlantDoc + PlantVillage Field Image Collector")
    print(f"📁 Save to: {BASE_DIR}")
    print(f"🔢 기존 수집: {downloaded_count}장")
    print("=" * 60)
    
    # 1단계: PlantDoc GitHub에서 수집
    collect_from_plantdoc()
    
    # 2단계: PlantVillage에서 보완
    copy_from_plantvillage()
    
    # 결과 저장
    print(f"\n{'=' * 60}")
    print(f"✅ 새로 추가: {new_downloads}장")
    print(f"✅ 총 수집: {downloaded_count}장")
    print(f"❌ 실패: {failed_count}건")
    
    save_labels()
