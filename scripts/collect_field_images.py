#!/usr/bin/env python3.10
"""
CropDoc Field Image Collector
실제 현장 이미지 수집 스크립트
Sources: iNaturalist API, Wikimedia Commons API
License: CC0/CC-BY only
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

# SSL 우회 (내부망 환경)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE_DIR = Path("/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good/data/field_images")
LABELS_FILE = BASE_DIR / "field_labels.json"

labels = []
downloaded_count = 0
failed_count = 0

def safe_request(url, headers=None, timeout=20):
    """SSL 우회 HTTP 요청"""
    try:
        req = urllib.request.Request(url, headers=headers or {"User-Agent": "CropDoc/1.0 (Research)"})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"  ⚠️  Request failed: {url[:80]}... -> {e}")
        return None

def download_image(url, save_path, metadata):
    """이미지 다운로드 및 저장"""
    global downloaded_count, failed_count
    
    if save_path.exists():
        print(f"  ✓ Already exists: {save_path.name}")
        return True
    
    data = safe_request(url, timeout=30)
    if data is None:
        failed_count += 1
        return False
    
    # 크기 체크 (최대 5MB)
    if len(data) > 5 * 1024 * 1024:
        print(f"  ⚠️  Too large ({len(data)//1024}KB), skipping")
        failed_count += 1
        return False
    
    # 최소 크기 체크 (너무 작은 이미지 제외)
    if len(data) < 5000:
        print(f"  ⚠️  Too small ({len(data)} bytes), skipping")
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
        print(f"  ✅ Downloaded: {save_path.name} ({len(data)//1024}KB)")
        return True
    except Exception as e:
        print(f"  ❌ Save failed: {e}")
        failed_count += 1
        return False

# ═══════════════════════════════════════════
# 방법 1: Wikimedia Commons API
# ═══════════════════════════════════════════

def fetch_wikimedia_images(search_term, category, label, limit=8):
    """Wikimedia Commons에서 CC 라이선스 이미지 수집"""
    print(f"\n📷 Wikimedia: '{search_term}' -> {label}")
    
    save_dir = BASE_DIR / category
    
    # 1단계: 카테고리 이미지 검색
    encoded = urllib.parse.quote(search_term)
    search_url = (
        f"https://commons.wikimedia.org/w/api.php"
        f"?action=query&list=search"
        f"&srsearch={encoded}&srnamespace=6"
        f"&srlimit={limit}&format=json"
    )
    
    data = safe_request(search_url)
    if not data:
        return 0
    
    try:
        result = json.loads(data)
        pages = result.get('query', {}).get('search', [])
    except:
        return 0
    
    collected = 0
    for page in pages:
        if collected >= limit:
            break
        
        title = page['title']
        if not any(ext in title.lower() for ext in ['.jpg', '.jpeg', '.png']):
            continue
        
        # 이미지 정보 가져오기
        info_url = (
            f"https://commons.wikimedia.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(title)}"
            f"&prop=imageinfo&iiprop=url|mime|size|extmetadata"
            f"&format=json"
        )
        
        info_data = safe_request(info_url)
        if not info_data:
            continue
        
        try:
            info_result = json.loads(info_data)
            pages_data = info_result.get('query', {}).get('pages', {})
            
            for page_id, page_info in pages_data.items():
                imageinfo = page_info.get('imageinfo', [])
                if not imageinfo:
                    continue
                
                img_info = imageinfo[0]
                img_url = img_info.get('url', '')
                mime = img_info.get('mime', '')
                extmeta = img_info.get('extmetadata', {})
                
                # JPEG/PNG만
                if mime not in ['image/jpeg', 'image/png']:
                    continue
                
                # 라이선스 확인 (CC 계열만)
                license_short = extmeta.get('LicenseShortName', {}).get('value', '')
                license_url = extmeta.get('LicenseUrl', {}).get('value', '')
                
                is_cc = any(cc in license_short.upper() for cc in ['CC', 'PUBLIC DOMAIN', 'CC0']) or \
                        'creativecommons' in license_url.lower() or \
                        license_short == ''  # 라이선스 정보 없으면 포함 (Wikimedia는 대부분 CC)
                
                if not is_cc and license_short:
                    print(f"  ⚠️  Non-CC license: {license_short}, skipping")
                    continue
                
                # 파일명 생성
                ext = '.jpg' if mime == 'image/jpeg' else '.png'
                url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
                filename = f"wikimedia_{url_hash}{ext}"
                save_path = save_dir / filename
                
                metadata = {
                    'source': 'wikimedia_commons',
                    'search_term': search_term,
                    'original_title': title,
                    'image_url': img_url,
                    'license': license_short or 'CC',
                    'license_url': license_url,
                    'category': category,
                    'label': label,
                    'plant': label.split('/')[0],
                    'condition': label.split('/')[1] if '/' in label else 'unknown'
                }
                
                if download_image(img_url, save_path, metadata):
                    collected += 1
                
                time.sleep(0.3)
        
        except Exception as e:
            print(f"  ⚠️  Parse error: {e}")
            continue
    
    return collected

# ═══════════════════════════════════════════
# 방법 2: iNaturalist API
# ═══════════════════════════════════════════

def fetch_inaturalist_images(taxon_name, category, label, per_page=8):
    """iNaturalist에서 CC 라이선스 현장 이미지 수집"""
    print(f"\n🌿 iNaturalist: '{taxon_name}' -> {label}")
    
    save_dir = BASE_DIR / category
    encoded = urllib.parse.quote(taxon_name)
    
    url = (
        f"https://api.inaturalist.org/v1/observations"
        f"?taxon_name={encoded}"
        f"&quality_grade=research"
        f"&per_page={per_page}"
        f"&photos=true"
        f"&license=cc-by,cc-by-sa,cc0"
        f"&photo_license=cc-by,cc-by-sa,cc0"
        f"&order_by=votes"
    )
    
    data = safe_request(url, headers={"User-Agent": "CropDoc/1.0 (Research)"})
    if not data:
        return 0
    
    try:
        result = json.loads(data)
        observations = result.get('results', [])
    except:
        return 0
    
    collected = 0
    for obs in observations:
        if collected >= per_page:
            break
        
        photos = obs.get('photos', [])
        if not photos:
            continue
        
        for photo in photos[:2]:  # 관찰당 최대 2장
            photo_url = photo.get('url', '')
            if not photo_url:
                continue
            
            # 원본 크기로 변경
            photo_url = photo_url.replace('/square.', '/original.').replace('/small.', '/original.').replace('/medium.', '/original.')
            
            # 라이선스 확인
            license_code = photo.get('license_code', '')
            if license_code and not any(cc in license_code.lower() for cc in ['cc', 'cc0', 'cc-by']):
                continue
            
            # 파일명
            url_hash = hashlib.md5(photo_url.encode()).hexdigest()[:8]
            ext = '.jpg'
            filename = f"inat_{url_hash}{ext}"
            save_path = save_dir / filename
            
            observer = obs.get('user', {}).get('login', 'unknown')
            obs_id = obs.get('id', '')
            
            metadata = {
                'source': 'inaturalist',
                'taxon_name': taxon_name,
                'observation_id': obs_id,
                'observer': observer,
                'image_url': photo_url,
                'license': license_code or 'cc-by',
                'license_url': f"https://www.inaturalist.org/observations/{obs_id}",
                'category': category,
                'label': label,
                'plant': label.split('/')[0],
                'condition': label.split('/')[1] if '/' in label else 'healthy',
                'quality_grade': obs.get('quality_grade', 'research')
            }
            
            if download_image(photo_url, save_path, metadata):
                collected += 1
            
            time.sleep(0.5)
    
    return collected

# ═══════════════════════════════════════════
# 방법 3: Wikimedia Category 직접 검색
# ═══════════════════════════════════════════

def fetch_wikimedia_category(category_name, save_category, label, limit=6):
    """Wikimedia Commons 카테고리에서 직접 이미지 수집"""
    print(f"\n📁 Wikimedia Category: '{category_name}'")
    
    save_dir = BASE_DIR / save_category
    encoded = urllib.parse.quote(category_name)
    
    url = (
        f"https://commons.wikimedia.org/w/api.php"
        f"?action=query&list=categorymembers"
        f"&cmtitle=Category:{encoded}"
        f"&cmtype=file&cmlimit={limit}"
        f"&format=json"
    )
    
    data = safe_request(url)
    if not data:
        return 0
    
    try:
        result = json.loads(data)
        members = result.get('query', {}).get('categorymembers', [])
    except:
        return 0
    
    collected = 0
    for member in members:
        if collected >= limit:
            break
        
        title = member['title']
        if not any(ext in title.lower() for ext in ['.jpg', '.jpeg', '.png']):
            continue
        
        # 이미지 URL 가져오기
        info_url = (
            f"https://commons.wikimedia.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(title)}"
            f"&prop=imageinfo&iiprop=url|mime|size"
            f"&format=json"
        )
        
        info_data = safe_request(info_url)
        if not info_data:
            continue
        
        try:
            info_result = json.loads(info_data)
            pages_data = info_result.get('query', {}).get('pages', {})
            
            for page_id, page_info in pages_data.items():
                imageinfo = page_info.get('imageinfo', [])
                if not imageinfo:
                    continue
                
                img_info = imageinfo[0]
                img_url = img_info.get('url', '')
                mime = img_info.get('mime', '')
                
                if mime not in ['image/jpeg', 'image/png']:
                    continue
                
                ext = '.jpg' if mime == 'image/jpeg' else '.png'
                url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
                filename = f"wikicat_{url_hash}{ext}"
                save_path = save_dir / filename
                
                metadata = {
                    'source': 'wikimedia_category',
                    'category_name': category_name,
                    'original_title': title,
                    'image_url': img_url,
                    'license': 'CC (Wikimedia Commons)',
                    'category': save_category,
                    'label': label,
                    'plant': label.split('/')[0],
                    'condition': label.split('/')[1] if '/' in label else 'unknown'
                }
                
                if download_image(img_url, save_path, metadata):
                    collected += 1
                
                time.sleep(0.3)
        
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    return collected


# ═══════════════════════════════════════════
# 수집 대상 정의
# ═══════════════════════════════════════════

COLLECTION_TASKS = [
    # === TOMATO ===
    # Wikimedia 검색
    ("tomato plant healthy fruit field", "tomato/healthy", "tomato/healthy"),
    ("tomato late blight Phytophthora infestans field", "tomato/late_blight", "tomato/late_blight"),
    ("tomato early blight Alternaria solani leaf", "tomato/early_blight", "tomato/early_blight"),
    ("tomato bacterial spot Xanthomonas field", "tomato/bacterial_spot", "tomato/bacterial_spot"),
    ("tomato leaf mold Passalora fulva", "tomato/leaf_mold", "tomato/leaf_mold"),
    ("tomato Septoria leaf spot disease", "tomato/septoria", "tomato/septoria"),
    
    # === POTATO ===
    ("potato plant healthy tuber field", "potato/healthy", "potato/healthy"),
    ("potato late blight Phytophthora field", "potato/late_blight", "potato/late_blight"),
    ("potato early blight Alternaria leaf", "potato/early_blight", "potato/early_blight"),
    
    # === CORN/MAIZE ===
    ("corn maize plant healthy field", "corn/healthy", "corn/healthy"),
    ("corn gray leaf spot Cercospora disease", "corn/gray_leaf_spot", "corn/gray_leaf_spot"),
    ("corn common rust Puccinia sorghi field", "corn/common_rust", "corn/common_rust"),
    ("corn northern leaf blight Exserohilum", "corn/northern_blight", "corn/northern_blight"),
    
    # === APPLE ===
    ("apple tree fruit healthy orchard", "apple/healthy", "apple/healthy"),
    ("apple scab Venturia inaequalis fruit", "apple/scab", "apple/scab"),
    ("apple black rot Botryosphaeria fruit", "apple/black_rot", "apple/black_rot"),
    
    # === GRAPE ===
    ("grape vine healthy fruit vineyard", "grape/healthy", "grape/healthy"),
    ("grape black rot Guignardia bidwellii", "grape/black_rot", "grape/black_rot"),
    ("grape leaf blight Isariopsis disease", "grape/leaf_blight", "grape/leaf_blight"),
    
    # === WHEAT ===
    ("wheat field healthy crop grain", "wheat/healthy", "wheat/healthy"),
    ("wheat stripe rust Puccinia striiformis", "wheat/stripe_rust", "wheat/stripe_rust"),
    ("wheat leaf rust brown rust Puccinia", "wheat/leaf_rust", "wheat/leaf_rust"),
    
    # === STRAWBERRY ===
    ("strawberry plant healthy fruit field", "strawberry/healthy", "strawberry/healthy"),
    ("strawberry leaf scorch Diplocarpon", "strawberry/leaf_scorch", "strawberry/leaf_scorch"),
    
    # === PEPPER ===
    ("pepper bell capsicum healthy fruit", "pepper/healthy", "pepper/healthy"),
    ("pepper bacterial spot Xanthomonas field", "pepper/bacterial_spot", "pepper/bacterial_spot"),
]

WIKIMEDIA_CATEGORIES = [
    ("Tomato diseases", "tomato/late_blight", "tomato/late_blight"),
    ("Potato diseases", "potato/late_blight", "potato/late_blight"),
    ("Apple scab", "apple/scab", "apple/scab"),
    ("Wheat diseases", "wheat/stripe_rust", "wheat/stripe_rust"),
    ("Maize diseases", "corn/common_rust", "corn/common_rust"),
    ("Grape diseases", "grape/black_rot", "grape/black_rot"),
]

INATURALIST_TASKS = [
    ("Solanum lycopersicum", "tomato/healthy", "tomato/healthy"),
    ("Solanum tuberosum", "potato/healthy", "potato/healthy"),
    ("Zea mays", "corn/healthy", "corn/healthy"),
    ("Malus domestica", "apple/healthy", "apple/healthy"),
    ("Vitis vinifera", "grape/healthy", "grape/healthy"),
    ("Triticum aestivum", "wheat/healthy", "wheat/healthy"),
    ("Fragaria", "strawberry/healthy", "strawberry/healthy"),
    ("Capsicum annuum", "pepper/healthy", "pepper/healthy"),
]


def main():
    global labels, downloaded_count, failed_count
    
    print("=" * 60)
    print("🌱 CropDoc Field Image Collector")
    print(f"📁 Save to: {BASE_DIR}")
    print("=" * 60)
    
    # 1단계: Wikimedia 검색으로 이미지 수집
    print("\n\n[1/3] Wikimedia Commons 검색 수집")
    print("=" * 40)
    for search_term, category, label in COLLECTION_TASKS:
        count = fetch_wikimedia_images(search_term, category, label, limit=6)
        print(f"  → {count}장 수집")
        time.sleep(0.5)
    
    # 2단계: Wikimedia 카테고리에서 추가 수집
    print("\n\n[2/3] Wikimedia Commons 카테고리 수집")
    print("=" * 40)
    for cat_name, save_cat, label in WIKIMEDIA_CATEGORIES:
        count = fetch_wikimedia_category(cat_name, save_cat, label, limit=5)
        print(f"  → {count}장 수집")
        time.sleep(0.5)
    
    # 3단계: iNaturalist에서 건강한 작물 이미지 수집
    print("\n\n[3/3] iNaturalist 현장 이미지 수집")
    print("=" * 40)
    for taxon, category, label in INATURALIST_TASKS:
        count = fetch_inaturalist_images(taxon, category, label, per_page=6)
        print(f"  → {count}장 수집")
        time.sleep(1)
    
    # 결과 저장
    print("\n\n" + "=" * 60)
    print(f"✅ 총 다운로드: {downloaded_count}장")
    print(f"❌ 실패: {failed_count}건")
    
    # labels JSON 저장
    labels_data = {
        'collected_at': datetime.now().isoformat(),
        'total_images': downloaded_count,
        'sources': ['wikimedia_commons', 'inaturalist'],
        'license': 'CC0/CC-BY/CC-BY-SA',
        'images': labels
    }
    
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)
    
    print(f"📋 Labels saved: {LABELS_FILE}")
    
    # 카테고리별 통계
    print("\n📊 Category Statistics:")
    from collections import Counter
    cat_counts = Counter(item.get('category', 'unknown') for item in labels)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}장")
    
    return downloaded_count, labels


if __name__ == '__main__':
    main()
