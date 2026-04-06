#!/usr/bin/env python3
"""
CropDoc 확장 데이터셋 다운로더
신규 16종 질병 데이터 수집용
"""
import os, ssl, glob, shutil, json
from pathlib import Path

# 환경 설정
os.environ['KAGGLE_USERNAME'] = 'nothink_ivan'
os.environ['KAGGLE_KEY'] = 'os.environ.get('KAGGLE_KEY', '')'
ssl._create_default_https_context = ssl._create_unverified_context

# Kaggle 자격증명 파일
os.makedirs(os.path.expanduser('~/.config/kaggle'), exist_ok=True)
cred = {"username": "nothink_ivan", "key": "os.environ.get('KAGGLE_KEY', '')"}
with open(os.path.expanduser('~/.config/kaggle/kaggle.json'), 'w') as f:
    json.dump(cred, f)
os.chmod(os.path.expanduser('~/.config/kaggle/kaggle.json'), 0o600)

import kagglehub

PROJECT_DIR = Path("/root/.openclaw/workspace/companies/Hackathon/projects/gemma4good")
OUTPUT_DIR = PROJECT_DIR / "data" / "extended_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 다운로드 대상 데이터셋 (검색 결과 기반 선정)
DATASETS = [
    # Coffee (195MB) - 가장 많은 클래스 보유
    ("badasstechie/coffee-leaf-diseases", "Coffee"),
    # Rice (36MB) - 핵심 Rice 질병
    ("vbookshelf/rice-leaf-diseases", "Rice"),
    # Rice (175MB) - 추가 Rice 데이터
    ("jay7080dev/rice-plant-diseases-dataset", "Rice"),
    # Rice (4MB) - 소형 추가
    ("thegoanpanda/rice-crop-diseases", "Rice"),
    # Mango (103MB) - 주요 Mango 데이터셋
    ("aryashah2k/mango-leaf-disease-dataset", "Mango"),
    # Mango (58MB) - 추가
    ("shuvokumarbasak4004/the-mango-leaf-disease-classification-dataset", "Mango"),
    # Mango (34MB) - 추가
    ("ahmadzargar/mango-leaf-diseases-dataset", "Mango"),
    # Banana (36MB) - 바나나 질병
    ("shifatearman/bananalsd", "Banana"),
    # Banana (24MB) - 추가
    ("harshitbana/banana-leaf-disease", "Banana"),
    # Banana (121MB) - 추가
    ("sujaykapadnis/banana-disease-recognition-dataset", "Banana"),
    # Citrus (41MB) - 감귤 질병
    ("myprojectdictionary/citrus-leaf-disease-image", "Citrus"),
    # Citrus (350MB) - 추가
    ("jonathansilva2020/dataset-for-classification-of-citrus-diseases", "Citrus"),
    # Wheat - 밀 질병 (별도 검색 필요)
    # Cassava - 카사바 (대회 데이터)
]

# Wheat 전용 검색 결과 기반 추가
WHEAT_DATASETS = [
    ("henshengcheng/wheat-diseases-dataset", "Wheat"),
    ("kushagrapandya/wheat-disease-dataset", "Wheat"),
    ("mathiasbendombo/wheat-leaf-disease-classification-dataset", "Wheat"),
    ("oluwafemidiakhoa/wheatdiseasedataset", "Wheat"),
]

# Cassava 추가
CASSAVA_DATASETS = [
    ("nirmalsankalana/cassava-leaf-disease-dataset", "Cassava"),
    ("gauravduttakiit/cassava-leaf-disease-classification", "Cassava"),
]

ALL_DATASETS = DATASETS + WHEAT_DATASETS + CASSAVA_DATASETS

downloaded = {}
failed = {}

print(f"=== 총 {len(ALL_DATASETS)}개 데이터셋 다운로드 시작 ===\n")

for ds_ref, crop_type in ALL_DATASETS:
    print(f"📥 다운로드: {ds_ref} ({crop_type})")
    try:
        path = kagglehub.dataset_download(ds_ref)
        # 이미지 카운트
        total_imgs = (
            len(glob.glob(f"{path}/**/*.jpg", recursive=True)) +
            len(glob.glob(f"{path}/**/*.JPG", recursive=True)) +
            len(glob.glob(f"{path}/**/*.jpeg", recursive=True)) +
            len(glob.glob(f"{path}/**/*.JPEG", recursive=True)) +
            len(glob.glob(f"{path}/**/*.png", recursive=True)) +
            len(glob.glob(f"{path}/**/*.PNG", recursive=True))
        )
        # 서브 디렉토리 (클래스) 탐색
        all_items = list(Path(path).rglob("*"))
        subdirs = [d for d in all_items if d.is_dir()]
        
        downloaded[ds_ref] = {
            'path': path,
            'crop': crop_type,
            'images': total_imgs,
            'dirs': len(subdirs)
        }
        print(f"  ✅ {total_imgs}장, {len(subdirs)}폴더 → {path}")
    except Exception as e:
        failed[ds_ref] = str(e)
        print(f"  ❌ {e}")

print(f"\n=== 다운로드 결과 ===")
print(f"성공: {len(downloaded)}개, 실패: {len(failed)}개")

# 결과 저장
result = {'downloaded': downloaded, 'failed': failed}
with open(PROJECT_DIR / "data" / "download_results.json", 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\n결과 저장: data/download_results.json")
