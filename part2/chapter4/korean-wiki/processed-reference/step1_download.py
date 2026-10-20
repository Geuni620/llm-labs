"""
한국어 위키백과 다운로드 스크립트 (HuggingFaceFW/finewiki).

`datasets` 라이브러리가 Python 3.14에서 dill 호환성 문제로 작동하지 않으므로,
huggingface_hub로 parquet 파일을 직접 다운로드한다.
parquet 내용을 분석/변환하는 작업은 별도 스크립트에서 수행.

데이터셋: HuggingFaceFW/finewiki (LLM 학습용으로 정제된 다국어 위키 코퍼스)
서브셋:   kowiki (한국어 위키백과)

저장 경로 (이 파일이 있는 디렉터리 기준):
    parquet/data/kowiki/    - 원본 parquet 파일
"""

import os
import time

from huggingface_hub import hf_hub_download

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")

REPO_ID = "HuggingFaceFW/finewiki"
FILES = [
    "data/kowiki/000_00000.parquet",
    "data/kowiki/000_00001.parquet",
]

os.makedirs(PARQUET_DIR, exist_ok=True)

print("=" * 60)
print("Korean FineWiki download (parquet only)")
print("=" * 60)

print(f"\nDownloading {len(FILES)} parquet files -> {PARQUET_DIR}")
for i, fname in enumerate(FILES):
    t0 = time.time()
    print(f"  ({i + 1}/{len(FILES)}) {fname} ...")
    p = hf_hub_download(
        repo_id=REPO_ID,
        filename=fname,
        repo_type="dataset",
        local_dir=PARQUET_DIR,
    )
    sz = os.path.getsize(p) / 1024 / 1024
    print(f"        done in {time.time() - t0:.1f}s, {sz:.1f} MB")

print("\n" + "=" * 60)
print("All done.")
print("=" * 60)
