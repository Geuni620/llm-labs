"""강의 Step 1: Hugging Face에서 한국어 위키 원본 Parquet 두 개 내려받기.

2025년 8월 스냅샷을 제공하는 revision을 고정한다.
실행: uv run --no-project --with huggingface_hub python step1_download.py
"""

from pathlib import Path
import time

from huggingface_hub import hf_hub_download


REPO_ID = "HuggingFaceFW/finewiki"
REVISION = "8bd13e72e6a002407649b3e898535f42ceb1aeb9"
PARQUET_DIR = Path(__file__).resolve().parent / "parquet"
FILES = [
    "data/kowiki/000_00000.parquet",
    "data/kowiki/000_00001.parquet",
]


def main():
    PARQUET_DIR.mkdir(exist_ok=True)
    print(f"데이터셋: {REPO_ID}", flush=True)
    print(f"고정 revision: {REVISION}", flush=True)
    for index, filename in enumerate(FILES, 1):
        start = time.monotonic()
        print(f"[{index}/{len(FILES)}] 다운로드 시작: {filename}", flush=True)
        downloaded = Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=filename,
            local_dir=PARQUET_DIR,
        ))
        print(f"완료: {downloaded} ({downloaded.stat().st_size:,} bytes, "
              f"{time.monotonic() - start:.1f}초)", flush=True)
    print("원본 Parquet 다운로드 완료", flush=True)


if __name__ == "__main__":
    main()
