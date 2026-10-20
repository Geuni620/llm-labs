# Step 1. 다운로드

HuggingFace 에서 한국어 위키백과 parquet 파일을 로컬에 저장한다.

## 데이터 소스

- 데이터셋: `HuggingFaceFW/finewiki` (LLM 학습용으로 정제된 다국어 위키 코퍼스)
- 서브셋: `kowiki` (한국어 위키백과)
- parquet 2개: `000_00000.parquet`, `000_00001.parquet`

## 왜 `datasets` 대신 직접 다운로드?

`datasets` 라이브러리가 Python 3.14 에서 `dill` 호환성 문제로 작동하지 않아서,
`huggingface_hub.hf_hub_download` 로 parquet 파일을 직접 받는다.

## 흐름

```python
from huggingface_hub import hf_hub_download

FILES = [
    "data/kowiki/000_00000.parquet",
    "data/kowiki/000_00001.parquet",
]

for fname in FILES:
    hf_hub_download(
        repo_id="HuggingFaceFW/finewiki",
        filename=fname,
        repo_type="dataset",
        local_dir="parquet/",
    )
```

각 파일마다 다운로드 시간과 파일 크기를 출력한다.

## parquet 파일 구조

다운로드한 parquet 파일은 14개 컬럼을 가진다:

```
text           본문 텍스트 (클리닝 대상)
id             "kowiki/282586"
wikiname       "kowiki"
page_id        282586
title          "고야마못"
url            "https://ko.wikipedia.org/wiki/..."
date_modified  "2024-10-09T23:30:35Z"
in_language    "ko"
wikidata_id    "Q3214989"
bytes_html     44879
wikitext       위키 마크업 원본
version        37976232
infoboxes      JSON (인포박스 데이터)
has_math       False
```

이후 파이프라인에서는 `text` 컬럼만 사용한다.

## 입출력

```
입력: (인터넷) HuggingFace
출력: parquet/data/kowiki/000_00000.parquet  (1.1 GB, 310,045행)
      parquet/data/kowiki/000_00001.parquet  (1.0 GB, 285,041행)
```
