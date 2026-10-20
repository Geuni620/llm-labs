"""
FineWiki parquet에서 앞쪽 N개 행을 추출해 사람이 읽기 좋은 형태로 저장.

- 모든 14개 컬럼 보존 (text만 추출하지 않음)
- 짧은 메타 필드: `key: value` 리스트
- 긴 텍스트 필드(text, wikitext, infoboxes): `--- <col> ---` 섹션으로 분리
- 원본 데이터 수동 검토용
"""

import os
import sys

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# (parquet 파일, 앞에서 뽑을 행 수)
FILES = [
    ("parquet/data/kowiki/000_00000.parquet", 10),
    ("parquet/data/kowiki/000_00001.parquet", 5),
]
OUTPUT = "sample_15rows.txt"

# 짧은 메타 필드 (한 줄 key: value)
META_COLS = [
    "id", "wikiname", "page_id", "title", "url", "date_modified",
    "in_language", "wikidata_id", "bytes_html", "version", "has_math",
]
# 긴 텍스트 필드 (별도 블록)
LONG_COLS = ["text", "wikitext", "infoboxes"]

ALL_COLS = META_COLS + LONG_COLS
TOTAL = sum(n for _, n in FILES)

out_path = os.path.join(DATA_DIR, OUTPUT)
row_num = 0
with open(out_path, "w", encoding="utf-8") as out:
    for fpath, n in FILES:
        abs_fpath = os.path.join(DATA_DIR, fpath)
        pf = pq.ParquetFile(abs_fpath)
        batch = next(pf.iter_batches(batch_size=n, columns=ALL_COLS))
        for i in range(len(batch)):
            row_num += 1
            out.write("=" * 80 + "\n")
            out.write(
                f"=== Row {row_num}/{TOTAL}   "
                f"source: {os.path.basename(fpath)}   local index: {i}\n"
            )
            out.write("=" * 80 + "\n\n")
            for col in META_COLS:
                v = batch.column(col)[i].as_py()
                out.write(f"{col:<15}: {v}\n")
            out.write("\n")
            for col in LONG_COLS:
                v = batch.column(col)[i].as_py()
                out.write(f"--- {col} ---\n")
                if v is None:
                    out.write("(None)\n")
                else:
                    s = str(v)
                    out.write(s)
                    if not s.endswith("\n"):
                        out.write("\n")
                out.write("\n")
            out.write("\n")

sz = os.path.getsize(out_path)
print(f"Wrote {OUTPUT} ({sz:,} bytes, {sz / 1024:.1f} KB, {row_num} rows)")
