# 가공본 해제 폴더 정리 기록

원본 Parquet과 가공본 압축 파일은 상위 `korean-wiki/`에 보관합니다.
이 디렉토리는 이전 압축 해제본에서 보존한 설명·전처리 코드·라이선스·검토 기록입니다.

## 확인한 동일성과 보존 범위

- 원본 데이터 출처: `HuggingFaceFW/finewiki`, revision `8bd13e72e6a002407649b3e898535f42ceb1aeb9`.
- 보관한 원본 Parquet 2개는 다운로드 기록 및 기존 출처 기록의 SHA-256과 일치했습니다.
- 보관한 `ko_wiki_cleaning.7z`는 Downloads의 제공 원본과 SHA-256이 일치했습니다.
- 압축 안의 337개 항목과 기존 해제본의 경로·크기가 모두 일치했습니다. 비어 있지 않은 335개는 CRC32도 재계산해 대조했고, 빈 2개는 0 bytes로 확인했습니다.
- 중복 해제 파일 306개, 총 9,072,041,118 bytes를 제거했습니다.
- 기존 문서·코드·검토 파일 37개는 내용을 보존해 이 디렉토리로 옮겼습니다. 압축에 없는 추가 검토 파일 6개도 모두 포함합니다.
- 기존 `part2/chapter4/datasets/korean_wiki/`는 제거했습니다. 원본 Parquet과 압축 파일은 유지합니다.

전체 파일별 검증과 보존 파일의 SHA-256은 [extracted-copy-check.json](extracted-copy-check.json)에 기록했습니다.

## 이전 검토 기록을 읽는 방법

[DATASET_INFO.md](DATASET_INFO.md), [PROVENANCE.json](PROVENANCE.json),
[archive_manifest.json](archive_manifest.json), [중복 검사 기록](audit/duplicates.json)은 이전 해제본을 검토한 당시의 기록입니다.
이 문서들에 적힌 이전 로컬 경로나 파일 존재 여부를 현재 상태로 해석하지 않습니다.

원래 전처리 코드를 실행하려면 압축을 다시 해제해 필요한 입력과 원래 디렉토리 구조를 준비해야 합니다.
현재 보관한 원본 Parquet의 정보는 [다운로드 기록](../download_manifest.json)을 참고합니다.
