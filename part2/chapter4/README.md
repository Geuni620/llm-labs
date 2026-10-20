# 챕터 4 — 신경망 언어 모델

실습별 디렉토리를 나누어 데이터와 구현, 실험 기록을 함께 관리합니다.

## 학습 순서와 실습

1. [toy-lm — 여섯 문장으로 비교하는 트라이그램과 NPLM](toy-lm/README.md): 작은 한국어 데이터로 횟수 기반 확률과 신경망의 순전파·역전파를 직접 구현합니다.
2. [학습 데이터 확인](toy-lm/data/README.md): 철수는·영희는·짱구는을 주어로 하는 문장 6개와 학습 쌍 24개를 확인합니다.
3. [학습 회고](toy-lm/docs/learning-journey.md): 이번 세션에서 논의한 임베딩, loss, 학습률, 스무딩, PPL, 생성과 평가의 차이를 살펴봅니다.

## toy-lm 실행

프로젝트 루트에서 시작합니다. Node.js 24.12 이상이 필요합니다.

```sh
cd part2/chapter4/toy-lm
npm ci
npm run data
npm run typecheck
npm run gradcheck
```

실험과 학습 재생은 실습 README에서 이어집니다. toy-lm의 모델 학습 데이터는 원래 여섯 문장으로 유지합니다.

## 한국어 위키백과 데이터셋

`korean-wiki/`에 원본 Parquet과 제공받은 `ko_wiki_cleaning.7z`를 보관합니다.
이전 이름 `korea-wiki`를 `korean-wiki`로 통일했습니다. 출처는
`HuggingFaceFW/finewiki`의 한국어 데이터이며, 고정 revision과 실제 파일 해시는
[다운로드 기록](korean-wiki/download_manifest.json)에서 확인합니다.

기존 `datasets/korean_wiki/`는 같은 압축 파일을 해제한 가공본이었습니다.
337개 파일의 크기·CRC 검증 후 중복 데이터 약 9.07GB를 제거했습니다.
기존 문서·전처리 코드·검토 기록 37개는 [processed-reference](korean-wiki/processed-reference/ARCHIVE_STATUS.md)에 보존했습니다.

`toy-lm/`의 `data`·`compare`·`animate`는 원래 6문장 예제를 계속 사용합니다.
