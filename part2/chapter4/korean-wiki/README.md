# 한국어 위키백과 전처리 실습

`HuggingFaceFW/finewiki`의 **2025년 8월 한국어 위키백과 스냅샷**을 직접 처리합니다.
원본 Parquet, 확인용 TXT, 정제 결과를 나누어 보관합니다.
제공받은 `ko_wiki_cleaning.7z`와 `processed-reference/`는 강의 기준을 확인하는 참고 자료입니다.

2026-10-18 사용자 결정으로 **실제 모델 학습·시간별 결과 비교는 다음 챕터로 미뤘습니다.**
전체 전처리, Kiwi 토큰화, 숫자 자릿수 분리, 빈도 집계와 보존율 분석은 완료했습니다.
실행 중이던 학습은 체크포인트를 저장하고 정상 중단했으며, 자동 확인 예약도 삭제했습니다.
원본·전체 처리 결과·학습 입력·가중치는 보존했습니다. 3시간·8시간 비교는 아직 완료하지 않았습니다.
중단 근거와 재개용 체크포인트 위치는 [다음 챕터 인계 기록](reports/chapter4-handoff.json)에 있습니다.

## 단계와 결과 위치

사용자 요청 전체의 완료 조건·현재 상태·결과 위치는 [작업 체크리스트](reports/task-checklist.json)에 기록합니다.
샘플/부분 결과, 코드 구현, 전체 데이터 처리, 실제 모델 학습, 결과 전달을 각각 구분하며 최종 보고 전 전 항목을 점검합니다.
전체 적용 순서는 [순서 정리 TXT](reports/applied-steps.txt), 실제 토큰과 ID의 대응은
[정수 인코딩 예시 TXT](reports/integer-encoding-example.txt)에서 볼 수 있습니다.

| 단계 | 입력 → 출력 | 위치 |
| --- | --- | --- |
| 1. 다운로드 | 14개 열을 가진 Parquet 2개, 실제 문서 582,035개 | `parquet/data/kowiki/`, `download_manifest.json` |
| 2. 원문 확인 | 5개 문서의 `text`만 UTF-8 TXT로 추출. 내용은 원본과 동일 | `examples/*.text.txt` |
| 3. 문단·문장 추출 | 원본 전체에 8개 필터 적용 → 2,618,729개 문단 → 7,099,328개 문장 | `paragraphs_raw/`, `paragraphs/`, `sentences/` |
| 4. 전역 중복 제거 | 421,696회 중복 제외(5.94%) → 6,677,632개 문장 | `sentences_dedup/`, `sentences_dedup_removed.txt` |
| 5. 형태소·품사 토큰화 | 중복 제거 후 6,677,632개 문장·2,496,587개 문단 → 292,363,011개 형태소/품사·공백 토큰 | `tokens/`, `reports/step5-summary.json` |
| 6. 전체 토큰 빈도 집계 | 전체 1,570,824종을 등장 횟수 내림차순으로 집계 | `token_frequencies.txt`, `reports/token-frequency-summary.json` |
| 7. 숫자 자릿수 분리 | SN·W_SERIAL 분리 → 302,281,587개 토큰·1,484,881종 | `tokens_numeric/`, `token_frequencies.numeric.txt` |
| 8. 사전 크기와 자원 분석 | 전체 보존 곡선·CPU/MPS 측정 → K=35,328·CPU 후보 | `reports/vocabulary-coverage.json`, `reports/nplm-benchmark.json` |
| 9. 문서별 자료 분리 | 원문 문서 연결 복원 → train/validation/test 분리, 평가 중복 제외 | `training-corpora/wiki-k35328/provenance/`, `training-corpora/wiki-k35328/manifest.json` |
| 10. 최종 사전과 OOV 필터 | train만으로 35,328종 사전 → 학습 문단 1,529,599개·문장 2,977,860개 | `training-corpora/wiki-k35328/vocabulary.json`, `training-corpora/wiki-k35328/train-token-frequencies.txt` |
| 11. 정수 인코딩과 NumPy 저장 | 고정 ID로 변환 → 1차원 uint16 배열 3개, 문단 경계 별도 보존 | `training-corpora/wiki-k35328/*.ids.npy`, `training-corpora/wiki-k35328/*.paragraphs.npy`, `npy_corpus.py` |
| 12. 독립 검증과 인계 | 전체 토큰·빈도 검증, NumPy 전체 ID 일치 확인, 학습 중단·보존 | `reports/integer-encoding-npy.json`, `reports/chapter4-handoff.json` |

2단계의 5개 문서는 사람이 형식을 확인하기 위한 샘플입니다. **전체 처리의 입력은 1단계 원본 582,035개 문서**입니다.
문단은 원본 `text`의 한 줄을 후보로 삼으며, 여러 문장을 포함할 수 있습니다.
각 전체 결과 디렉토리는 강의와 같은 배치 순서의 `part_00.txt`~`part_31.txt`로 나눕니다.
빈 줄은 문단 경계를 나타냅니다. 강의 압축본과 바이트 단위 비교가 가능하도록 전체 출력의 줄바꿈은 CRLF입니다.

## 8개 필터

아래 순서로 검사하며 **처음 탈락한 사유만** 집계합니다.

1. `has_formula`: 실제 문자열 `{\displaystyle`가 들어간 **문서 전체**를 제외합니다. 본문을 직접 검사합니다.
2. `is_structural`: 빈 행, `#` 제목, `|` 표, `- `·`* ` 목록, `1. ` 같은 번호 목록 행을 제외합니다.
3. `has_math_symbol`: `±∓≠≤≥∈∪∑√∞` 중 하나가 있는 행을 제외합니다. 일반적인 단위나 곱셈 기호까지 추가로 확대하지 않습니다.
4. `hanja_ratio`: 한자 / (한자 + 한글) 비율이 30%를 넘는 행을 제외합니다. 괄호 정리 전 계산합니다.
5. `clean_paragraph`: 괄호 안에 한글 이외의 알파벳 문자가 있으면 그 괄호를 제거합니다. 한글·숫자로 된 괄호는 보존합니다. 중첩 괄호는 안쪽부터 처리하며 공백·연속 마침표도 정리합니다. 정리 후 비어 버린 행만 이 단계의 제외 수로 셉니다.
6. `letter_ratio`: `isalpha()` 문자 수 / 전체 문자 수가 20% 미만이면 제외합니다.
7. `hangul_ratio`: 한글 수 / 전체 문자 수가 30% 미만이면 제외합니다.
8. `is_sentence_shape`: 정리 후 2자 이상이고 `.`, `?`, `!`로 끝나는 행만 남깁니다.

통과한 문단을 `.?!` 다음 공백에서 나누되 `D. Watson`, `J. D. Watson`처럼 대문자 이니셜 뒤에서 나뉜 조각을 합칩니다. `3.14` 같은 소수도 보존합니다.

```text
서울(Seoul)은 대한민국의 수도 (약 100만명)이다.
→ 서울은 대한민국의 수도 (약 100만명)이다.

1953년 제임스 D. Watson은 DNA를 연구했다. 다음 문장이다.
→ 1953년 제임스 D. Watson은 DNA를 연구했다.
→ 다음 문장이다.
```

## 중복 기준과 통계 해석

NFKC 정규화와 공백 정리 후 **완전히 일치하는 문장**을 전체 32개 파일에 걸쳐 한 번만 남깁니다.
첫 번째 문장의 원래 표기는 보존하며 영문 대소문자는 구분합니다. `sentences_dedup_removed.txt`의 `[Nx]`는 처음 보존한 1회를 포함한 총 등장 횟수입니다.

문서·본문행·분리된 문장은 서로 다른 단위입니다. 수식 필터는 전체 문서 582,035개를 분모로, 구조 필터는 수식 문서 제외 후 전체 행을 분모로 셉니다.
나머지 필터의 제외율은 구조 행 제거 후 본문행 3,227,444개가 분모입니다. 서로 다른 분모의 비율을 합산하지 않습니다.
원문행 전체 기준 집계에는 빈 행·제목·목록·표, 수식 문서 안의 행까지 포함됩니다.
괄호·공백이 바뀐 행은 뒤 필터에서 제외될 수도 있으므로 ‘변경’과 ‘제외’를 합산하지 않습니다.

이 처리는 강의의 규칙을 재현합니다. 문법적 완성도, 의미가 같은 문장의 중복 제거, 주제별 분포 균등까지 보장하는 기준은 아닙니다.

## 실행

`uv`와 Python을 사용합니다. 이 디렉토리에서 실행합니다.
이미 만들어진 전체 결과가 있으면 덮어쓰지 않고 중단합니다.

```sh
# 원본 다운로드: 고정 revision은 step1_download.py 참고
uv run --no-project --with huggingface_hub python step1_download.py

# 3단계: 전체 처리와 5개 원문 샘플의 전후 비교 자료
uv run --no-project --with pyarrow python step3_extract_paragraphs.py --workers 4

# 4단계: 파일 간 중복까지 제거
python3 step4_dedup_sentences.py

# 5개 샘플의 변경 내역만 다시 만들기
uv run --no-project --with pyarrow python step3_extract_paragraphs.py --samples-only
```

현재 확인용 원문 TXT 5개가 이미 있으므로 샘플 분석은 이 파일들과 원본의 일치도 함께 검사합니다.

## 검증과 읽을 자료

```sh
uv run --no-project --with pyarrow python -m unittest discover -s tests -v
uv run --no-project --with py7zr python verify_preprocessing.py
uv run --no-project --with pyarrow python build_preprocessing_report.py
```

- [필터별 실제 사례·제거 수·제거율](reports/preprocessing-summary.txt)
- [5개 문서의 1·2·3단계 전체 비교](reports/step1-3-examples.txt)
- [기계 판독 가능한 종합 기록](reports/preprocessing-results.json)
- [원본 보존·참고 압축본 출력 비교·잔여 중복 검증](reports/verification.json)
- `reports/examples/`: 문서별 원본행 처리 사유와 8단계 중간 결과, 최종 문단·문장 TXT

테스트는 괄호·이니셜·경계값과 전역 중복 처리를 확인하고, 강의 원본 코드와 샘플 5개·첫 1,000개 문서의 출력을 비교합니다.
전체 검증은 새로 생성한 129개 파일의 크기·CRC32를 강의 압축본 헤더와 비교한 뒤, 별도 SHA-256 기준으로 최종 문장의 잔여 중복을 다시 셉니다.

## Kiwi 형태소·품사 토큰과 전체 빈도

5단계는 중복 제거 후의 문장을 Kiwi `cong-global`로 분석해 `형태소/품사` 문자열로 저장합니다.
Kiwi와 모델 패키지는 모두 `0.24.0`으로 고정하며, 실행한 Python·CPU 구조·모델 파일 해시도 결과에 기록합니다.

```sh
uv venv --python 3.13 .venv-kiwi
uv pip install --python .venv-kiwi/bin/python -r requirements-tokenize.lock

# 실제 1,024개 문장과 별도 설명용 예시 먼저 확인
.venv-kiwi/bin/python step5_tokenize.py --samples-only --workers 8

# 전체 토큰화: 완료된 파일부터 tokens/에 저장
.venv-kiwi/bin/python step5_tokenize.py --workers 8 --batch-size 1024

# 중단한 실행의 완료된 파일을 검증하고 이어서 처리
.venv-kiwi/bin/python step5_tokenize.py --workers 8 --batch-size 1024 --resume

# 전체 문장·문단 경계와 토큰 형식 검사, 파일별 표본 복원 검사
.venv-kiwi/bin/python verify_tokenization.py
```

`tokens/part_00.txt`~`part_31.txt`에서 한 줄은 한 문장의 토큰열이고 빈 줄은 문단 경계입니다.
숫자는 `1987/SN`처럼 원래 숫자를 남깁니다. `<sp>/WS`는 강의 코드가 추가하는 공백 토큰입니다.
같은 문단의 다음 문장과 연결될 공백도 토큰으로 보존합니다.

형태소는 단어보다 작은 단위이므로 조사·어미도 각각의 토큰입니다.
자동차의 차와 마시는 차는 둘 다 `차/NNG`이며, 동사 차다는 `차/VV`로 구별됩니다.
현재 저장 형식은 형태소와 품사를 담고, 서로 다른 의미까지 별도의 토큰으로 나누지는 않습니다.

토큰을 `Kiwi.join()`으로 복원하면 원형 복원·축약·조사 결합 등의 영향으로 원문과 달라질 수 있습니다.
원문은 `sentences_dedup/`에 보존하고, 실제 차이는 표본 보고서에 기록합니다.
표본 복원의 원문 일치율은 문장 분석 정확도와 다른 지표입니다.

- [실제 토큰화 전후 예시와 복원 차이](reports/step5-examples.txt)
- [토큰화 표본 전체 기록](reports/step5-samples.json)
- `reports/step5-progress.json`: 실행 중인 파일과 누적 처리 문장 수
- `reports/step5-summary.json`: 전체 완료 후 토큰·문장·문단 수
- `reports/step5-verification.json`: 전체 파일 구조·집계 검증과 표본 복원 결과

6단계의 빈도표는 학습 입력의 구성을 확인하는 자료입니다.
키는 `형태소/품사` 전체이며, 조사·어미·공백·숫자·구두점까지 실제 토큰을 모두 셉니다.
같은 횟수의 토큰은 문자열 순서로 정렬해 결과를 일정하게 유지합니다.
빈도 순위는 이 보고서에서의 순위이며, 학습용 토큰 ID와는 별개입니다.

```sh
# 토큰화 중: 완료된 파일부터 빈도를 집계하고 전체 완료까지 이어서 처리
.venv-kiwi/bin/python step6_count_tokens.py --watch

# 전체 완료 후 빈도표 검증과 순위·등장 횟수 그래프 생성
.venv-kiwi/bin/python verify_token_frequencies.py
.venv-kiwi/bin/python build_token_frequency_plot.py
```

`token_frequencies.txt`에는 `rank`, `token`, `count`, `percent`, `cumulative_percent`를 탭으로 구분해 저장합니다.
빈도 내림차순으로 전체 종류를 포함합니다. `reports/token_frequencies.partial.all.txt`는 완료된 파일만 집계한 전체 종류 목록이며,
`reports/token_frequencies.partial.txt`는 그중 상위 1,000개를 빠르게 확인하는 용도입니다. 각 파일 머리말에 반영 범위를 표시합니다.

그래프는 X축을 토큰 순위, Y축을 등장 횟수로 표시하며 일반·로그 눈금을 선택할 수 있습니다.
빈도 1인 토큰은 종류 수 기준 비율과 전체 등장 횟수 기준 비율을 따로 표시합니다.
같은 빈도의 연속된 순위를 묶어 저장하므로 전체 순위의 빈도를 정확히 조회할 수 있습니다.

## 숫자 분리와 어휘 크기 비교

강의 6a처럼 `SN`·`W_SERIAL`의 숫자와 구분자를 한 글자씩 나눈 결과를 `tokens_numeric/`에 별도로 저장합니다.
원래 `tokens/`와 그 빈도표는 유지합니다.

```text
2010/SN       → 2/SN 0/SN 1/SN 0/SN
2-1/W_SERIAL  → 2/SN -/SW 1/SN
996,000/SN    → 9/SN 9/SN 6/SN ,/SW 0/SN 0/SN 0/SN
```

이는 숫자의 사전 항목 수를 줄이고 새로운 숫자도 조합으로 표현하는 처리입니다.
숫자의 의미·연산 능력이나 암기 방지는 학습과 검증에서 따로 확인해야 합니다.
일반 형태소·품사, 문장·문단 경계, 공백 토큰은 그대로 유지합니다.

```sh
.venv-kiwi/bin/python step6_split_numbers.py --watch
.venv-kiwi/bin/python verify_numeric_tokens.py
.venv-kiwi/bin/python analyze_vocabulary_coverage.py --workers 4
.venv-kiwi/bin/python build_vocabulary_coverage_plot.py
```

최종 숫자 분리 빈도표는 `token_frequencies.numeric.txt`입니다.
전체 32개 파일의 집계에서 토큰 등장량은 292,363,011개 → 302,281,587개,
고유 토큰 종류는 1,570,824개 → 1,484,881개(85,943개, 5.47% 감소)입니다.
숫자 관련 원본 85,958종은 자릿수·구분자 16종으로 표현하며 문장·문단·공백 수는 유지합니다.
집계 완료와 독립 검증 완료는 구분하며, 최종 검증 여부는 `reports/step5-verification.json`,
`reports/token-frequency-verification.json`, `reports/step6-numeric-verification.json`에서 확인합니다.
어휘 비교의 K는 `<pad>`, `<unk>`, `<bos>`, `<eos>` 4개를 포함한 크기입니다.
K=65,536이면 일반 토큰은 빈도 상위 65,532개를 사용합니다. 경계의 동률도 문자열 순서로 정확히 자릅니다.

보존율의 네 기준을 구분합니다.

- 토큰 커버리지: 선택한 어휘가 전체 등장 횟수 중 차지하는 비율
- 문장 유지율: 어휘 밖 토큰이 하나도 없는 문장의 비율
- 강의식 문단 유지율: OOV 문장을 제거한 뒤 한 문장이라도 남는 문단의 비율
- 문단 전체 보존율: 원래 문단의 모든 문장이 남는 문단의 비율

`reports/vocabulary-coverage.json`은 원본/숫자 분리 표현을 같은 입력 범위에서 비교합니다.
중간 분석은 이름에 `.partial`이 붙으며 전체 결과로 취급하지 않습니다.
급변 구간(knee)과 유지율별 최소 K는 데이터·사전 크기의 절충 후보이며, 모델 성능의 최적값을 증명하지 않습니다.

## 학습 자원 측정과 이후 비교

`benchmark_nplm.py`는 별도의 PyTorch 미니배치 NPLM으로 CPU·MPS의 계산 속도와 메모리를 측정하는 도구입니다.
작은 합성 입력의 실행 확인과 실제 말뭉치의 학습 품질 평가는 구분합니다.
전처리·분석이 실행 중이면 성능 측정을 차단해 자원 경쟁이 측정값을 왜곡하지 않도록 합니다.

비교 실험은 같은 데이터·사전 조건에서 8시간 학습 중 3시간 체크포인트를 저장하는 방식으로 준비합니다.
사전 크기, 데이터 양, 학습 시간을 따로 비교하고 검증 자료·시드·생성 조건을 고정해야 합니다.
문단 수와 함께 학습 토큰·예측 횟수·반복 횟수를 기록하며, 속도에서 계산한 시간 예산은 실제 학습 결과와 구분합니다.
세부 조건은 [학습 비교 계획](reports/training-comparison-plan.txt)에 기록합니다.

## 정수 인코딩과 NumPy 파일

숫자 분리를 마친 토큰을 **35,328종 사전의 고정 ID**로 바꾼 정수 배열도 저장했습니다.
사전은 학습 자료로만 만들었으며, 사전 밖 토큰이 있는 문장은 기존 규칙에 따라 제외한 결과입니다.
`vocabulary.json`의 `tokens[id]`로 해당 ID의 형태소/품사 토큰을 찾을 수 있습니다.

- 사전: `training-corpora/wiki-k35328/vocabulary.json`
- 학습: `training-corpora/wiki-k35328/train.ids.npy` — 136,469,327개 ID
- 검증: `training-corpora/wiki-k35328/validation.ids.npy` — 1,417,621개 ID
- 테스트: `training-corpora/wiki-k35328/test.ids.npy` — 1,371,753개 ID

세 배열은 모두 **1차원 `uint16`**입니다. 문단의 시작·길이·원문 문서 연결 정보는 같은 디렉토리의
`train.paragraphs.npy` 등에서 찾을 수 있습니다. 각 문단 앞에는 문맥용 BOS 8개, 끝에는 EOS 1개를 저장합니다.
문장들은 기존 공백·문장부호를 유지해 연결되며, 문장마다 EOS를 넣는 형식은 아닙니다.
학습 배열 136,469,327개 ID에는 BOS 12,236,792개가 포함됩니다. 이를 제외한 예측 타깃은
124,232,535개이며, 타깃에는 문단 끝 EOS가 포함됩니다.
NumPy 형식의 헤더에 자료형·배열 크기가 들어 있어 다음처럼 읽을 수 있습니다.

```python
import json
import numpy as np
from pathlib import Path

corpus = Path("training-corpora/wiki-k35328")
ids = np.load(corpus / "train.ids.npy", mmap_mode="r", allow_pickle=False)
vocabulary = json.loads((corpus / "vocabulary.json").read_text())["tokens"]
print(ids[:24])
print([vocabulary[int(index)] for index in ids[:24]])
```

`mmap_mode="r"`을 사용하면 전체 배열을 한 번에 RAM에 복사하지 않고 필요한 구간을 읽습니다.
형식 변환은 `export_integer_npy.py`로 수행했으며, **세 배열의 모든 정수 값이 기존 `.ids.u16`과 동일**한지
비교했습니다. 근거는 `reports/integer-encoding-npy.json`에 있습니다. 기존 정수 파일·사전·원본 manifest는 보존합니다.
훈련 코드에 컨텍스트·다음 토큰 타깃을 공급할 `.npy` 전용 로더도 추가했습니다.
사전·원본 manifest·내보내기 보고서의 연결과 배열·문단 경계를 검증하며 다음처럼 사용합니다.

```python
from npy_corpus import NpyArrayCorpus
from training_corpus import WindowSampler

corpus = NpyArrayCorpus("training-corpora/wiki-k35328", "train", 8)
sampler = WindowSampler(corpus.windows, seed=42)
contexts, targets, paragraph_ids = corpus.batch(sampler.take(128))
```

`npy_corpus.py`는 기존 배치 생성 로직을 공유합니다. 학습·검증·테스트와 10%·25%·100% 부분집합에서
같은 window의 컨텍스트·타깃·문단 ID가 기존 로더와 일치하는지 확인했습니다.
이 로더는 처음 열 때 무결성과 ID·문단 경계를 검사하기 위해 전체 배열을 순회합니다.
메모리 매핑은 배열 전체를 한 번에 RAM으로 복사하는 일을 피하며, 초기 검증의 파일 읽기까지 생략하지는 않습니다.
기존 체크포인트에 연결된 훈련 코드와 `.ids.u16` 경로는 보존하며, `.npy`를 사용할 때는 위 로더를 선택합니다.
여기까지는 데이터 로딩 검증이며 실제 모델 학습은 계속 중단 상태입니다.

## 실제 학습 비교: 데이터 양과 시간

사용자가 선택한 실험 범위는 아래 세 실행이며, 명목 학습 시간은 합계 14시간입니다.
준비·초기 평가 시간과 짧은 동작 확인은 이 합계에 별도로 더해집니다.

| 학습 후보에서 선택하는 문서 비율 | 실행 시간 | 보존하는 주요 결과 |
| --- | --- | --- |
| 10% | 3시간 | 0·1·3시간 모델과 공통 평가 결과 |
| 25% | 3시간 | 0·1·3시간 모델과 공통 평가 결과 |
| 100% | 8시간 | 0·1·3·8시간 모델과 공통 평가 결과 |

`prepare_training_experiment.py`는 원본 Parquet의 문서 ID를 다시 연결하고,
중복 제거 후의 **모든 문단을 원문과 대조**해 학습·검증·테스트를 문서별로 분리합니다.
문서 ID의 고정 해시로 약 98%·1%·1%를 배정하며, 사전은 학습 쪽 토큰만 세어 만듭니다.
숫자를 자릿수로 나눈 기존 `tokens_numeric/`를 사용하고, 토큰화 이후 새로 같아진 문장과
평가 쪽에 나타나는 동일 토큰열도 학습 쪽에서 제외합니다.
문장 끝의 공백 토큰 유무 때문에 같은 문장을 놓치지 않도록, 중복 비교에서만 양끝 공백 토큰을 무시합니다.
실제 학습 입력의 공백 토큰은 유지합니다.

그다음 사전 밖 토큰이 포함된 문장 전체를 제외하며, 한 문장이라도 남는 문단을 유지합니다.
여기서 **100%는 이 조건을 통과한 학습 후보 전체**이고 원본 문서 582,035개 전체와 다릅니다.
10%·25%는 별도 문서 해시로 고른 중첩 부분집합입니다. 문서 길이가 다르므로 실제 문단·토큰 비율은
정확히 10%·25%가 아닐 수 있어 각각의 개수를 함께 기록합니다.

실험 간 사전·초기 가중치·모델·평가 대상·생성 시드와 온도는 같게 고정합니다.
이때 사전은 100% 학습 후보에서 한 번 만들므로, 10% 실행의 사전에는 그 부분집합에서 아직 등장하지 않은 항목도 포함될 수 있습니다.
모델은 문맥 8토큰, 임베딩 64차원, 은닉 128차원인 NPLM입니다.
긴 문맥을 사용하는 Transformer LLM과는 구조와 표현력이 다릅니다.
한 문단의 토큰 ID를 uint16으로 저장하고, 학습 시 정수 인덱스와 float32 가중치를 사용합니다.
모델이 임베딩 행렬과 다음 토큰의 확률을 학습합니다. Kiwi 분석이나 ID 변환 자체는 임베딩 학습이 아닙니다.

다음 명령은 전체 전처리·독립 검증·자원 측정이 완료된 뒤 실행합니다.
K와 device는 완료된 보존율 분석과 실측 자원 결과에서 선택한 값을 전달해야 합니다.

```sh
uv pip install --python .venv-training/bin/python -r requirements-training.lock

# 예시: K=32,768 및 CPU의 실측 결과가 있을 때, 준비부터 세 실행까지 순차 진행
caffeinate -i .venv-training/bin/python -u run_training_comparison.py --vocab-size 32768 --device cpu

# 데이터 분리·경계·체크포인트·재개·비교 조건 검증
.venv-training/bin/python -m unittest discover -s tests -p test_training_experiment.py -v
```

- `training-corpora/wiki-kK/`: 원문 문서 연결 기록, 학습 전용 빈도표·사전, 문단별 ID 배열·통계
- `training-runs/wiki-kK/data-10pct/` 등: 모델 가중치, 업데이트·시간·반복 횟수, 평가 loss/PPL, 동일 프롬프트 생성 결과
- `reports/training-queue.json`: 실제 실행 중인 단계와 완료한 실행
- `reports/training-comparison.json`·`.txt`: 실제 3시간·8시간 체크포인트만 비교. 아직 도달하지 않은 결과는 빈칸

평가는 모든 실행에서 같은 검증 타깃 최대 16,384개를 사용합니다. 전체 토큰의 loss/PPL과 공백·EOS를
제외한 loss/PPL을 함께 기록하며, 최종 테스트 데이터는 이 비교에 사용하지 않습니다.
사전이 달라지면 평가 가능한 문장과 토큰의 정의가 달라지므로 PPL만 바로 비교해서는 안 됩니다.
이 14시간 실험은 **선택한 사전을 고정한 데이터 양·시간 비교**입니다. 사전 크기의 품질 최적값까지
증명하는 실험은 아니며, 한 시드의 결과라는 범위도 함께 해석해야 합니다.

장시간 실행을 멈추려면 해당 `training-runs/wiki-kK/` 아래에 `STOP` 파일을 만듭니다.
학습기는 체크포인트를 저장하고 다음 실행을 시작하지 않습니다. `STOP` 파일을 제거한 뒤 같은 명령에
`--resume`을 추가하면 마지막으로 저장한 가중치·옵티마이저·학습 순서를 이어갑니다.

이미 실행 중인 토큰화·빈도 집계에서 이어 갈 때는 `continue_experiment.py`가 완료 보고서를 기다린 뒤
독립 검증 → 전체 보존율·그래프 → CPU/MPS 측정 → 위 14시간 비교를 순서대로 실행합니다.
토큰화나 빈도 집계 프로세스를 추가로 만들지 않습니다. `reports/experiment-continuation.json`에 진행 상태를 남깁니다.
사전 K는 전체 숫자 분리 보존 곡선의 16,384~65,536 구간 꺾임점 후보로 정하고,
같은 모델 설정에서 실측이 빠른 장치를 선택합니다. 이 선택은 실험의 출발점이며 품질 최적값이라는 뜻은 아닙니다.
이 연속 실행 전체의 후속 단계를 멈추는 신호 파일은 `reports/STOP-EXPERIMENT`입니다.

후속 단계 오류를 수정한 뒤 `continue_experiment.py --reuse-step5-verification`로 재개하면,
전체 토큰화 검증의 32개 입력·출력 해시 기록, 현재 크기·수정 시각, 원본 참고 파일이 그대로인지 확인한 뒤
통과한 토큰화 재검사를 재사용합니다. 일부만 검증됐거나 파일이 바뀌었으면 재사용을 거부합니다.
빈도표의 9자리 비율 검증은 정확한 정수 비율과 표시 단위의 절반 오차에 부동소수점 연산 4 ULP를 더해 비교합니다.
횟수·순위·종류·합계·파일 해시는 계속 정확히 일치해야 하며, 2026-10-17 복구 근거는 `reports/pipeline-recovery.json`에 남깁니다.

T3 서버가 재시작돼도 별도 세션의 학습 큐·학습 프로세스는 살아 있을 수 있습니다.
이때는 저장된 PID와 명령·진행 파일의 갱신을 먼저 확인하고, 큐가 살아 있으면 `observe_training_queue.py`로
기존 큐의 절전 방지·완료 알림만 다시 연결합니다. 토큰화나 학습을 중복 실행하지 않으며,
인계 기록은 `reports/server-restart-recovery.json`에 남깁니다.
