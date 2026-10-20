# Korean Wiki 클리닝 절차

> ## 📦 압축 패키지로 받은 경우 — 빠진 파일 안내
>
> 용량 절약을 위해 다음 3가지가 패키지에서 **제외**되었습니다.
> 폴더 안의 `step*.py` 스크립트로 누구나 재생성할 수 있어, 압축본은 약 13 GB 대신 약 8.7 GB(7z 후 약 2 GB)로 줄어들었습니다.
>
> | 빠진 항목 | 원본 크기 | 재생성 명령 | 소요 시간 |
> |---|---:|---|---:|
> | `parquet/data/kowiki/*.parquet` | 2.1 GB | `python step1_download.py` | 약 1–5분 |
> | `tokens_filtered_{16384,32768,49152,65536}/` | 4.3 GB | `python step6a_build_vocab.py --top-k 16384` <br>(이어서 `--top-k 32768`, `49152`, `65536`) | 약 8.5분 합계 |
> | `tokens_verify_mismatches.txt` | 1.0 GB | `python step5b_verify.py` | 약 58초 |
>
> 필요한 Python 패키지: `kiwipiepy` (모델 `cong-global`), `huggingface_hub`, `pyarrow`, `numpy`
>
> 빈 폴더가 보이는 곳(`parquet/`, `tokens_filtered_*/`)에는 안내용 `README.txt`가 들어 있습니다 — 어떤 파일이 어떻게 채워지는지 폴더 단위로도 확인 가능합니다.
>
> ### `step1_download.py`가 작동하지 않을 때
>
> `huggingface_hub` 패키지 호환성 문제 등으로 자동 다운로드가 실패하면 HuggingFace 웹사이트에서 두 parquet 파일을 직접 받을 수 있습니다:
> - <https://huggingface.co/datasets/HuggingFaceFW/finewiki/tree/main/data/kowiki>
> - 파일: `000_00000.parquet`, `000_00001.parquet`
> - 받은 두 파일을 `parquet/data/kowiki/` 폴더에 배치
>
> 자세한 절차와 parquet 14개 컬럼 구조는 [step1_download.md](step1_download.md) 참고.
>
> **훈련만 바로 시작하려면** `token_ids_{16384..65536}/*.npy` + `vocab_*.txt`만으로 충분합니다. 위 재생성은 데이터를 직접 분석하거나 수정 실험할 때만 필요합니다.

## Step 1. 다운로드 ([step1_download.py](step1_download.py))

- 입력: HuggingFace 에서 인터넷으로 받음
    - 데이터셋: <https://huggingface.co/datasets/HuggingFaceFW/finewiki>
    - 서브셋: `kowiki` (한국어 위키백과)
- 출력: `parquet/data/kowiki/*.parquet` (로컬 저장)
- `huggingface_hub.hf_hub_download` 로 직접 받음 (`datasets` 라이브러리는 Python 3.14 호환성 문제)

## Step 2. 샘플 추출 ([step2_extract_sample_rows.py](step2_extract_sample_rows.py))

- 입력: `parquet/data/kowiki/*.parquet` 의 앞 15행 (모든 컬럼)
- 출력: `sample_15rows.txt`
- 원본 데이터 수동 검토용

## Step 3. 문단/문장 추출 ([step3_extract_paragraphs.py](step3_extract_paragraphs.py))

- 입력: `parquet/data/kowiki/*.parquet` 의 `text` 컬럼
- 출력:
    - `paragraphs_raw/` — 구조 라인만 제거한 비교용
    - `paragraphs/` — 최종 문단
    - `sentences/` — 문장 단위 분할
- 8단계로 필터링 한다 (자세한 내용은 코드 docstring 참고)
- 32 샤드 멀티프로세싱

## Step 4. 문장 dedup ([step4_dedup_sentences.py](step4_dedup_sentences.py))

- 입력: `sentences/part_*.txt`
- 출력:
    - `sentences_dedup/part_*.txt` — 중복 제거 후 본문 (입력과 같은 32 샤드 구조)
    - `sentences_dedup_removed.txt` — 제거된 문장과 그 원본(첫 등장) 쌍, 검증용
- 정규화 (NFKC + 공백 정리) 후 blake2b 16바이트 해시로 exact dedup (영문 case 보존)
- 샤드 간 전역 dedup, 첫 등장만 보존
- 문단 경계(빈 줄) 보존, 문단 전체가 dup이면 빈 줄도 함께 제거 → 연속 빈 줄 없음
- 콘솔에 가장 자주 잡힌 중복 TOP 30 출력
- 단일 프로세스 (전역 dedup이라 멀티프로세스 이득 적음)

## Step 5. 형태소/품사 토큰화 ([step5a_tokenize.py](step5a_tokenize.py))

- 입력: `sentences_dedup/part_*.txt`
- 출력: `tokens/part_*.txt` (32 샤드 구조 유지)
- Kiwi 형태소 분석기 사용 (`pip install kiwipiepy`, 모델: `cong-global`)
- 토큰 형식: `형태소/품사` (예: `국밥/NNG`, `먹/VV`, `었/EP`)
- 동형이의어가 품사 태그로 분리됨 (예: `차/NNG` 자동차·tea  vs  `차/VV` 차다)
- 숫자: 원본 그대로 보존 (예: `1987/SN` — 글자 단위 분해는 다음 단계에서 선택)
- 공백: `<sp>/WS` 토큰으로 명시 보존 (Kiwi 의 `SP`/`SF` 와 충돌 방지용 커스텀 태그)
- form 안 공백: `_` 로 escape (multi-word 고유명사 대비, 예: `쟈니즈_사무소/NNP`)
- 받침 합성 형태소 (예: "탔다" → 타/VV + 었/EP) 는 `start` 위치 겹침 →
  `next_tok.start >= cur_end` 조건으로 공백 검출
- 한 줄 = 한 문장 토큰열, 빈 줄 = 문단 경계 (Step 4 구조 그대로 보존)
- 특수 토큰 (`<bos>`, `<eos>` 등) 미삽입 → Step 6a 에서 추가
- 32 샤드 멀티프로세싱 (워커당 Kiwi 인스턴스 1개, `num_workers=1` 로 oversubscription 방지)
- 참고: 형태소 분석기는 문법적인 원형을 복원하기 위해 토큰을 추가할 수 있다.
  (예: "도로다" → `도로/NNG` + `이/VCP` + `다/EF`, 원문에 없는 서술격 조사 `이` 가 복원됨)

## Step 5b. 토큰화 round-trip 검증 ([step5b_verify.py](step5b_verify.py))

- 입력: `sentences_dedup/part_*.txt` (원본) + `tokens/part_*.txt` (Step 5 출력)
- Kiwi.join() 으로 토큰열 → 표면형 복원, 원본과 비교
- 분류: exact (완전 일치), mismatch (실제 불일치)
- 형태소 분석기의 표면형 정규화 ("되어" → "돼", "이다" → "다" 등) 때문에
  100% 일치는 원리적으로 불가능. 참고용 품질 지표로 사용
- `--samples N` 으로 콘솔 불일치 사례 출력 개수 조정
- 모든 불일치 사례를 `tokens_verify_mismatches.txt` 에 기록 (테스트 모드: `*_test.txt`)

## Step 6a. 어휘 구축 + OOV 문장 필터링 ([step6a_build_vocab.py](step6a_build_vocab.py))

- 입력: `tokens/part_*.txt` (Step 5a 출력, 읽기 전용)
- 출력:
    - `vocab_{VS}.txt` — 어휘 파일 (`토큰\t빈도`, 행 번호 = 인덱스)
    - `tokens_filtered_{VS}/part_*.txt` — OOV 문장 제거 + `<bos>`/`<eos>` 삽입 (32 샤드)
- SN/W_SERIAL 토큰을 개별 자릿수 + 구분자로 분리 (예: `2010/SN` → `2/SN 0/SN 1/SN 0/SN`)
  → ~86,000개 숫자 토큰이 ~16개로 통합
- `--top-k K` 로 vocab size 직접 지정 (CPU SIMD 정렬 고려: 32768, 49152, 65536 등)
- `--min-count N` 으로 최소 빈도 기반 필터링도 가능
- Pass 1: 단일 프로세스 빈도 카운팅, Pass 2: 32 워커 멀티프로세싱 문장 필터링
- 특수 토큰 4개 예약: `<pad>`(0), `<unk>`(1), `<bos>`(2), `<eos>`(3)
- `--top-k`를 바꿔가며 반복 실행 가능 (출력 경로에 vocab size 접미사)

## Step 6b. Vocab 시각화 보고서 ([step6b_vocab_report.py](step6b_vocab_report.py))

- 입력: `tokens/part_*.txt` + `vocab_*.txt` (Step 6a 출력)
- 출력:
    - `vocab_comparison.png` — 비교 보고서 (순위-빈도, 커버리지 vs 유지율, vocab size 곡선, POS 분포)
    - `vocab_report_full.png` — 전체 범위 보고서 (Zipf, 히스토그램, min_count 곡선)
    - `vocab_report_{VS}.png` — 설정별 보고서 (해당 vocab 범위에 맞춘 축 스케일)
- `vocab_*.txt` 파일 자동 탐색 (정규식 `^vocab_(\d+)\.txt$`)
- 2-pass 멀티프로세싱 분석: 빈도+유지율 (1st), 문장별 최소 빈도 (2nd)

## Step 7. 정수 인코딩 ([step7_encode.py](step7_encode.py))

- 입력: `vocab_{VS}.txt` + `tokens_filtered_{VS}/part_*.txt`
- 출력: `token_ids_{VS}/part_NN.npy` (uint16, 1-D)
- 각 샤드의 모든 토큰을 `vocab_{VS}.txt`의 인덱스로 변환하여 1차원 정수 배열로 저장
- 빈 줄(문단 경계)은 건너뜀 — `<bos>`/`<eos>` 토큰이 경계 역할
- dtype: uint16 (세 vocab size 32768, 49152, 65536 모두 0–65535 범위에 맞음)
- `.npy` 포맷은 헤더 + raw 바이너리로 C에서도 `fread`로 직접 읽을 수 있음
- 32 샤드 멀티프로세싱
- `--vocab-size VS` 로 대상 vocab 지정 (기본값 32768)

## Step 7b. 인코딩 round-trip 검증 ([step7b_verify.py](step7b_verify.py))

- 입력: `token_ids_{VS}/part_NN.npy` + `tokens_filtered_{VS}/part_NN.txt`
- `.npy`를 역변환(정수 → 토큰)하여 원본 `.txt`의 토큰 시퀀스와 비교
- flat 배열이므로 줄바꿈 복원은 하지 않고 토큰 시퀀스 일치만 검증
- 불일치 시 첫 번째 위치와 해당 토큰을 보고

---

## 파이프라인 실행 기록 (2026-04-10)

### 환경

- OS: Windows 11 Pro
- CPU: (RTX 4090 탑재 데스크탑)
- Python 3.14, kiwipiepy (cong-global 모델)

### 단계별 소요 시간 및 결과

| 단계 | 소요 시간 | 입력 | 출력 | 비고 |
|------|----------|------|------|------|
| step3 | 16s | parquet 2파일 (582K 문서) | 2,618,729 문단, 7,099,328 문장 | 32 샤드 멀티프로세싱 |
| step4 | 39s | 7,099,328 문장 | 6,677,632 문장 (5.9% 제거) | 단일 프로세스, 전역 dedup |
| step5a | 1158s (~19분) | 6,677,632 문장 (935 MB) | 292,630,218 토큰 (2,465 MB) | 32 샤드 멀티프로세싱, **병목 단계** |
| step5b | 58s | 6,677,632 문장 | exact 60.4%, mismatch 39.6% | 32 샤드 멀티프로세싱 |
| step6a | ~128s/vocab | 토큰 전체 | vocab별 필터링 결과 (아래 표) | Pass 1(카운팅) 115s + Pass 2(필터링) 13s |
| step6b | 49s | 토큰 + vocab 4개 | 비교 보고서 png 6개 | 2-pass 멀티프로세싱 |
| step7 | ~7s/vocab | tokens_filtered | token_ids npy | 32 샤드 멀티프로세싱 |
| step7b | ~20–40s/vocab | npy + txt | 전체 PASS | 32 샤드 멀티프로세싱 |

### Vocab별 최종 데이터셋

| Vocab Size | 유지 문장 수 | 유지율 | 토큰 수 | npy 크기 | 용도 |
|-----------|------------|--------|---------|---------|------|
| 16,384 | 1,864,988 | 27.9% | 73,437,479 | 140 MB | CPU 테스트용 |
| 32,768 | 2,961,181 | 44.3% | 124,445,087 | 237 MB | 기본 |
| 49,152 | 3,554,679 | 53.2% | 153,222,754 | 292 MB | |
| 65,536 | 3,931,296 | 58.9% | 171,806,989 | 328 MB | |

### 데이터 특성

- 평균 clean 문단 길이: 164.4자, 2.7 문장/문단
- 평균 토큰/문장: 43.8
- 전체 유니크 토큰: 1,431,690 (빈도 1이 55.0%)
- step5a가 전체 파이프라인의 대부분(~19분)을 차지. Kiwi 형태소 분석이 병목
- step6a의 Pass 1(빈도 카운팅)은 단일 프로세스, vocab마다 반복 실행됨 (~115s)
- step6a를 여러 vocab에 대해 병렬 실행하면 I/O 경합으로 오히려 느려질 수 있음 — 순차 실행 권장

### step5b 불일치 패턴

Kiwi의 표면형 정규화로 인한 것이며 데이터 품질 문제가 아님:

- `이다` → `다` (서술격 조사 축약): `화가이다` → `화가다`
- `되어` → `돼`, `하여` → `해` (축약)
- `되었` → `됐`, `하였` → `했` (축약)
- 간헐적 고유명사 분리: `라투르` → `라 투르`

### step3–5 재실행이 필요한 경우

step3–5의 출력은 vocab에 무관하므로, vocab size만 변경할 때는 step6a부터 실행하면 됨.
step3–5의 코드나 규칙을 수정한 경우에만 전체 재실행 필요.

---

## 설계 결정: 문장 저장 방식과 문장 간 공백 처리

### 문제

Step 3에서 문단을 문장으로 분할할 때 `(?<=[.!?])\s+` 정규식을 사용한다.
이 정규식은 구두점 뒤의 공백을 구분자로 소비하므로, 분할된 문장을 한 줄씩 저장하면
문장 사이의 공백 정보가 소실된다.

예: `"말한다. 그들은"` → `["말한다.", "그들은"]` — 마침표와 "그들은" 사이의 공백이 사라짐

이 공백이 복원되지 않으면 생성된 텍스트에서 `"말한다.그들은"` 처럼 마침표 뒤에 공백 없이
다음 문장이 이어지는 문제가 발생한다.

### 검토한 대안

**한 줄 = 한 문단** 방식으로 바꾸면 문장 분할 자체를 하지 않으므로 공백이 자연스럽게 보존된다.
그러나 이 방식은 Step 4의 **문장 단위 중복 제거**를 불가능하게 만든다.
문단 단위 중복 제거는 동일 문장이 다른 문단에 포함된 경우를 잡지 못해
dedup 효과가 크게 떨어진다.

### 채택한 방식

- **저장 형식은 유지**: 한 줄 = 한 문장, 문단 경계 = 빈 줄 (Step 3–4 변경 없음)
- **공백 복원은 토큰화 단계(Step 5a)에서 처리**: 비마지막 문장의 끝에 공백을 추가한 뒤 토크나이저에 전달
- 토크나이저(Kiwi)가 공백을 `<sp>/WS` 토큰으로 자연스럽게 처리하므로 별도 후처리 불필요

---

## 훈련 코드 작성 시 참고 사항

### 데이터 파일

- 훈련 입력: `token_ids_{VS}/part_*.npy` (uint16, 1-D flat array)
- vocab 역변환: `vocab_{VS}.txt` (행 번호 = 토큰 ID, `토큰\t빈도` 형식)
- 현재 생성 완료: VS=16384, 32768, 49152, 65536 (4개)

### 특수 토큰 ID

| ID | 토큰 | 의미 |
|----|------|------|
| 0 | `<pad>` | 패딩 |
| 1 | `<unk>` | 미등록 |
| 2 | `<bos>` | 문단 시작 |
| 3 | `<eos>` | 문단 끝 |
| 4 | `<sp>/WS` | 공백 (어절 경계) |

### 데이터 구조

- `.npy` 파일은 32개 샤드 전체를 concat하여 하나의 flat array로 사용
- 토큰 시퀀스 내 문단 구조: `<bos> 문장1토큰들 <sp>/WS 문장2토큰들 ... 문장N토큰들 <eos>`
- 비마지막 문장 끝에 `<sp>/WS`가 있고, 마지막 문장 끝에는 없거나 있을 수 있음 (OOV 필터링으로 원래 비마지막이었던 문장이 마지막이 된 경우 `<sp>/WS`가 남음 — 무해)
- 문단 간에 별도 구분자 없음. `<eos>` 다음 `<bos>`가 바로 이어짐

### 학습 윈도우 구성

- `<bos>`를 기준으로 `sent_id = cumsum(ids == BOS_ID)`를 계산하면 각 토큰이 속한 문단을 알 수 있음
- 슬라이딩 윈도우 `[i, i+context_len]`에서 첫 토큰과 마지막 토큰의 `sent_id`가 같으면 같은 문단 내 → 유효한 학습 윈도우

### 텍스트 복원 (생성 시)

- `<sp>/WS` → 공백 문자
- `형태소/품사` → 형태소만 추출 (`/` 기준 앞부분)
- `<bos>`, `<eos>` → 무시
- form 안의 `_`는 공백으로 복원 (multi-word 고유명사, 예: `쟈니즈_사무소/NNP`)
