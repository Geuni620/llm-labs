# Step 3. 문단/문장 추출

parquet 의 `text` 컬럼에서 본문 문단과 문장을 추출한다.
8단계 필터 파이프라인으로 노이즈를 제거한다.

## 필터 파이프라인

문서 단위 필터 1개 + 라인 단위 필터 7개, 총 8단계:

| # | 이름 | 단위 | 설명 |
|---|------|------|------|
| 1 | has_formula | 문서 | `{\displaystyle` 포함 문서 전체 제외 |
| 2 | is_structural | 라인 | 헤딩(`#`), 불릿(`- `, `* `), 번호(`1. `), 표(`|`), 빈 줄 제거 |
| 3 | has_math_symbol | 라인 | 수학 전용 기호 (±∓≠≤≥∈∪∑√∞) 포함 제거 |
| 4 | hanja_ratio | 라인 | 한자/(한자+한글) > 30% 제거 |
| 5 | clean_paragraph | 라인 | 비한글 letter 포함 괄호 제거, 공백 정리 |
| 6 | letter_ratio | 라인 | isalpha 비율 < 20% 제거 |
| 7 | hangul_ratio | 라인 | 한글 비율 < 30% 제거 |
| 8 | is_sentence_shape | 라인 | `.`, `?`, `!` 로 끝나지 않으면 제거 |

## 3종 출력

하나의 파이프라인에서 3가지 출력을 동시에 생성한다:

```
text (parquet)
  │
  ├─ step 1-2 통과 ──────────────→ paragraphs_raw/   (비교용)
  │
  └─ step 1-8 통과 ──────────────→ paragraphs/       (최종 문단)
                        │
                        └─ 문장 분할 ──→ sentences/    (한 줄 = 한 문장)
```

## 문장 분할

종결 부호(`.`, `?`, `!`) 뒤 공백에서 분할한 뒤, 두 단계 후처리:

```python
"1953년 제임스 D. Watson은 DNA를 발견했다."
# 1차 분할: ["1953년 제임스 D.", "Watson은 DNA를 발견했다."]
# 머지:     ["1953년 제임스 D. Watson은 DNA를 발견했다."]
#            (끝이 ' X.' 인 조각은 미들 이니셜로 보고 다음과 합침)

"방법은 다음과 같다. 1. 준비한다. 2. 시작한다."
# 1차 분할: ["방법은 다음과 같다.", "1.", "준비한다.", "2.", "시작한다."]
# 삭제:     ["방법은 다음과 같다.", "준비한다.", "시작한다."]
#            (bare digit marker 제거)
```

## 괄호 처리 (clean_paragraph)

비한글 letter 가 포함된 괄호를 안쪽부터 반복 제거한다.
순수 한글/숫자/기호 괄호는 보존한다.

```python
"서울(Seoul)은 대한민국의 수도(약 1000만명)이다."
→ "서울은 대한민국의 수도(약 1000만명)이다."
#  (Seoul) 제거, (약 1000만명) 보존
```

중첩 괄호는 placeholder 치환으로 안쪽 → 바깥쪽 순서로 처리한다.

## 멀티프로세싱

parquet 의 row group 을 32 샤드에 round-robin 배분하여 32 워커가 병렬 처리한다.

## 입출력

```
입력: parquet/data/kowiki/*.parquet (text 컬럼)
출력: paragraphs_raw/part_NN.txt   (32 샤드, 비교용)
      paragraphs/part_NN.txt       (32 샤드, 최종 문단)
      sentences/part_NN.txt        (32 샤드, 문장 단위)
```

## CLI

```bash
python step3_extract_paragraphs.py                     # 전체 실행 (32 샤드 멀티프로세싱)
python step3_extract_paragraphs.py --test               # 앞 1 row group만 단일 프로세스
python step3_extract_paragraphs.py --test 3             # 앞 3 row group
python step3_extract_paragraphs.py --test --shuffle 42  # 문서 셔플 (시드 42)
```

- `--shuffle SEED`: `--test` 전용. 문서 순서를 셔플 후 출력. 재현성을 위해 시드 필수
