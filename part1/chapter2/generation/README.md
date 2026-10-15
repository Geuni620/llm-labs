# 글자 단위 N-그램 문장 생성

일반 N, 최대 빈도 선택(`greedy`), 횟수 비례 추첨(`sample`), 배치 생성과 시드를 지원합니다. 마지막 N-1글자를 문맥으로 사용하고, 매 단계 새 글자 하나를 붙입니다.

## N=5, 7, 10에서 각 10개 생성

공통 시작 문구는 `마음이 조급해지면`입니다. 공백을 포함해 9글자라 N=10의 문맥도 채울 수 있습니다.

프로젝트 루트에서 실행합니다.

```sh
python3 part1/chapter2/generation/generate.py --n 5 --prefix '마음이 조급해지면' --method sample --seed 42 --count 10
python3 part1/chapter2/generation/generate.py --n 7 --prefix '마음이 조급해지면' --method sample --seed 42 --count 10
python3 part1/chapter2/generation/generate.py --n 10 --prefix '마음이 조급해지면' --method sample --seed 42 --count 10
```

- [N=5 결과](../ngram/n5/generation/sample_seed42_sentences.txt): 문맥 4글자, 서로 다른 결과 5개, 원본 답변 일치 4/10
- [N=7 결과](../ngram/n7/generation/sample_seed42_sentences.txt): 문맥 6글자, 서로 다른 결과 3개, 원본 답변 일치 6/10
- [N=10 결과](../ngram/n10/generation/sample_seed42_sentences.txt): 문맥 9글자, 서로 다른 결과 2개, 원본 답변 일치 10/10

각 N의 `generation/` 폴더에 문장 목록과 모든 선택 기록을 저장했습니다. 동일 조건과 전체 결과는 [비교 기록](../ngram/comparison_n5_n7_n10.json)에서 확인합니다. 이번 데이터와 접두어에서는 N=10의 출력이 원본의 두 답변을 재현했습니다.

임의의 N은 먼저 `count_ngrams.py --n N`으로 빈도표를 만듭니다. 접두어는 공백을 포함해 N-1글자 이상이어야 합니다. 추첨은 전체 후보의 횟수를 가중치로 사용하고, 배치 시작에 시드를 한 번 설정합니다.

`--trace`는 첫 문장의 선택 과정을 출력합니다. 파일에는 모든 문장의 후보·횟수·분모·확률, 추첨 티켓, 누적 텍스트, 종료 이유를 저장합니다. 같은 데이터·설정·시드로 배치의 순서까지 재현합니다.

## 기존 1단계 — 바이그램 최대 빈도 생성

기존 `bigram/counts.csv`에서 마지막 한 글자를 문맥으로 조회하고, 가장 많이 나온 다음 글자를 붙입니다. 같은 빈도에서는 빈도표에 먼저 저장된 후보를 고릅니다.

### 실행

프로젝트 루트에서 실행합니다.

```sh
python3 part1/chapter2/generation/generate.py --n 2 --prefix 하 --method greedy --max-chars 100 --trace
```

현재 데이터의 결과는 `하세요.`입니다.

```text
하     → 세 선택 → 하세
하세   → 요 선택 → 하세요
하세요 → . 선택  → 하세요.
```

각 단계의 로그에는 후보별 횟수와 데이터 비율, 분모, 누적 텍스트가 나옵니다. `--top 10`으로 화면에 표시하는 후보 수를 바꿀 수 있습니다. 선택은 전체 후보를 사용하며, JSON 기록에는 모든 후보를 저장합니다.

로그의 비율은 데이터에서 관측한 연결 비율입니다. 이번 최대 빈도 선택에서는 가장 많은 후보를 매번 고릅니다. 예를 들어 `하 → 세`의 데이터 비율은 약 18.60%지만, 현재 방식에서는 `하` 뒤에서 항상 `세`를 고릅니다.

### 결과 파일

`greedy`로 한 문장을 생성할 때 아래 파일이 갱신됩니다. 추첨 배치는 각 N의 `generation/sample_seed42_sentences.txt`, `generation/sample_seed42_traces.json`처럼 방법과 시드별 파일에 저장합니다.

- [result.txt](../bigram/generation/result.txt): 최종 생성 텍스트
- [trace.json](../bigram/generation/trace.json): 설정, 입력 해시, 모든 후보, 선택 과정, 종료 이유

입력한 접두어는 데이터와 같은 NFC 표현으로 통일합니다. 최대 길이는 접두어를 포함하며, 접두어가 이미 그 길이에 도달했다면 새 글자를 붙이지 않습니다.

종료 이유는 다음과 같습니다.

- `punctuation`: 마지막 글자가 `.`, `!`, `?`에 도달
- `max_length`: 접두어를 포함한 최대 글자 수에 도달
- `unseen_context`: 마지막 글자 뒤에 관측된 후보가 없음

문장부호 종료는 1단계의 규칙입니다. 시작·끝 표시를 데이터에서 학습하는 것은 [추가 계획](PLAN.md)의 4단계에서 다룹니다.

## 검증

```sh
python3 -B -m unittest discover -s part1/chapter2/generation -p 'test_generate.py' -v
```

손으로 계산한 작은 사례로 문맥 갱신, 조건부 비율, 동률, 공백, 반복과 종료를 검증합니다. 추가 검증은 N=5, 7, 10의 전체 문맥 사용, 정수 티켓의 2:1 선택 구간, 배치 시드 재현, 유니그램의 빈 문맥을 확인합니다. 실제 30개 출력에서는 모든 문맥과 선택 구간을 원본 빈도표에 대조했습니다.
