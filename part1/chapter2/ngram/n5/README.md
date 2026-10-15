# N=5 — 마지막 4글자로 다음 글자 생성

같은 일상 답변 2,380개로 학습했습니다. `counts.csv`에는 23,047회 등장한 17,103종류의 5글자 묶음이 있고, `summary.json`에는 입력 해시와 집계 조건이 있습니다.

```sh
python3 part1/chapter2/generation/generate.py --n 5 --prefix '마음이 조급해지면' --method sample --seed 42 --count 10 --trace
```

프로젝트 루트에서 실행합니다. 초기 문맥은 `급해지면`입니다. 매번 마지막 4글자를 조회하고 다음 글자 하나를 횟수 비례로 추첨합니다.

- [생성 문장 10개](generation/sample_seed42_sentences.txt)
- [모든 선택 기록](generation/sample_seed42_traces.json)

서로 다른 결과는 5개이고, 10회 중 원본 답변과 완전히 같은 출력은 4회였습니다. 입력 조건이 같은 다른 N의 결과는 [비교 기록](../comparison_n5_n7_n10.json)에서 확인합니다.
