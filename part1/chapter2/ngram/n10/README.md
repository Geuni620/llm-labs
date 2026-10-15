# N=10 — 마지막 9글자로 다음 글자 생성

같은 일상 답변 2,380개로 학습했습니다. `counts.csv`에는 11,895회 등장한 11,529종류의 10글자 묶음이 있고, `summary.json`에는 입력 해시와 집계 조건이 있습니다.

```sh
python3 part1/chapter2/generation/generate.py --n 10 --prefix '마음이 조급해지면' --method sample --seed 42 --count 10 --trace
```

프로젝트 루트에서 실행합니다. 초기 문맥은 `마음이 조급해지면` 전체 9글자입니다. 매번 마지막 9글자를 조회하고 다음 글자 하나를 횟수 비례로 추첨합니다.

- [생성 문장 10개](generation/sample_seed42_sentences.txt)
- [모든 선택 기록](generation/sample_seed42_traces.json)

서로 다른 결과는 2개이고, 10회 모두 원본 답변과 완전히 같았습니다. 이번 데이터와 접두어에서는 긴 문맥이 원본 답변 재현으로 이어졌습니다. [다른 N과의 비교 기록](../comparison_n5_n7_n10.json)도 함께 확인합니다.
