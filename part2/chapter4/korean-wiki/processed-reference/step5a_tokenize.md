# Step 5a. 형태소/품사 토큰화

Kiwi 형태소 분석기로 문장을 `형태소/품사` 토큰 시퀀스로 변환한다.

## 토큰 형식

```
"서울은 대한민국의 수도이다."
→ 서울/NNP 은/JX <sp>/WS 대한민국/NNP 의/JKG <sp>/WS 수도/NNG 이/VCP 다/EF ./SF
```

| 종류 | 형식 | 예시 |
|------|------|------|
| 일반 | `형태소/품사` | `국밥/NNG`, `먹/VV`, `었/EP` |
| 숫자 | `숫자/SN` | `1987/SN` (원본 보존, 숫자 분리는 이후 단계) |
| 공백 | `<sp>/WS` | 원문에서 형태소 사이에 공백이 있던 위치 |

## 공백 검출

형태소 사이 원문 위치를 비교해서 공백이 있었는지 판단한다.

```python
for i, tok in enumerate(tokens):
    out.append(f"{form}/{tok.tag}")

    if i + 1 < len(tokens):
        next_tok = tokens[i + 1]
        cur_end = tok.start + tok.len
        if next_tok.start >= cur_end:       # 겹치지 않을 때만 검사
            gap = sentence[cur_end : next_tok.start]
            if any(c.isspace() for c in gap):
                out.append("<sp>/WS")
```

받침 합성 형태소 (예: "탔다" → `타/VV` + `었/EP`) 는 start 가 겹치므로,
`next_tok.start >= cur_end` 조건이 거짓이 되어 공백을 삽입하지 않는다.

## Kiwi 특성 3가지

### 1. 원형 복원

분석기가 문법적 원형을 복원하기 위해 원문에 없는 토큰을 추가할 수 있다.

```
"도로다" → 도로/NNG + 이/VCP + 다/EF
                      ↑ 원문에 없는 서술격 조사
```

"도로다" 와 "도로이다" 가 동일한 토큰열이 된다 (의도적 손실).

### 2. 동형이의어 분리

같은 글자라도 품사가 다르면 다른 토큰이 된다.

```
차/NNG  → 자동차, 차(tea)
차/VV   → 차다 (kick)
```

### 3. 자모 단독 토큰

관형형 어미가 받침 하나로 분석되기도 한다.

```
"한 사람"  → 하/VV + ᆫ/ETM + <sp>/WS + 사람/NNG
"본 적"    → 보/VV + ᆫ/ETM + <sp>/WS + 적/NNB
```

"한", "본", "할" 등의 관형형이 동일 어미 `ᆫ/ETM`, `ᆯ/ETM` 으로 통일된다.

## form 안 공백 escape

Kiwi 가 multi-word 고유명사를 한 토큰으로 분석하는 경우, form 안의 공백을
`_` 로 치환한다 (토큰 구분자가 스페이스이므로).

```
"돗토리 대학" → 돗토리_대학/NNP
```

## 멀티프로세싱

32 샤드를 32 워커가 병렬 처리한다. 각 워커는 자체 Kiwi 인스턴스를 생성하되,
`Kiwi(num_workers=1)` 로 내부 스레드 풀을 비활성화하여 oversubscription 을 방지한다.

## 입출력

```
입력: sentences_dedup/part_NN.txt   (32 샤드, 한 줄 = 한 문장)
출력: tokens/part_NN.txt             (32 샤드, 한 줄 = 토큰열)
```

빈 줄(문단 경계)은 그대로 보존한다.

## CLI

```bash
python step5a_tokenize.py           # 전체 실행 (32 샤드 멀티프로세싱)
python step5a_tokenize.py --test     # 앞 1 샤드만 단일 프로세스
python step5a_tokenize.py --test 3   # 앞 3 샤드
```
