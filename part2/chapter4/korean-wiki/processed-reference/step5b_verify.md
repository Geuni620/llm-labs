# Step 5b. 토큰화 round-trip 검증

Step 5a 의 토큰열을 `Kiwi.join()` 으로 표면형 복원하여 원본과 비교한다.

## 왜 필요한가

형태소 토큰화는 비가역적이다. 얼마나 많은 정보가 손실되는지 정량적으로 파악해야
이후 단계 (생성, detokenization) 의 품질 기대치를 설정할 수 있다.

## 검증 흐름

```
원본 문장 (sentences_dedup/)
    ↓
토큰열 (tokens/) → Kiwi.join() → 복원 문장
    ↓
원본 문장 == 복원 문장?
    ├─ exact:    완전 일치
    └─ mismatch: 불일치
```

## detokenize 함수

`<sp>/WS` 위치를 어절 시작 마커로 변환하여 `Kiwi.join()` 에 전달한다.

```python
def detokenize(tokens_str, kiwi):
    tokens = tokens_str.split(" ")
    morphs = []
    next_is_word_start = True
    for tok in tokens:
        if tok == "<sp>/WS":
            next_is_word_start = True
            continue
        form, tag = tok.rsplit("/", 1)
        form = form.replace("_", " ")       # escape 역변환
        morphs.append((form, tag, next_is_word_start))
        next_is_word_start = False
    return kiwi.join(morphs)
```

## 100% 일치가 불가능한 이유

Kiwi 의 두 가지 특성 때문에 원본과 복원이 다를 수 있다:

```
원형 복원:  "도로다"  → 도로/NNG + 이/VCP + 다/EF → "도로이다"  (≠ 원본)
축약 선호:  "되어서"  → 되/VV + 어/EC + 서/JKB    → "돼서"      (≠ 원본)
```

이런 mismatch 는 분석기의 정규화에 의한 것이므로 품질 문제가 아니다.
참고용 지표로 사용한다.

## 출력 예시

콘솔 미리보기 (--samples 개수만큼):

```
[1] line 42 (mismatch)
  ORIG: 그는 되어서야 알았다.
  REST: 그는 돼서야 알았다.
  DIFF: '되어서' → '돼서' (pos 3)
```

로그 파일 (`tokens_verify_mismatches.txt`) 에도 동일 형식으로 전체 기록된다.
테스트 모드에서는 `tokens_verify_mismatches_test.txt` 에 기록.

## 입출력

```
입력: sentences_dedup/part_NN.txt (원본) + tokens/part_NN.txt (토큰열)
출력: tokens_verify_mismatches.txt (불일치 로그)
```

## CLI

```bash
python step5b_verify.py                # 전체 검증
python step5b_verify.py --test          # 앞 1 샤드 vs tokens_test.txt
python step5b_verify.py --samples 50   # 콘솔 불일치 사례 50개 출력
```
