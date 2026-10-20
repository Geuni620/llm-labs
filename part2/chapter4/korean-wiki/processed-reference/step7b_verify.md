# Step 7b. 인코딩 round-trip 검증

Step 7의 `.npy` 파일을 역변환(정수→토큰)하여 원본 `.txt`와 비교한다.

## 왜 필요한가

`.npy`는 바이너리 파일이라 눈으로 내용을 확인할 수 없다.
정수 인코딩이 정확한지 — vocab 사전이 일관되게 적용되었는지,
파일 I/O 과정에서 데이터가 손상되지 않았는지 — 자동 검증이 필수다.

## 구체적 예시

### 역변환 과정

`token_ids_32768/part_00.npy`의 처음 15개 정수:

```python
>>> ids = np.load("token_ids_32768/part_00.npy")
>>> ids[:15]
array([    2,   152,    21,     4,  9065,    16,    77,     4,  4093,
          12,     4, 13351,    16,     8,     5], dtype=uint16)
```

`vocab_32768.txt`로 역사전 `{인덱스→토큰}`을 만들어 복원:

```
인덱스  →  토큰
─────     ───────────
    2  →  <bos>
  152  →  이름/NNG
   21  →  은/JX
    4  →  <sp>/WS
 9065  →  못/NNG
   16  →  이/VCP
   77  →  지만/EC
    4  →  <sp>/WS
 4093  →  커다랗/VA-I
   12  →  ᆫ/ETM
    4  →  <sp>/WS
13351  →  석호/NNG
   16  →  이/VCP
    8  →  다/EF
    5  →  ./SF
```

복원된 시퀀스: `<bos> 이름/NNG 은/JX <sp>/WS 못/NNG 이/VCP 지만/EC ...`

### 원본과 비교

`tokens_filtered_32768/part_00.txt`에서 빈 줄을 제외하고 모든 토큰을 순서대로 추출:

```
원본: <bos> 이름/NNG 은/JX <sp>/WS 못/NNG 이/VCP 지만/EC ...
복원: <bos> 이름/NNG 은/JX <sp>/WS 못/NNG 이/VCP 지만/EC ...
      ────────────────────────────────────────────────────
      토큰 하나하나 비교 → 전부 일치 → PASS
```

`.npy`는 flat 배열이므로 줄바꿈(문장 경계)은 복원하지 않는다.
토큰 시퀀스의 순서와 값이 동일한지만 검증한다.

### 불일치가 발견된 경우 (가상 예시)

```
  part_05:  3,913,870 tokens  FAIL  MISMATCH at token 24831: npy=이/VCP txt=이/JKS (npy=3,913,870 vs txt=3,913,870 tokens)
```

토큰 24831번째에서 `.npy`의 값과 원본이 다르다는 뜻이다.
이 경우 step7의 인코딩 로직에 문제가 있는 것이므로 원인을 조사해야 한다.

## 검증 흐름

```
token_ids_{VS}/part_NN.npy          vocab_{VS}.txt
[2, 152, 21, 4, ...]                {인덱스→토큰} 역사전
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
        ["<bos>", "이름/NNG", "은/JX", "<sp>/WS", ...]
                   │
                   ▼  비교
tokens_filtered_{VS}/part_NN.txt
        → 빈 줄 제외, 모든 토큰을 순서대로 이어붙인 시퀀스
                   │
                   ▼
        일치 → PASS  /  불일치 → FAIL (첫 번째 위치 보고)
```

## 입출력

```
입력: token_ids_{VS}/part_NN.npy           (Step 7 출력)
      tokens_filtered_{VS}/part_NN.txt     (Step 6a 출력, 원본)
      vocab_{VS}.txt                       (역변환용)
출력: 콘솔 보고 (파일 출력 없음)
```

## CLI

```bash
python step7b_verify.py                      # VS=32768 (기본값)
python step7b_verify.py --vocab-size 49152   # 49152 vocab
python step7b_verify.py --test               # 앞 1 샤드만
```

## 실행 결과

```
============================================================
step7 인코딩 검증 [32 shards]
============================================================
vocab: vocab_32768.txt
npy:   token_ids_32768
txt:   tokens_filtered_32768

Vocab: 32,768 토큰

  part_00:  4,146,114 tokens  PASS  OK
  part_01:  3,906,425 tokens  PASS  OK
  ...
  part_31:  3,693,592 tokens  PASS  OK

============================================================
소요 시간:  24.7s
총 토큰:    122,573,403
PASS:       32  /  FAIL: 0
결과: 모든 샤드 round-trip 검증 통과
============================================================
```
