# Step 4. 문장 dedup

32 샤드 전체에서 동일 문장을 찾아 첫 등장만 남긴다.

## 왜 필요한가

위키백과에는 반복 문구가 많다. 예를 들어 "FIFA 라이선스" 관련 문장은
수백 개 문서에 동일하게 등장한다. 이런 boilerplate 를 제거해야
모델이 편향 없이 학습할 수 있다.

## 정규화 → 해시

비교 전에 문장을 정규화하되, 출력은 원문 그대로 보존한다.

```python
def normalize(s):
    s = unicodedata.normalize("NFKC", s)   # 전각/반각, 합성형 통일
    return re.sub(r"\s+", " ", s).strip()   # 공백 정리
    # 영문 case 보존 — "Apple"과 "apple"은 다른 문장

def hash_key(s):
    return hashlib.blake2b(s.encode(), digest_size=16).digest()
    # 16 byte → 1e9 문장에서도 충돌 확률 ~1e-19
```

## 핵심 로직

```python
seen = {}        # hash → 첫 등장 원문
dup_count = {}   # hash → 중복 횟수

for line in shard:
    key = hash_key(normalize(line))
    if key in seen:
        dup_count[key] += 1   # 중복 → 건너뜀
    else:
        seen[key] = line      # 첫 등장 → 출력
```

32 샤드를 하나의 `seen` dict 로 순차 처리하므로, 샤드 간 중복도 잡힌다.

## 구체적 예시

가장 많이 제거된 boilerplate — 축구 선수 문서 2,259개에 동일하게 등장:

```
# 서로 다른 문서에서 같은 문장이 반복 등장:

part_00.txt (line 34236):
  현재 인도 슈퍼리그에 참가하고 있다.
  (빈 줄)
  참고: FIFA 자격 규정에 따라 소속된 국가대표팀 국기를 표시합니다.   ← 첫 등장: 유지

part_00.txt (line 35557):
  가나의 국가대표 공격수이자 대표적인 스타인 아사모아 기안이 데뷔한 구단이기도 하다.
  (빈 줄)
  참고: FIFA 자격 규정에 따라 소속된 국가대표팀 국기를 표시합니다.   ← 2번째: 제거

part_00.txt (line 37217):
  현재 C리그에 참가하고 있다.
  (빈 줄)
  참고: FIFA 자격 규정에 따라 소속된 국가대표팀 국기를 표시합니다.   ← 3번째: 제거
```

## 문단 경계 보존

입력의 빈 줄(문단 경계) 을 그대로 유지하되, 문단 안의 문장이 전부 중복이면
빈 줄도 함께 제거한다. 결과적으로 연속 빈 줄은 발생하지 않는다.

```
# 입력                                          # 출력
고야마못은 돗토리현에 위치한 석호이다. (첫 등장)   고야마못은 돗토리현에 위치한 석호이다.
고대에는 만이었으나 ... (첫 등장)                  고대에는 만이었으나 ...
                                                  (빈 줄)
둘레 16 km, 면적 6.96 km2. (첫 등장)              둘레 16 km, 면적 6.96 km2.
                                                  ← 문단 전체 dup 시 빈 줄도 제거
참고: FIFA 자격 규정에 따라 ... (중복)
선수는 복수의 FIFA 비회원국 ... (중복)
```

## 제거 로그

어떤 문장이 얼마나 중복됐는지 `sentences_dedup_removed.txt` 에 기록한다.
빈도 내림차순으로, 가장 자주 나온 boilerplate 가 맨 위에 온다.

```
[2259x] 참고: FIFA 자격 규정에 따라 소속된 국가대표팀 국기를 표시합니다.
[2259x] 선수는 복수의 FIFA 비회원국 국적을 가지고 있을 수 있습니다.
[1217x] 최저 시청률과 최고 시청률은 시청률 조사회사와 지역별로 시청률에 차이가 있을 수 있습니다.
 [787x] 단선 승강장 1면 1선의 구조를 갖춘 지상역이다.
 [324x] 사인은 심근 경색이다.
 [304x] 로스앤젤레스에서 태어났다.
```

콘솔에도 TOP 30 을 출력한다.

## 입출력

```
입력: sentences/part_NN.txt           (32 샤드)
출력: sentences_dedup/part_NN.txt     (32 샤드, 중복 제거)
      sentences_dedup_removed.txt     (제거 로그)
```

## CLI

```bash
python step4_dedup_sentences.py           # 전체 실행
python step4_dedup_sentences.py --test     # 앞 1 샤드만
python step4_dedup_sentences.py --test 3   # 앞 3 샤드
```
