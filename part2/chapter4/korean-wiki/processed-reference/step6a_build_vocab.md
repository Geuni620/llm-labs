# Step 6a. 어휘 구축 + OOV 문장 필터링

빈도 기반으로 어휘를 구축하고, 어휘 밖(OOV) 토큰이 포함된 문장을 제거한다.

## 숫자 토큰 분리

SN(숫자)과 W_SERIAL(일련번호) 토큰을 개별 자릿수 + 구분자로 분리한다.
vocab의 ~5.7%(~86,000개)를 차지하던 숫자 토큰이 ~16개로 통합된다.

```
"2010/SN"       → 2/SN 0/SN 1/SN 0/SN
"2-1/W_SERIAL"  → 2/SN -/SW 1/SN
"996,000/SN"    → 9/SN 9/SN 6/SN ,/SW 0/SN 0/SN 0/SN
```

- 전각 숫자(０–９)는 반각(0–9)으로 정규화
- 숫자 내부 구분자는 `/SW`(기타 기호)로 태깅 (문장 부호 `,/SP`, `./SF` 등과 구분)
- 현대 LLM(LLaMA, GPT)도 숫자를 1–3자리 단위로 분리하는 것이 표준 관행

## 처리 흐름

### Pass 1: 토큰 빈도 카운팅 (단일 프로세스)

- 32 샤드 순차 읽기, `collections.Counter`로 집계
- 숫자 토큰 분리 적용 후 카운팅

### Vocab 구축

- `count >= min_count`인 토큰만 포함 (또는 `--top-k`로 상위 K개)
- 빈도 내림차순 정렬 (동률 시 사전순)
- 인덱스 0–3 특수 토큰 예약: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- 인덱스 4부터 일반 토큰

### Pass 2: 문장 필터링 + `<bos>`/`<eos>` 삽입 (멀티프로세싱)

- 문장의 **모든** 토큰이 vocab에 있으면 유지, 하나라도 없으면 제거
- 문단 경계 처리:
  - `<bos>` → 문단 첫 문장 앞에 삽입
  - `<eos>` → 문단 마지막 문장 뒤에 삽입
  - 문단 내 모든 문장이 제거되면 `<bos>`/`<eos>`도 생략
- 출력에 빈 줄 없음 (경계는 `<bos>`/`<eos>`가 담당)
- 32 샤드를 32 워커가 병렬 처리 (Pool `initializer` 패턴으로 vocab_set 공유)

## vocab 파일 형식

```
<pad>	0
<unk>	0
<bos>	0
<eos>	0
이/JKS	1234567
의/JKG	987654
...
```

`토큰\t빈도` (탭 구분), 행 번호(0-indexed) = 토큰 인덱스.

## 입출력

출력 경로에 vocab size가 접미사로 붙는다. `--top-k`를 바꿔가며 반복 실행 가능.

```
입력: tokens/part_NN.txt                  (32 샤드, 읽기 전용)
출력: vocab_{VS}.txt                      (어휘 파일)
      tokens_filtered_{VS}/part_NN.txt    (필터링된 토큰열, 32 샤드)
```

테스트 모드: `vocab_{VS}_test.txt` + `tokens_filtered_{VS}_test.txt` (단일 파일)

## CLI

```bash
python step6a_build_vocab.py --top-k 65536              # top-K 방식 (권장)
python step6a_build_vocab.py --min-count 5               # 최소 빈도 방식
python step6a_build_vocab.py --top-k 32768 --test        # 앞 1 샤드 테스트
python step6a_build_vocab.py --top-k 32768 --test 3      # 앞 3 샤드 테스트
```

- `--top-k K`: vocab size 상한 (특수 토큰 포함). 빈도순 상위 K개만 포함
- `--min-count N`: 최소 빈도 (기본값 0). `count < N`인 토큰 제외
- `--test [N]`: 앞 N개 샤드만 단일 프로세스로 처리 (기본 N=1)

## 설계 결정: 특수 토큰 `<pad>`, `<unk>`

### 현재 상태

4개의 특수 토큰(`<pad>`, `<unk>`, `<bos>`, `<eos>`)을 인덱스 0–3에 예약하지만,
실제로 데이터와 학습에서 사용되는 것은 `<bos>`와 `<eos>`뿐이다.

- **`<pad>` (ID=0)**: 데이터에는 존재하지 않음. `train.py`의 `insert_bos_padding()`이
  런타임에 `<bos>` 앞에 삽입할 뿐, `.npy` 파일 자체에는 없다.
- **`<unk>` (ID=1)**: 사용되지 않음. OOV 토큰이 포함된 문장은 `<unk>`로 치환하지 않고
  **통째로 제거**한다 (Pass 2의 `all(t in vocab_set for t in tokens)` 조건).

따라서 `<pad>`와 `<unk>`는 학습 데이터에 한 번도 등장하지 않으며,
softmax 출력에서 자연스럽게 극히 낮은 확률을 갖게 된다.
별도의 `-inf` 마스킹은 불필요하다.

### 대안 검토

- **`<pad>` 제거 + BOS 반복**: `[<pad>]*3 + [<bos>]` 대신 `[<bos>]*4`를 사용하면
  `<pad>`를 완전히 없앨 수 있다. 코드 변경은 `train.py`와 `generate.py` 두 파일뿐이지만,
  vocab 파일에서 `<pad>`를 빼면 전체 인덱스가 밀려서 step6a → step7 → `.npy` 파일을
  모두 재생성해야 한다.
- **`<unk>` 제거**: 마찬가지로 vocab에서 빼면 인덱스가 밀려 파이프라인 재실행 필요.

### 결론

vocab 16,384개에서 미사용 슬롯 2개의 비용은 무시할 수 있으므로,
**현재 구조를 유지**한다. Transformer 기반 모델에서는 가변 길이 배치를 위해
`<pad>` + attention mask가 필수적이므로, 이 단계에서 특수 토큰 체계에 익숙해지는 것이
교육적으로도 유리하다.
