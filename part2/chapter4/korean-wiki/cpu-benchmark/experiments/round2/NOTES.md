# Round 2: softmax 4행 ILP 후보

상태: **파일 작성과 읽기 검토만 했습니다.** 이 폴더의 코드는 컴파일, 테스트, 벤치마크를 한 번도 실행하지 않았습니다. 아래 수치 목표와 명령은 루트가 공유 lock 아래에서 직렬로 실행해야 할 계획입니다. 속도 향상은 측정하지 않았고 주장하지도 않습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `softmax_ilp.c` | `fw_softmax`와 같은 API를 가진 후보입니다. `../fixedwork_softmax.c`를 대신해 링크합니다. 그룹+tail 테스트용으로 `fw_softmax_ilp_rows`도 export합니다. |
| `test_softmax_ilp.c` | 원본을 `baseline_fw_softmax`로 이름을 바꿔 링크한 뒤 후보와 memcmp로 비교합니다. |
| `NOTES.md` | 이 문서입니다. |

동결 파일(`../fixedwork_softmax.c`, `../fixedwork_api.h`, `../experiment_kernels.h`, `../fast_exp.h`, `../vector_vforce.c`, `../fixedwork_benchmark.c`, `../test_fixedwork_softmax.c`)은 수정하지 않았습니다.

## 구현 범위

- **바꾼 것은 스케줄뿐입니다.** 4행을 한 그룹으로 묶고, 행마다 독립된 이름의 변수(`m0..m3`, `s0..s3`)를 둡니다. 이렇게 하면 max 사슬(행당 vocab번의 `>` 비교·선택)과 exp 합 사슬(행당 vocab번의 덧셈)이 서로 독립인 사슬 4개로 인터리브됩니다.
- **행 안에서는 원본과 같은 순서로 연산합니다.**
  - `row[v] += bias[v]`를 행 값이 왼쪽 피연산자가 되도록 계산한 뒤, 엄격한 `>`로 max를 갱신합니다.
  - target logit은 bias를 더한 뒤에 읽습니다.
  - `row[v] -= max`를 한 다음, **행마다 그 행의 vocab 길이로** in-place `vvexpf`를 한 번 호출합니다. 여러 행을 합쳐 호출하지 않습니다.
  - float32 합은 0.0f에서 시작해 v 오름차순으로 더합니다.
  - `logf(sum) + max - target_logit`을 계산합니다.
  - `(row/sum)*inverse_batch`를 계산한 뒤 target을 뺍니다.
  - loss는 원래 행 순서로 합하고, 마지막에 `/ (float)batch`로 나눕니다.
- **원본과 같은 동작:**
  - 2의 거듭제곱이 아닌 배치나 유효하지 않은 배치는 `experiment_softmax`로 위임합니다.
  - `fast_exp` 경로는 원본 단일 행 함수를 그대로 복사한 함수를 씁니다.
  - `EXPERIMENT_VFORCE_EXP`가 없거나 Apple 플랫폼이 아니거나 `__FAST_MATH__`가 켜져 있으면 `#error`입니다.
- **tail:** 4로 나눈 나머지 행 0~3개는 원본 단일 행 함수(vvexpf)로 처리합니다. 공개 경로의 2의 거듭제곱 배치에서는 B1과 B2만 tail에 해당합니다. 그룹과 tail이 섞이는 경우(예: 7행)는 export한 `fw_softmax_ilp_rows`로 테스트합니다.
- **OpenMP:** `parallel && threads>1`이면 `schedule(static)`으로 그룹 4행 단위를 worker에게 줍니다. 스레드1이거나 병렬을 끄면 같은 group4 코드를 직렬로 실행합니다.
- **명시 NEON(vmaxnm 등)은 넣지 않았습니다.** NaN과 signed zero 의미가 바뀔 수 있기 때문입니다. `#pragma STDC FP_CONTRACT OFF`를 방어용으로 넣었지만 빌드 플래그 `-ffp-contract=off`는 그대로 필요합니다.
- **전제:** logits는 bias나 targets와 메모리가 겹치면 안 됩니다. 원본은 행을 하나씩 처리하고 후보는 행을 인터리브하므로, 겹치면 결과가 달라질 수 있습니다. 실제 엔진은 서로 다른 할당 영역을 넘깁니다.

## 빌드·링크 방법 (루트가 lock 아래에서 실행, 미실행)

```sh
EXP=/Users/geuni/Desktop/hong-labs/part2/chapter4/korean-wiki/cpu-benchmark/experiments
R2=$EXP/round2
OUT=${TMPDIR:-/tmp}/softmax_ilp_build   # 저장소 밖에 산출물을 둠
mkdir -p "$OUT"
FLAGS="-O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -fno-math-errno -mcpu=native \
 -DEXPERIMENT_VFORCE_EXP=1 -DFW_FIXED_NORM=1 -DFW_SOFTMAX_INV_BATCH=1 -DFW_MINIMAL_ZERO=1 \
 -Xpreprocessor -fopenmp -isystem /opt/homebrew/opt/libomp/include"
LIBS="-framework Accelerate -L/opt/homebrew/opt/libomp/lib -Wl,-rpath,/opt/homebrew/opt/libomp/lib -lomp -lm"

# 1) 원본 softmax 오브젝트: 심볼 이름만 baseline_fw_softmax로 바꿈
cc $FLAGS -Dfw_softmax=baseline_fw_softmax -c "$EXP/fixedwork_softmax.c" -o "$OUT/baseline_fw_softmax.o"
# 2) 후보 오브젝트 (fw_softmax, fw_softmax_ilp_rows)
cc $FLAGS -c "$R2/softmax_ilp.c" -o "$OUT/softmax_ilp.o"
# 3) 양쪽이 공유하는 experiment_softmax (2의 거듭제곱이 아닌 배치 fallback)
cc $FLAGS -c "$EXP/vector_vforce.c" -o "$OUT/vector_vforce.o"
# 4) 테스트 링크
cc $FLAGS "$R2/test_softmax_ilp.c" "$OUT/baseline_fw_softmax.o" "$OUT/softmax_ilp.o" \
   "$OUT/vector_vforce.o" $LIBS -o "$OUT/test_softmax_ilp"
# 5) 실행: 종료코드 0=전부 일치, 1=유한 gate 실패, 3=유한 gate 통과 + 비유한 probe 비트 차이
"$OUT/test_softmax_ilp" > "$OUT/softmax-ilp-verification.json"; echo "exit=$?"
# 확인: 원본 오브젝트에 fw_softmax 심볼이 없어야 함
nm "$OUT/baseline_fw_softmax.o" | grep -E ' T _(baseline_)?fw_softmax$'
```

`-Dfw_softmax=...`는 토큰 `fw_softmax`만 바꿉니다. `fixedwork_api.h`의 선언도 같은 이름으로 바뀌므로 원본 소스와 일관됩니다. 원본의 `static fw_softmax_row`는 다른 토큰이므로 영향을 받지 않습니다.

### 학습 벤치마크 바이너리 (combined + 후보)

`fixedwork-build-manifest.v2.json`의 `combined` 명령과 같게 빌드하되, 소스 목록의 `fixedwork_softmax.c`만 `round2/softmax_ilp.c`로 바꿉니다. 다른 플래그·소스·순서는 그대로 둡니다.

```sh
cc $FLAGS -DBENCH_BUILD_FLAGS="\"$FLAGS\"" \
  "$EXP/fixedwork_benchmark.c" "$EXP/gemm_kernels.c" "$EXP/vector_vforce.c" \
  "$EXP/fixedwork_norm.c" "$R2/softmax_ilp.c" \
  $LIBS -o "$OUT/cpu_fixedwork_combined_softmax_ilp"
```

바이너리는 자기가 어떤 softmax 소스로 빌드됐는지 출력하지 않습니다. 루트 manifest에 `softmax_ilp.c`의 sha256을 반드시 기록하세요.

`check_fixedwork.py`의 `--report`와 `--markdown` 기본값은 기존 `fixedwork-model-verification.{json,md}`입니다. 후보를 검사할 때는 **다른 경로를 지정해야** 기존 보고서를 덮어쓰지 않습니다.

## 테스트 구성 (`test_softmax_ilp.c`)

- **유한 gate (종료코드에 반영):**
  - 판정: canary를 포함한 guarded logits 전체를 memcmp로 비교하고 loss 비트도 같아야 합니다. bias와 targets가 바뀌면 안 되고, 원본 출력은 유한해야 합니다(기존 테스트와 같은 정의). 허용오차는 쓰지 않습니다.
  - 작은 매트릭스: B {1,2,3,4,5,7,8,15,16,71,128,256,512} × V {1,2,3,4,5,7,8,16,17,64,257,1025} × 패턴 19개입니다. 설정은 (threads, parallel) {(1,0),(1,1),(4,0),(4,1),(8,1),(0,1)}이고, 여기에 fast_exp {(1,0),(4,1)}을 더합니다.
  - 실제 V35328: B {1,2,3,4,8,16,71,128} × 패턴 6개 × {(1,0),(4,1),(8,1)}입니다. 추가로 B128 fast_exp 2개, B256 2개, B512 1개를 검사합니다. B512 케이스의 최대 메모리는 약 150MB입니다.
  - 패턴 0~13은 기존 테스트와 같습니다. 14~18은 새로 넣었습니다: signed zero 동점(홀수 행의 max가 -0.0), 첫·끝 열의 중복 max, 학습 logit과 비슷한 종형 분포, 행마다 다른 max 위치, 그룹 안에서 행마다 다른 스케일.
  - target은 0, V-1, 해시 위치를 섞습니다.
  - 버퍼는 64바이트 정렬로 할당합니다. vvexpf가 정렬에 따라 다른 경로를 타더라도 양쪽이 같은 조건이 되도록 하기 위해서입니다.
- **경로별 카운트를 따로 보고합니다:** `grouped_vforce_cases`, `tail_only_vforce_cases_b1_b2`, `fast_exp_row_cases`, `delegated_non_power_of_two_cases`. B3, B5, B7, B15, B71은 원본과 후보 모두 `experiment_softmax`로 위임하므로 비교 결과가 자명합니다. 이 케이스를 숨기지 않고 별도 카운트로 표시합니다.
- **엔진 그룹+tail 검사:** `fw_softmax_ilp_rows(rows=1..7 / 9..15)`의 결과를 원본 B8·B16의 앞 rows행과 비트 단위로 비교합니다. 바로 다음 행 위치의 canary로 쓰기 overrun도 검사합니다. 같은 행의 loss를 tail 경로와 group 경로에서 계산해 비트가 같은지 확인합니다. full 호출의 loss 합을 B로 나눈 값도 원본과 비트 단위로 비교합니다.
- **음성 대조군:** 1ulp 변화, +0과 -0의 차이, canary 훼손을 비교 함수가 검출하는지 확인합니다.
- **비유한 probe (별도 판정, 유한 gate에 포함하지 않음):**
  - 입력: NaN 한 개(row1), +Inf, 행 전체 -Inf, 짝수 열 -Inf, ±Inf 혼합, bias NaN, 행 전체 +Inf입니다. B {1,4,8,16} × V {1,7,257}에서 검사합니다.
  - memcmp, NaN·Inf 종류 일치, 행 격리를 보고합니다. 행 격리는 오염되지 않은 행이 유한하게 남는지 보는 검사입니다.
  - NaN payload는 canonical `NAN` 하나만 씁니다. 서로 다른 두 NaN이 만날 때 어느 payload가 전파되는지는 IEEE 754가 정하지 않으므로, payload가 섞인 경우는 **테스트하지 않았습니다**. JSON에 그렇게 명시합니다.
- 크기가 큰 케이스는 일부만 넣었습니다. B512/V35328을 대량 반복하지 않습니다.

## 남은 검증과 위험 (루트 담당)

1. 위 테스트 실행: 컴파일 경고와 실행 결과가 아직 없습니다. 이 단계에서 컴파일 오류가 나올 수도 있습니다.
2. 동일 CPU 독립 gate `check_fixedwork.py`: tiny/irregular × threads 1/4 × clip 1/0.05 × 4 updates와 gradient NaN poison을 검사합니다. 허용오차는 그대로 둡니다.
3. 실제 Wiki 512 update × 3회를 같은 코호트에서 baseline과 교대로 실행해 대조합니다. 최종 weights, m1, m2 checksum이 baseline과 같아야 합니다. 이 후보가 목표로 하는 것은 비트 동일성이므로 checksum이 하나라도 다르면 탈락입니다.
4. 성능 불확실성: 이론 상한 5~9%는 추정이며 측정이 아닙니다.
   - 4행이면 동시에 접근하는 데이터가 약 565KB(4×141KB)입니다. P코어 L1(128KB)을 넘으므로 L2 대역폭 한계로 이득이 줄어들 수 있습니다.
   - 6스레드에서는 B128이 32그룹이므로 worker당 6/5그룹으로 나뉘어 불균형이 생깁니다. 행 단위(22/21)보다 거칩니다.
   - E코어는 4MB 공유 L2이므로 8스레드는 따로 확인해야 합니다.
5. 공개 경로가 2의 거듭제곱 배치만 group4로 처리하므로, B71 fixture에서는 후보와 원본이 같은 함수를 실행합니다. 독립 gate의 irregular fixture는 이 후보의 group 경로를 거치지 않습니다.
6. vvexpf가 입력 길이나 정렬에 따라 결과 비트가 달라지는지는 Accelerate가 문서로 보장하지 않습니다. 후보는 원본과 같은 행 포인터와 같은 길이로 호출하므로, 같은 버퍼에서는 같은 결과가 기대됩니다. 테스트에서는 64바이트 정렬을 맞췄습니다.
