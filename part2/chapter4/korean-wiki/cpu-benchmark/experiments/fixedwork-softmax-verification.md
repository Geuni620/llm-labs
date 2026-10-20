# 고정 작업량 softmax 배치 역수 후보 수치 검증

결과: **PASS**. 학습 시간 측정은 실행하지 않았습니다.

## 변경 범위

`fw_softmax`는 양의 2의 거듭제곱 배치에서만 호출당 한 번 `1.0f / batch`를 준비하고 gradient의 반복 배치 나눗셈을 해당 역수 곱셈으로 바꿉니다. 실제 배치128과 작은 배치4가 여기에 포함됩니다.

다음 순서와 연산은 원본 `vector_vforce.c`에 맞게 유지했습니다.

- 각 gradient의 `/sum`은 나눗셈을 유지합니다.
- bias 추가, 최댓값 및 지수 합산 순서와 float32 정밀도를 유지합니다.
- 기본 exp 경로는 Apple Accelerate의 CPU `vvexpf`입니다. GPU와 전역 fast-math는 사용하지 않습니다.
- target gradient 보정과 마지막 `loss_sum / batch`를 유지합니다.
- OpenMP 작업자가 행을 소유하며, 각 행의 loss는 원래 행 순서로 합칩니다.
- 2의 거듭제곱이 아닌 배치는 원본 함수로 직접 fallback합니다. 배치71 fixture도 검사합니다.
- 기존 `fast_exp` 옵션은 호환하지만 실제 성능 비교에서 사용할 값은 false입니다.

## 검증 입력과 결과

- 전체 **2,288개** 유효 입력 케이스.
- 2의 거듭제곱 배치 경로 **1,504개**, 원본 나눗셈 fallback **784개**.
- 기존 fast-exp 옵션 호환 **56개**.
- 원본과 비교한 전체 `dL/dlogits` 성분 **63,033,664개**.
- 26개 사전 크기1~65,536, 기본 배치1/3/4, 추가 배치71/128, 스레드1/4.
- 실제 **B128/V35,328** 조합은 zero·일반 random·underflow tail 패턴을 검사했습니다.
- 기본 grid의14개 패턴은 near-zero, ±1,000, ±1e30, 큰 공통 offset, 근접 최댓값, 지배 클래스, exp underflow/subnormal 경계 등을 포함합니다.
- 최대 loss 절대차 **0**, 최대 gradient 절대차 **0**.
- loss 반환 비트 차이 **0**, gradient 반환 비트 차이 **0**.
- 비유한 출력 쌍 **0**, canary 훼손 **0**, bias·target 변경 **0**.
- 유효하지 않은 차원의 원본 NaN 반환 guard도 통과했습니다.

성공 조건은 유한한 유효 입력에 대해 원본과 모든 반환 float32 비트가 같고, 경계와 읽기 전용 입력이 보존되는 것입니다. 진단용 허용오차도 실행 전에 loss 절대1e-5+상대2e-6, gradient 절대2e-7+상대2e-6으로 고정했습니다.

## 실행 조건

`-std=c11 -O3 -mcpu=native -ffp-contract=off -fno-math-errno`, OpenMP와 `EXPERIMENT_VFORCE_EXP=1`; 전역 fast-math 없음.

다른 unit 검사와 공유하는 `.timing.lock`을 획득한 동안만 컴파일·검사하고 즉시 해제했습니다. unit 수치 검사는 **2.342808초**였으며, 이는 모델 처리량 benchmark가 아닙니다. 원본 및 후보 관련 소스 해시는 검사 전후 일치했습니다.

## 해석 범위

검사한 입력에서의 float32 반환 비트 일치입니다. 모든 실수/float 입력에 대한 보편적인 증명이 아닙니다. 전체 모델의 기울기·optimizer 갱신 검사와 실제 처리량 측정은 부모의 공용 runner에서 별도로 진행해야 합니다. 이번 변경만으로 속도나 학습 품질이 좋아졌다고 주장하지 않습니다.

기계 판독 근거: `fixedwork-softmax-verification.json`.
