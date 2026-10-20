# CPU와 MPS 비교 실험

NPLM의 모델·초기화·실제 위키 입력 순서를 맞춰 수치와 짧은 실행 시간을 비교한 코드입니다. 전체 에폭 학습기는 아닙니다. 실제 에폭별 Loss와 이후 실행 계획은 [TODO](../TODO.md)에 남겼습니다.

저장소에는 소스와 검증·측정 기록을 보관합니다. 원본 데이터, Python 환경, 실행 바이너리와 공유 라이브러리는 로컬 산출물입니다. `.venv-training`은 상위 폴더의 `requirements-training.lock`을 참고해 준비하고, 데이터는 상위 README의 전처리·정수 인코딩 단계를 따릅니다.

macOS에서는 `korean-wiki/`에서 초기화·해시 도우미를 먼저 빌드합니다. 다음은 기존 빌드 기록의 명령이며 BLAS나 모델 학습을 실행하지 않습니다.

```sh
clang -O3 -std=c11 -ffp-contract=off -dynamiclib \
  gpu-benchmark/init_reference.c -o gpu-benchmark/init_reference.dylib
```

`corpus_fixture.py`는 도우미를 자동 컴파일하거나 다른 초기화로 대체하지 않습니다. Linux에서 별도 공유 라이브러리를 빌드한 경우 `KOWIKI_INIT_REFERENCE_LIBRARY`로 경로를 지정할 수 있지만, 현재 저장된 Intel/NVIDIA 실측은 없습니다.

- `verify_mps.py`, `verify_batch512.py`: 독립 CPU 기준과 비교하는 수치 검사입니다.
- `run_mps_comparison.py`, `run_confirmations.py`: 같은 실제 위키 작업량의 비교 실행기입니다.
- `optimization/`: 개별 최적화와 엄격한 검증에서 제외된 native-fused 후보의 기록을 보존합니다.
- `optimization_round2/`: Metal clipping·None gradient·모델 compilation 조합을 별도로 검증·측정합니다.

과거 결과에는 측정 당시의 절대 경로, 소스 해시와 실행 조건이 들어 있습니다. 다른 환경에서는 검증과 측정 기록을 새 경로로 다시 만들어야 하며, 기존 보고서를 덮어쓰거나 실패한 검사를 허용오차 변경으로 통과시키지 않습니다. 자세한 관측값은 `../reports/nplm-additional-time-summary.txt`와 각 원본 보고서를 확인합니다.

MPS와 CPU는 공유된 `../cpu-benchmark/.timing.lock` 아래에서 순서대로 측정합니다. 동시 모델 계산이나 미리보기를 곁들이지 않습니다. 짧은 처리량의 에폭 환산, 장시간 실제 훈련, 검증 Loss와 생성 품질은 서로 구분합니다.
