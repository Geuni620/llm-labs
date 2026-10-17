# Toy LM — 여섯 문장으로 비교하는 트라이그램과 NPLM

같은 한국어 문장 6개로 다음 단어의 확률을 비교합니다. 신경망의 임베딩, 완전연결, tanh, softmax, cross-entropy, 역전파를 `Float64Array`와 반복문으로 직접 구현했습니다. 실행 시 딥러닝 라이브러리를 사용하지 않습니다.

[챕터 4](../README.md)의 작은 언어 모델 실습입니다. `toy`는 원리를 확인하기 위한 작은 예제를, `lm`은 언어 모델을 뜻합니다.

이번 세션의 질문과 실험 해석은 [학습 회고](docs/learning-journey.md)에 정리했습니다. 데이터는 철수는·영희는·짱구는을 주어로 하는 완성된 문장 6개입니다. 일반 단어 8개와 학습 문장 6개, 학습 쌍 24개를 구분합니다.

학습·실험뿐 아니라 시각화도 외부 실행 라이브러리 없이 구현합니다. 이전에 사용한 D3는 제거했습니다. 좌표 변환·축 눈금·SVG 경로·재생 제어는 프로젝트 코드와 Node/브라우저 기본 API만 사용합니다. `typescript`, `@types/node`와 그 개발용 전이 의존성은 요청대로 타입 검사 도구로 유지합니다. 이 도구들은 모델 실행에 사용하지 않습니다.

그림·파일별 설명·실제 코드 발췌는 [전체 구현 해설](IMPLEMENTATION.md) 또는 [단계별 HTML 해설](docs/implementation-guide.html)에서 읽을 수 있습니다. 코드가 바뀌면 `npm run explain`으로 다시 생성합니다.

## 학습 순서와 파일

1. [data](data/README.md): 원본 여섯 문장, 어휘 11개, 학습 쌍 24개를 확인합니다.
2. [trigram](trigram/README.md): 문맥별 횟수로 확률을 계산하고 add-k 스무딩을 적용합니다.
3. [nplm](nplm/README.md): 임베딩·완전연결·loss·역전파·경사하강법을 따라갑니다.
4. [generation](generation/README.md): 확률로 문장을 생성하고 원문·새 조합·오류를 나눕니다.
5. [전체 구현 해설](IMPLEMENTATION.md): 실제 소스와 수식, 공통 실행·실험·시각화 코드를 연결합니다.

각 학습 디렉토리에 해당 구현과 README를 함께 두었습니다. 공통 수학·명령·비교 실험·학습 재생 코드는 `src/`, 수치 미분 검산은 [scripts/gradcheck.ts](scripts/gradcheck.ts)에 있습니다. [원본 ZIP](reference/zltTIWJVSp2vOwmaW8Fh_minimal_example_v2.zip)은 `reference/`에 보관합니다.

## 시작

Node.js **24.12 이상**이 필요합니다. Node는 TypeScript의 타입을 지우고 실행하며, 타입 검사는 별도로 수행합니다. [Node 공식 문서](https://nodejs.org/api/typescript.html)

```sh
cd part2/chapter4/toy-lm
npm ci
npm run typecheck
npm run gradcheck
npm run compare
```

`compare`는 기본 설정으로 36회 학습하고, 14개 모델·설정에서 10,000개씩 총 140,000개를 생성합니다. 결과는 이 실습의 `results/`에 저장합니다. 결과와 `node_modules/`는 Git에서 제외했습니다.

## 작은 단계로 확인

```sh
# 문장 6개, 어휘 11개, 학습 쌍 24개
npm run data

# 처음 보는 문맥: MLE는 미정, add-k는 균등 확률
npm run trigram -- --context 철수는,바나나를 --k 0.01

# 본 문맥의 실제 횟수와 확률
npm run trigram -- --context 철수는,사과를 --k 0.01

# 초기 신경망 한 쌍의 순전파·gradient·한 번 갱신한 loss
# D=2, H=6이면 W1 연결 수는 4 × 6 = 24
npm run trace -- --dim 2 --hidden 6
```

## 기본 실험 조건과 변경

- 임베딩 차원: `1, 2, 3`
- 은닉 뉴런 수: `8`. 변경 가능하며 6으로 고정하지 않았습니다.
- 학습률: `0.05, 0.1, 0.3, 1.0`
- full-batch gradient descent: 공통 학습 쌍 24개, 가중치 갱신 2,000회
- 학습 seed: `123, 456, 789`. 같은 D와 seed에서는 같은 초기 가중치로 시작합니다.
- 트라이그램: MLE와 add-k(`k=0.01`)를 함께 비교합니다.
- 각 설정당 10,000개 생성. 신경망 생성에는 첫 학습 seed인 123의 모델을 사용합니다.
- 샘플링 seed: `2026`. 확률에 비례해 다음 토큰을 추첨합니다.
- 본문 최대 길이: 20토큰. 마지막 허용 위치 뒤의 `<eos>`까지 확인합니다.

```sh
npm run compare -- --dims 1,2,3 --hidden 12 --lrs 0.05,0.3,1 \
  --epochs 2000 --seeds 123,456,789 --samples 10000 --k 0.01 --out results-h12
```

결과 폴더를 바꾸면 앞선 결과를 보존하면서 다른 설정을 비교할 수 있습니다. 같은 폴더로 다시 실행하면 같은 이름의 결과 파일을 갱신합니다.

## 생성 결과의 세 범주

```text
학습 문장       원래 6개 중 하나이며 <eos>로 종료
새 조합         주체 + 대상 + 행동의 3토큰이며 <eos>로 종료
                원래 6개에는 없는 조합
오류·미완성     특수 토큰 출력, 순서·길이 오류, 빈 문장, 길이 제한 등
```

주체 3개 × 대상 3개 × 행동 2개로 가능한 형식 조합은 18개입니다. 학습 문장은 6개, 새로운 조합은 최대 12개입니다. 분류 규칙은 생성 후에만 적용합니다. 출력 후보를 강제로 제한하거나 오류를 숨기지 않습니다.

형식에 맞는 새 조합은 문법·의미 이해가 입증된 문장과는 구분해야 합니다. 단순한 원문 불일치율(`literalNovelRate`)에는 오류도 포함되므로, 새 조합 비율과 별도로 기록합니다.

## 퍼플렉시티

```text
평균 loss = 평균(-log(정답 토큰의 확률))
PPL = exp(평균 loss)
```

평가는 학습 쌍 24개를 동일하게 사용합니다. 균등 예측의 PPL은 11이고, 원시 트라이그램의 MLE 기준은 약 **1.5651**입니다. 시작 문맥의 정답이 세 주체이고 일부 주체의 정답이 여러 과일이므로, 학습 데이터의 최적 PPL도 1보다 큽니다.

PPL은 조절하는 설정이 아니라 계산한 결과입니다. 학습률과 학습 횟수를 바꿔 비교합니다. 대표 세 사례는 `D=2, H=8`에서 `lr=0.05, 0.3, 1.0`입니다. 리포트의 PPL 평균은 각 학습 seed에서 계산한 PPL들의 평균입니다.

이 PPL은 학습 적합도입니다. `(철수는, 바나나를)`의 예측 분포와 새 문장 생성은 별도 탐색으로 확인합니다. 이 조합을 추가 학습하지 않습니다.

이번 실험에서 NPLM의 가장 낮은 단일 실행 PPL은 **1.5661**이었습니다. D=2, H=8, 학습률 1.0, seed=123, 2,000회 갱신의 결과이고, 같은 설정의 세 seed 평균은 **1.5662**입니다. 이론적 기준 1.5651에 가까워졌다는 뜻이며, 새 문장 평가 성능까지 측정한 값은 아닙니다.

## 결과 파일

아래 파일은 로컬에서 `npm run compare`를 실행하면 생성됩니다. `results/`는 Git에서 제외하며, 저장소에서 바로 읽을 대표 수치와 해석은 [학습 회고](docs/learning-journey.md)에 기록했습니다.

- `results/report.md`: 학습률·PPL·생성 비율, 예시와 임베딩 전후 좌표
- `results/report.html`: 동일한 결과의 표와 텍스트 시각화
- `results/learning_rates.csv`: 설정별 loss·PPL·생성 비율
- `results/comparison.json`: 모든 seed의 지표·좌표·문맥별 확률·생성 분포
- `results/loss.csv`: 갱신 0회부터 2,000회까지의 학습 경과
- `results/samples/*.jsonl`: 각 모델의 전체 생성 결과, 종료 상태와 분류
- `results/distributions/*.csv`: 문장별 실제 생성 횟수와 비율

## 학습 과정을 재생하기

```sh
npm run animate
```

대표 설정 `D=2, H=8, lr=0.3, seed=123`을 같은 순서로 학습하고, 20회 갱신마다 전체 가중치를 저장합니다. `results/training-animation.html`을 열면 재생·일시정지·슬라이더로 신경망의 값, 다음 토큰 확률, 임베딩 이동, loss·PPL 곡선을 볼 수 있습니다. 총 101개 가중치 기록과 갱신별 loss 2,001개를 사용합니다. 동작 줄이기 설정에서는 기록을 수동으로 넘깁니다.

`results/training-animation.json`에는 중간 가중치 `E, W1, b1, W2, b2`가 전부 있습니다. 기존 비교 결과가 있으면 최종 loss·임베딩·같은 조건의 10,000회 생성 집계를 대조합니다. 원본 6문장과 기존 결과 파일을 변경하지 않습니다.

하단의 자유 생성 확률은 주체 3개 × 과일 3개 × 동사 2개인 18문장 각각의 시작부터 `<eos>`까지 확률을 합한 값입니다. 형식에 맞는 18문장 이외의 출력은 오류·미완성으로 분류합니다. 이 계산값과 실제 10,000회 추첨 빈도는 서로 구분합니다. 선택한 문맥의 다음 토큰 확률도 전체 문장 생성 확률과 별도입니다.

## 공통 코드와 시각화

[src/cli.ts](src/cli.ts)는 명령줄의 입구이고, [src/math.ts](src/math.ts)는 난수·softmax·cross-entropy·추첨을 계산합니다. [src/experiment.ts](src/experiment.ts)가 D·학습률·seed 실험과 파일 저장을 맡습니다.

시각화는 [src/svg.js](src/svg.js)의 기본 좌표·SVG 함수, [src/training-charts.js](src/training-charts.js)의 차트 생성, [src/training-animation.js](src/training-animation.js)의 재생 제어로 나누었습니다. `npm run animate`가 실제 중간 가중치를 기록하고 외부 스크립트가 없는 HTML을 생성합니다.

일반 테스트 파일 대신 수치 미분 스크립트 하나를 남겼습니다. 실제 24쌍과 D=1·2·3, H=8에서 `E`, `W1`, `b1`, `W2`, `b2`의 모든 성분을 직접 계산한 기울기와 대조합니다. 자동 미분 없이 구현한 역전파를 독립적인 계산으로 확인하기 위한 실습입니다.

## 순전파 6단계와 역전파의 대응

`B`는 한 번에 계산하는 학습 쌍의 수입니다. `X`는 두 단어의 임베딩을 이어 붙인 `B × 2D` 배열, `h`는 `B × H`, `z`와 `p`는 `B × V`입니다. 코드에서는 한 쌍씩 반복 계산한 뒤 gradient를 누적합니다.

```text
1. 임베딩 조회      e = E[두 토큰 ID]
2. 벡터 이어 붙이기 X = concat(e[0], e[1])
3. 은닉층          h = tanh(X W1 + b1)
4. 출력층 점수     z = h W2 + b2
5. 확률            p = softmax(z)
6. 평균 loss       L = mean(logsumexp(z) - z[정답])

역전파 (T = 전치, * = 성분별 곱)
dz  = (p - one_hot(정답)) / B
dW2 = h^T dz                  db2 = 각 행의 dz 합
dh  = (dz W2^T) * (1 - h*h)
dW1 = X^T dh                  db1 = 각 행의 dh 합
dX  = dh W1^T
dE  = dX를 두 D차원 조각으로 나눠 해당 단어의 행에 합산

갱신: parameter = parameter - learningRate * gradient
```

같은 단어가 여러 입력 위치에 등장하면 `dE`를 덮어쓰지 않고 더합니다. 이 전체 미분을 작은 양쪽 변화로 구한 수치 미분 `(L(theta+epsilon)-L(theta-epsilon))/(2*epsilon)`과 대조합니다.

현재 학습은 24쌍 전체로 gradient를 계산하고 한 번 갱신하는 **full-batch GD**입니다. SGD의 한 쌍씩 갱신, mini-batch 또는 Adam은 아직 구현하지 않았습니다. 정리 자료의 SGD/Adam은 옵티마이저 선택지로 이해하며, 모두 구현해야 핵심 목적을 달성하는 것은 아닙니다.

## 추가 자료를 대조한 보완 항목

[plans.md](plans.md)의 추가 자료 대조에 정리 자료에 언급된 ZIP의 핵심 실험 9개와 현재 구현의 차이를 기록했습니다. 현재 보완 순서는 loss·PPL 곡선, 새 문장 전체 평가와 문맥 고정 추첨, 은닉 크기 비교, 임베딩의 중간 이동 기록입니다.

변수·함수 이름은 영어로 쓰고, 주석은 한국어로 목적과 배열 크기를 설명합니다. `E`, `W1`, `b1`, `W2`, `b2`는 수식과 같은 이름을 사용합니다. 실험은 짧은 계획 → 실행 → 기록 확인 → 다음 보완의 순서로 진행합니다.

모델 구조는 Bengio et al. (2003), [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)의 Section 2를 기반으로 했습니다. 입력 임베딩에서 출력층으로 이어지는 선택적 직접 연결은 생략했습니다. 원본 ZIP은 압축 상태로 보관하며 실습용 TypeScript 코드는 별도로 작성했습니다.
