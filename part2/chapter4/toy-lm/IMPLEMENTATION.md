# Node·TypeScript 언어 모델 전체 구현 해설

실행·학습·시각화는 외부 라이브러리 없이 구현했습니다. TypeScript와 Node 타입 정의는 개발 도구로 유지합니다. 코드 블록은 실제 파일에서 추출한 발췌입니다. 전체 코드는 각 파일 링크에서 확인할 수 있습니다.

## 1. 전체 구조와 외부 의존성

```text
                 ┌→ 트라이그램 ─┐
6문장 → 24학습쌍 ─┤              ├→ 생성 → 분류·리포트
                 └→ NPLM ──────┘
```

언어는 TypeScript이고 실행 환경은 Node.js입니다. Node가 .ts 파일의 타입 표기를 지우고 실행합니다. TypeScript 컴파일러는 별도의 타입 검사에만 사용합니다.

같은 24개 학습 문제를 트라이그램과 NPLM에 제공합니다. 두 모델은 입력 두 토큰으로 다음 토큰의 확률을 계산한다는 인터페이스를 공유합니다. 이후 생성·분류·집계 코드는 함께 사용합니다.

모델 계산에는 처음부터 외부 딥러닝 라이브러리를 쓰지 않았습니다. 시각화에 사용했던 D3는 제거했고, SVG 생성·좌표 변환·축·경로·재생도 직접 구현했습니다.

- 실행·학습·시각화용 외부 패키지: 0개. React·차트 라이브러리·자동 미분 라이브러리를 사용하지 않습니다.
- 요청대로 개발 도구 TypeScript와 @types/node는 유지합니다. 이들의 개발용 전이 의존성도 포함됩니다.
- node:fs, node:path, node:zlib 등은 Node 기본 모듈입니다. Math·Float64Array·Map·SVG·requestAnimationFrame은 실행 환경의 기본 기능입니다.

**Node가 실행하는 명령과 개발 전용 도구** — [package.json](package.json), 8~22행

```json
  "scripts": {
    "data": "node src/cli.ts data",
    "trigram": "node src/cli.ts trigram",
    "trace": "node src/cli.ts trace",
    "compare": "node src/cli.ts compare",
    "animate": "node src/animate.ts",
    "explain": "node scripts/build-guide.ts",
    "typecheck": "tsc --noEmit",
    "gradcheck": "node scripts/gradcheck.ts"
  },
  "devDependencies": {
    "@types/node": "24.19.2",
    "typescript": "7.0.2"
  }
}
```

## 2. data.ts — 6문장을 24개 학습 문제로

```text
문장 → 토큰 ID → 경계 표시 → 학습 쌍
```

일반 단어는 주체 3개·과일 3개·동사 2개로 총 8개입니다. 패딩 <pad>, 문장 시작 <bos>, 문장 끝 <eos>를 더해 출력 후보는 11개입니다. ID는 단어를 가리키는 번호이며 임베딩 좌표나 확률이 아닙니다.

철수는 사과를 좋아해 앞뒤에 표시를 붙이면 [pad, bos, 철수는, 사과를, 좋아해, eos]가 됩니다. 두 칸씩 문맥을 읽고 바로 다음 칸을 정답으로 삼습니다.

한 문장에서 4쌍이 나오므로 6문장에서 24쌍이 됩니다. 문장 경계를 넘어서 서로 다른 문장을 이어 붙이지 않습니다.

- (pad, bos) → 철수는 / (bos, 철수는) → 사과를 / (철수는, 사과를) → 좋아해 / (사과를, 좋아해) → eos.
- 번호는 같게 유지하고, 학습 중에는 그 번호로 찾아오는 임베딩 숫자가 바뀝니다.
- 원본 6문장을 유지합니다. 탐색용 철수는+바나나를 조합은 학습 쌍에 추가하지 않습니다.

**두 토큰에서 다음 토큰을 맞히는 문제 만들기** — [data/data.ts](data/data.ts), 47~55행

```typescript
  const pairs: Pair[] = [];
  for (const sentence of words) {
    const sequence = [PAD, BOS, ...sentence, EOS].map((word) => ids.get(word)!);
    for (let i = 0; i < sequence.length - 2; i++) {
      pairs.push({ context: [sequence[i], sequence[i + 1]], target: sequence[i + 2] });
    }
  }
  return { sentences: words.map((words) => words.join(' ')), tokens, ids, pairs };
}
```

**실제 학습 문장 6개** — [data/data.ts](data/data.ts), 9~16행

```typescript
export const SENTENCES = [
  '철수는 사과를 좋아해',
  '철수는 딸기를 좋아해',
  '영희는 사과를 좋아해',
  '영희는 딸기를 좋아해',
  '영희는 바나나를 좋아해',
  '짱구는 바나나를 싫어해',
] as const;
```

## 3. trigram.ts — MLE와 add-k

```text
문맥 2개 → 등장 횟수 → 보정 선택 → 확률
```

fit은 각 문맥 뒤에 어떤 토큰이 몇 번 나왔는지 Map과 배열에 셉니다. 트라이그램은 신경망의 가중치를 학습하지 않으므로 역전파나 경사하강법을 사용하지 않습니다.

MLE는 관측 횟수를 문맥 전체 횟수로 나눕니다. add-k는 모든 후보 횟수에 같은 k를 더하고, 분모에도 k×어휘 수를 더합니다. 따라서 add-k를 적용한 모델도 트라이그램입니다.

한 번도 본 적 없는 문맥에서 MLE는 미정으로 표시합니다. k가 양수라면 그 문맥에서 11개 후보가 균등해집니다.

- bos+철수는 다음에는 사과를·딸기를 각각 1회 관측했습니다. MLE는 각각 1/2입니다.
- k=0.01이면 분모는 2+0.01×11=2.11입니다. 사과를은 1.01/2.11≈47.87%, 바나나를은 0.01/2.11≈0.47%입니다.
- 스무딩은 원하는 단어에 임의로 확률을 얹는 학습이 아닙니다. 여기서는 분포를 균등 분포 쪽으로 당기는 고정 계산 규칙입니다.

```text
MLE = count / contextCount
add-k = (count + k) / (contextCount + k × V)
```

**실제 MLE·add-k 확률 계산** — [trigram/trigram.ts](trigram/trigram.ts), 48~63행

```typescript
  distribution(context: Context, k = 0): Distribution {
    if (!Number.isFinite(k) || k < 0) throw new Error('k는 0 이상의 유한한 값이어야 합니다.');
    const key = this.key(context);
    const counts = Array.from(this.counts.get(key) ?? new Float64Array(this.vocabularySize));
    const contextCount = this.totals.get(key) ?? 0;
    const denominator = contextCount + k * this.vocabularySize;
    return {
      contextCount, counts, k, denominator,
      // MLE의 미관측 문맥은 미정이다. add-k의 균등 분포와 구분한다.
      probabilities: denominator === 0 ? null : counts.map((count) => (count + k) / denominator),
    };
  }

  evaluate(pairs: readonly Pair[], k = 0): Evaluation {
    if (pairs.length === 0) throw new Error('평가 쌍이 필요합니다.');
    let totalLoss = 0;
```

## 4. math.ts — 난수·softmax·추첨의 기본 계산

```text
출력 점수 → 최댓값 빼기 → exp와 합 → 정규화
```

모델은 처음부터 확률을 출력하지 않고 후보별 점수인 logit을 만듭니다. softmax가 exp를 취한 뒤 합으로 나누어 확률로 바꿉니다. 최댓값을 먼저 빼도 결과 확률은 같고, 지나치게 큰 exp 계산을 피할 수 있습니다.

randomGenerator는 seed로 재현 가능한 의사난수를 만들고, normal은 Box–Muller 식으로 정규분포 초기값을 만듭니다. 외부 난수·수학 패키지를 불러오지 않습니다.

sample은 확률의 누적합과 난수 티켓을 비교합니다. 예를 들어 [0.6,0.4]라면 티켓이 0.6보다 작을 때 첫 후보, 그 이후에는 둘째 후보를 고릅니다.

- 점수 0.6·0.4를 softmax에 넣으면 약 54.98%·45.02%입니다. 이미 확률 60%·40%인 값을 다시 softmax에 넣는 것과 구분합니다.
- Math.exp·Math.log·Math.tanh는 JavaScript의 기본 수학 함수입니다. 행렬 라이브러리나 텐서 엔진은 쓰지 않습니다.
- 같은 seed와 같은 계산 순서를 사용하면 같은 초기값과 추첨 결과를 재현할 수 있습니다.

**직접 구현한 안정적인 softmax** — [src/math.ts](src/math.ts), 29~41행

```typescript
export function softmax(logits: ArrayLike<number>): Float64Array {
  positiveInteger(logits.length, '출력 크기');
  let maximum = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (!Number.isFinite(logits[i])) throw new Error('logit은 유한한 값이어야 합니다.');
    maximum = Math.max(maximum, logits[i]);
  }
  const probabilities = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    probabilities[i] = Math.exp(logits[i] - maximum);
    sum += probabilities[i];
  }
```

**누적 확률로 다음 후보 추첨** — [src/math.ts](src/math.ts), 71~78행

```typescript
  const threshold = draw * total;
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (threshold < cumulative) return i;
  }
  return lastPositive;
}
```

## 5. nlm.ts — 학습하는 숫자와 완전연결 구조

```text
E 임베딩 표 → 입력 벡터 → 은닉층 → 출력층
```

대표 설정은 어휘 V=11, 임베딩 D=2, 은닉 뉴런 H=8입니다. E는 모든 단어가 공유하는 임베딩 표입니다. 단어 두 개의 행을 가져와 이어 붙인 입력 크기는 2D=4입니다.

W1은 입력의 모든 성분과 모든 은닉 뉴런을 연결하고, W2는 모든 은닉 뉴런과 모든 출력 후보를 연결합니다. 둘 다 완전연결층입니다. b1과 b2는 각 뉴런 계산에 더해지는 편향입니다.

행렬은 Float64Array 하나에 행 단위로 펼쳐 저장합니다. W1[i×H+h]는 입력 성분 i에서 은닉 뉴런 h로 가는 가중치입니다.

- E: 22개 / W1: 32개 / b1: 8개 / W2: 88개 / b2: 11개. 총 161개 숫자를 학습합니다.
- H=6으로 바꾸면 입력→은닉 연결은 4×6=24개입니다. 6개로 고정한 구조는 아닙니다.
- 같은 단어는 입력의 첫 번째·두 번째 위치에서 같은 E 행을 사용합니다.

**다섯 종류의 학습 변수 초기화** — [nplm/nlm.ts](nplm/nlm.ts), 30~36행

```typescript
    this.parameters = {
      E: weights(vocabularySize * dimension, 0.5),
      W1: weights(2 * dimension * hiddenSize, Math.sqrt(2 / (2 * dimension + hiddenSize))),
      b1: new Float64Array(hiddenSize),
      W2: weights(hiddenSize * vocabularySize, Math.sqrt(2 / (hiddenSize + vocabularySize))),
      b2: new Float64Array(vocabularySize),
    };
```

## 6. 출력 뉴런은 왜 11개인가?

```text
입력 단어 → 임베딩 입력 → 은닉 뉴런 → 출력 뉴런
```

우리처럼 다음 토큰을 softmax로 예측하는 언어 모델의 출력 뉴런 수는 어휘 크기 V로 정해집니다. V는 지금 입력한 두 단어의 개수가 아니라, 학습 데이터로 구성한 어휘표 전체의 크기입니다.

이 데이터에는 일반 단어 8개와 특수 토큰 3개가 있으므로 V=11입니다. 입력이 철수는·사과를 두 단어이고 임베딩이 2차원이어도 출력은 11개입니다. 두 단어를 2차원으로 바꾼 결과는 은닉층에 들어가는 숫자 4개입니다.

같은 신경망이 문장의 모든 위치를 예측합니다. 시작에서는 주체, 그다음에는 과일, 이어서 동사, 마지막에는 eos가 필요하므로 동사 두 개만 출력하는 구조로 만들지 않았습니다.

각 출력 뉴런은 후보 하나의 점수를 만듭니다. 11개 점수를 확률로 바꾸고 그중 다음 토큰 하나를 추첨합니다. 한 번에 단어 11개를 생성한다는 뜻은 아닙니다.

- 임베딩 차원 D: 한 단어를 몇 개의 숫자로 나타낼지 정합니다.
- 은닉 뉴런 수 H: 설계자가 정하는 모델 크기입니다. 어휘 8개와 은닉 8개가 같아야 하는 규칙은 없습니다.
- 출력 후보 수 V: 모델이 예측할 토큰 종류 수입니다. 어휘가 100개인 같은 방식의 언어 모델이면 출력 뉴런도 100개입니다.
- 좋아해·싫어해 두 종류만 분류하는 별도 문제라면 softmax 출력 2개로 설계할 수 있습니다. 현재는 문장 전체의 다음 토큰을 생성하므로 모든 11개 후보를 사용합니다.

**데이터의 전체 토큰 종류 수를 모델에 전달** — [src/experiment.ts](src/experiment.ts), 111~111행

```typescript
        const model = new NeuralLanguageModel(dataset.tokens.length, dimension, settings.hidden, seed);
```

**어휘 수 V만큼 출력 점수를 만드는 실제 반복문** — [nplm/nlm.ts](nplm/nlm.ts), 64~70행

```typescript
    const logits = new Float64Array(V);
    for (let v = 0; v < V; v++) {
      let value = b2[v];
      for (let h = 0; h < H; h++) value += hidden[h] * W2[h * V + v];
      logits[v] = value;
    }
    return { input, hiddenInput, hidden, logits, probabilities: softmax(logits) };
```

## 7. nlm.ts — 순전파: 임베딩에서 확률까지

```text
단어 ID → 임베딩 조회 → 은닉 계산 → 출력 점수 → 확률
```

forward는 입력에서 출력 방향으로 계산합니다. 먼저 두 ID에 해당하는 E의 행을 복사해 길이 2D의 input을 만듭니다.

각 은닉 뉴런은 입력 4개를 모두 가중합하고 편향을 더한 뒤 tanh를 적용합니다. tanh의 결과는 -1~1 범위지만, E의 임베딩 좌표가 이 범위에 제한되는 것은 아닙니다.

출력층은 은닉 값 8개를 후보마다 다른 가중치로 합쳐 logit 11개를 만듭니다. softmax 결과가 다음 토큰 확률 벡터입니다.



```text
x = concat(E[앞 단어], E[뒤 단어])
h = tanh(x W1 + b1)
z = h W2 + b2
p = softmax(z)
```

**배열과 반복문으로 임베딩·은닉층·출력층 계산** — [nplm/nlm.ts](nplm/nlm.ts), 52~70행

```typescript
    const input = new Float64Array(2 * D);
    for (let position = 0; position < 2; position++) {
      for (let d = 0; d < D; d++) input[position * D + d] = E[context[position] * D + d];
    }
    const hiddenInput = new Float64Array(H);
    const hidden = new Float64Array(H);
    for (let h = 0; h < H; h++) {
      let value = b1[h];
      for (let i = 0; i < input.length; i++) value += input[i] * W1[i * H + h];
      hiddenInput[h] = value;
      hidden[h] = Math.tanh(value);
    }
    const logits = new Float64Array(V);
    for (let v = 0; v < V; v++) {
      let value = b2[v];
      for (let h = 0; h < H; h++) value += hidden[h] * W2[h * V + v];
      logits[v] = value;
    }
    return { input, hiddenInput, hidden, logits, probabilities: softmax(logits) };
```

## 8. math.ts·nlm.ts — 크로스 엔트로피와 PPL

```text
정답 토큰 → 정답 확률 → 한 문제 loss → 24쌍 평가
```

정답 토큰 하나가 정해진 다중분류 문제의 크로스 엔트로피는 -ln(정답 확률)입니다. 정답에 높은 확률을 줄수록 0에 가까워집니다. loss 자체는 퍼센트가 아닙니다.

코드의 crossEntropy는 출력 점수를 받아 logsumexp(z)-z[target]을 계산합니다. softmax를 구한 뒤 -ln(p[target])을 계산하는 것과 같은 식이지만, 작은 확률의 반올림·언더플로에 더 안정적입니다.

모델의 loss는 24쌍 손실의 평균입니다. PPL은 그 평균에 exp를 적용한 값입니다. 실험 표에서는 seed별 PPL을 계산한 후 평균내므로, 재생에 표시되는 seed123 하나의 값과 구분합니다.

- 정답 확률 99% → loss 약 0.01 / 1% → 약 4.61. 틀렸다고 loss가 100%가 되는 계산은 아닙니다.
- 표시가 0.00%라고 실제 수학적 확률이 정확히 0이라고 단정할 수는 없습니다.
- 학습 데이터 PPL은 학습 적합도입니다. 새 문장 생성 개수나 별도 평가 데이터 성능과 같은 지표가 아닙니다.

```text
한 문제 loss = -ln(p[정답])
배치 loss = (모든 문제의 loss 합) / 24
PPL = exp(배치 loss)
```

**안정적인 크로스 엔트로피 계산** — [src/math.ts](src/math.ts), 47~57행

```typescript
export function crossEntropy(logits: ArrayLike<number>, target: number): number {
  tokenId(target, logits.length);
  let maximum = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (!Number.isFinite(logits[i])) throw new Error('logit은 유한한 값이어야 합니다.');
    maximum = Math.max(maximum, logits[i]);
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - maximum);
  return maximum - logits[target] + Math.log(sum);
}
```

**24개 문제의 평균 손실** — [nplm/nlm.ts](nplm/nlm.ts), 73~78행

```typescript
  loss(pairs: readonly Pair[]): number {
    if (pairs.length === 0) throw new Error('학습 쌍이 필요합니다.');
    let sum = 0;
    for (const pair of pairs) sum += crossEntropy(this.forward(pair.context).logits, pair.target);
    return sum / pairs.length;
  }
```

## 9. nlm.ts — 역전파를 직접 계산하기

```text
출력 기울기 → W2·b2 → 은닉 기울기 → W1·b1 → E에 누적
```

역전파는 각 학습 변수가 조금 바뀔 때 loss가 얼마나 변하는지 계산합니다. 이 민감도를 기울기라고 합니다. 자동 미분 라이브러리를 사용하지 않고 미분식을 반복문으로 작성했습니다.

softmax와 크로스 엔트로피를 합친 출력 점수의 기울기는 p-y입니다. y는 정답 위치만 1인 벡터입니다. 코드에서는 정답 위치에만 1을 빼고, 평균 손실에 맞춰 문제 수 B로 나눕니다.

그 기울기를 W2, tanh, W1 순서로 거슬러 보내고 마지막에는 E의 해당 단어 행에 더합니다. 같은 단어가 여러 위치에 나와도 기울기를 덮어쓰지 않고 누적합니다.



```text
dz = (p - y) / B
dh = (dz W2ᵀ) ⊙ (1 - h²)
dE[해당 단어] += 입력 쪽으로 전달된 기울기
```

**출력에서 은닉층으로 기울기 전달** — [nplm/nlm.ts](nplm/nlm.ts), 92~110행

```typescript
      const dz = probabilities.slice();
      dz[target] -= 1;
      for (let v = 0; v < V; v++) {
        dz[v] /= pairs.length;
        gradients.b2[v] += dz[v];
      }

      const dh = new Float64Array(H);
      for (let h = 0; h < H; h++) {
        let value = 0;
        for (let v = 0; v < V; v++) {
          gradients.W2[h * V + v] += hidden[h] * dz[v];
          value += W2[h * V + v] * dz[v];
        }
        dh[h] = value * (1 - hidden[h] * hidden[h]);
        gradients.b1[h] += dh[h];
      }

      const dx = new Float64Array(2 * D);
```

**공유 임베딩 표에 기울기 누적** — [nplm/nlm.ts](nplm/nlm.ts), 118~124행

```typescript
      // 같은 단어의 여러 위치·여러 쌍에서 온 gradient를 같은 E 행에 누적한다.
      for (let position = 0; position < 2; position++) {
        for (let d = 0; d < D; d++) {
          gradients.E[context[position] * D + d] += dx[position * D + d];
        }
      }
    }
```

## 10. nlm.ts — 경사하강법으로 값을 갱신하기

```text
24쌍 전체 → 역전파 완료 → 경사하강법 → 다음 갱신
```

step은 역전파가 계산한 기울기를 사용해 실제 숫자를 바꿉니다. 기울기가 양수이면 값을 줄이고, 음수이면 값을 늘립니다. 작게 만들려는 대상은 모든 가중치가 아니라 loss입니다.

현재는 24쌍 전체로 한 번의 평균 기울기를 구하는 full-batch 경사하강법입니다. 한 문제마다 가중치를 바꾸는 SGD나 Adam을 구현한 것은 아닙니다.

모든 기울기를 구한 뒤 E·W1·b1·W2·b2를 갱신합니다. 역전파 중간에 가중치를 먼저 바꾸지 않습니다. 학습률이 너무 크면 loss가 오히려 커질 수 있으므로 실험에서는 여러 값을 비교했습니다.



```text
새 값 = 기존 값 - learningRate × gradient
```

**다섯 변수 그룹을 갱신하는 실제 코드** — [nplm/nlm.ts](nplm/nlm.ts), 128~136행

```typescript
  step(gradients: Parameters, learningRate: number): void {
    if (!Number.isFinite(learningRate) || learningRate <= 0) throw new Error('학습률은 양수여야 합니다.');
    for (const name of PARAMETER_NAMES) {
      if (gradients[name].length !== this.parameters[name].length) throw new Error('gradient 크기가 다릅니다.');
      for (let i = 0; i < gradients[name].length; i++) {
        this.parameters[name][i] -= learningRate * gradients[name][i];
      }
    }
  }
```

**손실·역전파·갱신의 반복** — [nplm/nlm.ts](nplm/nlm.ts), 143~150행

```typescript
  for (let epoch = 0; epoch < epochs; epoch++) {
    const { loss, gradients } = model.lossAndGradients(pairs);
    if (!Number.isFinite(loss)) throw new Error(`${epoch + 1}번째 학습에서 loss가 유한하지 않습니다.`);
    // 업데이트 직전의 loss는 앞선 업데이트 횟수에 대응한다.
    losses.push(loss);
    model.step(gradients, learningRate);
  }
  const finalLoss = model.loss(pairs);
```

## 11. generation.ts — 생성과 분류를 분리하기

```text
시작 문맥 → 확률 계산 → 추첨 → 종료 → 분류
```

생성기는 모델에 두 토큰을 주고 확률을 받습니다. sample로 다음 토큰을 추첨하고 문맥을 한 칸 옮겨 반복합니다. 원문 문장을 검색해서 다음 토큰을 고르는 과정은 없습니다.

생성이 끝난 뒤 원문 6개와 비교합니다. 원문과 같으면 training, 주체+과일+동사의 3토큰이며 원문에 없으면 novel, 나머지는 invalid입니다. <eos>로 끝나야 원문·새 조합으로 인정합니다.

출력 후보를 주체·과일·동사로 강제 제한하지 않습니다. 따라서 특수 토큰이 본문에 나오거나 순서·길이가 틀린 결과도 그대로 집계합니다.

- 새 조합 17개는 17종을 뜻하지 않습니다. 대표 설정에서는 6종이 중복을 포함해 총 17회 나왔습니다.
- 원문과 다른 107개 중 새 조합은 17개, 오류·미완성은 90개입니다. 1.07%와 0.17%의 분류 기준이 다릅니다.
- 최대 본문 길이는 20토큰이며, 그 뒤에 오는 eos도 한 번 확인합니다.

**모델 확률로 다음 토큰을 생성** — [generation/generation.ts](generation/generation.ts), 52~61행

```typescript
    const next = sample(probabilities, random);
    if (next === dataset.ids.get(EOS)) {
      termination = 'eos';
      break;
    }
    if (position === maxTokens) break;
    tokens.push(dataset.tokens[next]);
    context = [context[1], next];
  }
  return { tokens, sentence: tokens.join(' '), termination, category: classify(tokens, termination, dataset) };
```

**생성이 끝난 뒤에만 원문과 비교** — [generation/generation.ts](generation/generation.ts), 26~35행

```typescript
export function classify(tokens: readonly string[], termination: Termination, dataset: Dataset): Category {
  if (termination !== 'eos') return 'invalid';
  if (dataset.sentences.includes(tokens.join(' '))) return 'training';
  // 이 규칙은 생성 후의 분류에만 사용한다. 생성 확률을 제한하지 않는다.
  if (tokens.length === 3 && SUBJECTS.some((word) => word === tokens[0])
    && OBJECTS.some((word) => word === tokens[1]) && VERBS.some((word) => word === tokens[2])) {
    return 'novel';
  }
  return 'invalid';
}
```

## 12. experiment.ts — 같은 조건으로 모델 비교

```text
설정 조합 → 신규 모델 → 학습 → 평가·생성 → 저장
```

임베딩 차원 1·2·3, 학습률 0.05·0.1·0.3·1.0, 초기화 seed 123·456·789로 신경망을 36회 학습합니다. 같은 D와 seed에서는 학습률이 달라도 같은 초기 가중치로 시작합니다.

각 설정의 loss·PPL은 세 학습 seed를 기록합니다. 문장 생성에는 첫 번째 학습 seed의 모델 하나를 사용하고, 추첨 seed는 2026으로 따로 고정합니다.

신경망 12설정과 트라이그램 2종에서 각각 10,000개를 생성하므로 총 140,000개입니다. D가 바뀌면 매개변수 수도 바뀌므로 모든 조건이 완전히 같다고 해석하지 않습니다.

- comparison.json: 모든 설정·좌표·확률·생성 집계. loss.csv: 갱신별 loss·PPL.
- learning_rates.csv: 설정별 표. samples/*.jsonl: 생성 결과 한 개당 한 줄.
- distributions/*.csv: 문장별 횟수. 각 설정의 생성 개수와 분류 합계를 함께 기록합니다.

**조건별 독립 모델을 학습하는 반복문** — [src/experiment.ts](src/experiment.ts), 106~113행

```typescript
  for (const dimension of settings.dimensions) {
    for (const learningRate of settings.learningRates) {
      const runs: Run[] = [];
      let primary: NeuralLanguageModel | undefined;
      for (const seed of settings.seeds) {
        const model = new NeuralLanguageModel(dataset.tokens.length, dimension, settings.hidden, seed);
        const embeddingBefore = dataset.tokens.map((_, id) => Array.from(model.parameters.E.slice(id * dimension, (id + 1) * dimension)));
        const losses = train(model, dataset.pairs, settings.epochs, learningRate);
```

**학습 지표와 임베딩 좌표 기록** — [src/experiment.ts](src/experiment.ts), 114~121행

```typescript
        const finalLoss = losses.at(-1)!;
        runs.push({
          seed, initialLoss: losses[0], finalLoss, perplexity: Math.exp(finalLoss), embeddingBefore,
          embeddingAfter: dataset.tokens.map((_, id) => Array.from(model.parameters.E.slice(id * dimension, (id + 1) * dimension))),
        });
        losses.forEach((loss, update) => lossRows.push([dimension, settings.hidden, learningRate, seed, update, loss, Math.exp(loss)]));
        if (!primary) primary = model;
      }
```

## 13. cli.ts·report.ts — 명령과 결과 문서

```text
명령 인자 → 검사 → 실행 → 문서화
```

cli.ts는 명령줄의 입구입니다. 라이브러리 없이 process.argv를 읽고 허용한 옵션인지 검사합니다. data는 데이터 구조, trigram은 횟수와 확률, trace는 초기 모델 한 문제의 계산, compare는 전체 비교 실험을 실행합니다.

report.ts는 이미 계산된 비교 결과를 표·텍스트·HTML로 바꿉니다. 여기서 새로 학습하지 않습니다. 1·2·3차원 임베딩의 좌표와 ASCII 위치도 문자열로 그립니다.

trace는 학습이 끝난 저장 모델의 조회가 아니라, 초기 모델의 한 문제 순전파·역전파·한 번 갱신을 보여주는 명령이라는 점을 구분합니다.



**한 문제의 계산을 살펴보는 명령** — [src/cli.ts](src/cli.ts), 70~77행

```typescript
  if (command === 'trace') {
    const context = encodeContext(dataset, (flags.get('context') ?? '철수는,사과를').split(',').map((word) => word.trim()));
    const target = dataset.ids.get(flags.get('target') ?? '좋아해');
    if (target === undefined) throw new Error('정답 토큰이 어휘에 없습니다.');
    const model = new NeuralLanguageModel(dataset.tokens.length, numeric('dim', 2), numeric('hidden', DEFAULT_SETTINGS.hidden), numeric('seed', 123));
    const forward = model.forward(context);
    const { loss, gradients } = model.lossAndGradients([{ context, target }]);
    const vector = (values: ArrayLike<number>) => Array.from(values, (value) => Number(value.toFixed(6)));
```

**표도 직접 문자열로 생성** — [src/report.ts](src/report.ts), 22~26행

```typescript
function markdownTable(headers: readonly string[], rows: readonly (readonly (string | number)[])[]): string {
  return `| ${headers.join(' | ')} |\n| ${headers.map(() => '---').join(' | ')} |\n` +
    rows.map((row) => `| ${row.join(' | ')} |`).join('\n');
}

```

## 14. animation.ts — 학습 중간 가중치 기록

```text
같은 초기화 → 원래 역전파 → 20회마다 복사 → 매 갱신 loss
```

애니메이션을 위해 새로운 학습 알고리즘을 만든 것이 아닙니다. 원래 NeuralLanguageModel의 lossAndGradients와 step을 같은 순서로 호출하면서 중간 숫자를 복사합니다.

0회 갱신의 초기 상태부터 2,000회까지 20회 간격으로 101개 가중치 스냅샷을 저장합니다. loss는 0~2,000회에 대해 모두 기록하므로 2,001개입니다.

가중치 배열을 복사해 저장해야 이후 학습이 이전 기록까지 바꾸지 않습니다. animate.ts는 끝점 가중치를 복원하고, 기존 비교 결과가 있으면 loss·임베딩·생성 집계를 대조합니다.



**원래 계산을 재사용하면서 중간 상태 복사** — [src/animation.ts](src/animation.ts), 37~50행

```typescript
  for (let update = 0; update <= settings.updates; update++) {
    const step = update < settings.updates ? model.lossAndGradients(dataset.pairs) : undefined;
    const loss = step?.loss ?? model.loss(dataset.pairs);
    if (!Number.isFinite(loss)) throw new Error(`${update}회 갱신에서 loss가 유한하지 않습니다.`);
    losses.push(loss);
    if (update % settings.recordEvery === 0 || update === settings.updates) {
      frames.push({ update, loss,
        parameters: Object.fromEntries(PARAMETER_NAMES.map(name => [name, Array.from(model.parameters[name])])) as Record<ParameterName, number[]> });
    }
    if (step) model.step(step.gradients, settings.learningRate);
  }
  return { settings: { ...settings }, data: { tokens: dataset.tokens, sentences: dataset.sentences, pairs: dataset.pairs.length }, losses, frames };
}

```

## 15. animate.ts — 기록 복원·검산·재생 파일

```text
학습 기록 → 모델 복원 → 확률 검산 → 10,000회 생성 → HTML·JSON
```

animate.ts는 기록을 만들고 최종 가중치를 복원해 10,000개를 다시 생성합니다. 기존 comparison.json이 같은 설정이면 최종 loss·임베딩·생성 집계를 대조합니다.

중간 시점마다 세 입력 문맥의 확률도 계산해 브라우저의 순전파와 대조할 기준으로 저장합니다. 또한 주체×과일×동사의 18문장에 대해 시작부터 eos까지 확률을 곱해 원문·새 조합·기타 출력 확률을 합산합니다.

재생 HTML은 프로젝트의 SVG·차트·재생 코드를 포함합니다. 대화용 파일은 Node 기본 gzip과 브라우저 기본 압축 해제 기능으로 크기만 줄입니다. 값의 정밀도를 줄이거나 외부 서버를 호출하지 않습니다.

- 화면의 새 조합 확률 약 0.15%는 확률 합산값입니다. 실제 10,000회 추첨에서는 17회, 0.17%였습니다.
- 선택 문맥 다음 좋아해 확률과 전체 문장 생성 확률은 다릅니다. 전체 생성은 그 문맥까지 먼저 도달해야 합니다.
- TypeScript 모델의 끝점과 loss 기록은 기존 실험과 일치합니다.

**전체 문장 확률은 각 단계의 확률을 곱해 계산** — [src/animate.ts](src/animate.ts), 47~56행

```typescript
    let probability = 1;
    for (const word of [...words, EOS]) {
      const next = dataset.ids.get(word)!;
      probability *= model.forward(context).probabilities[next];
      context = [context[1], next];
    }
    if (dataset.sentences.includes(words.join(' '))) training += probability;
    else novel += probability;
  }
  const invalid = 1 - training - novel;
```

## 16. svg.js — 그래프의 바닥 계산도 직접 구현

```text
모델 숫자 → 범위 계산 → 픽셀 변환 → SVG 생성
```

제거한 D3 대신 document.createElementNS로 SVG 요소를 직접 만듭니다. add는 요소 생성, set은 속성 지정, linear는 데이터 숫자와 화면 좌표 사이의 변환을 담당합니다.

niceDomain은 데이터 전체를 포함하는 축 범위와 읽기 쉬운 눈금을 고릅니다. bottomAxis·leftAxis는 선과 글자를 직접 배치합니다. path는 좌표들을 M·L 명령으로 연결합니다.

이 파일은 학습이나 확률을 계산하지 않습니다. 같은 모델 숫자를 화면의 어느 위치에 그릴지 결정하는 작은 내부 도구입니다. 외부 차트 엔진을 감싼 래퍼가 아닙니다.



```text
픽셀 = r0 + (데이터값-d0)/(d1-d0) × (r1-r0)
```

**브라우저 기본 API로 SVG 요소 생성** — [src/svg.js](src/svg.js), 13~20행

```javascript
  function add(parent, name, attributes = {}, text) {
    const node = document.createElementNS(namespace, name);
    set(node, attributes);
    if (text !== undefined) node.textContent = String(text);
    parent.append(node);
    return node;
  }
  function range(count) {
```

**직접 작성한 좌표 변환과 역변환** — [src/svg.js](src/svg.js), 34~41행

```javascript
  function linear(domain, output) {
    const [d0, d1] = domain, [r0, r1] = output;
    if (![d0,d1,r0,r1].every(Number.isFinite)) throw new Error('좌표 변환 범위는 유한해야 합니다.');
    const scale = value => d0 === d1 ? (r0+r1)/2 : r0+(value-d0)/(d1-d0)*(r1-r0);
    scale.invert = value => r0 === r1 ? (d0+d1)/2 : d0+(value-r0)/(r1-r0)*(d1-d0);
    return scale;
  }
  function niceDomain(minimum, maximum, count = 5) {
```

## 17. training-charts.js — 신경망·확률·곡선·산점도

```text
가중치 기록 → 화면 구조 → 현재 시점 → 차트 표시
```

drawNetwork는 4×8과 8×11 연결선을 이중 반복문으로 만듭니다. 각 시점의 실제 가중치 절댓값으로 선 진하기를, 실제 tanh 결과로 은닉 뉴런의 값을 갱신합니다.

drawProbabilities는 11개 토큰의 확률 막대를 그립니다. drawLoss는 loss와 PPL의 실제 기록을 선으로 연결하고 현재 시점까지 보여줍니다. drawEmbedding은 E의 두 성분을 x·y 좌표로 바꿔 단어 위치와 이동 경로를 그립니다.

축 범위는 재생 전체의 데이터를 보고 고정해, 학습 중 축 자체가 바뀌어 움직임을 왜곡하지 않도록 했습니다. 이름이 겹치면 주변 후보 위치를 검사하고 연결선을 사용합니다.



**완전연결 선을 직접 생성하는 반복문** — [src/training-charts.js](src/training-charts.js), 37~41행

```javascript
    const inputEdges=range(4).flatMap(i=>range(H).map(h=>({i,h,node:add(svg,'line',{class:'weight',x1:eX,y1:eY[i],x2:hX,y2:hY(h),stroke:'var(--muted-foreground)','stroke-width':.8})})));
    const outputEdges=range(H).flatMap(h=>range(V).map(v=>({h,v,node:add(svg,'line',{class:'weight',x1:hX,y1:hY(h),x2:oX,y2:oY(v),'stroke-width':.8})})));
    const contextLabels=[106,266].map(y=>add(svg,'text',{x:eX,y,'text-anchor':'middle'}));
    const inputs=range(4).map(i=>{add(svg,'circle',{cx:eX,cy:eY[i],r:8,fill:'var(--chart-1)'});return add(svg,'text',{x:eX-13,y:eY[i]+4,'text-anchor':'end'});});
    const hidden=range(H).map(h=>add(svg,'circle',{cx:hX,cy:hY(h),r:15,fill:'var(--chart-4)',stroke:'var(--border)'}));
```

**loss·PPL 값을 화면 좌표로 변환** — [src/training-charts.js](src/training-charts.js), 74~87행

```javascript
  function drawLoss(ppl=false){
    const{svg,width}=canvas(ppl?'[data-ppl-chart]':'[data-loss-chart]',320);
    const values=ppl?data.losses.map(value=>Math.exp(value)):data.losses;
    const left=64,right=width-20,top=22,bottom=266;
    const [minimum,maximum]=extent(values),padding=(maximum-minimum)*.08;
    const domain=niceDomain(Math.max(0,minimum-padding),maximum+padding,5);
    const x=linear([0,data.settings.updates],[left+5,right-5]),y=linear(domain.domain,[bottom-5,top+5]);
    frame(svg,left,top,right-left,bottom-top);
    const ticks=width<500?[0,data.settings.updates/2,data.settings.updates]:range(5).map(i=>i*data.settings.updates/4);
    bottomAxis(svg,x,ticks,bottom,value=>String(Math.round(value)));leftAxis(svg,y,domain.ticks,left);
    label(svg,'x','가중치 갱신 (회)',(left+right)/2,313);
    label(svg,'y',ppl?'PPL':'평균 loss',15,(top+bottom)/2,`rotate(-90,15,${(top+bottom)/2})`);
    const clipId=ppl?'chapter4-ppl-reveal':'chapter4-loss-reveal';
    const clip=add(add(add(svg,'defs'),'clipPath',{id:clipId,clipPathUnits:'userSpaceOnUse'}),'rect',{class:'moving',x:0,y:0,width:right-left,height:bottom-top});
```

## 18. training-animation.js·HTML — 재생 조작

```text
재생·슬라이더 → 가중치 선택 → 순전파 재계산 → 그래프 갱신
```

HTML 템플릿은 버튼·선택 상자·슬라이더와 SVG가 들어갈 자리를 정의합니다. training-animation.js는 그 요소들에 이벤트를 연결하고 현재 시점·문맥·그래프 종류를 관리합니다.

requestAnimationFrame으로 재생 상태를 확인하며 200ms마다 다음 저장 기록을 보여줍니다. 끝에 도달하면 멈추고 자동 반복하지 않습니다. 슬라이더를 움직이면 일시정지한 뒤 선택한 기록으로 이동합니다.

브라우저는 학습을 새로 하지 않고 저장된 가중치의 순전파를 계산합니다. 이 확률을 TypeScript 모델이 저장한 기준값과 모든 기록 시점에서 비교합니다. 동작 줄이기 설정에서는 수동으로 한 기록씩 넘깁니다.

- 재생 20초는 학습 실행 시간 측정값이 아닙니다. 학습 결과를 설명하기 위한 재생 속도입니다.
- 입력 문맥 선택은 관찰 대상을 바꿉니다. 학습 데이터나 가중치를 수정하지 않습니다.
- 화면의 값은 표시를 위해 반올림하지만, 저장된 가중치와 모델 계산은 원래 정밀도를 유지합니다.

**재생·끝점 정지** — [src/training-animation.js](src/training-animation.js), 89~99행

```javascript
  function tick(time) {
    if (!state.playing) return;
    if (document.hidden) { pause(); return; }
    if (time - state.lastTime >= 200) {
      state.lastTime = time;
      setFrame(state.index + 1);
      if (state.index === frames.length - 1) { pause(); remember(); return; }
    }
    state.raf = requestAnimationFrame(tick);
  }
  function play() {
```

**기본 HTML 요소로 조작부 정의** — [src/training-animation.template.html](src/training-animation.template.html), 40~52행

```html
  <div class="controls">
    <label class="button-hit" for="training-play"><button class="btn btn-primary" type="button" id="training-play" data-play>재생</button></label>
    <label class="button-hit" for="training-reset"><button class="btn" type="button" id="training-reset" data-reset>처음으로</button></label>
    <label class="field form-label">그래프
      <select class="form-select" data-view-select>
        <option value="network">신경망·다음 토큰 확률</option>
        <option value="embedding">임베딩 이동</option>
        <option value="loss">loss·PPL 곡선</option>
      </select>
    </label>
  </div>
  <label class="timeline form-label">가중치 갱신 <span data-update>0 / 2,000회</span>
    <input class="form-range" type="range" min="0" max="100" step="1" value="0" data-time aria-label="학습 기록 시점">
```

## 19. gradcheck.ts — 직접 미분한 기울기 검산

```text
직접 역전파 → 양쪽으로 이동 → 수치 미분 → 오차 대조
```

일반 테스트 파일 대신 scripts/gradcheck.ts 하나를 남겼습니다. 모델 실행에 테스트 파일이 필수인 것은 아니지만, 자동 미분 없이 작성한 역전파를 독립적인 계산으로 검산하는 것은 이 실습의 핵심입니다.

원래 여섯 문장에서 만든 24쌍을 사용합니다. 임베딩 D=1·2·3, 은닉 H=8에서 E·W1·b1·W2·b2의 전체 483개 성분을 중심 차분과 대조합니다. 같은 단어가 여러 입력에 반복되는 경우도 포함됩니다.

npm run gradcheck로 실행합니다. 일반 테스트 러너는 사용하지 않습니다. TypeScript의 타입 검사는 npm run typecheck로 따로 수행합니다.



```text
수치 기울기 = (L(w + epsilon) - L(w - epsilon)) / (2 × epsilon)
```

**양쪽 loss로 수치 미분을 구해 직접 미분과 대조** — [scripts/gradcheck.ts](scripts/gradcheck.ts), 17~28행

```typescript
      const original = model.parameters[name][i];
      let plus = 0, minus = 0;
      try {
        model.parameters[name][i] = original + epsilon;
        plus = model.loss(dataset.pairs);
        model.parameters[name][i] = original - epsilon;
        minus = model.loss(dataset.pairs);
      } finally {
        model.parameters[name][i] = original;
      }
      const numerical = (plus - minus) / (2 * epsilon);
      maximumError = Math.max(maximumError, Math.abs(numerical - gradients[name][i]));
```

## 20. 이 해설 문서의 생성과 코드 읽는 순서

```text
해설 정의 → 실제 파일 읽기 → 그림·문서 → 검토
```

이 해설의 코드 블록은 별도의 예제 코드를 다시 작성한 것이 아니라, 현재 파일에서 실제 줄을 읽어 발췌한 것입니다. 각 블록에 파일명과 줄 범위를 표시합니다.

scripts/build-guide.ts가 설명 정의와 실제 소스를 합쳐 IMPLEMENTATION.md와 이 HTML을 만듭니다. 문서의 도형도 svg.js로 직접 그립니다. 문서 생성에도 마크다운·다이어그램 라이브러리를 사용하지 않습니다.

처음에는 data.ts → trigram.ts → math.ts → nlm.ts → generation.ts 순서로 읽고, 그다음 실험·리포트·재생 코드를 읽는 것이 좋습니다. 모델 계산과 화면 표시를 구분하면서 한 입력 문맥을 따라가면 흐름을 확인할 수 있습니다.



**현재 소스에서 직접 발췌해 문서 생성** — [scripts/build-guide.ts](scripts/build-guide.ts), 18~27행

```typescript
const prepared = definitions.map(section => ({
  ...section,
  snippets: section.snippets.map(snippet => {
    const source = readFileSync(join(root, snippet.file), 'utf8').split('\n');
    const start = source.findIndex(line => line.includes(snippet.anchor));
    assert.ok(start >= 0, `${snippet.file}: 코드 구간을 찾을 수 없습니다: ${snippet.anchor}`);
    const end = Math.min(start + snippet.lines, source.length);
    return { ...snippet, start: start + 1, end, code: source.slice(start, end).join('\n') };
  }),
}));
```
