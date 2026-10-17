import type { Context, Pair } from '../data/data.ts';
import { crossEntropy, normal, positiveInteger, randomGenerator, softmax, tokenId } from '../src/math.ts';

export const PARAMETER_NAMES = ['E', 'W1', 'b1', 'W2', 'b2'] as const;
export type ParameterName = typeof PARAMETER_NAMES[number];
export type Parameters = Record<ParameterName, Float64Array>;
export interface Forward {
  input: Float64Array;
  hiddenInput: Float64Array;
  hidden: Float64Array;
  logits: Float64Array;
  probabilities: Float64Array;
}

export class NeuralLanguageModel {
  readonly vocabularySize: number;
  readonly dimension: number;
  readonly hiddenSize: number;
  readonly parameters: Parameters;

  constructor(vocabularySize: number, dimension: number, hiddenSize: number, seed = 123) {
    positiveInteger(vocabularySize, '어휘 크기');
    positiveInteger(dimension, '임베딩 차원');
    positiveInteger(hiddenSize, '은닉 뉴런 수');
    this.vocabularySize = vocabularySize;
    this.dimension = dimension;
    this.hiddenSize = hiddenSize;
    const random = randomGenerator(seed);
    const weights = (size: number, scale: number) => Float64Array.from({ length: size }, () => normal(random) * scale);
    this.parameters = {
      E: weights(vocabularySize * dimension, 0.5),
      W1: weights(2 * dimension * hiddenSize, Math.sqrt(2 / (2 * dimension + hiddenSize))),
      b1: new Float64Array(hiddenSize),
      W2: weights(hiddenSize * vocabularySize, Math.sqrt(2 / (hiddenSize + vocabularySize))),
      b2: new Float64Array(vocabularySize),
    };
  }

  get parameterCount(): number {
    return PARAMETER_NAMES.reduce((sum, name) => sum + this.parameters[name].length, 0);
  }

  snapshot(): Parameters {
    return Object.fromEntries(PARAMETER_NAMES.map((name) => [name, this.parameters[name].slice()])) as Parameters;
  }

  forward(context: Context): Forward {
    if (context.length !== 2) throw new Error('입력 문맥 길이는 2입니다.');
    context.forEach((id) => tokenId(id, this.vocabularySize));
    const D = this.dimension, H = this.hiddenSize, V = this.vocabularySize;
    const { E, W1, b1, W2, b2 } = this.parameters;
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
  }

  loss(pairs: readonly Pair[]): number {
    if (pairs.length === 0) throw new Error('학습 쌍이 필요합니다.');
    let sum = 0;
    for (const pair of pairs) sum += crossEntropy(this.forward(pair.context).logits, pair.target);
    return sum / pairs.length;
  }

  lossAndGradients(pairs: readonly Pair[]): { loss: number; gradients: Parameters } {
    if (pairs.length === 0) throw new Error('학습 쌍이 필요합니다.');
    const gradients = Object.fromEntries(PARAMETER_NAMES.map((name) =>
      [name, new Float64Array(this.parameters[name].length)])) as Parameters;
    const D = this.dimension, H = this.hiddenSize, V = this.vocabularySize;
    const { W1, W2 } = this.parameters;
    let loss = 0;
    for (const { context, target } of pairs) {
      const { input, hidden, logits, probabilities } = this.forward(context);
      loss += crossEntropy(logits, target);

      // softmax + cross-entropy의 미분. 배치 평균도 여기에서 적용한다.
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
      for (let i = 0; i < input.length; i++) {
        for (let h = 0; h < H; h++) {
          gradients.W1[i * H + h] += input[i] * dh[h];
          dx[i] += W1[i * H + h] * dh[h];
        }
      }

      // 같은 단어의 여러 위치·여러 쌍에서 온 gradient를 같은 E 행에 누적한다.
      for (let position = 0; position < 2; position++) {
        for (let d = 0; d < D; d++) {
          gradients.E[context[position] * D + d] += dx[position * D + d];
        }
      }
    }
    return { loss: loss / pairs.length, gradients };
  }

  step(gradients: Parameters, learningRate: number): void {
    if (!Number.isFinite(learningRate) || learningRate <= 0) throw new Error('학습률은 양수여야 합니다.');
    for (const name of PARAMETER_NAMES) {
      if (gradients[name].length !== this.parameters[name].length) throw new Error('gradient 크기가 다릅니다.');
      for (let i = 0; i < gradients[name].length; i++) {
        this.parameters[name][i] -= learningRate * gradients[name][i];
      }
    }
  }
}

export function train(model: NeuralLanguageModel, pairs: readonly Pair[], epochs: number, learningRate: number): number[] {
  positiveInteger(epochs, '학습 횟수');
  if (!Number.isFinite(learningRate) || learningRate <= 0) throw new Error('학습률은 양수여야 합니다.');
  const losses: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    const { loss, gradients } = model.lossAndGradients(pairs);
    if (!Number.isFinite(loss)) throw new Error(`${epoch + 1}번째 학습에서 loss가 유한하지 않습니다.`);
    // 업데이트 직전의 loss는 앞선 업데이트 횟수에 대응한다.
    losses.push(loss);
    model.step(gradients, learningRate);
  }
  const finalLoss = model.loss(pairs);
  if (!Number.isFinite(finalLoss)) throw new Error('최종 loss가 유한하지 않습니다.');
  losses.push(finalLoss);
  return losses;
}
