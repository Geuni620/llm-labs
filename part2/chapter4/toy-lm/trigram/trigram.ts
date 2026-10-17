import type { Context, Pair } from '../data/data.ts';
import { positiveInteger, tokenId } from '../src/math.ts';

export interface Distribution {
  contextCount: number;
  counts: number[];
  k: number;
  denominator: number;
  probabilities: number[] | null;
}
export interface Evaluation {
  loss: number | 'Infinity' | null;
  perplexity: number | 'Infinity' | null;
  undefinedContexts: number;
  zeroProbabilityTargets: number;
}

export class Trigram {
  readonly vocabularySize: number;
  private readonly counts = new Map<string, Float64Array>();
  private readonly totals = new Map<string, number>();

  constructor(vocabularySize: number) {
    positiveInteger(vocabularySize, '어휘 크기');
    this.vocabularySize = vocabularySize;
  }

  private key(context: Context): string {
    if (context.length !== 2) throw new Error('트라이그램 문맥 길이는 2입니다.');
    context.forEach((id) => tokenId(id, this.vocabularySize));
    return context.join(',');
  }

  fit(pairs: readonly Pair[]): void {
    if (pairs.length === 0) throw new Error('학습 쌍이 필요합니다.');
    this.counts.clear();
    this.totals.clear();
    for (const { context, target } of pairs) {
      const key = this.key(context);
      tokenId(target, this.vocabularySize);
      const counts = this.counts.get(key) ?? new Float64Array(this.vocabularySize);
      counts[target]++;
      this.counts.set(key, counts);
      this.totals.set(key, (this.totals.get(key) ?? 0) + 1);
    }
  }

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
    let undefinedContexts = 0;
    let zeroProbabilityTargets = 0;
    for (const { context, target } of pairs) {
      tokenId(target, this.vocabularySize);
      const distribution = this.distribution(context, k);
      if (distribution.probabilities === null) {
        undefinedContexts++;
      } else if (distribution.probabilities[target] === 0) {
        zeroProbabilityTargets++;
      } else {
        totalLoss -= Math.log(distribution.probabilities[target]);
      }
    }
    const loss = undefinedContexts > 0 ? null
      : zeroProbabilityTargets > 0 ? 'Infinity' : totalLoss / pairs.length;
    return {
      loss,
      perplexity: typeof loss === 'number' ? Math.exp(loss) : loss,
      undefinedContexts, zeroProbabilityTargets,
    };
  }
}
