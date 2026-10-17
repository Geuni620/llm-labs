import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { BOS, PAD, buildDataset, encodeContext, type Dataset } from '../data/data.ts';
import { explainGeneration, generateBatch, type GenerationSummary, type ProbabilityFunction } from '../generation/generation.ts';
import { positiveInteger, randomGenerator } from './math.ts';
import { NeuralLanguageModel, train } from '../nplm/nlm.ts';
import { Trigram, type Distribution, type Evaluation } from '../trigram/trigram.ts';

export interface Settings {
  dimensions: number[];
  hidden: number;
  learningRates: number[];
  epochs: number;
  seeds: number[];
  k: number;
  samples: number;
  sampleSeed: number;
  maxTokens: number;
}
export const DEFAULT_SETTINGS: Settings = {
  dimensions: [1, 2, 3], hidden: 8, learningRates: [0.05, 0.1, 0.3, 1],
  epochs: 2000, seeds: [123, 456, 789], k: 0.01, samples: 10000, sampleSeed: 2026, maxTokens: 20,
};
export interface Run {
  seed: number;
  initialLoss: number;
  finalLoss: number;
  perplexity: number;
  embeddingBefore: number[][];
  embeddingAfter: number[][];
}
export interface NeuralResult {
  dimension: number;
  hidden: number;
  learningRate: number;
  parameterCount: number;
  runs: Run[];
  lossMean: number;
  lossStd: number;
  perplexityMean: number;
  generation: GenerationSummary;
  sampleFile: string;
  probes: { context: string[]; probabilities: number[] }[];
  exampleSteps: ReturnType<typeof explainGeneration>[];
}
export interface Comparison {
  settings: Settings;
  nodeVersion: string;
  data: { sentences: string[]; tokens: string[]; pairs: number; contextLength: number };
  uniformPerplexity: number;
  trigram: {
    mle: { metrics: Evaluation; generation: GenerationSummary; sampleFile: string };
    smoothed: { metrics: Evaluation; generation: GenerationSummary; sampleFile: string };
    probes: { context: string[]; mle: Distribution; smoothed: Distribution }[];
  };
  neural: NeuralResult[];
}

function validate(settings: Settings): void {
  positiveInteger(settings.hidden, '은닉 뉴런 수');
  positiveInteger(settings.epochs, '학습 횟수');
  positiveInteger(settings.samples, '생성 개수');
  positiveInteger(settings.maxTokens, '최대 토큰 수');
  for (const [name, values] of [
    ['dimensions', settings.dimensions], ['learningRates', settings.learningRates], ['seeds', settings.seeds],
  ] as const) {
    if (values.length === 0 || new Set(values).size !== values.length) throw new Error(`${name}에는 서로 다른 값들이 필요합니다.`);
  }
  for (const d of settings.dimensions) if (![1, 2, 3].includes(d)) throw new Error('실험 임베딩 차원은 1, 2, 3입니다.');
  for (const lr of settings.learningRates) if (!Number.isFinite(lr) || lr <= 0) throw new Error('학습률은 양수여야 합니다.');
  for (const seed of [...settings.seeds, settings.sampleSeed]) randomGenerator(seed);
  if (!Number.isFinite(settings.k) || settings.k <= 0) throw new Error('스무딩 비교의 k는 양수여야 합니다.');
}

function csvCell(value: string | number): string {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
export function csv(rows: readonly (readonly (string | number)[])[]): string {
  return rows.map((row) => row.map(csvCell).join(',')).join('\n') + '\n';
}

function sampleAndWrite(name: string, probability: ProbabilityFunction, dataset: Dataset, settings: Settings, output: string) {
  const { records, summary } = generateBatch(probability, dataset, settings.samples, settings.sampleSeed, settings.maxTokens);
  const sampleFile = `samples/${name}.jsonl`;
  writeFileSync(join(output, sampleFile), records.map((record, i) => JSON.stringify({ index: i + 1, ...record })).join('\n') + '\n');
  writeFileSync(join(output, 'distributions', `${name}.csv`), csv([
    ['category', 'sentence', 'count', 'rate'],
    ...summary.distribution.map((row) => [row.category, row.sentence, row.count, row.count / settings.samples]),
  ]));
  return { summary, sampleFile };
}

export function runExperiment(settings: Settings, output: string, progress: (message: string) => void = () => {}): Comparison {
  validate(settings);
  mkdirSync(join(output, 'samples'), { recursive: true });
  mkdirSync(join(output, 'distributions'), { recursive: true });
  const dataset = buildDataset();
  const trigram = new Trigram(dataset.tokens.length);
  trigram.fit(dataset.pairs);
  const probeWords = [[PAD, BOS], [BOS, '철수는'], ['철수는', '사과를'], ['철수는', '바나나를'], ['짱구는', '사과를']];
  const mle = sampleAndWrite('trigram-mle', (ctx) => trigram.distribution(ctx).probabilities, dataset, settings, output);
  const smoothed = sampleAndWrite('trigram-add-k', (ctx) => trigram.distribution(ctx, settings.k).probabilities, dataset, settings, output);
  const lossRows: (string | number)[][] = [['dimension', 'hidden', 'learning_rate', 'seed', 'updates', 'loss', 'perplexity']];
  const neural: NeuralResult[] = [];
  for (const dimension of settings.dimensions) {
    for (const learningRate of settings.learningRates) {
      const runs: Run[] = [];
      let primary: NeuralLanguageModel | undefined;
      for (const seed of settings.seeds) {
        const model = new NeuralLanguageModel(dataset.tokens.length, dimension, settings.hidden, seed);
        const embeddingBefore = dataset.tokens.map((_, id) => Array.from(model.parameters.E.slice(id * dimension, (id + 1) * dimension)));
        const losses = train(model, dataset.pairs, settings.epochs, learningRate);
        const finalLoss = losses.at(-1)!;
        runs.push({
          seed, initialLoss: losses[0], finalLoss, perplexity: Math.exp(finalLoss), embeddingBefore,
          embeddingAfter: dataset.tokens.map((_, id) => Array.from(model.parameters.E.slice(id * dimension, (id + 1) * dimension))),
        });
        losses.forEach((loss, update) => lossRows.push([dimension, settings.hidden, learningRate, seed, update, loss, Math.exp(loss)]));
        if (!primary) primary = model;
      }
      const model = primary!;
      const probability: ProbabilityFunction = (ctx) => model.forward(ctx).probabilities;
      const generated = sampleAndWrite(`nlm-d${dimension}-lr${learningRate}`, probability, dataset, settings, output);
      const lossMean = runs.reduce((sum, run) => sum + run.finalLoss, 0) / runs.length;
      neural.push({
        dimension, hidden: settings.hidden, learningRate, parameterCount: model.parameterCount, runs, lossMean,
        lossStd: Math.sqrt(runs.reduce((sum, run) => sum + (run.finalLoss - lossMean) ** 2, 0) / runs.length),
        perplexityMean: runs.reduce((sum, run) => sum + run.perplexity, 0) / runs.length,
        generation: generated.summary, sampleFile: generated.sampleFile,
        probes: probeWords.map((words) => ({ context: words, probabilities: Array.from(model.forward(encodeContext(dataset, words)).probabilities) })),
        exampleSteps: [...generated.summary.examples.training.slice(0, 1), ...generated.summary.examples.novel.slice(0, 1)]
          .map((example) => explainGeneration(example, probability, dataset)),
      });
      progress(`D=${dimension}, H=${settings.hidden}, lr=${learningRate}: loss=${lossMean.toFixed(6)}, PPL=${neural.at(-1)!.perplexityMean.toFixed(4)}; ${settings.samples}개 생성`);
    }
  }
  const comparison: Comparison = {
    settings, nodeVersion: process.version,
    data: { sentences: dataset.sentences, tokens: dataset.tokens, pairs: dataset.pairs.length, contextLength: 2 },
    uniformPerplexity: dataset.tokens.length,
    trigram: {
      mle: { metrics: trigram.evaluate(dataset.pairs), generation: mle.summary, sampleFile: mle.sampleFile },
      smoothed: { metrics: trigram.evaluate(dataset.pairs, settings.k), generation: smoothed.summary, sampleFile: smoothed.sampleFile },
      probes: probeWords.map((words) => {
        const ctx = encodeContext(dataset, words);
        return { context: words, mle: trigram.distribution(ctx), smoothed: trigram.distribution(ctx, settings.k) };
      }),
    }, neural,
  };
  writeFileSync(join(output, 'loss.csv'), csv(lossRows));
  writeFileSync(join(output, 'comparison.json'), JSON.stringify(comparison, null, 2) + '\n');
  writeFileSync(join(output, 'learning_rates.csv'), csv([
    ['dimension', 'hidden', 'learning_rate', 'parameters', 'epochs', 'seeds', 'loss_mean', 'loss_std', 'perplexity_mean',
      'generation_training_seed', 'generation_seed', 'samples', 'training_rate', 'novel_pattern_rate', 'invalid_rate', 'unique_novel'],
    ...neural.map((result) => [result.dimension, result.hidden, result.learningRate, result.parameterCount, settings.epochs,
      settings.seeds.join(';'), result.lossMean, result.lossStd, result.perplexityMean, settings.seeds[0], settings.sampleSeed,
      result.generation.total, result.generation.rates.training, result.generation.rates.novel, result.generation.rates.invalid,
      result.generation.uniqueNovelSentences]),
  ]));
  return comparison;
}
