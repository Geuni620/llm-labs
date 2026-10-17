import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildDataset, encodeContext } from '../data/data.ts';
import { DEFAULT_SETTINGS, runExperiment, type Settings } from './experiment.ts';
import { crossEntropy } from './math.ts';
import { NeuralLanguageModel, PARAMETER_NAMES } from '../nplm/nlm.ts';
import { renderHtml, renderMarkdown } from './report.ts';
import { Trigram } from '../trigram/trigram.ts';

const help = `학습 실습 (Node.js 24.12 이상)
  npm run data
  npm run trigram -- --context 철수는,바나나를 --k 0.01
  npm run trace -- --dim 2 --hidden 6
  npm run compare

compare 옵션:
  --dims 1,2,3 --hidden 8 --lrs 0.05,0.1,0.3,1
  --epochs 2000 --seeds 123,456,789 --k 0.01
  --samples 10000 --sample-seed 2026 --max-tokens 20 --out results
`;

function parse(command: string, argv: string[]): Map<string, string> {
  const allowed: Record<string, readonly string[]> = {
    data: [], trigram: ['context', 'k'], trace: ['context', 'target', 'dim', 'hidden', 'seed', 'lr'],
    compare: ['dims', 'hidden', 'lrs', 'epochs', 'seeds', 'k', 'samples', 'sample-seed', 'max-tokens', 'out'],
  };
  if (!(command in allowed)) throw new Error(`알 수 없는 명령: ${command}\n${help}`);
  const flags = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '');
    const value = argv[i + 1];
    if (!argv[i].startsWith('--') || !allowed[command].includes(key) || value === undefined || value.startsWith('--') || flags.has(key)) {
      throw new Error(`잘못된 옵션: ${argv[i]}`);
    }
    if (!value.trim()) throw new Error(`${key}에 값이 필요합니다.`);
    flags.set(key, value);
  }
  return flags;
}

function main(): void {
  const [command = 'help', ...argv] = process.argv.slice(2);
  if (command === 'help' || command === '--help') { console.log(help); return; }
  const flags = parse(command, argv);
  const numeric = (name: string, fallback: number) => flags.has(name) ? Number(flags.get(name)) : fallback;
  const list = (name: string, fallback: number[]) => flags.has(name) ? flags.get(name)!.split(',').map((part) => {
    if (!part.trim()) throw new Error(`${name}의 목록에 빈 값이 있습니다.`);
    return Number(part);
  }) : [...fallback];
  const dataset = buildDataset();
  if (command === 'data') {
    console.log(`문장 ${dataset.sentences.length}개 / 일반 어휘 ${dataset.tokens.length - 3}개 / 전체 어휘 ${dataset.tokens.length}개 / 학습 쌍 ${dataset.pairs.length}개`);
    console.table(dataset.tokens.map((token, id) => ({ id, token })));
    console.table(dataset.pairs.map((pair, i) => ({ index: i + 1, context: pair.context.map((id) => dataset.tokens[id]).join(' '), next: dataset.tokens[pair.target] })));
    return;
  }
  if (command === 'trigram') {
    const context = encodeContext(dataset, (flags.get('context') ?? '철수는,사과를').split(',').map((word) => word.trim()));
    const model = new Trigram(dataset.tokens.length);
    model.fit(dataset.pairs);
    const raw = model.distribution(context);
    const smoothed = model.distribution(context, numeric('k', DEFAULT_SETTINGS.k));
    console.log(`문맥 횟수=${raw.contextCount}, MLE 분모=${raw.denominator}, add-k 분모=${smoothed.denominator}`);
    console.table(dataset.tokens.map((token, id) => ({ token, count: raw.counts[id],
      MLE: raw.probabilities?.[id] ?? '미정', addK: smoothed.probabilities?.[id] ?? '미정' })));
    console.log('학습 데이터 평가', { mle: model.evaluate(dataset.pairs), addK: model.evaluate(dataset.pairs, smoothed.k) });
    return;
  }
  if (command === 'trace') {
    const context = encodeContext(dataset, (flags.get('context') ?? '철수는,사과를').split(',').map((word) => word.trim()));
    const target = dataset.ids.get(flags.get('target') ?? '좋아해');
    if (target === undefined) throw new Error('정답 토큰이 어휘에 없습니다.');
    const model = new NeuralLanguageModel(dataset.tokens.length, numeric('dim', 2), numeric('hidden', DEFAULT_SETTINGS.hidden), numeric('seed', 123));
    const forward = model.forward(context);
    const { loss, gradients } = model.lossAndGradients([{ context, target }]);
    const vector = (values: ArrayLike<number>) => Array.from(values, (value) => Number(value.toFixed(6)));
    console.log(`임베딩 입력 ${2 * model.dimension}개 → 은닉 ${model.hiddenSize}개; W1 연결 ${model.parameters.W1.length}개`);
    console.log('초기 가중치에서 한 학습 쌍의 순전파', {
      context: context.map((id) => dataset.tokens[id]), target: dataset.tokens[target], input: vector(forward.input),
      hiddenInput: vector(forward.hiddenInput), hidden: vector(forward.hidden), logits: vector(forward.logits),
      probabilities: vector(forward.probabilities), loss: crossEntropy(forward.logits, target),
    });
    console.table(PARAMETER_NAMES.map((name) => ({ parameter: name, values: gradients[name].length,
      gradientNorm: Math.hypot(...gradients[name]), maxAbsoluteGradient: Math.max(...Array.from(gradients[name], Math.abs)) })));
    model.step(gradients, numeric('lr', 0.3));
    console.log({ beforeLoss: loss, afterOneUpdateLoss: model.loss([{ context, target }]) });
    return;
  }
  const settings: Settings = {
    dimensions: list('dims', DEFAULT_SETTINGS.dimensions), hidden: numeric('hidden', DEFAULT_SETTINGS.hidden),
    learningRates: list('lrs', DEFAULT_SETTINGS.learningRates), epochs: numeric('epochs', DEFAULT_SETTINGS.epochs),
    seeds: list('seeds', DEFAULT_SETTINGS.seeds), k: numeric('k', DEFAULT_SETTINGS.k), samples: numeric('samples', DEFAULT_SETTINGS.samples),
    sampleSeed: numeric('sample-seed', DEFAULT_SETTINGS.sampleSeed), maxTokens: numeric('max-tokens', DEFAULT_SETTINGS.maxTokens),
  };
  const output = flags.has('out') ? resolve(flags.get('out')!) : fileURLToPath(new URL('../results/', import.meta.url));
  const comparison = runExperiment(settings, output, console.log);
  writeFileSync(resolve(output, 'report.md'), renderMarkdown(comparison));
  writeFileSync(resolve(output, 'report.html'), renderHtml(comparison));
  console.table(comparison.neural.map((result) => ({ D: result.dimension, H: result.hidden, lr: result.learningRate,
    lossMean: Number(result.lossMean.toFixed(6)), PPL: Number(result.perplexityMean.toFixed(4)),
    training: `${(result.generation.rates.training * 100).toFixed(2)}%`,
    novel: `${(result.generation.rates.novel * 100).toFixed(2)}%`, invalid: `${(result.generation.rates.invalid * 100).toFixed(2)}%` })));
  console.log(`결과 저장: ${output}`);
}

try { main(); } catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
