import { BOS, EOS, OBJECTS, PAD, SUBJECTS, VERBS, type Context, type Dataset } from '../data/data.ts';
import { positiveInteger, randomGenerator, sample } from '../src/math.ts';

export type ProbabilityFunction = (context: Context) => ArrayLike<number> | null;
export type Category = 'training' | 'novel' | 'invalid';
export type Termination = 'eos' | 'maxTokens' | 'unseenContext';
export interface GeneratedSentence {
  tokens: string[];
  sentence: string;
  termination: Termination;
  category: Category;
}
export interface GenerationSummary {
  total: number;
  counts: Record<Category, number>;
  rates: Record<Category, number>;
  // 원문 불일치는 오류도 포함한다. 형식에 맞는 새 조합 비율과 별도로 표시한다.
  literalNovelRate: number;
  uniqueSentences: number;
  uniqueNovelSentences: number;
  terminations: Record<Termination, number>;
  distribution: { sentence: string; category: Category; count: number }[];
  examples: Record<Category, GeneratedSentence[]>;
}

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

export function generateSentence(
  probability: ProbabilityFunction, dataset: Dataset, random: () => number, maxTokens = 20,
): GeneratedSentence {
  positiveInteger(maxTokens, '최대 토큰 수');
  let context: Context = [dataset.ids.get(PAD)!, dataset.ids.get(BOS)!];
  const tokens: string[] = [];
  let termination: Termination = 'maxTokens';
  // 최대 maxTokens개의 본문 뒤에 <eos>가 오는지도 확인한다.
  for (let position = 0; position <= maxTokens; position++) {
    const probabilities = probability(context);
    if (probabilities === null) {
      termination = 'unseenContext';
      break;
    }
    if (probabilities.length !== dataset.tokens.length) throw new Error('출력 어휘 크기가 다릅니다.');
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
}

export function generateBatch(
  probability: ProbabilityFunction, dataset: Dataset, count: number, seed: number, maxTokens = 20,
): { records: GeneratedSentence[]; summary: GenerationSummary } {
  positiveInteger(count, '생성 개수');
  const random = randomGenerator(seed);
  const records: GeneratedSentence[] = [];
  const counts: Record<Category, number> = { training: 0, novel: 0, invalid: 0 };
  const terminations: Record<Termination, number> = { eos: 0, maxTokens: 0, unseenContext: 0 };
  const histogram = new Map<string, { sentence: string; category: Category; count: number }>();
  const examples: Record<Category, GeneratedSentence[]> = { training: [], novel: [], invalid: [] };
  const unique = new Set<string>();
  const uniqueNovel = new Set<string>();
  for (let i = 0; i < count; i++) {
    const generated = generateSentence(probability, dataset, random, maxTokens);
    records.push(generated);
    counts[generated.category]++;
    terminations[generated.termination]++;
    unique.add(generated.sentence);
    if (generated.category === 'novel') uniqueNovel.add(generated.sentence);
    if (examples[generated.category].length < 3) examples[generated.category].push(generated);
    const key = `${generated.category}\0${generated.sentence}`;
    const item = histogram.get(key) ?? { sentence: generated.sentence, category: generated.category, count: 0 };
    item.count++;
    histogram.set(key, item);
  }
  return {
    records,
    summary: {
      total: count, counts,
      rates: { training: counts.training / count, novel: counts.novel / count, invalid: counts.invalid / count },
      literalNovelRate: 1 - counts.training / count,
      uniqueSentences: unique.size, uniqueNovelSentences: uniqueNovel.size, terminations,
      distribution: [...histogram.values()].sort((a, b) => b.count - a.count || a.sentence.localeCompare(b.sentence, 'ko')),
      examples,
    },
  };
}

export function explainGeneration(record: GeneratedSentence, probability: ProbabilityFunction, dataset: Dataset) {
  let context: Context = [dataset.ids.get(PAD)!, dataset.ids.get(BOS)!];
  const steps: { context: string[]; chosen: string; probability: number }[] = [];
  const words = record.termination === 'eos' ? [...record.tokens, EOS] : record.tokens;
  for (const word of words) {
    const probabilities = probability(context);
    if (!probabilities) break;
    const next = dataset.ids.get(word)!;
    steps.push({ context: context.map((id) => dataset.tokens[id]), chosen: word, probability: probabilities[next] });
    context = [context[1], next];
  }
  return steps;
}
