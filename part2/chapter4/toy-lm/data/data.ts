export const PAD = '<pad>';
export const BOS = '<bos>';
export const EOS = '<eos>';
export const SUBJECTS = ['철수는', '영희는', '짱구는'] as const;
export const OBJECTS = ['사과를', '딸기를', '바나나를'] as const;
export const VERBS = ['좋아해', '싫어해'] as const;

// ZIP 원본의 여섯 문장. 테스트용 새 조합은 여기에 추가하지 않는다.
export const SENTENCES = [
  '철수는 사과를 좋아해',
  '철수는 딸기를 좋아해',
  '영희는 사과를 좋아해',
  '영희는 딸기를 좋아해',
  '영희는 바나나를 좋아해',
  '짱구는 바나나를 싫어해',
] as const;

export type Context = readonly [number, number];
export interface Pair {
  context: Context;
  target: number;
}
export interface Dataset {
  sentences: string[];
  tokens: string[];
  ids: Map<string, number>;
  pairs: Pair[];
}

export function buildDataset(sentences: readonly string[] = SENTENCES): Dataset {
  if (sentences.length === 0) throw new Error('학습 문장이 필요합니다.');
  const tokens = [PAD, BOS, EOS];
  const ids = new Map(tokens.map((token, id) => [token, id]));
  const words = sentences.map((sentence) => {
    if (!sentence.trim()) throw new Error('빈 문장은 사용할 수 없습니다.');
    return sentence.trim().split(/\s+/);
  });
  for (const sentence of words) {
    for (const word of sentence) {
      if ([PAD, BOS, EOS].includes(word)) throw new Error('문장에 특수 토큰을 넣을 수 없습니다.');
      if (!ids.has(word)) {
        ids.set(word, tokens.length);
        tokens.push(word);
      }
    }
  }
  const pairs: Pair[] = [];
  for (const sentence of words) {
    const sequence = [PAD, BOS, ...sentence, EOS].map((word) => ids.get(word)!);
    for (let i = 0; i < sequence.length - 2; i++) {
      pairs.push({ context: [sequence[i], sequence[i + 1]], target: sequence[i + 2] });
    }
  }
  return { sentences: words.map((words) => words.join(' ')), tokens, ids, pairs };
}

export function encodeContext(dataset: Dataset, words: readonly string[]): Context {
  if (words.length !== 2) throw new Error('문맥에는 토큰 두 개가 필요합니다.');
  const ids = words.map((word) => {
    const id = dataset.ids.get(word);
    if (id === undefined) throw new Error(`어휘에 없는 토큰: ${word}`);
    return id;
  });
  return [ids[0], ids[1]];
}
