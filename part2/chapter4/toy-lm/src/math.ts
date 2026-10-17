export function positiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${label}은 양의 정수여야 합니다.`);
}

export function tokenId(value: number, vocabularySize: number): void {
  if (!Number.isInteger(value) || value < 0 || value >= vocabularySize) {
    throw new Error(`잘못된 토큰 ID: ${value}`);
  }
}

// 임베딩 초기화와 문장 샘플링을 각각 고정 seed로 재현한다.
export function randomGenerator(seed: number): () => number {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('seed는 0부터 2^32-1 사이의 정수여야 합니다.');
  }
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function normal(random: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

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
  for (let i = 0; i < logits.length; i++) probabilities[i] /= sum;
  return probabilities;
}

// max + log(sum(exp(z-max))) - z[target]는 작은 정답 확률에서도 안정적이다.
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

export function sample(probabilities: ArrayLike<number>, random: () => number): number {
  let total = 0;
  let lastPositive = -1;
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i];
    if (!Number.isFinite(p) || p < 0) throw new Error('잘못된 샘플링 확률입니다.');
    total += p;
    if (p > 0) lastPositive = i;
  }
  if (total <= 0) throw new Error('확률의 합은 양수여야 합니다.');
  const draw = random();
  if (!Number.isFinite(draw) || draw < 0 || draw >= 1) throw new Error('난수는 [0, 1) 범위여야 합니다.');
  const threshold = draw * total;
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (threshold < cumulative) return i;
  }
  return lastPositive;
}
