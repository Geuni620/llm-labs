import type { Comparison, NeuralResult } from './experiment.ts';
import type { GenerationSummary } from '../generation/generation.ts';

const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
const number = (value: number | string | null, digits = 4) => typeof value === 'number' ? value.toFixed(digits) : value ?? '미정';
const escape = (value: string) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');

export function representativeCases(comparison: Comparison): NeuralResult[] {
  const dimension = comparison.settings.dimensions.includes(2) ? 2 : comparison.settings.dimensions[0];
  const rows = comparison.neural.filter((result) => result.dimension === dimension).sort((a, b) => a.learningRate - b.learningRate);
  const middle = rows.find((row) => row.learningRate === 0.3) ?? rows[Math.floor((rows.length - 1) / 2)];
  return [...new Set([rows[0], middle, rows.at(-1)!])];
}

function generationRows(comparison: Comparison): (string | number)[][] {
  const row = (name: string, summary: GenerationSummary) => [name, summary.total, percent(summary.rates.training),
    percent(summary.rates.novel), percent(summary.rates.invalid), summary.uniqueNovelSentences];
  return [row('트라이그램 MLE', comparison.trigram.mle.generation), row(`트라이그램 add-k (${comparison.settings.k})`, comparison.trigram.smoothed.generation),
    ...comparison.neural.map((result) => row(`D${result.dimension}, lr ${result.learningRate}`, result.generation))];
}

function markdownTable(headers: readonly string[], rows: readonly (readonly (string | number)[])[]): string {
  return `| ${headers.join(' | ')} |\n| ${headers.map(() => '---').join(' | ')} |\n` +
    rows.map((row) => `| ${row.join(' | ')} |`).join('\n');
}

function htmlTable(headers: readonly string[], rows: readonly (readonly (string | number)[])[]): string {
  return `<div class="table-scroll"><table><thead><tr>${headers.map((header) => `<th>${escape(header)}</th>`).join('')}</tr></thead><tbody>` +
    rows.map((row) => `<tr>${row.map((cell) => `<td>${escape(String(cell))}</td>`).join('')}</tr>`).join('') + '</tbody></table></div>';
}

function rateRows(comparison: Comparison) {
  return comparison.neural.map((result) => [result.dimension, result.learningRate,
    `${result.lossMean.toFixed(4)} ± ${result.lossStd.toFixed(4)}`, result.perplexityMean.toFixed(4), result.parameterCount]);
}

function caseRows(comparison: Comparison) {
  return representativeCases(comparison).map((result) => [result.dimension, result.learningRate,
    number(result.runs.reduce((sum, run) => sum + Math.exp(run.initialLoss), 0) / result.runs.length),
    number(result.perplexityMean), percent(result.generation.rates.novel), percent(result.generation.rates.invalid)]);
}

function anchorModel(comparison: Comparison, dimension: number): NeuralResult {
  return comparison.neural.find((result) => result.dimension === dimension && result.learningRate === 0.3)
    ?? comparison.neural.find((result) => result.dimension === dimension)!;
}

function contextRows(comparison: Comparison) {
  const likes = comparison.data.tokens.indexOf('좋아해');
  const dislikes = comparison.data.tokens.indexOf('싫어해');
  const probe = comparison.trigram.probes.find((row) => row.context.join(' ') === '철수는 바나나를')!;
  const raw = probe.mle.probabilities;
  const smoothed = probe.smoothed.probabilities!;
  const row = (name: string, p: ArrayLike<number>) => [name, percent(p[likes]), percent(p[dislikes]), percent(Math.max(0, 1 - p[likes] - p[dislikes]))];
  return [['MLE', raw ? percent(raw[likes]) : '미정 (문맥 횟수 0)', raw ? percent(raw[dislikes]) : '미정', '미정'],
    row('add-k', smoothed), ...comparison.settings.dimensions.map((dimension) => {
      const result = anchorModel(comparison, dimension);
      return row(`NLM D${dimension}, lr ${result.learningRate}`, result.probes.find((probe) => probe.context.join(' ') === '철수는 바나나를')!.probabilities);
    })];
}

function novelRows(comparison: Comparison) {
  const observed = new Set(comparison.data.sentences.map((sentence) => sentence.split(' ').slice(0, 2).join(' ')));
  const row = (label: string, generation: GenerationSummary) => {
    let unseenContext = 0;
    let seenContextNewVerb = 0;
    for (const item of generation.distribution) {
      if (item.category !== 'novel') continue;
      if (observed.has(item.sentence.split(' ').slice(0, 2).join(' '))) seenContextNewVerb += item.count;
      else unseenContext += item.count;
    }
    return [label, `${unseenContext} (${percent(unseenContext / generation.total)})`,
      `${seenContextNewVerb} (${percent(seenContextNewVerb / generation.total)})`];
  };
  return [row('MLE', comparison.trigram.mle.generation), row('add-k', comparison.trigram.smoothed.generation),
    ...comparison.settings.dimensions.map((dimension) => {
      const model = anchorModel(comparison, dimension);
      return row(`NLM D${dimension}, lr ${model.learningRate}`, model.generation);
    })];
}

function prefixExplanation(comparison: Comparison): string {
  const model = anchorModel(comparison, comparison.settings.dimensions.includes(2) ? 2 : comparison.settings.dimensions[0]);
  const prefix = model.probes.find((probe) => probe.context.join(' ') === '<bos> 철수는')!;
  const banana = prefix.probabilities[comparison.data.tokens.indexOf('바나나를')];
  return `NLM D${model.dimension}, lr=${model.learningRate}에서 P(바나나를 | <bos>, 철수는)는 ${percent(banana)}다. ` +
    '새 문맥에서 좋아해의 확률이 높아도, 먼저 그 문맥까지 생성해야 전체 문장이 나온다.';
}

function embeddingSketch(points: number[][], tokens: string[], dimension: number): string {
  const width = 41, height = 11;
  const minX = Math.min(...points.map((point) => point[0]));
  const maxX = Math.max(...points.map((point) => point[0]));
  const minY = dimension === 1 ? 0 : Math.min(...points.map((point) => point[1]));
  const maxY = dimension === 1 ? 0 : Math.max(...points.map((point) => point[1]));
  const cells = new Map<string, string[]>();
  points.forEach((point, index) => {
    const x = Math.round((point[0] - minX) / (maxX - minX || 1) * (width - 1));
    const y = dimension === 1 ? 0 : Math.round((maxY - point[1]) / (maxY - minY || 1) * (height - 1));
    const key = `${x},${y}`;
    cells.set(key, [...(cells.get(key) ?? []), String.fromCharCode(65 + index)]);
  });
  const grid = Array.from({ length: dimension === 1 ? 1 : height }, () => Array(width).fill(' '));
  for (const [key, labels] of cells) {
    const [x, y] = key.split(',').map(Number);
    grid[y][x] = labels.length === 1 ? labels[0] : '*';
  }
  const border = '+' + '-'.repeat(width) + '+';
  const title = dimension === 1 ? '학습 후 1차원 위치' : dimension === 2 ? '학습 후 2차원 위치' : '학습 후 3차원 좌표의 XY 투영 (Z는 위 좌표 참고)';
  const collisions = [...cells.values()].filter((labels) => labels.length > 1).map((labels) => labels.join('/'));
  return `${title}\n` + (dimension === 1 ? '' : `y: ${minY.toFixed(3)} ~ ${maxY.toFixed(3)} (위가 최댓값)\n`) +
    `${border}\n${grid.map((row) => '|' + row.join('') + '|').join('\n')}\n${border}\n` +
    `x: ${minX.toFixed(3)} ~ ${maxX.toFixed(3)}\n` + tokens.map((token, i) => `${String.fromCharCode(65 + i)}=${token}`).join(' ') +
    (collisions.length ? `\n같은 칸의 점(*): ${collisions.join(', ')}` : '');
}

function embeddingText(comparison: Comparison): string {
  return comparison.settings.dimensions.map((dimension) => {
    const result = anchorModel(comparison, dimension);
    const run = result.runs[0];
    const rows = comparison.data.tokens.slice(3).map((token, index) => {
      const before = run.embeddingBefore[index + 3].map((value) => value.toFixed(3)).join(', ');
      const after = run.embeddingAfter[index + 3].map((value) => value.toFixed(3)).join(', ');
      return `${token}: [${before}] -> [${after}]`;
    });
    return `D=${dimension}, H=${result.hidden}, lr=${result.learningRate}, seed=${run.seed}\n${rows.join('\n')}\n\n` +
      embeddingSketch(run.embeddingAfter.slice(3), comparison.data.tokens.slice(3), dimension);
  }).join('\n\n');
}

function exampleText(comparison: Comparison): string {
  const selected = comparison.neural.find((row) => row.dimension === 2 && row.learningRate === 0.3) ?? comparison.neural[0];
  const sources = [
    ['트라이그램 MLE', comparison.trigram.mle.generation],
    ['트라이그램 add-k', comparison.trigram.smoothed.generation],
    [`NLM D${selected.dimension}, lr=${selected.learningRate}`, selected.generation],
  ] as const;
  return sources.map(([name, summary]) => `${name}\n` + (['training', 'novel', 'invalid'] as const).map((category) => {
    const label = { training: '학습 문장', novel: '형식에 맞는 새 조합', invalid: '형식 오류·미완성' }[category];
    return `  ${label}: ${summary.examples[category].map((example) => example.sentence || '(빈 문장)').join(' / ') || '(없음)'}`;
  }).join('\n')).join('\n\n');
}

export function renderMarkdown(comparison: Comparison): string {
  const s = comparison.settings;
  return `# 트라이그램과 신경망의 실제 실험 결과

문장 6개, 어휘 11개, 공통 학습 쌍 24개. H=${s.hidden}, full-batch GD ${s.epochs}회.
학습 seed: ${s.seeds.join(', ')}. 각 신경망 설정의 문장 생성에는 학습 seed ${s.seeds[0]}와 샘플링 seed ${s.sampleSeed}를 사용했다.
각 모델·설정당 ${s.samples.toLocaleString('en-US')}개를 생성했다. 최대 본문 길이는 ${s.maxTokens}토큰이다.

## 퍼플렉시티 기준

PPL = exp(24개 학습 쌍의 평균 cross-entropy). 신경망의 PPL은 각 seed에서 계산한 PPL의 평균이다.
균등 예측 기준은 ${comparison.uniformPerplexity}. MLE 트라이그램은 ${number(comparison.trigram.mle.metrics.perplexity)}, add-k(${s.k})는 ${number(comparison.trigram.smoothed.metrics.perplexity)}다.
동일 문맥에 여러 정답이 있어 MLE 기준도 1보다 크다. 이 표는 학습 데이터 적합도이며, 별도 검증 데이터 성능을 측정한 값은 아니다.

## 대표적인 세 가지 학습률 사례

${markdownTable(['D', '학습률', '초기 PPL 평균', '최종 PPL 평균', '새 조합', '오류·미완성'], caseRows(comparison))}

## 임베딩 차원과 학습률 비교

${markdownTable(['D', '학습률', '최종 loss 평균 ± 표준편차', 'PPL 평균', '매개변수'], rateRows(comparison))}

동일한 D와 seed에서는 학습률에 관계없이 같은 초기 가중치로 시작했다. D를 바꿀 때는 매개변수 수도 바뀐다.

## 생성 비율

- 학습 문장: 원래 6문장과 일치하고 <eos>로 종료.
- 새 조합: <eos>로 종료하고 주체 1개 + 대상 1개 + 행동 1개의 형식이며, 원래 6문장에 없음.
- 오류·미완성: 나머지 모두. 특수 토큰이 본문에 나오거나 길이가 다르거나 끝나지 않은 결과도 포함.

가능한 형식 조합은 3 × 3 × 2 = 18개이며, 그중 학습 문장 6개와 새로운 조합 12개가 있다. 이 형식 검사는 샘플링 이후에만 적용했다.

${markdownTable(['모델', '생성 개수', '학습 문장', '새 조합', '오류·미완성', '새 조합 종류'], generationRows(comparison))}

단순한 원문 불일치율은 오류도 포함한다. comparison.json의 literalNovelRate와 위의 새 조합 비율을 구분한다.

### 새 조합을 다시 나눠 보면

미관측 문맥은 학습에 없던 주체+대상 조합이다. 본 문맥·새 동사는 기존 주체+대상에서 학습과 다른 동사가 나온 경우다. 두 경우 모두 형식에 맞는 새 조합이지만, 이것만으로 의미가 올바르다고 판단하지 않는다. 아래 비율의 분모도 전체 ${s.samples}개다.

${markdownTable(['모델', '미관측 문맥', '본 문맥·새 동사'], novelRows(comparison))}

## 관측하지 못한 문맥: (철수는, 바나나를)

${markdownTable(['모델', '좋아해', '싫어해', '기타 9토큰'], contextRows(comparison))}

이 문맥의 다음 단어 분포와 전체 문장 생성 빈도는 서로 다른 관찰이다. 전체 생성에서는 먼저 ( <bos>, 철수는 ) 다음에 바나나를이 선택되어야 한다. 이 앞 단계의 분포도 comparison.json의 probes에 기록했다.

${prefixExplanation(comparison)}

## 생성 예시

\`\`\`text
${exampleText(comparison)}
\`\`\`

## 학습 전후 임베딩 좌표

\`\`\`text
${embeddingText(comparison)}
\`\`\`

## 원본 기록

- comparison.json: 모든 학습 seed의 지표, 좌표, 문맥별 확률, 생성 분포·예시.
- learning_rates.csv: 학습률별 loss·PPL·생성 비율 표.
- loss.csv: 가중치 갱신 0회부터 ${s.epochs}회까지의 loss·PPL.
- samples/*.jsonl: 모든 생성 결과, 종료 상태, 분류.
- distributions/*.csv: 결과별 실제 생성 횟수와 비율.
`;
}

export function renderHtml(comparison: Comparison): string {
  const s = comparison.settings;
  return `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root { color-scheme: light dark; }
body { margin: 0; font-family: var(--font-sans, system-ui); color: var(--foreground, CanvasText); background: transparent; }
h2 { font-size: inherit; font-weight: 500; margin: 24px 0 12px; }
p { margin: 10px 0; }
.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
td, th { padding: 8px 6px; text-align: left; border-bottom: 1px solid var(--border, GrayText); font-weight: 400; }
th { font-weight: 500; }
pre { font-family: var(--font-mono, monospace); font-size: inherit; line-height: inherit; overflow-x: auto; }
details { margin: 16px 0; }
summary { font-weight: 500; }
.muted { color: var(--muted-foreground, GrayText); }
</style></head><body>
<p>문장 6개 · 24개 학습 쌍 · H=${s.hidden} · ${s.epochs}회 학습 · seed ${s.seeds.join('/')}<br>
각 모델·설정당 ${s.samples.toLocaleString('en-US')}개 생성 · 생성에 사용한 학습 seed ${s.seeds[0]}</p>
<h2>퍼플렉시티 대표 사례 3개</h2>
<p>PPL = exp(평균 loss). 균등 기준 ${comparison.uniformPerplexity} · MLE ${number(comparison.trigram.mle.metrics.perplexity)} · add-k ${number(comparison.trigram.smoothed.metrics.perplexity)}</p>
${htmlTable(['D', '학습률', '초기 PPL', '최종 PPL', '새 조합', '오류·미완성'], caseRows(comparison))}
<p class="muted">같은 학습 데이터에서 측정했습니다. PPL은 학습 seed별 값의 평균입니다.</p>
<h2>학습률과 임베딩 차원 비교</h2>
${htmlTable(['D', '학습률', 'loss 평균 ± 표준편차', 'PPL 평균', '매개변수'], rateRows(comparison))}
<h2>10,000개씩 생성한 결과</h2>
${htmlTable(['모델', '개수', '학습 문장', '새 조합', '오류·미완성', '새 조합 종류'], generationRows(comparison))}
<pre>학습 문장       원래 6개 중 하나 + 종료
새 조합         주체 + 대상 + 행동 + 종료
                형식에 맞고 원래 6개에 없음
오류·미완성     나머지 모두

가능한 형식: 3 × 3 × 2 = 18개
학습 6개 / 학습에 없던 조합 12개</pre>
<details><summary>새 조합: 미관측 문맥과 본 문맥의 새 동사 구분</summary>
${htmlTable(['모델', '미관측 문맥', '본 문맥·새 동사'], novelRows(comparison))}
<p class="muted">전체 ${s.samples}개를 분모로 계산했습니다. 두 범주의 합이 새 조합의 수입니다.</p></details>
<h2>처음 보는 문맥: 철수는 + 바나나를</h2>
${htmlTable(['모델', '좋아해', '싫어해', '기타 9토큰'], contextRows(comparison))}
<p class="muted">${escape(prefixExplanation(comparison))}</p>
<details open><summary>실제 생성 예시</summary><pre>${escape(exampleText(comparison))}</pre></details>
<details><summary>임베딩 1·2·3차원: 학습 전 → 학습 후</summary><pre>${escape(embeddingText(comparison))}</pre></details>
<p class="muted">전체 생성 결과와 횟수는 samples/*.jsonl 및 distributions/*.csv에 기록했습니다.</p>
</body></html>`;
}
