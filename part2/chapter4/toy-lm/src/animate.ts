import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { ANIMATION_SETTINGS, captureTraining } from './animation.ts';
import { BOS, EOS, OBJECTS, PAD, SUBJECTS, VERBS, buildDataset, encodeContext } from '../data/data.ts';
import { generateBatch } from '../generation/generation.ts';
import { NeuralLanguageModel, PARAMETER_NAMES } from '../nplm/nlm.ts';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = join(root, 'results');
mkdirSync(output, { recursive: true });
const recording = captureTraining();
const dataset = buildDataset();
const final = recording.frames.at(-1)!;
const model = new NeuralLanguageModel(dataset.tokens.length, 2, recording.settings.hidden);
for (const name of PARAMETER_NAMES) model.parameters[name].set(final.parameters[name]);
const generated = generateBatch(ctx => model.forward(ctx).probabilities, dataset, 10000, 2026, 20);

// 기존 리포트가 있으면 같은 설정의 끝점과 생성 결과를 대조한다.
let verifiedExisting = false;
try {
  const saved = JSON.parse(readFileSync(join(output, 'comparison.json'), 'utf8'));
  const result = saved.neural.find((row: { dimension: number; hidden: number; learningRate: number }) =>
    row.dimension === 2 && row.hidden === ANIMATION_SETTINGS.hidden && row.learningRate === ANIMATION_SETTINGS.learningRate);
  if (result && saved.settings.epochs === ANIMATION_SETTINGS.updates && result.runs[0].seed === ANIMATION_SETTINGS.seed) {
    assert.equal(final.loss, result.runs[0].finalLoss);
    assert.deepEqual(final.parameters.E, result.runs[0].embeddingAfter.flat());
    if (saved.settings.samples === 10000 && saved.settings.sampleSeed === 2026 && saved.settings.maxTokens === 20) {
      assert.deepEqual(generated.summary.counts, result.generation.counts);
    }
    verifiedExisting = true;
  }
} catch (error) {
  if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT')) throw error;
}

const probes = [['철수는', '사과를'], ['철수는', '바나나를'], ['짱구는', '사과를']];
const evaluations = recording.frames.map(frame => {
  for (const name of PARAMETER_NAMES) model.parameters[name].set(frame.parameters[name]);
  const reference = probes.map(words => Array.from(model.forward(encodeContext(dataset, words)).probabilities));
  let training = 0, novel = 0;
  for (const subject of SUBJECTS) for (const object of OBJECTS) for (const verb of VERBS) {
    const words = [subject, object, verb];
    let context = encodeContext(dataset, [PAD, BOS]);
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
  assert.ok(training >= 0 && novel >= 0 && invalid >= 0);
  return { reference, mass: { training, novel, invalid } };
});
const data = { ...recording, probes, reference: evaluations.map(row => row.reference),
  generationMass: evaluations.map(row => row.mass), generated: generated.summary, verifiedExisting };
const template = readFileSync(join(root, 'src/training-animation.template.html'), 'utf8');
const svg = readFileSync(join(root, 'src/svg.js'), 'utf8');
const charts = readFileSync(join(root, 'src/training-charts.js'), 'utf8');
const player = readFileSync(join(root, 'src/training-animation.js'), 'utf8');
const fragment = template.replace('__SVG_SOURCE__', () => svg)
  .replace('__CHART_SOURCE__', () => charts)
  .replace('__PLAYER_SOURCE__', () => player)
  .replace('__TRAINING_DATA__', () => JSON.stringify(data).replaceAll('<', '\\u003c'));
assert.ok(Buffer.byteLength(fragment) < 1024 * 1024, '시각화는 1MB 미만이어야 합니다.');
writeFileSync(join(output, 'training-animation.json'), JSON.stringify(data));
writeFileSync(join(output, 'training-animation.fragment.html'), fragment);
// 대화 렌더러의 512,000자 제한을 위해 원본 전체를 손실 없이 압축한다.
const packed = gzipSync(Buffer.from(fragment)).toString('base64');
const inline = `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><span id="training-loading">학습 기록을 불러오는 중…</span><script>(async()=>{try{const packed="${packed}";const bytes=Uint8Array.from(atob(packed),c=>c.charCodeAt(0));const source=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();const host=document.createElement("div");host.innerHTML=source;document.body.append(host);document.getElementById("training-loading").remove();for(const old of [...host.querySelectorAll("script")]){const script=document.createElement("script");script.textContent=old.textContent;old.replaceWith(script);}document.documentElement.dataset.trainingReady="true";}catch(error){console.error(error);document.getElementById("training-loading").textContent="학습 기록 재생을 불러오지 못했습니다.";}})();</script></body></html>`;
assert.ok(inline.length <= 512000, '대화용 HTML은 512,000자 이하여야 합니다.');
writeFileSync(join(output, 'training-animation.inline.html'), inline);
// 대화 밖에서도 열 수 있도록 테마 기본값을 제공한다. 대화에서는 호스트의 테마를 사용한다.
const theme = `:root{color-scheme:light dark;--background:light-dark(#fff,#18181b);--foreground:light-dark(#18181b,#f4f4f5);--muted:light-dark(#f4f4f5,#27272a);--muted-foreground:light-dark(#52525b,#b4b4be);--border:light-dark(#d4d4d8,#484850);--primary:light-dark(#18181b,#f4f4f5);--primary-foreground:light-dark(#fff,#18181b);--popover:var(--background);--popover-foreground:var(--foreground);--font-sans:system-ui,sans-serif;--font-size-base:14px;--chart-1:light-dark(#2563eb,#76b5fa);--chart-2:light-dark(#0d9488,#5ed4c2);--chart-3:light-dark(#b45309,#f8b45d);--chart-4:light-dark(#7c3aed,#bb9cf3);--chart-5:#dc2626;--chart-6:#ca8a04}body{margin:24px auto;max-width:940px;padding:0 16px;background:var(--background);color:var(--foreground);font:14px var(--font-sans)}button,input,select{font:inherit}button,select{padding:7px 10px}button{border:1px solid var(--border);border-radius:6px;background:var(--background);color:var(--foreground)}button[aria-pressed=true]{background:var(--primary);color:var(--primary-foreground)}input[type=range]{width:100%}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--chart-1);outline-offset:3px}`;
writeFileSync(join(output, 'training-animation.html'), `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NPLM 학습 기록 재생</title><style>${theme}</style></head><body>${fragment}</body></html>`);
console.log(`학습 기록 ${recording.frames.length}개 · ${recording.losses.length}개 loss · 기존 결과 대조 ${verifiedExisting ? '일치' : '해당 없음'}`);
console.log(`생성 10,000개: 원문 ${generated.summary.counts.training}, 새 조합 ${generated.summary.counts.novel}, 오류 ${generated.summary.counts.invalid}`);
console.log(join(output, 'training-animation.html'));
