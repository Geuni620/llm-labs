import { buildDataset, type Dataset } from '../data/data.ts';
import { positiveInteger } from './math.ts';
import { NeuralLanguageModel, PARAMETER_NAMES, type ParameterName } from '../nplm/nlm.ts';

export interface AnimationSettings {
  dimension: number;
  hidden: number;
  learningRate: number;
  updates: number;
  seed: number;
  recordEvery: number;
}
export const ANIMATION_SETTINGS: AnimationSettings = {
  dimension: 2, hidden: 8, learningRate: 0.3, updates: 2000, seed: 123, recordEvery: 20,
};
export interface TrainingFrame {
  update: number;
  loss: number;
  parameters: Record<ParameterName, number[]>;
}
export interface TrainingRecording {
  settings: AnimationSettings;
  data: { tokens: string[]; sentences: string[]; pairs: number };
  losses: number[];
  frames: TrainingFrame[];
}

// 기존 train과 같은 순서로 갱신한다. losses[t]는 t회 갱신을 마친 뒤의 손실이다.
export function captureTraining(settings: AnimationSettings = ANIMATION_SETTINGS, dataset: Dataset = buildDataset()): TrainingRecording {
  positiveInteger(settings.updates, '학습 갱신 수');
  positiveInteger(settings.recordEvery, '기록 간격');
  if (settings.dimension !== 2) throw new Error('이 시각화는 2차원 임베딩을 기록합니다.');
  if (!Number.isFinite(settings.learningRate) || settings.learningRate <= 0) throw new Error('학습률은 양수여야 합니다.');
  const model = new NeuralLanguageModel(dataset.tokens.length, settings.dimension, settings.hidden, settings.seed);
  const frames: TrainingFrame[] = [];
  const losses: number[] = [];
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
