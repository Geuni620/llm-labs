import { buildDataset } from '../data/data.ts';
import { NeuralLanguageModel, PARAMETER_NAMES } from '../nplm/nlm.ts';

// 직접 미분한 역전파를 독립적인 수치 미분과 대조한다.
const dataset = buildDataset();
const epsilon = 1e-5;
const tolerance = 1e-7;
const rows: { dimension: number; parameter: string; values: number; maximumError: string }[] = [];
let checkedValues = 0;

for (const dimension of [1, 2, 3]) {
  const model = new NeuralLanguageModel(dataset.tokens.length, dimension, 8, 42);
  const { gradients } = model.lossAndGradients(dataset.pairs);
  for (const name of PARAMETER_NAMES) {
    let maximumError = 0;
    for (let i = 0; i < model.parameters[name].length; i++) {
      const original = model.parameters[name][i];
      let plus = 0, minus = 0;
      try {
        model.parameters[name][i] = original + epsilon;
        plus = model.loss(dataset.pairs);
        model.parameters[name][i] = original - epsilon;
        minus = model.loss(dataset.pairs);
      } finally {
        model.parameters[name][i] = original;
      }
      const numerical = (plus - minus) / (2 * epsilon);
      maximumError = Math.max(maximumError, Math.abs(numerical - gradients[name][i]));
    }
    if (!Number.isFinite(maximumError) || maximumError >= tolerance) {
      throw new Error(`D=${dimension} ${name}: 기울기 오차 ${maximumError}가 허용값 ${tolerance} 이상입니다.`);
    }
    const values = model.parameters[name].length;
    checkedValues += values;
    rows.push({ dimension, parameter: name, values, maximumError: maximumError.toExponential(3) });
  }
}

console.table(rows);
console.log(`6문장·24쌍, D=1·2·3, H=8: ${checkedValues}개 변수 성분의 수치 미분 대조 완료`);
