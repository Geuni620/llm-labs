(() => {
  'use strict';
  const data = chapter4TrainingData;
  const root = document.getElementById('training-player');
  const frames = data.frames;
  const tokens = data.data.tokens;
  const D = data.settings.dimension, H = data.settings.hidden, V = tokens.length;
  const query = selector => root.querySelector(selector);
  const playButton = query('[data-play]');
  const slider = query('[data-time]');
  const viewSelect = query('[data-view-select]');
  const probeSelect = query('[data-probe]');
  const tooltip = query('.tooltip');
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  const saved = window.openai?.widgetState;
  const remembered = saved?.modelContent ?? saved ?? {};
  const state = { index: 0, view: 'network', probe: 0, playing: false, raf: 0, lastTime: 0 };
  let updateVisual = () => {};
  let lastWidth = 0;
  let tooltipPinned = false;
  let constructing = false;
  let scatterBoxes = [];
  const percent = value => (100 * value).toFixed(2) + '%';
  const currentFrame = () => frames[state.index];
  const trackedId = () => tokens.indexOf(state.probe === 2 ? '싫어해' : '좋아해');

  // 저장한 가중치의 순전파만 수행한다. 학습은 원래 TypeScript 모델에서 기록했다.
  function infer(frame, probeIndex) {
    const context = data.probes[probeIndex].map(word => tokens.indexOf(word));
    const p = frame.parameters;
    const input = context.flatMap(id => p.E.slice(id * D, (id + 1) * D));
    const hidden = Array.from({ length: H }, (_, h) => {
      let value = p.b1[h];
      for (let i = 0; i < input.length; i++) value += input[i] * p.W1[i * H + h];
      return Math.tanh(value);
    });
    const logits = Array.from({ length: V }, (_, v) => {
      let value = p.b2[v];
      for (let h = 0; h < H; h++) value += hidden[h] * p.W2[h * V + v];
      return value;
    });
    const maximum = Math.max(...logits);
    const weights = logits.map(value => Math.exp(value - maximum));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return { context, input, hidden, logits, probabilities: weights.map(value => value / total) };
  }

  function remember() {
    if (window.openai?.setWidgetState) window.openai.setWidgetState({
      modelContent: { frameIndex: state.index, update: currentFrame().update, view: state.view, probe: state.probe },
    }).catch(() => {});
  }
  function refreshPlay() {
    root.classList.toggle('is-playing', state.playing && !media.matches && !constructing);
    playButton.textContent = media.matches ? '다음 기록' : state.playing ? '일시정지' : state.index === frames.length - 1 ? '다시 재생' : '재생';
    playButton.setAttribute('aria-pressed', String(state.playing));
    query('[data-motion-note]').textContent = media.matches
      ? '동작 줄이기 설정 · 버튼 또는 슬라이더로 기록을 한 시점씩 확인'
      : `${data.settings.recordEvery}회 갱신마다 기록 · 재생 속도 200ms/기록 · 자동 반복 없음`;
  }
  function setFrame(index) {
    state.index = Math.max(0, Math.min(frames.length - 1, Math.round(index)));
    const frame = currentFrame();
    const prediction = infer(frame, state.probe);
    slider.value = String(state.index);
    slider.setAttribute('aria-valuetext', `${frame.update}회 갱신`);
    query('[data-update]').textContent = `${frame.update.toLocaleString('ko-KR')} / ${data.settings.updates.toLocaleString('ko-KR')}회`;
    query('[data-loss]').textContent = frame.loss.toFixed(4);
    query('[data-ppl]').textContent = Math.exp(frame.loss).toFixed(4);
    query('[data-probe-label]').textContent = `‘${tokens[trackedId()]}’ 다음 토큰 확률`;
    query('[data-probe-p]').textContent = percent(prediction.probabilities[trackedId()]);
    const mass = data.generationMass[state.index];
    let offset = 0;
    for (const category of ['training', 'novel', 'invalid']) {
      const element = query(`[data-mass="${category}"]`);
      element.style.transform = `translateX(${offset * 100}%) scaleX(${mass[category]})`;
      query(`[data-mass-value="${category}"]`).textContent = percent(mass[category]);
      offset += mass[category];
    }
    query('.mass-track').setAttribute('aria-label', `학습 문장 ${percent(mass.training)}, 새 조합 ${percent(mass.novel)}, 오류 ${percent(mass.invalid)}`);
    updateVisual(frame, prediction);
    refreshPlay();
  }
  function pause() {
    state.playing = false;
    cancelAnimationFrame(state.raf);
    refreshPlay();
  }
  function tick(time) {
    if (!state.playing) return;
    if (document.hidden) { pause(); return; }
    if (time - state.lastTime >= 200) {
      state.lastTime = time;
      setFrame(state.index + 1);
      if (state.index === frames.length - 1) { pause(); remember(); return; }
    }
    state.raf = requestAnimationFrame(tick);
  }
  function play() {
    if (media.matches) { pause(); setFrame(state.index === frames.length - 1 ? 0 : state.index + 1); remember(); return; }
    if (state.playing) { pause(); remember(); return; }
    if (state.index === frames.length - 1) setFrame(0);
    state.playing = true;
    state.lastTime = performance.now();
    refreshPlay();
    state.raf = requestAnimationFrame(tick);
  }
  const { drawNetwork, drawProbabilities, drawLoss, drawEmbedding } = globalThis.createNplmCharts(root, data, {
    state, currentFrame,
    getPinned: () => tooltipPinned,
    setPinned: value => { tooltipPinned = value; },
    setBoxes: boxes => { scatterBoxes = boxes; },
  });

  function drawActive() {
    constructing = true;
    root.classList.remove('is-playing');
    tooltipPinned = false; tooltip.hidden = true;
    root.querySelectorAll('[data-panel]').forEach(panel => { panel.hidden = panel.dataset.panel !== state.view; });
    if (state.view === 'network') {
      const network = drawNetwork(), probabilities = drawProbabilities();
      updateVisual = (frame, prediction) => { network(frame, prediction); probabilities(frame, prediction); };
    } else if (state.view === 'embedding') updateVisual = drawEmbedding();
    else {
      const loss = drawLoss(), ppl = drawLoss(true);
      updateVisual = frame => { loss(frame);ppl(frame); };
    }
    setFrame(state.index);
    root.querySelectorAll('.moving').forEach(node => { getComputedStyle(node).transform; });
    constructing = false;
    refreshPlay();
  }
  function setView(view) {
    state.view = ['network', 'embedding', 'loss'].includes(view) ? view : 'network';
    viewSelect.value = state.view; drawActive();
  }
  function setProbe(probe) {
    state.probe = Math.max(0, Math.min(data.probes.length - 1, Number(probe) || 0));
    probeSelect.value = String(state.probe); setFrame(state.index);
  }
  slider.max = String(frames.length - 1);
  slider.addEventListener('input', () => { pause();setFrame(Number(slider.value));remember(); });
  playButton.addEventListener('click', play);
  query('[data-reset]').addEventListener('click', () => { pause();setFrame(0);remember(); });
  viewSelect.addEventListener('change', () => { setView(viewSelect.value);remember(); });
  probeSelect.addEventListener('change', () => { setProbe(probeSelect.value);remember(); });
  media.addEventListener('change', () => { pause();refreshPlay(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  const generated = data.generated;
  query('[data-model-settings]').textContent = `임베딩 ${D}차원 · 은닉 뉴런 ${H}개 · 학습률 ${data.settings.learningRate} · 학습 seed ${data.settings.seed}`;
  query('[data-connections-note]').textContent = `연결: ${2*D}×${H}=${2*D*H}개, ${H}×${V}=${H*V}개 · 선 진하기: 가중치 절댓값 · 은닉 원 안: tanh 값(소수 1자리)`;
  query('[data-final-result]').textContent = `최종 모델의 실제 10,000회 추첨: 새 조합 ${generated.counts.novel}개 (${percent(generated.rates.novel)}) · 서로 다른 새 문장 ${generated.uniqueNovelSentences}종 · 추첨 seed 2026`;
  state.index = Number.isInteger(remembered.frameIndex) ? Math.max(0,Math.min(frames.length-1,remembered.frameIndex)) : 0;
  state.probe = Number.isInteger(remembered.probe) ? Math.max(0,Math.min(data.probes.length-1,remembered.probe)) : 0;
  probeSelect.value = String(state.probe);
  setView(remembered.view ?? 'network');
  lastWidth = root.clientWidth;
  new ResizeObserver(() => { const width=root.clientWidth;if(Math.abs(width-lastWidth)>0.5){lastWidth=width;drawActive();} }).observe(root);
  window.addEventListener('openai:set_globals', event => {
    const incoming=event.detail?.globals?.widgetState;
    const next=incoming?.modelContent ?? incoming;
    if(!next)return;
    pause();if(Number.isInteger(next.frameIndex))state.index=Math.max(0,Math.min(frames.length-1,next.frameIndex));
    setProbe(next.probe ?? 0);setView(next.view ?? 'network');
  });

  // 미리보기에서 수치와 조작을 검사하는 작은 인터페이스.
  window.chapter4Animation = {
    state: () => ({ index:state.index,update:currentFrame().update,view:state.view,probe:state.probe,playing:state.playing,
      loss:currentFrame().loss,ppl:Math.exp(currentFrame().loss),probabilities:infer(currentFrame(),state.probe).probabilities,
      mass:data.generationMass[state.index],reducedMotion:media.matches }),
    seek: update => { pause();setFrame(globalThis.NplmSvg.nearestIndex(frames,frame=>Math.abs(frame.update-update))); },
    play, pause, setView, setProbe,
    validate: () => {
      let maximumError=0;
      frames.forEach((frame,index)=>data.probes.forEach((_,probe)=>{
        infer(frame,probe).probabilities.forEach((p,v)=>{maximumError=Math.max(maximumError,Math.abs(p-data.reference[index][probe][v]));});
      }));
      return {frames:frames.length,losses:data.losses.length,maximumProbabilityError:maximumError,existingResultMatched:data.verifiedExisting,
        massSumsValid:data.generationMass.every(mass=>Math.abs(mass.training+mass.novel+mass.invalid-1)<1e-12),
        overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth};
    },
    labelBoxes: () => scatterBoxes,
  };
  console.log(JSON.stringify({animation:window.chapter4Animation.validate()}));
})();
