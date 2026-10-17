// 이 실습에서 필요한 SVG 생성·좌표 변환·축·경로만 직접 구현한다.
// 브라우저에서는 전역 객체 NplmSvg로 사용한다.
(() => {
  'use strict';
  const namespace = 'http://www.w3.org/2000/svg';
  function set(node, attributes) {
    for (const [name, value] of Object.entries(attributes)) {
      if (value === null || value === undefined) node.removeAttribute(name);
      else node.setAttribute(name, String(value));
    }
    return node;
  }
  function add(parent, name, attributes = {}, text) {
    const node = document.createElementNS(namespace, name);
    set(node, attributes);
    if (text !== undefined) node.textContent = String(text);
    parent.append(node);
    return node;
  }
  function range(count) {
    if (!Number.isInteger(count) || count < 0) throw new Error('개수는 0 이상의 정수여야 합니다.');
    return Array.from({ length: count }, (_, i) => i);
  }
  function extent(values, accessor = value => value) {
    let minimum = Infinity, maximum = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const value = accessor(values[i], i);
      if (!Number.isFinite(value)) throw new Error('좌표에는 유한한 숫자가 필요합니다.');
      minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
    }
    if (minimum === Infinity) throw new Error('좌표 범위를 구할 데이터가 없습니다.');
    return [minimum, maximum];
  }
  function linear(domain, output) {
    const [d0, d1] = domain, [r0, r1] = output;
    if (![d0,d1,r0,r1].every(Number.isFinite)) throw new Error('좌표 변환 범위는 유한해야 합니다.');
    const scale = value => d0 === d1 ? (r0+r1)/2 : r0+(value-d0)/(d1-d0)*(r1-r0);
    scale.invert = value => r0 === r1 ? (d0+d1)/2 : d0+(value-r0)/(r1-r0)*(d1-d0);
    return scale;
  }
  function niceDomain(minimum, maximum, count = 5) {
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum || !Number.isInteger(count) || count < 2) throw new Error('축 범위가 잘못되었습니다.');
    if (minimum === maximum) { minimum -= 1; maximum += 1; }
    if (count === 2) return { domain:[minimum,maximum], ticks:[minimum,maximum] };
    const raw = (maximum-minimum)/(count-1);
    const power = 10 ** Math.floor(Math.log10(raw));
    const unit = [1,2,2.5,5,10].find(value => value*power >= raw) ?? 10;
    let step = unit*power;
    let first = Math.floor(minimum/step)*step, last = Math.ceil(maximum/step)*step;
    // 눈금이 너무 많으면 간격을 키운다. 좁은 화면에서도 요청한 개수를 넘기지 않는다.
    while (Math.round((last-first)/step)+1 > count) {
      step *= 2;
      first = Math.floor(minimum/step)*step; last = Math.ceil(maximum/step)*step;
    }
    const number = Math.round((last-first)/step);
    const ticks = Array.from({ length: number+1 }, (_, i) => Number((first+i*step).toPrecision(12)));
    return { domain:[ticks[0],ticks.at(-1)], ticks };
  }
  function nearestIndex(values, score) {
    let index = -1, best = Infinity;
    for (let i=0;i<values.length;i++) { const value=score(values[i],i);if(value<best){best=value;index=i;} }
    return index;
  }
  function maxIndex(values) { return nearestIndex(values, value => -value); }
  function path(points) {
    return points.map((point,i) => {
      if (point.length !== 2 || !point.every(Number.isFinite)) throw new Error('경로 좌표가 잘못되었습니다.');
      return `${i===0?'M':'L'}${point[0]},${point[1]}`;
    }).join(' ');
  }
  function format(value) {
    if (Math.abs(value)<1e-10) return '0';
    return Number(value.toFixed(3)).toString();
  }
  function bottomAxis(svg, scale, values, bottom, formatter = format) {
    const group=add(svg,'g',{'data-axis-ticks':'x'});
    values.forEach((value,i) => {
      const x=scale(value);
      add(group,'line',{x1:x,x2:x,y1:bottom,y2:bottom+5,stroke:'var(--border)'});
      add(group,'text',{x,y:bottom+20,'text-anchor':i===0?'start':i===values.length-1?'end':'middle'},formatter(value));
    });
    return group;
  }
  function leftAxis(svg, scale, values, left, formatter = format) {
    const group=add(svg,'g',{'data-axis-ticks':'y'});
    values.forEach(value => {
      const y=scale(value);
      add(group,'line',{x1:left-5,x2:left,y1:y,y2:y,stroke:'var(--border)'});
      add(group,'text',{x:left-9,y:y+4,'text-anchor':'end'},formatter(value));
    });
    return group;
  }
  function pointer(event, svg) {
    const bounds=svg.getBoundingClientRect();
    const box=svg.viewBox?.baseVal;
    const width=box?.width || bounds.width, height=box?.height || bounds.height;
    return [(event.clientX-bounds.left)*width/bounds.width+(box?.x??0),
      (event.clientY-bounds.top)*height/bounds.height+(box?.y??0)];
  }
  globalThis.NplmSvg = Object.freeze({add,set,range,extent,linear,niceDomain,nearestIndex,maxIndex,path,format,bottomAxis,leftAxis,pointer});
})();
