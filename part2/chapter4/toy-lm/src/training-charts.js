// 브라우저 기본 DOM·SVG API와 이 프로젝트의 svg.js만 사용한다.
globalThis.createNplmCharts = (root, data, controls) => {
  'use strict';
  const {add,set,range,extent,linear,niceDomain,nearestIndex,maxIndex,path,bottomAxis,leftAxis,pointer} = globalThis.NplmSvg;
  const {state,currentFrame,getPinned,setPinned,setBoxes}=controls;
  const frames=data.frames,tokens=data.data.tokens,D=data.settings.dimension,H=data.settings.hidden,V=tokens.length;
  const ordinary=tokens.map((token,id)=>({token,id})).filter(row=>!row.token.startsWith('<'));
  const semanticColor=token=>token.endsWith('는')?'var(--chart-1)':token.endsWith('를')?'var(--chart-2)':'var(--chart-3)';
  const percent=value=>(100*value).toFixed(2)+'%';
  const maximumWeight=name=>Math.max(1e-12,...frames.flatMap(frame=>frame.parameters[name].map(Math.abs)));
  const maxW1=maximumWeight('W1'),maxW2=maximumWeight('W2');
  const tooltip=root.querySelector('.tooltip');
  function canvas(selector,height){
    const svg=root.querySelector(selector),width=svg.getBoundingClientRect().width;
    svg.replaceChildren();set(svg,{viewBox:`0 0 ${width} ${height}`,height});return{svg,width};
  }
  function label(svg,axis,text,x,y,transform){return add(svg,'text',{class:'axis-title','data-axis':axis,x,y,'text-anchor':'middle',transform},text);}
  function frame(svg,left,top,width,height){add(svg,'rect',{'data-chart-frame':'',x:left,y:top,width,height,fill:'none',stroke:'var(--border)'});}
  function showTooltip(event,text){
    tooltip.textContent=text;tooltip.style.whiteSpace='pre-line';tooltip.hidden=false;
    const bounds=root.getBoundingClientRect();
    tooltip.style.left=`${Math.max(0,Math.min(root.clientWidth-tooltip.offsetWidth-4,event.clientX-bounds.left+12))}px`;
    tooltip.style.top=`${Math.max(0,event.clientY-bounds.top+12)}px`;
  }
  function hideTooltip(){if(!getPinned())tooltip.hidden=true;}
  function interactions(hit,detail,leave=()=>{}){
    hit.addEventListener('pointermove',event=>{if(!getPinned()&&event.pointerType!=='touch')detail(event);});
    hit.addEventListener('pointerleave',()=>{if(!getPinned())leave();hideTooltip();});
    hit.addEventListener('click',event=>{setPinned(!getPinned());if(getPinned())detail(event);else{tooltip.hidden=true;leave();}});
  }

  function drawNetwork(){
    const {svg,width}=canvas('[data-network]',430);
    const eX=width*.18,hX=width*.5,oX=width*.8,eY=[130,182,290,342];
    const hY=linear([0,H-1],[70,390]),oY=linear([0,V-1],[70,390]);
    for(const [x,text] of [[eX,'임베딩 4값'],[hX,`은닉 ${H}개`],[oX,`출력 ${V}개`]])add(svg,'text',{x,y:25,'text-anchor':'middle'},text);
    const inputEdges=range(4).flatMap(i=>range(H).map(h=>({i,h,node:add(svg,'line',{class:'weight',x1:eX,y1:eY[i],x2:hX,y2:hY(h),stroke:'var(--muted-foreground)','stroke-width':.8})})));
    const outputEdges=range(H).flatMap(h=>range(V).map(v=>({h,v,node:add(svg,'line',{class:'weight',x1:hX,y1:hY(h),x2:oX,y2:oY(v),'stroke-width':.8})})));
    const contextLabels=[106,266].map(y=>add(svg,'text',{x:eX,y,'text-anchor':'middle'}));
    const inputs=range(4).map(i=>{add(svg,'circle',{cx:eX,cy:eY[i],r:8,fill:'var(--chart-1)'});return add(svg,'text',{x:eX-13,y:eY[i]+4,'text-anchor':'end'});});
    const hidden=range(H).map(h=>add(svg,'circle',{cx:hX,cy:hY(h),r:15,fill:'var(--chart-4)',stroke:'var(--border)'}));
    const hiddenText=range(H).map(h=>add(svg,'text',{x:hX,y:hY(h)+4,'text-anchor':'middle'}));
    const outputs=range(V).map(v=>add(svg,'circle',{cx:oX,cy:oY(v),r:8,fill:'var(--chart-2)'}));
    tokens.forEach((token,v)=>add(svg,'text',{x:oX+12,y:oY(v)+4},token));
    return(record,prediction)=>{
      const best=maxIndex(prediction.probabilities);
      inputEdges.forEach(({i,h,node})=>set(node,{opacity:.03+.32*Math.abs(record.parameters.W1[i*H+h])/maxW1}));
      outputEdges.forEach(({h,v,node})=>set(node,{stroke:v===best?'var(--chart-2)':'var(--muted-foreground)',opacity:(v===best?.18:.025)+.28*Math.abs(record.parameters.W2[h*V+v])/maxW2}));
      contextLabels.forEach((node,i)=>{node.textContent=data.probes[state.probe][i];});
      inputs.forEach((node,i)=>{node.textContent=prediction.input[i].toFixed(2);});
      hidden.forEach((node,h)=>set(node,{'fill-opacity':.1+.3*Math.abs(prediction.hidden[h])}));
      hiddenText.forEach((node,h)=>{const value=prediction.hidden[h].toFixed(1);node.textContent=Number(value)===0?'0.0':value;});
      outputs.forEach((node,v)=>set(node,{'fill-opacity':.08+.72*prediction.probabilities[v],stroke:v===best?'var(--chart-2)':'var(--border)','stroke-width':v===best?2:1}));
    };
  }

  function drawProbabilities(){
    const{svg,width}=canvas('[data-probabilities]',430);
    const left=64,right=width-54,top=47,bottom=390;
    const x=linear([0,1],[left+2,right-2]),step=(bottom-top)/V,barHeight=step*.7;
    const y=v=>top+v*step+step*.15;
    frame(svg,left,top,right-left,bottom-top);
    bottomAxis(svg,x,[0,.5,1],bottom,value=>`${Math.round(100*value)}%`);
    tokens.forEach((token,v)=>add(svg,'text',{x:left-6,y:y(v)+barHeight/2+4,'text-anchor':'end'},token));
    const bars=range(V).map(()=>{const node=add(svg,'rect',{class:'moving',x:0,y:0,height:barHeight,width:right-left-4,fill:'var(--chart-1)'});node.style.transformOrigin='0px 0px';return node;});
    const values=range(V).map(v=>add(svg,'text',{x:right+6,y:y(v)+barHeight/2+4}));
    label(svg,'x','확률 (%)',(left+right)/2,426);label(svg,'y','다음 토큰',35,25);
    return(_,prediction)=>{
      bars.forEach((node,v)=>{node.style.transform=`translate(${left+2}px,${y(v)}px) scaleX(${prediction.probabilities[v]})`;});
      values.forEach((node,v)=>{node.textContent=percent(prediction.probabilities[v]);});
    };
  }

  function drawLoss(ppl=false){
    const{svg,width}=canvas(ppl?'[data-ppl-chart]':'[data-loss-chart]',320);
    const values=ppl?data.losses.map(value=>Math.exp(value)):data.losses;
    const left=64,right=width-20,top=22,bottom=266;
    const [minimum,maximum]=extent(values),padding=(maximum-minimum)*.08;
    const domain=niceDomain(Math.max(0,minimum-padding),maximum+padding,5);
    const x=linear([0,data.settings.updates],[left+5,right-5]),y=linear(domain.domain,[bottom-5,top+5]);
    frame(svg,left,top,right-left,bottom-top);
    const ticks=width<500?[0,data.settings.updates/2,data.settings.updates]:range(5).map(i=>i*data.settings.updates/4);
    bottomAxis(svg,x,ticks,bottom,value=>String(Math.round(value)));leftAxis(svg,y,domain.ticks,left);
    label(svg,'x','가중치 갱신 (회)',(left+right)/2,313);
    label(svg,'y',ppl?'PPL':'평균 loss',15,(top+bottom)/2,`rotate(-90,15,${(top+bottom)/2})`);
    const clipId=ppl?'chapter4-ppl-reveal':'chapter4-loss-reveal';
    const clip=add(add(add(svg,'defs'),'clipPath',{id:clipId,clipPathUnits:'userSpaceOnUse'}),'rect',{class:'moving',x:0,y:0,width:right-left,height:bottom-top});
    clip.style.transformOrigin='0px 0px';
    add(svg,'path',{d:path(values.map((value,i)=>[x(i),y(value)])),'clip-path':`url(#${clipId})`,fill:'none',stroke:'var(--chart-1)','stroke-width':2});
    const marker=add(svg,'g',{class:'moving'});add(marker,'circle',{r:4,fill:'var(--chart-1)'});
    const guide=add(svg,'line',{y1:top,y2:bottom,stroke:'var(--border)',visibility:'hidden'});
    const hit=add(svg,'rect',{'data-chart-hit':'',x:left,y:top,width:right-left,height:bottom-top,fill:'transparent'});
    interactions(hit,event=>{
      const [px]=pointer(event,svg),update=Math.max(0,Math.min(currentFrame().update,Math.round(x.invert(px))));
      set(guide,{x1:x(update),x2:x(update),visibility:'visible'});
      showTooltip(event,`${update}회 갱신\nloss ${data.losses[update].toFixed(6)}\nPPL ${Math.exp(data.losses[update]).toFixed(6)}`);
    },()=>set(guide,{visibility:'hidden'}));
    return record=>{
      clip.style.transform=`translate(${left}px,${top}px) scaleX(${(x(record.update)-left)/(right-left)})`;
      marker.style.transform=`translate(${x(record.update)}px,${y(values[record.update])}px)`;
    };
  }

  function drawEmbedding(){
    const{svg,width}=canvas('[data-embedding]',350);
    const left=64,right=width-20,top=22,bottom=296;
    const all=frames.flatMap(record=>ordinary.map(row=>[record.parameters.E[row.id*D],record.parameters.E[row.id*D+1]]));
    const xe=extent(all,point=>point[0]),ye=extent(all,point=>point[1]);
    const xp=(xe[1]-xe[0])*.22,yp=(ye[1]-ye[0])*.22;
    const xd=niceDomain(xe[0]-xp,xe[1]+xp,width<500?3:5),yd=niceDomain(ye[0]-yp,ye[1]+yp,5);
    const x=linear(xd.domain,[left+6,right-6]),y=linear(yd.domain,[bottom-6,top+6]);
    frame(svg,left,top,right-left,bottom-top);bottomAxis(svg,x,xd.ticks,bottom);leftAxis(svg,y,yd.ticks,left);
    label(svg,'x','임베딩 첫 번째 성분',(left+right)/2,343);
    label(svg,'y','임베딩 두 번째 성분',15,(top+bottom)/2,`rotate(-90,15,${(top+bottom)/2})`);
    const trails=ordinary.map(row=>add(svg,'path',{fill:'none',stroke:semanticColor(row.token),'stroke-width':1,opacity:.3}));
    const leaders=ordinary.map(()=>add(svg,'line',{stroke:'var(--border)','stroke-width':1}));
    const dots=ordinary.map(row=>{const g=add(svg,'g',{class:'moving'});add(g,'circle',{r:5,fill:semanticColor(row.token)});return g;});
    const texts=ordinary.map(row=>add(svg,'text',{class:'moving'},row.token));
    const widths=texts.map(node=>node.getComputedTextLength());
    let points=[];
    const hit=add(svg,'rect',{'data-chart-hit':'',x:left,y:top,width:right-left,height:bottom-top,fill:'transparent'});
    interactions(hit,event=>{
      const [px,py]=pointer(event,svg),i=nearestIndex(points,p=>(p.x-px)**2+(p.y-py)**2),point=points[i];
      if(point)showTooltip(event,`${point.token}\n[${point.values[0].toFixed(4)}, ${point.values[1].toFixed(4)}]\n${currentFrame().update}회 갱신`);
    });
    return record=>{
      points=ordinary.map(row=>{const values=record.parameters.E.slice(row.id*D,row.id*D+D);return{...row,values,x:x(values[0]),y:y(values[1])};});
      trails.forEach((node,i)=>set(node,{d:path(frames.slice(0,state.index+1).map(history=>[x(history.parameters.E[ordinary[i].id*D]),y(history.parameters.E[ordinary[i].id*D+1])]))}));
      dots.forEach((node,i)=>{node.style.transform=`translate(${points[i].x}px,${points[i].y}px)`;});
      const placed=[],overlaps=(a,b)=>a.x<b.x+b.width+3&&a.x+a.width+3>b.x&&a.y<b.y+b.height+3&&a.y+a.height+3>b.y;
      texts.forEach((node,i)=>{
        const point=points[i],w=widths[i];
        const offsets=[[8,-8],[-w-8,-8],[8,19],[-w-8,19],[8,-25],[-w-8,-25],[8,36],[-w-8,36],[-w/2,-38],[-w/2,48],[24,-8],[-w-24,-8],[24,25],[-w-24,25],[8,-52],[-w-8,-52],[8,62],[-w-8,62]];
        const valid=box=>box.x>=left+3&&box.x+w<=right-3&&box.y>=top+3&&box.y+14<=bottom-3&&!placed.some(other=>overlaps(box,other))&&!points.some(p=>p.x>box.x-5&&p.x<box.x+w+5&&p.y>box.y-5&&p.y<box.y+19);
        let box=offsets.map(([dx,dy])=>({x:point.x+dx,y:point.y+dy-12,width:w,height:14})).find(valid);
        if(!box){const candidates=[];for(let cy=top+3;cy<=bottom-17;cy+=8)for(let cx=left+3;cx<=right-w-3;cx+=8)candidates.push({x:cx,y:cy,width:w,height:14});candidates.sort((a,b)=>(a.x+w/2-point.x)**2+(a.y+7-point.y)**2-((b.x+w/2-point.x)**2+(b.y+7-point.y)**2));box=candidates.find(valid);}
        box??={x:left+3,y:top+3,width:w,height:14};placed.push(box);
        node.style.transform=`translate(${box.x}px,${box.y+12}px)`;
      });
      leaders.forEach((node,i)=>{const box=placed[i],point=points[i];set(node,{x1:point.x,y1:point.y,x2:Math.max(box.x,Math.min(box.x+box.width,point.x)),y2:Math.max(box.y,Math.min(box.y+box.height,point.y)),visibility:Math.hypot(box.x-point.x,box.y-point.y)>24?'visible':'hidden'});});
      setBoxes(placed);
    };
  }
  return{drawNetwork,drawProbabilities,drawLoss,drawEmbedding};
};
