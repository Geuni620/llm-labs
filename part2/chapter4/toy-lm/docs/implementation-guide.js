(() => {
  const root=document.getElementById('implementation-guide');
  const data=chapter4GuideData;
  const {add,set}=globalThis.NplmSvg;
  const select=root.querySelector('[data-section]');
  const codeSelect=root.querySelector('[data-code-section]');
  const previous=root.querySelector('[data-previous]'),next=root.querySelector('[data-next]');
  let index=0,codeIndex=0,lastWidth=0;
  function textElement(tag,text,parent){const node=document.createElement(tag);node.textContent=text;parent.append(node);return node;}
  data.sections.forEach((section,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`${i+1}. ${section.title}`;select.append(option);});
  function draw(){
    const section=data.sections[index],svg=root.querySelector('svg'),width=svg.getBoundingClientRect().width,wide=width>=650;
    svg.replaceChildren();
    const defs=add(svg,'defs'),marker=add(defs,'marker',{id:'implementation-arrow',viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});
    add(marker,'path',{d:'M0,0 L10,5 L0,10 Z',fill:'var(--muted-foreground)'});
    let boxes,height,connections;
    if(section.kind==='branches'){
      if(wide){const w=140;boxes=[{x:0,y:90,w},{x:width*.27,y:22,w},{x:width*.27,y:158,w},{x:width*.56,y:90,w},{x:width-w,y:90,w}];height=250;}
      else{const w=(width-20)/2,center=(width-w)/2;boxes=[{x:center,y:8,w},{x:0,y:105,w},{x:width-w,y:105,w},{x:center,y:202,w},{x:center,y:299,w}];height=390;}
      connections=[[0,1],[0,2],[1,3],[2,3],[3,4]];
    }else{
      const n=section.nodes.length,gap=18,w=wide?(width-(n-1)*gap)/n:width-32;
      boxes=section.nodes.map((_,i)=>({x:wide?i*(w+gap):16,y:wide?18:i*88+5,w}));
      height=wide?112:n*88+4;connections=section.nodes.slice(1).map((_,i)=>[i,i+1]);
    }
    set(svg,{viewBox:`0 0 ${width} ${height}`,height});
    for(const[a,b]of connections){const from=boxes[a],to=boxes[b];const d=wide?`M${from.x+from.w+2},${from.y+32} L${to.x-5},${to.y+32}`:`M${from.x+from.w/2},${from.y+66} L${to.x+to.w/2},${to.y-5}`;add(svg,'path',{d,fill:'none',stroke:'var(--muted-foreground)','stroke-width':1.3,'marker-end':'url(#implementation-arrow)'});}
    section.nodes.forEach((node,i)=>{
      const b=boxes[i],center=b.x+b.w/2;
      add(svg,'rect',{x:b.x+.5,y:b.y+.5,width:b.w-1,height:65,rx:7,fill:'var(--muted)',stroke:'var(--border)'});
      add(svg,'text',{x:center,y:b.y+23,'text-anchor':'middle','font-weight':500},node[0]);
      const detail=add(svg,'text',{x:center,y:b.y+43,'text-anchor':'middle','font-size':12});
      const lines=[];let line='';
      for(const character of node[1]){detail.textContent=line+character;if(detail.getComputedTextLength()>b.w-24&&line){lines.push(line);line=character;}else line+=character;}
      if(line)lines.push(line);detail.textContent='';
      lines.forEach((value,j)=>add(detail,'tspan',{x:center,y:b.y+43+j*14},value));
    });
  }
  function showCode(value){
    const section=data.sections[index];codeIndex=Math.max(0,Math.min(section.snippets.length-1,Number(value)||0));codeSelect.value=String(codeIndex);
    const snippet=section.snippets[codeIndex],snippets=root.querySelector('[data-snippets]');snippets.replaceChildren();
    textElement('div',snippet.label,snippets).className='snippet-label';textElement('div',`${snippet.file} · ${snippet.start}~${snippet.end}행`,snippets).className='source-range';const pre=document.createElement('pre');textElement('code',snippet.code,pre);snippets.append(pre);
  }
  function show(value,save=false){
    index=Math.max(0,Math.min(data.sections.length-1,Number(value)||0));
    const section=data.sections[index];select.value=String(index);previous.disabled=index===0;next.disabled=index===data.sections.length-1;
    root.querySelector('[data-progress]').textContent=`${index+1} / ${data.sections.length}`;
    root.querySelector('[data-title]').textContent=section.title;
    const paragraphs=root.querySelector('[data-paragraphs]');paragraphs.replaceChildren();section.paragraphs.forEach(text=>textElement('p',text,paragraphs));
    const list=root.querySelector('[data-bullets]');list.replaceChildren();(section.bullets??[]).forEach(text=>textElement('li',text,list));list.hidden=!section.bullets?.length;
    const formula=root.querySelector('[data-formula]');formula.hidden=!section.formula;formula.textContent=section.formula??'';
    codeSelect.replaceChildren();section.snippets.forEach((snippet,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=snippet.label;codeSelect.append(option);});
    root.querySelector('[data-code-picker]').hidden=section.snippets.length<=1;showCode(0);
    draw();
    if(save&&window.openai?.setWidgetState)window.openai.setWidgetState({modelContent:{chapter:index,title:section.title}}).catch(()=>{});
  }
  previous.addEventListener('click',()=>show(index-1,true));next.addEventListener('click',()=>show(index+1,true));select.addEventListener('change',()=>show(select.value,true));
  codeSelect.addEventListener('change',()=>showCode(codeSelect.value));
  const saved=window.openai?.widgetState;show(saved?.modelContent?.chapter??saved?.chapter??0);
  lastWidth=root.clientWidth;new ResizeObserver(()=>{if(root.clientWidth!==lastWidth){lastWidth=root.clientWidth;draw();}}).observe(root);
  window.addEventListener('openai:set_globals',event=>{const state=event.detail?.globals?.widgetState;const chapter=state?.modelContent?.chapter??state?.chapter;if(chapter!==undefined)show(chapter);});
  window.chapter4Guide={show,showCode,count:data.sections.length,snippetCount:()=>data.sections[index].snippets.length,check:()=>({index,codeIndex,overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth})};
  console.log(JSON.stringify({sourceGuide:data.sections.length,runtimeLibraries:data.runtimeDependencies.length}));
})();
