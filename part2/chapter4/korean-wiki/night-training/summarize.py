"""Audit a stopped night run and export actual held-out curves; never trains."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

BASE=Path(__file__).resolve().parent
WIKI=BASE.parent

def identity(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(4*1024*1024):h.update(data)
    return {'bytes':Path(path).stat().st_size,'sha256':h.hexdigest()}

def assert_not_running(run,control_path=None):
    control_path=Path(control_path) if control_path is not None else WIKI/'reports/night-20261022-control.json'
    control=json.loads(control_path.read_text())
    if Path(control['run_dir']).resolve()!=run:raise ValueError('Unexpected run directory')
    progress=run/'progress.json'
    current=json.loads(progress.read_text()) if progress.exists() else {}
    for pid in set(filter(None,[current.get('pid'),control.get('pid')])):
        result=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=10)
        if result.returncode not in (0,1):raise RuntimeError('Cannot establish process termination')
        if 'night-training/run.py' in result.stdout and str(run) in result.stdout:
            raise RuntimeError('Actual trainer is still alive; final results remain pending')
    if current.get('pid') and current['pid']!=control.get('pid'):
        raise RuntimeError('Supervisor terminal record belongs to an earlier PID')
    if control.get('exit_code') is None:raise RuntimeError('Supervisor has not recorded a terminal result')
    return control

def curve_points(metrics,expected_hash):
    points=[];previous=-1
    for m in metrics:
        if m['kind']!='ACTUAL_CPU_C_CHECKPOINT' or m['evaluation_window_sha256']!=expected_hash:
            raise ValueError('Mixed or unverified evaluation conditions')
        if m['consumed']<previous:raise ValueError('Training targets moved backwards')
        previous=m['consumed'];v=m['validation']
        for prefix in ('','content_'):
            n=v[prefix+'targets'];loss=v[prefix+'loss'];nll=v[prefix+'negative_log_likelihood_sum']
            if not n or not math.isfinite(loss) or not math.isclose(loss,nll/n,rel_tol=1e-12):
                raise ValueError('Invalid held-out loss/count')
            if not math.isclose(v[prefix+'perplexity'],math.exp(loss),rel_tol=1e-12):
                raise ValueError('Invalid held-out perplexity')
        points.append({'label':m['label'],'at':m['at'],'actual_elapsed_seconds':m['actual_elapsed_seconds'],
                       'epochs_equivalent':m['epochs_equivalent'],'target_predictions':m['consumed'],
                       'updates':m['updates'],'visited_paragraphs':m['visited_paragraphs'],'validation':v,
                       'generation':m['generation'],'peak_rss_bytes':m['peak_rss_bytes']})
    return points

def inspect_checkpoint(path):
    import numpy as np  # Only after the real trainer has stopped; no model runtime.
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!={'metadata','parameters','moment1','moment2','visited'}:raise ValueError('Checkpoint schema differs')
        m=json.loads(a['metadata'].tobytes());shape=a['parameters'].shape
        for key in ('parameters','moment1','moment2'):
            x=a[key]
            if x.dtype!=np.float32 or x.ndim!=1 or x.shape!=shape or not np.isfinite(x).all():
                raise ValueError('Invalid checkpoint '+key)
            if key=='moment2' and np.any(x<0):raise ValueError('Negative second moment')
        visited=a['visited']
        if visited.dtype!=np.uint8 or visited.ndim!=1 or np.any(visited>1) or int(visited.sum())!=m['visited_paragraphs']:
            raise ValueError('Invalid durable visited count')
        if m['sampler'][4]!=m['consumed']:raise ValueError('Invalid durable sampler count')
    return {'path':str(path),'identity':identity(path),'metadata':m,'finite_arrays':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--control',type=Path)
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve()
    if output.exists() or output.with_suffix('.txt').exists():raise FileExistsError('Preserve prior result')
    control=assert_not_running(run,args.control)
    metadata=json.loads((run/'run.json').read_text())
    summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
    last=json.loads((run/'progress.json').read_text())
    raw=(run/'metrics.jsonl').read_text().splitlines();metrics=[];incomplete_tail=None
    for index,line in enumerate(raw):
        try:metrics.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(raw)-1 or summary is not None:raise
            incomplete_tail={'line':index+1,'reason':'Interrupted final metric record retained on disk'}
    points=curve_points(metrics,metadata['evaluation_window_sha256'])
    scripts={name:identity(BASE/name)==expected for name,expected in metadata['identity']['scripts'].items()}
    data={name:identity(Path(metadata['corpus'])/name)==expected for name,expected in metadata['identity']['data'].items()}
    library=identity(BASE/'.artifacts/libnight_engine.dylib')==metadata['identity']['library']
    checkpoint_path=run/'latest.npz'
    if not checkpoint_path.exists():checkpoint_path=run/'latest.previous.npz'
    checkpoint=inspect_checkpoint(checkpoint_path)
    if checkpoint['metadata']['identity']!=metadata['identity'] or checkpoint['metadata']['config']!=metadata['config']:
        raise ValueError('Durable checkpoint does not match actual run')
    final=summary['final'] if summary is not None else last
    if checkpoint['metadata']['consumed']>final['consumed']:raise ValueError('Durable count exceeds last recorded count')
    target=metadata['plan']['training']['target_epochs']*final['windows']
    last_evaluated=points[-1] if points else None
    checkpoint_evaluated=bool(last_evaluated and checkpoint['metadata']['consumed']==last_evaluated['target_predictions']
                              and checkpoint['metadata']['updates']==last_evaluated['updates'])
    outcome={'kind':metadata.get('result_kind','ACTUAL_OVERNIGHT_CPU_C_RESULT'),'at':datetime.now(timezone.utc).astimezone().isoformat(),
             'control':control,'run':metadata,'summary':summary,'last_recorded_progress':final,
             'final_count_confirmed':summary is not None,'target_predictions':target,
             'target_progress_percent':100*final['consumed']/target,
             'completed_target_epochs':summary is not None and final['consumed']>=target,
             'final_saved_weights_have_recorded_evaluation':checkpoint_evaluated,
             'last_actual_evaluation_at':last_evaluated['at'] if last_evaluated else None,
             'validation_curve':points,'checkpoint':checkpoint,'incomplete_metric_tail':incomplete_tail,
             'preservation':{'scripts':scripts,'data':data,'library':library,
                             'old_queue_STOP_exists':(WIKI/'training-runs/wiki-k35328/STOP').exists()},
             'scope':metadata.get('scope','One fresh CPU/C H128 run with four requested threads. No 2/3/5-epoch or CPU/MPS quality claim unless the corresponding actual checkpoint exists. Content loss excludes WS/EOS targets.')}
    if not all(scripts.values()) or not all(data.values()) or not library:raise ValueError('Run source/input binding changed')
    output.write_text(json.dumps(outcome,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['CPU/C NPLM 실제 야간 결과',f"실행 상태: {control['status']} / exit={control['exit_code']}",
           f"종료 사유: {summary['reason'] if summary else 'Final summary missing; last recorded progress and durable checkpoint only'}",
           f"실제 기록된 경과 시간: {final['elapsed_seconds']:.3f}초",
           f"학습량: {final['consumed']:,} 타깃 / {final['epochs_equivalent']:.6f} epoch / 목표 대비 {outcome['target_progress_percent']:.3f}%",
           f"종료 저장본 자체의 평가 기록: {'있음' if checkpoint_evaluated else '없음; 최근 평가값을 종료 시점 품질로 해석하지 않음'}",
           f"최신 완전한 체크포인트: {checkpoint['metadata']['consumed']:,} 타깃",'',
           '단계\t실제초\tepoch\t타깃예측\t검증Loss\tPPL\t공백/EOS제외Loss\t제외PPL']
    for p in points:
        v=p['validation'];lines.append(f"{p['label']}\t{p['actual_elapsed_seconds']:.3f}\t{p['epochs_equivalent']:.6f}\t{p['target_predictions']}\t{v['loss']:.6f}\t{v['perplexity']:.3f}\t{v['content_loss']:.6f}\t{v['content_perplexity']:.3f}")
    lines.extend(['',outcome['scope']]);output.with_suffix('.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'result':str(output),'actual_metric_points':len(points),'target_progress_percent':outcome['target_progress_percent'],'completed_target_epochs':outcome['completed_target_epochs']}))

if __name__=='__main__':main()
