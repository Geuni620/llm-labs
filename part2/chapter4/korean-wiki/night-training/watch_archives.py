"""Preserve two reached evaluation states; never starts or changes training."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;WIKI=BASE.parent
from archive_checkpoint import metadata

def stamp():return datetime.now(timezone.utc).astimezone().isoformat()

def main():
    c=json.loads((WIKI/'reports/night-20261022-control.json').read_text());run=Path(c['run_dir']);pid=c['pid']
    info=json.loads((run/'run.json').read_text())
    jobs=[{'name':'elapsed-3h','args':['--min-elapsed','10800'],'path':str(run/'checkpoint-elapsed-3h.npz'),'status':'waiting','attempts':0},
          {'name':'epoch-1','args':['--metric-label','epoch_1'],'path':str(run/'checkpoint-epoch-1.npz'),'status':'waiting','attempts':0}]
    progress=WIKI/'reports/night-20261022-archive-progress.json'
    def save(status,**more):
        value={'at':stamp(),'status':status,'pid':os.getpid(),'training_pid':pid,'run_dir':str(run),'jobs':jobs,**more}
        temporary=progress.with_suffix('.json.partial')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,progress)
    with (run/'.checkpoint-archive.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                q=subprocess.run(['/bin/ps','-p',str(pid),'-o','command='],capture_output=True,text=True,timeout=10)
                if q.returncode not in (0,1):raise OSError('Training process query failed')
                alive='night-training/run.py' in q.stdout and str(run) in q.stdout
                metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
            except (OSError,ValueError,subprocess.TimeoutExpired) as error:
                save('observation-unresolved',error=f'{type(error).__name__}: {error}');time.sleep(30);continue
            for job in jobs:
                if job['status']=='archived':continue
                path=Path(job['path']);sidecar=path.with_suffix('.json')
                if path.exists() or sidecar.exists():
                    # Existing files are evidence to validate, never overwrite.
                    if not path.exists() or not sidecar.exists():
                        job['status']='failed-incomplete-archive';save('failed');return 1
                    with path.open('rb') as f:m=metadata(f)
                    saved=json.loads(sidecar.read_text());metric=saved['actual_metric']
                    if (m['identity']!=info['identity'] or m['config']!=info['config']
                        or m['consumed']!=metric['consumed'] or m['updates']!=metric['updates']
                        or metric['evaluation_window_sha256']!=info['evaluation_window_sha256']):
                        job['status']='failed-archive-binding';save('failed');return 1
                    job.update(status='archived',consumed=m['consumed'],updates=m['updates'],
                               actual_elapsed_seconds=metric['actual_elapsed_seconds'],sha256=saved['sha256'])
                    continue
                reached=any(m['consumed']>0 and (m['actual_elapsed_seconds']>=10800 if job['name']=='elapsed-3h' else m['label']=='epoch_1') for m in metrics)
                if reached:
                    command=[sys.executable,str(BASE/'archive_checkpoint.py'),'--run',str(run),*job['args'],'--output',str(path)]
                    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
                    job['attempts']+=1
                    if result.returncode:
                        job['last_error']=(result.stdout+result.stderr)[-2000:]
                        if job['attempts']>=5:job['status']='failed';save('failed');return 1
                    else:
                        print(json.dumps({'at':stamp(),'archived':job['name'],'result':result.stdout.strip()},ensure_ascii=False),flush=True)
                elif not alive:job['status']='not-reached-before-training-ended'
            if all(j['status']=='archived' for j in jobs):save('complete');return 0
            if not alive:save('training-ended');return 0
            save('waiting-for-reached-saved-states');time.sleep(30)

if __name__=='__main__':raise SystemExit(main())
