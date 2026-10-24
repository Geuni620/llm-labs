"""Read-only process/RSS observations for the already-running owned night job.

No training, model import, restart, signal, thermal policy or deadline changes.
An observation error is not evidence that the trainer stopped.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

WIKI=Path(__file__).resolve().parent.parent

def main():
    control=json.loads((WIKI/'reports/night-20261022-control.json').read_text())
    run=Path(control['run_dir']);pid=int(control['pid'])
    observations=run/'resource-observations.jsonl'
    # One reader only. This does not acquire or change the trainer's locks.
    import fcntl
    with (run/'.resource-observer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            at=datetime.now(timezone.utc).astimezone().isoformat()
            try:
                process=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid=,rss=,%cpu=,etime=,command='],
                                       capture_output=True,text=True,timeout=10)
                if process.returncode not in (0,1):raise RuntimeError('ps query failed: '+process.stderr)
                fields=process.stdout.strip().split(None,4)
                alive=(len(fields)==5 and int(fields[0])==pid and 'night-training/run.py' in fields[4] and str(run) in fields[4])
                p=json.loads((run/'progress.json').read_text())
                item={'at':at,'run_dir':str(run),'pid':pid,'process_identity_confirmed':alive,
                      'rss_bytes':int(fields[1])*1024 if alive else None,
                      'cpu_percent':float(fields[2]) if alive else None,
                      'process_elapsed':fields[3] if alive else None,'source_progress_at':p['at'],
                      'recorded_status':p['status'],'target_predictions':p['consumed'],
                      'target_epoch_goal':2,'epoch_windows':p['windows'],
                      'epochs_equivalent':p['epochs_equivalent'],'target_progress_percent':100*p['consumed']/(2*p['windows']),
                      'peak_rss_bytes':p['peak_rss_bytes'],'guard_at_progress':p['guard']}
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired) as error:
                item={'at':at,'pid':pid,'run_dir':str(run),'observation_error':f'{type(error).__name__}: {error}',
                      'process_identity_confirmed':None,'policy':'Status unresolved; retry the same PID, never restart.'}
                alive=None
            with observations.open('a',encoding='utf-8') as f:
                f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            print(json.dumps({k:item.get(k) for k in ['at','pid','process_identity_confirmed','target_predictions','target_progress_percent','rss_bytes','cpu_percent','observation_error']},ensure_ascii=False),flush=True)
            if alive is False:return
            time.sleep(60)

if __name__=='__main__':main()
