"""Own one detached trainer and persist its real exit result; never auto-restart."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parent
WIKI=BASE.parent
CONTROL=WIKI/'reports/night-20261022-control.json'

def stamp():return datetime.now(timezone.utc).astimezone().isoformat()

def write(value):
    temporary=CONTROL.with_suffix('.json.partial')
    with temporary.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,CONTROL)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=a.plan.resolve();output=a.output.resolve()
    with (WIKI/'reports/.night-20261022-launch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():raise FileExistsError('Preserve existing run: '+str(output))
        control=json.loads(CONTROL.read_text())
        if str(output)!=control['run_dir']:raise ValueError('Unexpected run directory')
        settings=json.loads(plan.read_text())
        if not settings.get('physical_check_confirmed'):raise ValueError('Missing physical confirmation')
        if datetime.now(timezone.utc)>=datetime.fromisoformat(settings['stop_training_by']):raise ValueError('Training deadline passed')
        command=[sys.executable,'-u',str(BASE/'run.py'),'--plan',str(plan),'--output',str(output)]
        log=WIKI/'reports/night-20261022-training.log'
        with log.open('ab',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=WIKI,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,
                                   close_fds=True,start_new_session=True)
        control.update(status='launched',pid=child.pid,supervisor_pid=os.getpid(),command=command,
                       execution_started=True,started_at=stamp(),plan=str(plan),log=str(log))
        write(control)
        try:
            assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)
        except OSError:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            control.update(status='failed',exit_code=child.returncode,ended_at=stamp(),error='Sleep assertion failed; owned child stopped')
            write(control);raise
        control.update(caffeinate_pid=assertion.pid);write(control)
        result=child.wait()
        try:assertion.wait(timeout=5)
        except subprocess.TimeoutExpired:assertion.terminate();assertion.wait(timeout=5)
        summary_path=output/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        control.update(status=summary.get('status','protective-stop' if result in (86,87,88) else 'failed'),
                       exit_code=result,ended_at=stamp(),reason=summary.get('reason'),
                       exit_meanings={'86':'critical thermal state','87':'hard morning deadline','88':'protective stop acknowledgement timeout'})
        write(control)
        return 0 if result==0 else result

if __name__=='__main__':raise SystemExit(main())
