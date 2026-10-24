"""Read-only small progress and process snapshot; imports no model runtime."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

WIKI=Path(__file__).resolve().parent.parent

def main():
    control=json.loads((WIKI/'reports/night-20261022-control.json').read_text())
    output=Path(control['run_dir']);report={'at':datetime.now(timezone.utc).astimezone().isoformat(),'control':control}
    progress_path=output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    report['progress']=progress
    pid=progress.get('pid') or control.get('pid')
    if pid:
        process=subprocess.run(['/bin/ps','-p',str(int(pid)),'-o','pid=,rss=,%cpu=,etime=,command='],capture_output=True,text=True,timeout=5)
        raw=process.stdout.strip();parts=raw.split(None,4)
        matches=len(parts)==5 and str(WIKI/'night-training/run.py') in parts[4] and str(output) in parts[4]
        report['process']={'pid':pid,'alive_and_identity_matches':matches,'snapshot':raw}
        if matches:report['process'].update(rss_mib=int(parts[1])/1024,cpu_percent=float(parts[2]),elapsed=parts[3])
    if progress.get('windows'):
        plan=json.loads((output/'run-plan.json').read_text())
        target=plan['training']['target_epochs']*progress['windows']
        report['target_predictions']=target
        report['target_progress_percent']=100*progress['consumed']/target
    for name,command in [('swap',['/usr/sbin/sysctl','vm.swapusage']),('memory_pressure',['/usr/bin/memory_pressure','-Q'])]:
        result=subprocess.run(command,capture_output=True,text=True,timeout=5)
        report[name]={'exit_code':result.returncode,'output':(result.stdout+result.stderr).strip()}
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
