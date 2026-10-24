"""Own the authorized 1.512 -> cumulative 2/3/5 epoch continuation.

The original trainer, optimizer, library, corpus and completed night run stay
unchanged. This controller never retries training after a protective stop.
"""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

BASE=Path(__file__).resolve().parent
WIKI=BASE.parent
TAG='resume-20261023'
CONTROL=WIKI/f'reports/{TAG}-control.json'
PLAN=WIKI/'resume-run-20261023.execution.json'
SOURCE=WIKI/'training-runs/night-cpu-20261022-H128-B512'
RUN=WIKI/'training-runs/resume-cpu-20261023-H128-B512-to5'
PREFLIGHT=WIKI/'training-runs/resume-preflight-20261023-H128-B512'
MILESTONES=(2,3,5)
SOURCE_RESULT=WIKI/'reports/night-20261022-result.json'
FIXED_DEADLINES=None
RESTART_APPROVAL=None


def stamp():return datetime.now(timezone.utc).astimezone().isoformat()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def identity(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(4*1024*1024):h.update(data)
    return {'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def patch_control(**changes):
    value=read(CONTROL);value.update(changes);write(CONTROL,value);return value


def process_snapshot(pid,run):
    p=subprocess.run(['/bin/ps','-p',str(int(pid)),'-o','pid=,rss=,%cpu=,etime=,command='],capture_output=True,text=True,timeout=10)
    if p.returncode not in (0,1):raise OSError('Cannot determine process state: '+p.stderr)
    parts=p.stdout.strip().split(None,4)
    matches=(len(parts)==5 and int(parts[0])==pid and str(BASE/'run.py') in parts[4].split() and str(run) in parts[4].split())
    return {'pid':pid,'process_identity_confirmed':matches,'actual_command':parts[4] if len(parts)==5 else None,
            'process_rss_bytes':int(parts[1])*1024 if matches else None,
            'cpu_percent':float(parts[2]) if matches else None,'process_elapsed':parts[3] if matches else None}


def assert_no_trainer():
    p=subprocess.run(['/bin/ps','-axo','pid=,command='],capture_output=True,text=True,timeout=10)
    if p.returncode:raise OSError('Cannot inventory trainers')
    matches=[line for line in p.stdout.splitlines() if str(BASE/'run.py') in line.split() and '--output' in line.split()]
    if matches:raise RuntimeError('Existing trainer; refusing duplicate: '+str(matches))


def bound_sources(info):
    expected={str(BASE/name):v for name,v in info['identity']['scripts'].items()}
    expected[str(BASE/'.artifacts/libnight_engine.dylib')]=info['identity']['library']
    expected.update({str(Path(info['corpus'])/name):v for name,v in info['identity']['data'].items()})
    return expected


def verify_bindings(expected):
    checks={path:identity(path)==value for path,value in expected.items()}
    if not all(checks.values()):raise ValueError('Preserved source/input changed: '+str(checks))
    return checks


def evaluation_reference(source_metrics,checkpoint_metadata,resumed):
    if resumed is None:raise ValueError('No actual resumed evaluation record')
    match=next((m for m in reversed(source_metrics)
                if m['consumed']==checkpoint_metadata['consumed'] and m['updates']==checkpoint_metadata['updates']),None)
    # A protective stop can save newer weights than the most recent evaluation.
    # Compare two fresh zero-update evaluations, never the stale earlier score.
    return (match if match is not None else resumed),match is not None


def prepare():
    if CONTROL.exists() or RUN.exists() or PREFLIGHT.exists() or PLAN.exists() or any((WIKI/f'reports/{TAG}-{name}.json').exists() for name in ['result','comparison','generation-examples','resume-verification']):
        raise FileExistsError('Continuation already prepared; inspect existing control instead')
    assert_no_trainer()
    from summarize import inspect_checkpoint
    info=read(SOURCE/'run.json');old=read(SOURCE_RESULT)
    cp=inspect_checkpoint(SOURCE/'latest.npz')
    if cp['identity']!=old['checkpoint']['identity'] or cp['metadata']['identity']!=info['identity']:
        raise ValueError('Source checkpoint changed after completed night review')
    baseline_paths=[SOURCE/n for n in ['latest.npz','latest.previous.npz','run.json','summary.json','metrics.jsonl']]
    baseline_paths += [WIKI/'reports'/n for n in ['night-20261022-control.json','night-20261022-result.json','night-20261022-acceptance.json','experiment-continuation.json','training-queue.json']]
    baseline_paths.append(WIKI/'training-runs/wiki-k35328/STOP')
    baseline_paths.append(SOURCE_RESULT)
    baseline={str(p):identity(p) for p in baseline_paths}
    expected=bound_sources(info);verify_bindings(expected)
    now=datetime.now(timezone.utc).astimezone();plan=json.loads(json.dumps(info['plan']))
    plan.update(status='authorized_continuation_plan',requested_start=now.isoformat(),
                stop_training_by=(now+timedelta(hours=18)).isoformat(),exit_by=(now+timedelta(hours=18,minutes=30)).isoformat(),
                deadline_policy='Stop at cumulative 5 epochs or the fixed 18-hour training guard, whichever is first. Final cleanup has a separate 30-minute allowance. Never move these deadlines or auto-restart after thermal/monitor stop.',
                execution_started=False,source_run=str(SOURCE),cumulative_milestones=list(MILESTONES),
                physical_confirmation='Existing user confirmation in this thread: ventilated solid desk, no observed physical abnormalities. Reused for the explicitly authorized continuation; current exterior not remotely observed.')
    plan['training'].update(target_epochs=5,output_policy='Separate exact-state continuation; preserve the completed night run and stopped old 14-hour queue.')
    if FIXED_DEADLINES is not None:
        if not RESTART_APPROVAL or not RESTART_APPROVAL.get('approved'):
            raise ValueError('Explicit protective-stop restart authorization missing')
        for key in ['stop_training_by','exit_by']:
            if FIXED_DEADLINES[key]!=info['plan'][key]:raise ValueError('Do not extend inherited deadlines')
            plan[key]=FIXED_DEADLINES[key]
        if now>=datetime.fromisoformat(plan['stop_training_by']):raise ValueError('Inherited training deadline has passed')
        plan['deadline_policy']='User-approved full-state restart after protective stop; retain the exact original training/hard deadlines and guard. Stop at cumulative 5 epochs or the inherited limit. No further automatic restart.'
        plan['restart_approval']=RESTART_APPROVAL
    write(PLAN,plan)
    lineage={'source_run':str(SOURCE),'source_checkpoint':str(SOURCE/'latest.npz'),'source_checkpoint_identity':cp['identity'],
             'start_consumed':cp['metadata']['consumed'],'start_updates':cp['metadata']['updates'],
             'start_elapsed_seconds':cp['metadata']['state']['elapsed_seconds'],
             'baseline_preserved_files':baseline,'bound_training_files':expected,'prepared_at':stamp(),
             'source_result':str(SOURCE_RESULT),'parent_continuation':info.get('continuation'),
             'restart_approval':RESTART_APPROVAL}
    for output in [PREFLIGHT,RUN]:
        output.mkdir();shutil.copy2(SOURCE/'latest.npz',output/'latest.npz')
        metadata=json.loads(json.dumps(info));metadata.update(plan=plan,pid=None,continuation=lineage,
            result_kind='ACTUAL_CPU_C_CONTINUATION_RESULT',
            scope='Exact full-state continuation of the same CPU4 H128/V35328/B512 run. Elapsed seconds are cumulative active trainer time and exclude the offline pause. Same evaluation targets and prompts; no vocabulary/model-quality optimum claim.')
        write(output/'run.json',metadata);write(output/'run-plan.json',plan);write(output/'lineage.json',lineage)
    shutil.copy2(SOURCE/'metrics.jsonl',RUN/'metrics.jsonl')
    command=[sys.executable,'-u',str(BASE/'run.py'),'--plan',str(PLAN),'--output',str(PREFLIGHT),'--resume','--max-updates',str(lineage['start_updates'])]
    with (WIKI/f'reports/{TAG}-preflight.log').open('xb') as log:
        checked=subprocess.run(command,cwd=WIKI,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if checked.returncode:raise RuntimeError('Full-state restore preflight failed; inspect its log')
    restored=inspect_checkpoint(PREFLIGHT/'latest.npz');summary=read(PREFLIGHT/'summary.json')
    import numpy as np
    arrays={}
    with np.load(SOURCE/'latest.npz',allow_pickle=False) as a,np.load(PREFLIGHT/'latest.npz',allow_pickle=False) as b:
        for key in ['parameters','moment1','moment2','visited']:arrays[key]=bool(np.array_equal(a[key],b[key]))
    before=cp['metadata'];after=restored['metadata']
    evaluations=metrics(PREFLIGHT)
    resumed=next((m for m in evaluations if m['label']=='resumed'),None)
    reference,source_evaluated=evaluation_reference(metrics(SOURCE),before,resumed)
    old_validation=reference['validation'];new_validation=after['state']['last_validation']
    val_equal={k:new_validation[k]==v for k,v in old_validation.items() if k!='evaluation_seconds'}
    checks={'zero_optimizer_updates':after['updates']==before['updates']==lineage['start_updates'],
            'healthy_step_limited_preflight':summary['reason']=='step-limited-test',
            'real_resumed_evaluation_recorded':resumed is not None and resumed['consumed']==before['consumed'] and resumed['evaluation_window_sha256']==info['evaluation_window_sha256'],
            'zero_new_training_targets':after['consumed']==before['consumed'],
            'sampler_bit_exact':after['sampler']==before['sampler'],
            'all_arrays_bit_exact':all(arrays.values()),'same_heldout_metrics':all(val_equal.values()),
            'same_seeded_generations':after['state']['last_generation']==reference['generation'],
            'finite_checkpoint':restored['finite_arrays']}
    if not all(checks.values()):raise ValueError('Exact real-model resume check failed: '+str(checks))
    verify_bindings(baseline);verify_bindings(expected)
    proof={'at':stamp(),'passed':True,'checks':checks,'array_checks':arrays,'validation_checks':val_equal,
           'preflight_exit_code':checked.returncode,'source_checkpoint':cp['identity'],
           'consumed':before['consumed'],'updates':before['updates'],'evaluation':new_validation,
           'source_checkpoint_had_matching_evaluation':source_evaluated,
           'comparison_reference':'source checkpoint evaluation' if source_evaluated else 'first fresh zero-update resumed evaluation; source score was stale',
           'scope':'Full actual model restored and evaluated; no training update or additional target consumed.'}
    write(WIKI/f'reports/{TAG}-resume-verification.json',proof)
    write(CONTROL,{'status':'prepared','run_dir':str(RUN),'control_path':str(CONTROL),'plan':str(PLAN),
          'execution_started':False,'pid':None,'supervisor_pid':None,'memory_schedule_id':None,
          'stop_training_by':plan['stop_training_by'],'exit_by':plan['exit_by'],'target_epochs':5,
          'milestones':[{'epoch':e,'status':'pending'} for e in MILESTONES],
          'resume_verification':str(WIKI/f'reports/{TAG}-resume-verification.json'),'lineage':lineage,
          'hourly_log':str(WIKI/f'reports/{TAG}-hourly.jsonl'),
          'log':str(WIKI/f'reports/{TAG}-training.log')})
    print(json.dumps({'status':'prepared','control':str(CONTROL),'preflight_passed':True,
                      'start_epochs':before['consumed']/old['last_recorded_progress']['windows'],'deadline':plan['stop_training_by']}))


def metrics(run):
    path=run/'metrics.jsonl'
    if not path.exists():return []
    lines=path.read_text().splitlines();out=[]
    for index,line in enumerate(lines):
        try:out.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
    return out


def preserve_milestones(control):
    from archive_checkpoint import metadata
    run=Path(control['run_dir']);info=read(run/'run.json');seen=metrics(run);jobs=control['milestones']
    for job in jobs:
        if job['status']=='archived':continue
        epoch=job['epoch'];wanted=next((m for m in seen if m['label']==f'epoch_{epoch}'),None)
        if wanted is None:continue
        if wanted['consumed']!=epoch*read(run/'progress.json')['windows']:raise ValueError('Milestone count differs')
        output=run/f'checkpoint-epoch-{epoch}.npz';sidecar=output.with_suffix('.json')
        if not output.exists() and not sidecar.exists():
            command=[sys.executable,str(BASE/'archive_checkpoint.py'),'--run',str(run),'--metric-label',f'epoch_{epoch}','--output',str(output)]
            p=subprocess.run(command,capture_output=True,text=True,timeout=120)
            if p.returncode:
                job['attempts']=job.get('attempts',0)+1;job['last_error']=(p.stdout+p.stderr)[-2000:]
                if job['attempts']>=5:raise RuntimeError('Cannot preserve required milestone: '+str(job))
                continue
        if not output.exists() or not sidecar.exists():raise ValueError('Incomplete archive; refusing overwrite')
        s=read(sidecar)
        with output.open('rb') as stream:m=metadata(stream)
        if (identity(output)!={'bytes':s['bytes'],'sha256':s['sha256']} or m['identity']!=info['identity']
            or m['config']!=info['config'] or m['consumed']!=wanted['consumed'] or m['updates']!=wanted['updates']
            or m['state']['last_validation']!=wanted['validation'] or m['state']['last_generation']!=wanted['generation']):
            raise ValueError('Archived checkpoint/evaluation mismatch')
        job.update(status='archived',path=str(output),identity=identity(output),
                   consumed=wanted['consumed'],updates=wanted['updates'],at=wanted['at'],
                   cumulative_active_seconds=wanted['actual_elapsed_seconds'],validation=wanted['validation'])
    return jobs


def observe(control):
    run=Path(control['run_dir']);p=read(run/'progress.json') if (run/'progress.json').exists() else {}
    proc=process_snapshot(control['pid'],run) if control.get('pid') else {'pid':None,'process_identity_confirmed':False,'process_rss_bytes':None,'cpu_percent':None}
    base=control['lineage'];n=p.get('windows');consumed=p.get('consumed')
    root_base=base
    while root_base.get('parent_continuation') is not None:root_base=root_base['parent_continuation']
    row={'kind':'MEASUREMENT_NOT_DELIVERY_RECEIPT','at':stamp(),'run_dir':str(run),**proc,
         'control_status':control['status'],'source_progress_at':p.get('at'),'recorded_status':p.get('status'),
         'consumed':consumed,'windows':n,'target_epochs':5,'epochs_equivalent':p.get('epochs_equivalent'),
         'updates':p.get('updates'),'new_updates':p['updates']-base['start_updates'] if 'updates' in p else None,
         'target_progress_percent':100*consumed/(5*n) if n else None,
         'continuation_progress_percent':100*(consumed-base['start_consumed'])/(5*n-base['start_consumed']) if n else None,
         'whole_continuation_progress_percent':100*(consumed-root_base['start_consumed'])/(5*n-root_base['start_consumed']) if n else None,
         'whole_continuation_new_targets':consumed-root_base['start_consumed'] if consumed is not None else None,
         'whole_continuation_start_epochs':root_base['start_consumed']/n if n else None,
         'peak_rss_bytes':p.get('peak_rss_bytes'),'cumulative_active_seconds':p.get('elapsed_seconds'),
         'resume_wall_elapsed_seconds':(datetime.now(timezone.utc)-datetime.fromisoformat(control['started_at'])).total_seconds() if control.get('started_at') else None,
         'continuation_active_seconds':p['elapsed_seconds']-base['start_elapsed_seconds'] if 'elapsed_seconds' in p else None,
         'validation':p.get('last_validation'),'guard':p.get('guard'),'milestones':control['milestones']}
    seen=metrics(run);row['validation_at']=seen[-1]['at'] if seen else None
    return row


def comparison():
    from summarize import assert_not_running,inspect_checkpoint
    assert_not_running(RUN,CONTROL);control=read(CONTROL);result=read(WIKI/f'reports/{TAG}-result.json');info=read(RUN/'run.json')
    jobs=preserve_milestones(control);rows=[];audits=[]
    for epoch in MILESTONES:
        job=next(j for j in jobs if j['epoch']==epoch)
        point=next((p for p in result['validation_curve'] if p['label']==f'epoch_{epoch}'),None)
        if job['status']!='archived' or point is None:
            rows.append({'total_epochs':epoch,'status':'not_reached','actual_result':None});continue
        cp=inspect_checkpoint(job['path']);m=cp['metadata']
        if (cp['identity']!=job['identity'] or m['identity']!=info['identity'] or m['config']!=info['config']
            or m['consumed']!=point['target_predictions'] or m['updates']!=point['updates']
            or m['state']['last_validation']!=point['validation'] or m['state']['last_generation']!=point['generation']):
            raise ValueError('Independent full milestone audit failed')
        rows.append({'total_epochs':epoch,'status':'actual_reached_and_audited','actual_result':point,
                     'additional_active_seconds':point['actual_elapsed_seconds']-control['lineage']['start_elapsed_seconds'],
                     'checkpoint':{'path':cp['path'],'identity':cp['identity']}})
        audits.append({'epoch':epoch,'finite_arrays':cp['finite_arrays'],'identity':cp['identity']})
    preserved=verify_bindings(control['lineage']['baseline_preserved_files']);verify_bindings(control['lineage']['bound_training_files'])
    report={'kind':'ACTUAL_SAME_RUN_CUMULATIVE_2_3_5_COMPARISON','at':stamp(),'rows':rows,
            'milestone_audits':audits,'original_night_files_preserved':preserved,
            'all_requested_epochs_reached':all(row['status']=='actual_reached_and_audited' for row in rows),
            'scope':info['scope'],'evaluation_window_sha256':info['evaluation_window_sha256']}
    output=WIKI/f'reports/{TAG}-comparison.json'
    if output.exists():raise FileExistsError('Preserve existing comparison')
    write(output,report)
    text=['동일 실행 누적 2·3·5에폭 실제 비교','에폭\t상태\t추가실행초\t누적활성초\tLoss\tPPL\t공백/EOS제외Loss\t제외PPL']
    for row in rows:
        p=row['actual_result']
        if p:
            v=p['validation'];text.append(f"{row['total_epochs']}\t{row['status']}\t{row['additional_active_seconds']:.3f}\t{p['actual_elapsed_seconds']:.3f}\t{v['loss']:.6f}\t{v['perplexity']:.6f}\t{v['content_loss']:.6f}\t{v['content_perplexity']:.6f}")
        else:text.append(f"{row['total_epochs']}\t미도달\t-\t-\t-\t-\t-\t-")
    output.with_suffix('.txt').write_text('\n'.join(text)+'\n')
    return report


def finalize():
    control=read(CONTROL)
    if control.get('exit_code') is None or process_snapshot(control['pid'],RUN)['process_identity_confirmed']:
        raise RuntimeError('Cannot finalize a live or unconfirmed trainer')
    output=WIKI/f'reports/{TAG}-result.json'
    if not output.exists():
        subprocess.run([sys.executable,str(BASE/'summarize.py'),'--control',str(CONTROL),'--run',str(RUN),'--output',str(output)],cwd=WIKI,check=True,timeout=180)
    decoded=WIKI/f'reports/{TAG}-generation-examples.json'
    if not decoded.exists():
        subprocess.run([str(WIKI/'.venv-kiwi/bin/python'),str(BASE/'decode_generations.py'),'--control',str(CONTROL),'--result',str(output),'--output',str(decoded)],cwd=WIKI,check=True,timeout=180)
    report_path=WIKI/f'reports/{TAG}-comparison.json'
    report=read(report_path) if report_path.exists() else comparison()
    patch_control(status='training-complete-review-pending' if report['all_requested_epochs_reached'] else 'stopped-review-pending',
                  finalized_files=[str(output),str(decoded),str(report_path)],finalization_completed_at=stamp(),
                  note='Actual numerical artifacts ready; native loss graph publication, user report and only this monitor cleanup still required.')


def supervise():
    with (WIKI/f'reports/.{TAG}-launch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=read(CONTROL)
        if c['status']!='prepared' or c.get('execution_started'):raise RuntimeError('Already started; never duplicate')
        assert_no_trainer();verify_bindings(c['lineage']['baseline_preserved_files']);verify_bindings(c['lineage']['bound_training_files'])
        if not read(c['resume_verification'])['passed']:raise ValueError('Resume preflight missing')
        proof=read(c['resume_verification'])
        if not all(proof['checks'].values()) or not proof['checks'].get('real_resumed_evaluation_recorded'):
            raise ValueError('Actual healthy full-state evaluation preflight missing')
        if datetime.now(timezone.utc)>=datetime.fromisoformat(c['stop_training_by']):raise ValueError('Fixed deadline passed')
        command=[sys.executable,'-u',str(BASE/'run.py'),'--plan',str(PLAN),'--output',str(RUN),'--resume']
        with Path(c['log']).open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=WIKI,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
        patch_control(status='launched',execution_started=True,pid=child.pid,supervisor_pid=os.getpid(),command=command,started_at=stamp())
        info=read(RUN/'run.json');info['pid']=child.pid;write(RUN/'run.json',info)
        try:
            assertion=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except OSError:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            patch_control(status='failed',exit_code=child.returncode,ended_at=stamp(),error='Sleep assertion failed; owned child stopped');raise
        patch_control(caffeinate_pid=assertion.pid)
        observation_errors=[]
        while child.poll() is None:
            c=read(CONTROL)
            try:
                row=observe(c)
                with (RUN/'resource-observations.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                jobs=preserve_milestones(c);patch_control(milestones=jobs,last_supervisor_observation_at=stamp())
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired) as error:
                observation_errors.append({'at':stamp(),'error':f'{type(error).__name__}: {error}'})
                patch_control(observation_errors=observation_errors[-10:])
            except RuntimeError as error:
                (RUN/'STOP').write_text('Required archive preservation failed: '+str(error)+'\n')
                patch_control(artifact_stop_reason=str(error))
            time.sleep(20)
        result=child.wait()
        try:assertion.wait(timeout=5)
        except subprocess.TimeoutExpired:assertion.terminate();assertion.wait(timeout=5)
        summary=read(RUN/'summary.json') if (RUN/'summary.json').exists() else {}
        patch_control(status='trainer-ended',exit_code=result,ended_at=stamp(),reason=summary.get('reason'))
        try:
            jobs=preserve_milestones(read(CONTROL));patch_control(milestones=jobs)
            if result in (86,87,88) or summary.get('reason') in ['thermal_critical','thermal_serious','monitor_error','low_battery']:
                patch_control(status='protective-stop-review-pending',restart_confirmation_required=True,
                              note='No automatic restart or heavy final processing after protective stop. Preserve latest complete checkpoint.')
            else:finalize()
        except Exception as error:
            patch_control(status='finalization-error',finalization_error=f'{type(error).__name__}: {error}')
            raise
        print(json.dumps({'control':str(CONTROL),'status':read(CONTROL)['status'],'trainer_exit':result}),flush=True)


def status(record=False):
    c=read(CONTROL);row=observe(c)
    for name,command in [('memory_pressure',['/usr/bin/memory_pressure','-Q']),('swap',['/usr/sbin/sysctl','vm.swapusage'])]:
        p=subprocess.run(command,capture_output=True,text=True,timeout=10);row[name]={'exit_code':p.returncode,'output':(p.stdout+p.stderr).strip()}
    if record:
        with Path(c['hourly_log']).open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False,indent=2))


def wait_for_end():
    with (RUN/'.completion-observer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            c=read(CONTROL)
            if c.get('exit_code') is not None and not process_snapshot(c['pid'],RUN)['process_identity_confirmed']:
                if c['status'] in ['training-complete-review-pending','stopped-review-pending','protective-stop-review-pending','finalization-error']:
                    print(json.dumps({'status':c['status'],'control':str(CONTROL),'exit_code':c['exit_code'],
                        'next_action':'Inspect actual results; publish real loss curve with visualize and native preview/render, compare cumulative 2/3/5 and saved generations, report then delete only this continuation hourly monitor. Never restart old training.'}),flush=True);return
            time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','supervise','status','finalize','wait'])
    p.add_argument('--record',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='supervise':supervise()
    elif a.action=='status':status(a.record)
    elif a.action=='finalize':finalize()
    else:wait_for_end()


if __name__=='__main__':main()
