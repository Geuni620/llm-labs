"""Actual CPU/C NPLM training, atomic resume, held-out metrics and local guards.

Never starts the previous comparison queue. Absolute morning deadlines include
preparation/evaluation. This process does not import Torch or Kiwi/OpenMP runtimes.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time

import numpy as np

BASE=Path(__file__).resolve().parent;WIKI=BASE.parent
sys.path.insert(0,str(WIKI))
from npy_corpus import NpyArrayCorpus
from engine import Engine,LIBRARY,file_identity
import checkpoint
from thermal_guard import NightGuard

CONFIG_KEYS=('vocab_size','context_size','embedding_dim','hidden_dim','batch_size','cpu_threads','seed')
PROMPTS=[('서울은',['서울/NNP','은/JX','<sp>/WS']),('대한민국의',['대한민국/NNP','의/JKG','<sp>/WS']),
         ('역사는',['역사/NNG','는/JX','<sp>/WS']),('그는',['그/NP','는/JX','<sp>/WS']),
         ('2025년',['2/SN','0/SN','2/SN','5/SN','년/NNB','<sp>/WS']),
         ('2037년',['2/SN','0/SN','3/SN','7/SN','년/NNB','<sp>/WS'])]

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

def now_iso():return datetime.now(timezone.utc).astimezone().isoformat()

def corpus_identity(corpus):
    manifest=json.loads((corpus/'manifest.json').read_text())
    export=json.loads((WIKI/'reports/integer-encoding-npy.json').read_text())
    if manifest['status']!='complete' or export['status']!='complete':raise ValueError('Incomplete corpus/export')
    actual_manifest=file_identity(corpus/'manifest.json')
    if actual_manifest!={k:export['source_manifest'][k] for k in ('bytes','sha256')}:raise ValueError('Manifest binding changed')
    identities={'manifest.json':actual_manifest,'vocabulary.json':file_identity(corpus/'vocabulary.json')}
    if identities['vocabulary.json']!=manifest['outputs']['vocabulary.json']:raise ValueError('Vocabulary changed')
    for split in ('train','validation'):
        entry=next(x for x in export['arrays'] if x['split']==split)
        for suffix,expected in [('ids.npy',entry['output']),('paragraphs.npy',manifest['outputs'][f'{split}.paragraphs.npy'])]:
            path=corpus/f'{split}.{suffix}';actual=file_identity(path)
            if actual!=expected:raise ValueError('Array binding changed '+path.name)
            identities[path.name]=actual
    return identities,manifest

def evaluate(engine,validation,windows,space_id,should_stop):
    totals=np.zeros(6,dtype=np.float64);begun=time.monotonic()
    for start in range(0,len(windows),engine.config['batch_size']):
        if should_stop():return None
        x,y,_=validation.batch(windows[start:start+engine.config['batch_size']])
        totals+=engine.evaluate(x,y,space_id)
    if not np.isfinite(totals).all():raise FloatingPointError('Invalid validation totals')
    result={}
    for prefix,offset in [('',0),('content_',3)]:
        count,nll,correct=totals[offset:offset+3]
        loss=float(nll/count) if count else None
        result.update({prefix+'targets':int(count),prefix+'negative_log_likelihood_sum':float(nll),
                       prefix+'loss':loss,prefix+'perplexity':math.exp(loss) if loss is not None and loss<700 else None,
                       prefix+'accuracy':float(correct/count) if count else None})
    result['evaluation_seconds']=time.monotonic()-begun
    return result

def generate(engine,tokens,seed,should_stop):
    lookup={token:i for i,token in enumerate(tokens)};results=[]
    for index,(label,prompt) in enumerate(PROMPTS):
        if should_stop():break
        rng=np.random.default_rng(seed+1000+index)
        context=[2]*engine.config['context_size']+[lookup.get(t,1) for t in prompt];generated=[]
        for _ in range(64):
            if should_stop():return results
            logits=engine.logits(np.asarray([context[-engine.config['context_size']:]],dtype=np.uint32))[0].astype(np.float64)
            if not np.isfinite(logits).all():raise FloatingPointError('Invalid generation logits')
            logits[:3]=-np.inf
            order=np.argsort(-logits,kind='stable')[:40];values=logits[order]/.8
            probabilities=np.exp(values-values.max());probabilities/=probabilities.sum()
            token=int(rng.choice(order,p=probabilities))
            if token==3:break
            generated.append(tokens[token]);context.append(token)
        trigrams=list(zip(generated,generated[1:],generated[2:]))
        results.append({'prompt':label,'prompt_tokens':prompt,'prompt_oov_tokens':[x for x in prompt if x not in lookup],
                        'generated_tokens':generated,'seed':seed+1000+index,'temperature':.8,'top_k':40,
                        'repeated_trigram_fraction':1-len(set(trigrams))/len(trigrams) if trigrams else 0,
                        'surface_restoration':'Deferred to a separate post-run Kiwi process; raw form/POS keys retained.'})
    return results

def run(args):
    plan=json.loads(args.plan.read_text());training=plan['training']
    if not plan.get('physical_check_confirmed'):raise ValueError('Physical placement/check confirmation missing')
    if training['device']!='cpu':raise ValueError('This run is CPU only')
    config={k:training[k] for k in ('vocab_size','context_size','embedding_dim','hidden_dim','batch_size','seed')}
    config['cpu_threads']=training['night_cpu_threads']
    if config!={**{k:training[k] for k in ('vocab_size','context_size','embedding_dim','hidden_dim','batch_size','seed')},'cpu_threads':4}:
        raise ValueError('Tonight is the explicitly planned four-thread configuration')
    if (training['learning_rate']!=.001 or training['weight_decay']!=.01 or training['gradient_clip_norm']!=1
        or training['betas']!=[.9,.999] or training['epsilon']!=1e-8):raise ValueError('C optimizer settings differ')
    training_stop=datetime.fromisoformat(plan['stop_training_by']);hard_stop=datetime.fromisoformat(plan['exit_by'])
    if datetime.now(timezone.utc)>=training_stop:raise ValueError('Training deadline already passed')
    output=args.output.resolve();corpus=(WIKI/training['corpus']).resolve()
    if output.exists() and not args.resume:raise FileExistsError('Preserving existing run '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    log_lock=threading.RLock()
    def emit(event):
        with log_lock:
            with (output/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'at':now_iso(),**event},ensure_ascii=False,allow_nan=False)+'\n')
    requested={'reason':None}
    def on_signal(signum,_frame):requested['reason']='signal_'+str(signum)
    signal.signal(signal.SIGINT,on_signal);signal.signal(signal.SIGTERM,on_signal)
    engine=None;guard=None
    with ExitStack() as stack:
        for lock_path in [WIKI/'cpu-benchmark/.timing.lock',WIKI/'reports/.nplm-benchmark.lock']:
            lock=stack.enter_context(lock_path.open('a+'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard=NightGuard(training_stop,hard_stop,emit=emit).start()
        if not guard.wait_ready(10):
            atomic_json(output/'summary.json',{'status':'preflight-stopped','guard':guard.metadata(),'at':now_iso()})
            guard.mark_stopped();guard.close();raise RuntimeError('Thermal/power monitor not ready')
        state={'status':'preparing','started_at':now_iso(),'elapsed_seconds':0.0,'update_seconds':0.0,
               'evaluation_seconds':0.0,'checkpoint_seconds':0.0,'last_validation':None,'last_evaluation_at':0.0,
               'last_checkpoint_at':0.0,'completed_epoch_snapshots':[],'last_generation':[]}
        begun=time.monotonic();elapsed_before=0.0
        def elapsed():return elapsed_before+time.monotonic()-begun
        def stop_auxiliary():
            return (requested['reason'] is not None or (output/'STOP').exists()
                    or (guard.should_stop and guard.reason!='training_deadline')
                    or datetime.now(timezone.utc)>=hard_stop or (hard_stop-datetime.now(timezone.utc)).total_seconds()<30)
        try:
            data_identity,manifest=corpus_identity(corpus)
            vocabulary=json.loads((corpus/'vocabulary.json').read_text())['tokens']
            if len(vocabulary)!=config['vocab_size']:raise ValueError('Vocabulary size changed')
            identity={'data':data_identity,'library':file_identity(LIBRARY),
                      'scripts':{name:file_identity(BASE/name) for name in ['engine.c','engine.py','checkpoint.py','thermal_guard.py','run.py']}}
            engine=Engine(corpus,config)
            if engine.info['windows']!=manifest['statistics']['train_target_windows']:raise ValueError('Target count mismatch')
            validation=NpyArrayCorpus(corpus,'validation',config['context_size'],verify=True)
            rng=np.random.default_rng(config['seed']+7000)
            windows=rng.choice(validation.windows,min(training['validation_windows'],validation.windows),replace=False)
            eval_hash=hashlib.sha256(windows.astype('<i8').tobytes()).hexdigest()
            metadata={'kind':'ACTUAL_CPU_C_NPLM_TRAINING','config':config,'plan':plan,'identity':identity,
                      'evaluation_window_sha256':eval_hash,'evaluation_windows':len(windows),
                      'initial_parameter_sha256':hashlib.sha256(engine.arrays['parameters'].tobytes()).hexdigest(),
                      'corpus':str(corpus),'parameters':engine.info['parameters'],'pid':os.getpid(),
                      'math':'FP32 full-softmax NPLM with dense AdamW; no sampled softmax or Word2Vec',
                      'test_split_used':False,'timing':'Actual elapsed time including setup, evaluation and checkpoint I/O; absolute morning deadlines.'}
            if args.resume:
                old=json.loads((output/'run.json').read_text())
                if old['identity']!=identity or old['config']!=config or old['evaluation_window_sha256']!=eval_hash:
                    raise ValueError('Resume identities differ')
                state,path,rejected=checkpoint.restore_latest(output,engine,identity)
                state.pop('stop_reason',None)
                elapsed_before=state['elapsed_seconds'];begun=time.monotonic()
                emit({'kind':'resumed','updates':engine.info['updates'],'checkpoint':str(path),'rejected':rejected})
            else:
                atomic_json(output/'run.json',metadata);atomic_json(output/'run-plan.json',plan)
            recent=deque(maxlen=100);last_progress=0.0
            def progress():
                info=engine.info
                payload={**state,'pid':os.getpid(),'at':now_iso(),'elapsed_seconds':elapsed(),**info,
                         'epochs_equivalent':info['consumed']/info['windows'],'config':config,
                         'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'guard':guard.metadata(),
                         'recent_training_loss':sum(a*b for a,b in recent)/sum(b for _,b in recent) if recent else None}
                atomic_json(output/'progress.json',payload);return payload
            def save(label):
                start=time.monotonic();state['elapsed_seconds']=elapsed()
                checkpoint.save(output/'latest.npz',engine,identity,state)
                state['checkpoint_seconds']+=time.monotonic()-start;state['last_checkpoint_at']=elapsed()
                emit({'kind':'checkpoint','label':label,'updates':engine.info['updates'],'consumed':engine.info['consumed']})
            def measure(label,with_generation=True):
                if stop_auxiliary():return False
                start=time.monotonic();space=vocabulary.index('<sp>/WS') if '<sp>/WS' in vocabulary else -1
                result=evaluate(engine,validation,windows,space,stop_auxiliary)
                if result is None:emit({'kind':'evaluation_interrupted','label':label});return False
                generation=generate(engine,vocabulary,config['seed'],stop_auxiliary) if with_generation else []
                state['last_validation']=result;state['last_evaluation_at']=elapsed();state['last_generation']=generation
                state['evaluation_seconds']+=time.monotonic()-start
                metric={'kind':'ACTUAL_CPU_C_CHECKPOINT','label':label,'at':now_iso(),'actual_elapsed_seconds':elapsed(),
                        **engine.info,'epochs_equivalent':engine.info['consumed']/engine.info['windows'],
                        'recent_training_loss':sum(a*b for a,b in recent)/sum(b for _,b in recent) if recent else None,
                        'validation':result,'generation':generation,'evaluation_window_sha256':eval_hash,
                        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'guard':guard.metadata()}
                with (output/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(metric,ensure_ascii=False,allow_nan=False)+'\n')
                print(json.dumps({k:metric[k] for k in ['label','at','actual_elapsed_seconds','updates','epochs_equivalent','validation']},ensure_ascii=False),flush=True)
                return True
            state['status']='initial-evaluation';progress()
            measure('resumed' if args.resume else 'initial');save('resumed' if args.resume else 'initial')
            state['status']='pilot-running'
            limit=training['target_epochs']*engine.info['windows']
            initial_consumed=engine.info['consumed'];pilot_started=time.monotonic();pilot_reported=False
            while engine.info['consumed']<limit:
                if requested['reason'] or (output/'STOP').exists():state['stop_reason']=requested['reason'] or 'STOP_file';break
                if guard.should_stop:state['stop_reason']=guard.reason;break
                if args.max_updates is not None and engine.info['updates']>=args.max_updates:
                    state['stop_reason']='step-limited-test';break
                start=time.monotonic();result=engine.step(limit);state['update_seconds']+=time.monotonic()-start
                if result is None:break
                recent.append((result['loss'],int(result['batch_size'])))
                if not math.isfinite(result['loss']):raise FloatingPointError('Nonfinite training loss')
                at=elapsed();info=engine.info
                if not pilot_reported and time.monotonic()-pilot_started>=args.pilot_seconds:
                    pilot_reported=True;state['status']='training-running'
                    pilot={'at':now_iso(),'actual_seconds':time.monotonic()-pilot_started,
                           'target_predictions':info['consumed']-initial_consumed,'guard':guard.metadata(),
                           'scope':'Observed four-thread process/thermal progress; desktop UI latency not directly measured.'}
                    atomic_json(output/'pilot.json',pilot);emit({'kind':'pilot_complete',**pilot});save('pilot')
                epoch=info['consumed']//info['windows']
                if epoch>0 and epoch not in state['completed_epoch_snapshots']:
                    measure('epoch_'+str(epoch));state['completed_epoch_snapshots'].append(epoch);save('epoch_'+str(epoch))
                elif at-state['last_evaluation_at']>=training['validation_interval_seconds']:
                    measure('periodic');save('evaluation')
                elif at-state['last_checkpoint_at']>=training['checkpoint_interval_seconds']:save('periodic')
                if at-last_progress>=30:progress();last_progress=at
            state['status']='complete' if engine.info['consumed']>=limit else 'stopped'
            if state['status']=='complete':state['stop_reason']='target_epochs'
            else:state.setdefault('stop_reason',guard.reason or requested['reason'] or 'stopped')
            if not stop_auxiliary():measure('final')
            if guard.reason!='thermal_critical':save('final')
            state['elapsed_seconds']=elapsed();final=progress()
            atomic_json(output/'summary.json',{'status':state['status'],'reason':state['stop_reason'],
                                             'kind':'ACTUAL_CPU_C_TRAINING_RESULT','final':final,'run':metadata})
            emit({'kind':'finished','status':state['status'],'reason':state['stop_reason']})
            return 0
        except BaseException as error:
            emit({'kind':'failure','error':f'{type(error).__name__}: {error}'})
            atomic_json(output/'failure.json',{'status':'failed','at':now_iso(),'error':f'{type(error).__name__}: {error}',
                                               'guard':guard.metadata(),'checkpoint_policy':'Latest completed checkpoint preserved; no automatic restart.'})
            raise
        finally:
            if engine is not None:engine.close()
            guard.mark_stopped();guard.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--pilot-seconds',type=float,default=1800);p.add_argument('--max-updates',type=int)
    a=p.parse_args()
    if a.pilot_seconds<1 or (a.max_updates is not None and a.max_updates<1):p.error('Positive bounds required')
    return run(a)

if __name__=='__main__':raise SystemExit(main())
