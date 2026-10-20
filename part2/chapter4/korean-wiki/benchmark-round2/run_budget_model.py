"""One explicit model-capacity experiment for the 6-8 hour budget.

Compare H128/H64 at B512, six requested threads, same full vocabulary/data.
Different parameter counts and initial arrays: never a same-model speedup.
"""

from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys

WIKI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WIKI / "gpu-benchmark"))
from run_mps_comparison import capture, parse_json, identity, atomic_json, require

OUTPUT = WIKI / "reports/nplm-budget-model-20261018-v1.json"
BINARY = WIKI / "cpu-benchmark/experiments/cpu_fixedwork_combined"
TARGETS, EVENTS, BATCH, STEPS, THREADS = 124232535, 65536, 512, 128, 6


def main():
    require(not OUTPUT.exists(), "Preserve earlier budget experiment")
    source = {str(p): identity(p) for p in (Path(__file__), BINARY,
               WIKI / "cpu-benchmark/experiments/fixedwork_benchmark.c")}
    report = {"kind": "NPLM_CAPACITY_CHANGE_SHORT_TIMING", "status": "running",
              "started_at": datetime.now().astimezone().isoformat(), "source_before": source,
              "planned_order": [[128,64],[64,128],[128,64]], "trials": [], "summary": [],
              "policy": {"changed_condition": "Hidden width 128 vs 64; not the same model or initialization arrays. Same V35328/C8/D64/B512/FP32/dense AdamW/clip1/real Wiki target stream.",
                         "quality": "No convergence or 2/3/5 epoch loss measured. H64 may underfit or yield worse held-out loss; smaller architecture must be evaluated.",
                         "numeric_scope": "Unchanged generic C engine, existing tiny/irregular independent checks, B512 full-V softmax unit coverage and full-state finiteness. No new independent full-model H64/B512 gate.",
                         "timing_scope": "Short real data throughput only; full epoch times extrapolated. Setup/evaluation/checkpoint/long thermal excluded."}}
    logs = OUTPUT.parent / (OUTPUT.stem + "-logs")
    require(not logs.exists(), "Preserve logs")
    logs.mkdir()
    atomic_json(OUTPUT, report)
    sampling = None
    env = dict(os.environ, OMP_NUM_THREADS="6", VECLIB_MAXIMUM_THREADS="6", OPENBLAS_NUM_THREADS="6",
               MKL_NUM_THREADS="6", OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
    env.pop("OMP_THREAD_LIMIT",None)
    with (WIKI / "cpu-benchmark/.timing.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for repeat, order in enumerate(report['planned_order'],1):
            for hidden in order:
                label=f"repeat-{repeat}-hidden-{hidden}"
                command=[str(BINARY),'--corpus',str(WIKI/'training-corpora/wiki-k35328'),
                         '--steps',str(STEPS),'--batch-size',str(BATCH),'--hidden-dim',str(hidden),
                         '--threads',str(THREADS),'--warmup-steps','2','--seed','42','--clip-norm','1',
                         '--softmax-parallel','--reduction-parallel','--adam-parallel']
                print(label,flush=True)
                raw=capture(['/usr/bin/time','-l',*command],timeout=120,env=env,cwd=WIKI)
                (logs/(label+'.stdout.json')).write_text(raw['stdout'])
                (logs/(label+'.stderr.txt')).write_text(raw['stderr'])
                row={'repeat':repeat,'hidden_dim':hidden,'command':command,'passed':False,
                     'exit_code':raw['exit_code'],'stdout_log':str(logs/(label+'.stdout.json')),
                     'stderr_log':str(logs/(label+'.stderr.txt'))}
                report['trials'].append(row)
                try:
                    require(raw['exit_code']==0 and not raw['timed_out'],'Timing child failed')
                    data=parse_json(raw['stdout'])
                    for key,value in {'vocab_size':35328,'context_size':8,'embedding_dim':64,'hidden_dim':hidden,
                                      'batch_size':BATCH,'threads':THREADS,'seed':42,'warmup_steps':2,'steps':STEPS}.items():
                        require(data['config'][key]==value,f'Configuration changed: {key}')
                    expected=35328*64+hidden*(8*64)+hidden+35328*hidden+35328
                    require(data['parameter_count']==expected,'Parameter count mismatch')
                    require(data['steps']==STEPS and data['target_prediction_events']==EVENTS,'Completed work mismatch')
                    require(data['finite_flag'] and data['warmup_reset'] and not data['fast_math_build'],'State/FP32 audit failed')
                    require(data['corpus']['target_windows']==TARGETS,'Corpus changed')
                    require(math.isclose(data['targets_per_second'],EVENTS/data['elapsed_seconds'],rel_tol=1e-12),'Rate mismatch')
                    actual_sampling={key:data[key] for key in ('input_checksum','final_input_checksum','sampler_initial','sampler_final')}
                    if sampling is None: sampling=actual_sampling
                    require(sampling==actual_sampling,'Actual target stream changed across architectures')
                    row.update(passed=True,measurement=data,outside_10_30_seconds=not 10<=data['elapsed_seconds']<=30)
                    print(f"  {data['elapsed_seconds']:.6f}s / {data['targets_per_second']:.2f} targets/s",flush=True)
                except Exception as error:
                    row['error']=f'{type(error).__name__}: {error}'; print(row['error'],flush=True)
                atomic_json(OUTPUT,report)
        for hidden in (128,64):
            rows=[r['measurement'] for r in report['trials'] if r['hidden_dim']==hidden and r['passed']]
            summary={'hidden_dim':hidden,'passed':len(rows)==3,'successful_repeats':len(rows)}
            if len(rows)==3:
                rates=[r['targets_per_second'] for r in rows]; rate=statistics.median(rates)
                summary.update(actual_seconds=[r['elapsed_seconds'] for r in rows],median_targets_per_second=rate,
                               parameter_count=rows[0]['parameter_count'],
                               epoch_hours={str(e):e*TARGETS/rate/3600 for e in (1,2,3,5)},
                               epoch2_observed_range=[2*TARGETS/max(rates)/3600,2*TARGETS/min(rates)/3600],
                               initial_checksums=[r['initial_parameter_checksum'] for r in rows])
                require(len(set(summary['initial_checksums']))==1,'Same architecture initialization not repeatable')
            report['summary'].append(summary)
        report['source_after']={p:identity(Path(p)) for p in source}
        require(report['source_before']==report['source_after'],'Source changed during timing')
        report.update(status='complete' if all(r['passed'] for r in report['summary']) else 'complete_with_failures',
                      completed_at=datetime.now().astimezone().isoformat(),shared_actual_sampling=sampling)
        atomic_json(OUTPUT,report)
    print(json.dumps(report['summary'],indent=2),flush=True)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
