"""Full V35328/B512 H128/H64 four-update independent CPU-oracle gates."""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WIKI=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(WIKI/'gpu-benchmark'))
import verify_mps as oracle
from corpus_fixture import Corpus,checksum_batch,checksum_f32,parameter_layout,create_initial

OUT=WIKI/'reports/nplm-budget-model-verification-20261018-v1.json'
HERE=Path(__file__).resolve().parent

def main():
    if OUT.exists(): raise FileExistsError('Preserve prior verification')
    report={'kind':'FULL_WIKI_B512_CAPACITY_CPU_ORACLE_CHECK','status':'running','passed':False,
            'started_at':datetime.now().astimezone().isoformat(),'cases':[],
            'tolerances_predeclared':oracle.TOLERANCES,
            'scope':'H128/H64 x clip1/.05 x four consecutive updates after two warmups and exact reset. Gradient NaN poison. Existing oracle and tolerances unchanged.'}
    source_paths=[Path(__file__),HERE/'capture_budget.c',WIKI/'cpu-benchmark/experiments/fixedwork_benchmark.c',
                  WIKI/'cpu-benchmark/experiments/vector_vforce.c',WIKI/'cpu-benchmark/experiments/fixedwork_norm.c',
                  WIKI/'cpu-benchmark/experiments/fixedwork_softmax.c',WIKI/'gpu-benchmark/verify_mps.py',
                  WIKI/'gpu-benchmark/corpus_fixture.py']
    report['sources_before']={str(p):oracle.identity(p) for p in source_paths}
    oracle.atomic_json(OUT,report)
    with oracle.numerical_lock():
        manifest=json.loads((WIKI/'cpu-benchmark/experiments/fixedwork-build-manifest.v2.json').read_text())
        binary=HERE/'.artifacts/capture_budget'
        binary.parent.mkdir(exist_ok=True)
        if binary.exists(): raise FileExistsError('Preserve capture executable')
        cmd=[str(HERE/'capture_budget.c') if x==str(WIKI/'cpu-benchmark/experiments/fixedwork_benchmark.c') else
             str(binary) if x==str(WIKI/'cpu-benchmark/experiments/cpu_fixedwork_combined') else x
             for x in manifest['variants']['combined']['command']]
        build=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        report['build']={'command':cmd,'exit_code':build.returncode,'stdout':build.stdout,'stderr':build.stderr}
        if build.returncode: raise RuntimeError('Capture build failed')
        import torch
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        corpus=Corpus(WIKI/'training-corpora/wiki-k35328',8,35328)
        for hidden in (128,64):
            config={'vocab_size':35328,'context_size':8,'embedding_dim':64,'hidden_dim':hidden,
                    'batch_size':512,'threads':6,'seed':42,'warmup_steps':2,'steps':4}
            initial,layout=create_initial(config)
            for clip in (1,.05):
                directory=HERE/'.artifacts'/f'capture-h{hidden}-clip{clip}'
                directory.mkdir()
                env=dict(os.environ,KOWIKI_CAPTURE_DIRECTORY=str(directory),OMP_NUM_THREADS='6',
                         VECLIB_MAXIMUM_THREADS='6',OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0')
                command=[str(binary),'--corpus',str(WIKI/'training-corpora/wiki-k35328'),'--hidden-dim',str(hidden),
                         '--batch-size','512','--threads','6','--steps','4','--warmup-steps','2','--seed','42',
                         '--clip-norm',str(clip),'--softmax-parallel','--reduction-parallel','--adam-parallel']
                process=subprocess.run(command,capture_output=True,text=True,env=env,timeout=120)
                (directory/'stdout.json').write_text(process.stdout)
                (directory/'stderr.txt').write_text(process.stderr)
                if process.returncode: raise RuntimeError('Capture child failed')
                data=json.loads(process.stdout)
                load=lambda name:np.fromfile(directory/(name+'.bin'),dtype='<f4')
                assert load('initial').tobytes()==initial.tobytes()==load('reset_params').tobytes()
                assert np.all(load('reset_m1')==0) and np.all(load('reset_m2')==0)
                assert data['initial_checksum']==checksum_f32(initial)
                reference=oracle.build_reference(torch,config,initial,layout)
                sampler=corpus.sampler(42)
                comparisons=[]
                for step,scalars in enumerate(data['steps'],1):
                    contexts,targets=sampler.next_batch(512)
                    assert np.fromfile(directory/f'step{step}.contexts.bin',dtype='<u4').reshape(512,8).tolist()==contexts.tolist()
                    assert np.fromfile(directory/f'step{step}.targets.bin',dtype='<u4').tolist()==targets.tolist()
                    assert scalars['batch_checksum']==checksum_batch(contexts,targets)
                    expected=oracle.reference_update(torch,reference,contexts,targets,clip)
                    for field in ('loss','gradient_norm','clipping_scale','raw_gradients','clipped_gradients','updated_params','moment1','moment2'):
                        actual=torch.tensor(scalars[field],dtype=torch.float32) if field in scalars else torch.from_numpy(load(f'step{step}.{field}'))
                        comparisons.append(oracle.compare(torch,f'update{step}.{field}',actual,expected[field],field if field in ('moment1','moment2') else 'ordinary'))
                    assert clip!=.05 or scalars['clipping_scale']<1
                case={'hidden_dim':hidden,'clip_norm':clip,'passed':all(r['passed'] for r in comparisons),
                      'comparison_count':len(comparisons),'failed_comparisons':sum(not r['passed'] for r in comparisons),
                      'comparisons':comparisons,'initial_bits_equal':True,'warmup_reset_checked':True,'gradient_poison':True,
                      'actual_Wiki_batches_equal':True,'capture_config':data['config'],
                      'capture_files':{p.name:oracle.identity(p) for p in directory.iterdir()}}
                report['cases'].append(case);oracle.atomic_json(OUT,report)
                print(hidden,clip,case['passed'],case['failed_comparisons'],flush=True)
                # Derived verification arrays are reproducible; preserve hashes, not 2GB of scratch.
                shutil.rmtree(directory)
                del reference,expected,actual
        report['sources_after']={str(p):oracle.identity(p) for p in source_paths}
        assert report['sources_before']==report['sources_after']
        report.update(status='complete',passed=all(r['passed'] for r in report['cases']),
                      comparison_count=sum(r['comparison_count'] for r in report['cases']),
                      failed_comparisons=sum(r['failed_comparisons'] for r in report['cases']),
                      completed_at=datetime.now().astimezone().isoformat())
        oracle.atomic_json(OUT,report)
    return 0 if report['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
