"""Serial full-corpus ABI check against an isolated Torch CPU oracle."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

BASE=Path(__file__).resolve().parent;WIKI=BASE.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(WIKI/'gpu-benchmark'))
from engine import Engine,build,file_identity
from corpus_fixture import Corpus,create_initial
import verify_mps as oracle

def main():
    report_path=WIKI/'reports/night-full-verification-20261022.json'
    if report_path.exists():raise FileExistsError('Preserve prior gate')
    config={'vocab_size':35328,'context_size':8,'embedding_dim':64,'hidden_dim':128,
            'batch_size':512,'cpu_threads':4,'seed':42}
    report={'kind':'NIGHT_ACTUAL_C_ABI_FULL_WIKI_CPU_ORACLE','status':'running','passed':False,
            'config':config,'tolerances_predeclared':oracle.TOLERANCES,'comparisons':[]}
    write=lambda:report_path.write_text(json.dumps(report,indent=2)+'\n')
    write()
    with oracle.numerical_lock():
        build();engine=Engine(WIKI/'training-corpora/wiki-k35328',config)
        try:
            corpus=Corpus(WIKI/'training-corpora/wiki-k35328',8,35328);sampler=corpus.sampler(42)
            with tempfile.TemporaryDirectory(prefix='night-full-',dir=BASE/'.artifacts') as temp:
                directory=Path(temp);(directory/'config.json').write_text(json.dumps(config))
                initial,layout=create_initial(config)
                np.testing.assert_array_equal(initial,engine.arrays['parameters']);np.save(directory/'initial.npy',initial)
                for i in range(1,5):
                    result=engine.step(2*engine.info['windows']);x,y=engine.batch(512)
                    xx,yy=sampler.next_batch(512);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy)
                    np.save(directory/f'x{i}.npy',x);np.save(directory/f'y{i}.npy',y)
                    np.savez(directory/f'actual{i}.npz',loss=result['loss'],gradient_norm=result['gradient_norm'],clipping_scale=result['clipping_scale'],
                             clipped_gradients=engine.arrays['gradients'],updated_params=engine.arrays['parameters'],
                             moment1=engine.arrays['moment1'],moment2=engine.arrays['moment2'])
                proc=subprocess.run([sys.executable,str(BASE/'oracle_worker.py'),str(directory)],capture_output=True,text=True,timeout=180)
                if proc.returncode:raise RuntimeError(proc.stderr)
                for i in range(1,5):
                    with np.load(directory/f'actual{i}.npz') as actual,np.load(directory/f'expected{i}.npz') as expected:
                        for name in actual.files:
                            a=actual[name].astype(np.float64);b=expected[name].astype(np.float64)
                            family=name if name in ('moment1','moment2') else 'ordinary';tol=oracle.TOLERANCES[family]
                            error=np.abs(a-b);bad=error>tol['atol']+tol['rtol']*np.abs(b)
                            row={'step':i,'field':name,'elements':int(a.size),'nonfinite':int((~np.isfinite(a)).sum()),
                                 'max_absolute_error':float(error.max()),'outside_tolerance':int(bad.sum())}
                            row['passed']=row['outside_tolerance']==0 and row['nonfinite']==0
                            report['comparisons'].append(row);write()
                before={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in engine.arrays.items()} if False else None
                with np.load(directory/'evaluation.npz') as expected:
                    stats=engine.evaluate(x,y,4);losses=expected['losses'];content=(y!=4)&(y!=3)
                    report['evaluation']={'all_nll_error':abs(float(stats[1])-float(losses.astype(np.float64).sum())),
                                          'content_nll_error':abs(float(stats[4])-float(losses[content].astype(np.float64).sum())),
                                          'targets':int(stats[0]),'content_targets':int(stats[3])}
                    assert report['evaluation']['all_nll_error']/512<2e-6
                    assert report['evaluation']['content_nll_error']/max(1,int(content.sum()))<2e-6
                report.update(status='complete',passed=all(x['passed'] for x in report['comparisons']),
                              comparison_count=len(report['comparisons']),actual_inputs_match_independent_sampler=True,
                              sources={str(p):file_identity(p) for p in [BASE/'engine.c',BASE/'engine.py',BASE/'oracle_worker.py',BASE/'verify_full.py']})
                write()
        finally:engine.close()
    print(json.dumps({k:report[k] for k in ['status','passed','comparison_count','evaluation']},indent=2),flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
