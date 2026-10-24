"""Independent numerical, epoch-edge, evaluation and exact-resume checks."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE.parent/'gpu-benchmark'))
from engine import Engine,build
import checkpoint
import verify_mps as oracle
from corpus_fixture import Corpus,create_initial

CONFIG={'vocab_size':11,'context_size':2,'embedding_dim':3,'hidden_dim':5,
        'batch_size':4,'cpu_threads':1,'seed':42}

def fixture(directory):
    rows=[];ids=[]
    for i,line in enumerate([[4,5,6,7,8],[5,4,9],[10,4,5,6]]):
        tokens=[2,2]+line+[3]
        rows.append((len(ids),len(tokens),i,1,i+20));ids.extend(tokens)
    dtype=np.dtype([('start','<u8'),('length','<u4'),('subset_key','<u4'),('sentences','<u4'),('doc_key','<u8')])
    np.save(directory/'train.ids.npy',np.asarray(ids,dtype='<u2'))
    np.save(directory/'train.paragraphs.npy',np.asarray(rows,dtype=dtype))

class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name);fixture(self.directory)
        self.engine=Engine(self.directory,CONFIG)
    def tearDown(self):
        self.engine.close();self.temp.cleanup()
    def test_four_updates_and_eval_match_independent_torch(self):
        config={**CONFIG,'threads':1}
        initial,layout=create_initial(config)
        np.testing.assert_array_equal(self.engine.arrays['parameters'],initial)
        corpus=Corpus.from_arrays(np.load(self.directory/'train.ids.npy'),np.load(self.directory/'train.paragraphs.npy'),2,11);sampler=corpus.sampler(42)
        (self.directory/'config.json').write_text(json.dumps(config))
        np.save(self.directory/'initial.npy',initial)
        actual_steps=[]
        for i in range(4):
            result=self.engine.step(self.engine.info['windows']*2)
            n=int(result['batch_size']);x,y=self.engine.batch(n)
            expected_x,expected_y=sampler.next_batch(n)
            np.testing.assert_array_equal(x,expected_x);np.testing.assert_array_equal(y,expected_y)
            np.save(self.directory/f'x{i+1}.npy',x);np.save(self.directory/f'y{i+1}.npy',y)
            actual_steps.append(dict([(name,np.array(actual,copy=True)) for name,actual in [('loss',result['loss']),('gradient_norm',result['gradient_norm']),('clipping_scale',result['clipping_scale']),
                                ('clipped_gradients',self.engine.arrays['gradients']),('updated_params',self.engine.arrays['parameters']),
                                ('moment1',self.engine.arrays['moment1']),('moment2',self.engine.arrays['moment2'])]]))
        process=subprocess.run([sys.executable,str(BASE/'oracle_worker.py'),str(self.directory)],capture_output=True,text=True,timeout=120)
        self.assertEqual(process.returncode,0,process.stderr)
        for i,actual in enumerate(actual_steps,1):
            with np.load(self.directory/f'expected{i}.npz') as expected:
                for name,a in actual.items():
                    tol=oracle.TOLERANCES[name if name in ('moment1','moment2') else 'ordinary']
                    self.assertTrue(np.all(np.abs(a.astype(np.float64)-expected[name].astype(np.float64))<=tol['atol']+tol['rtol']*np.abs(expected[name].astype(np.float64))),name)
        before={k:v.copy() for k,v in self.engine.arrays.items()};state=self.engine.sampler
        logits=self.engine.logits(x)
        with np.load(self.directory/'evaluation.npz') as expected_eval:
            reference_logits=expected_eval['logits'];losses=expected_eval['losses']
        np.testing.assert_allclose(logits,reference_logits,atol=2e-6,rtol=2e-4)
        stats=self.engine.evaluate(x,y,4);content=(y!=4)&(y!=3)
        self.assertAlmostEqual(stats[1],float(losses.sum()),places=5)
        self.assertAlmostEqual(stats[4],float(losses[content].sum()),places=5)
        self.assertEqual(stats[0],n);self.assertEqual(stats[3],int(content.sum()))
        self.assertEqual(state,self.engine.sampler)
        for k,v in before.items():np.testing.assert_array_equal(v,self.engine.arrays[k])
    def test_epoch_edges_and_target_limit_are_exact(self):
        windows=self.engine.info['windows'];sizes=[]
        while (result:=self.engine.step(2*windows)) is not None:sizes.append(int(result['batch_size']))
        self.assertEqual(sum(sizes),2*windows);self.assertEqual(self.engine.info['consumed'],2*windows)
        self.assertIn(windows%4,sizes);self.assertEqual(self.engine.info['visited_paragraphs'],3)
        self.assertEqual(self.engine.info['updates'],2*((windows+3)//4))
    def test_abi_rejects_wrong_shapes_ranges_and_capacity(self):
        good=np.asarray([[2,4]],dtype=np.uint32)
        for x in [good[:,:1],np.tile(good,(5,1)),np.asarray([2,4]),np.asarray([[-1,4]]),np.asarray([[2,11]]),np.asarray([[2.,4.]])]:
            with self.assertRaises(ValueError):self.engine.logits(x)
        for y in [np.asarray([],dtype=np.uint32),np.asarray([11]),np.asarray([-1]),np.asarray([[4]])]:
            with self.assertRaises(ValueError):self.engine.evaluate(good,y,4)
    def test_invalid_checkpoint_does_not_mutate_and_corrupt_latest_falls_back(self):
        self.engine.step(30);identity={'corpus':'fixture'};path=self.directory/'latest.npz'
        checkpoint.save(path,self.engine,identity,{'elapsed':1})
        self.engine.step(30);checkpoint.save(path,self.engine,identity,{'elapsed':2})
        with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        metadata=json.loads(arrays['metadata'].tobytes());metadata['sampler'][2]=0
        arrays['metadata']=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8)
        arrays['parameters'][:]=123
        bad=self.directory/'invalid.npz';np.savez(bad,**arrays)
        before={k:v.copy() for k,v in self.engine.arrays.items()};sampler=self.engine.sampler;info=self.engine.info
        with self.assertRaises(ValueError):checkpoint.restore(bad,self.engine,identity)
        for k,v in before.items():np.testing.assert_array_equal(v,self.engine.arrays[k])
        self.assertEqual(sampler,self.engine.sampler);self.assertEqual(info,self.engine.info)
        path.write_bytes(b'damaged partial archive')
        state,used,rejected=checkpoint.restore_latest(self.directory,self.engine,identity)
        self.assertEqual(state['elapsed'],1);self.assertEqual(used.name,'latest.previous.npz')
        self.assertEqual(len(rejected),1);self.assertTrue(list(self.directory.glob('latest.rejected-*.npz')))
        checkpoint.save(path,self.engine,identity,{'elapsed':3})
        self.assertEqual(checkpoint.restore(self.directory/'latest.previous.npz',self.engine,identity)['elapsed'],1)
    def test_atomic_checkpoint_resume_is_bit_exact_across_epoch(self):
        limit=2*self.engine.info['windows']
        for _ in range(3):self.engine.step(limit)
        path=self.directory/'latest.npz';identity={'corpus':'fixture','code':'checked'}
        checkpoint.save(path,self.engine,identity,{'elapsed':1.25})
        other=Engine(self.directory,CONFIG)
        try:
            restored=checkpoint.restore(path,other,identity);self.assertEqual(restored['elapsed'],1.25)
            with self.assertRaises(ValueError):checkpoint.restore(path,other,{'corpus':'changed'})
            while True:
                a=self.engine.step(limit);b=other.step(limit);self.assertEqual(a,b)
                for k in ['parameters','moment1','moment2']:np.testing.assert_array_equal(self.engine.arrays[k],other.arrays[k])
                self.assertEqual(self.engine.sampler,other.sampler)
                if a is None:break
            checkpoint.save(path,other,identity,{'elapsed':2.5})
            self.assertTrue(path.with_name('latest.previous.npz').is_file())
        finally:other.close()

if __name__=='__main__':unittest.main(verbosity=2)
