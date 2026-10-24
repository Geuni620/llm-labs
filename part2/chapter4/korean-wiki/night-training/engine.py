"""Build/load the actual C training ABI; no implicit training on import."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

BASE=Path(__file__).resolve().parent
WIKI=BASE.parent
EXP=WIKI/'cpu-benchmark/experiments'
LIBRARY=BASE/'.artifacts/libnight_engine.dylib'

def file_identity(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return {'bytes':path.stat().st_size,'sha256':h.hexdigest()}

def build():
    LIBRARY.parent.mkdir(exist_ok=True)
    sources=[BASE/'engine.c',EXP/'fixedwork_benchmark.c',EXP/'gemm_kernels.c',EXP/'vector_vforce.c',
             EXP/'fixedwork_norm.c',EXP/'fixedwork_softmax.c',EXP/'fixedwork_api.h',EXP/'experiment_kernels.h',EXP/'fast_exp.h']
    identities={str(p):file_identity(p) for p in sources}
    manifest_path=LIBRARY.with_suffix('.build.json')
    if LIBRARY.exists() and manifest_path.exists():
        prior=json.loads(manifest_path.read_text())
        if prior['sources']==identities and prior['library']==file_identity(LIBRARY):return prior
        raise FileExistsError('Preserve existing build: use a new artifact location for changed sources')
    command=['clang','-O3','-std=c11','-ffp-contract=off','-fno-math-errno','-mcpu=native',
             '-DEXPERIMENT_VFORCE_EXP=1','-DFW_FIXED_NORM=1','-DFW_SOFTMAX_INV_BATCH=1','-DFW_MINIMAL_ZERO=1',
             '-Xpreprocessor','-fopenmp','-isystem','/opt/homebrew/opt/libomp/include','-dynamiclib',
             str(BASE/'engine.c'),*[str(EXP/name) for name in ('gemm_kernels.c','vector_vforce.c','fixedwork_norm.c','fixedwork_softmax.c')],
             '-framework','Accelerate','-L/opt/homebrew/opt/libomp/lib','-Wl,-rpath,/opt/homebrew/opt/libomp/lib','-lomp','-lm','-o',str(LIBRARY)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    record={'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'sources':identities}
    if result.returncode:
        manifest_path.write_text(json.dumps(record,indent=2)+'\n');raise RuntimeError(result.stderr)
    assert identities=={str(p):file_identity(p) for p in sources}
    record['library']=file_identity(LIBRARY);manifest_path.write_text(json.dumps(record,indent=2)+'\n')
    return record

class Engine:
    def __init__(self,corpus,config):
        self.config=dict(config)
        threads=config['cpu_threads']
        for name in ['OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
            os.environ[name]=str(threads)
        os.environ.update(OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0')
        self.library=ctypes.CDLL(str(LIBRARY))
        lib=self.library;ptr=ctypes.c_void_p;u64p=ctypes.POINTER(ctypes.c_uint64);f32p=ctypes.POINTER(ctypes.c_float)
        u32=np.ctypeslib.ndpointer(dtype=np.uint32,ndim=2,flags='C_CONTIGUOUS')
        targets=np.ctypeslib.ndpointer(dtype=np.uint32,ndim=1,flags='C_CONTIGUOUS')
        f32=np.ctypeslib.ndpointer(dtype=np.float32,ndim=2,flags='C_CONTIGUOUS')
        f64=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
        lib.kw_abi.restype=ctypes.c_uint32
        if lib.kw_abi()!=1:raise RuntimeError('Unknown training ABI')
        lib.kw_create.argtypes=[ctypes.c_char_p,*([ctypes.c_int]*6),ctypes.c_uint32];lib.kw_create.restype=ptr
        lib.kw_destroy.argtypes=[ptr]
        lib.kw_array.argtypes=[ptr,ctypes.c_int];lib.kw_array.restype=f32p
        lib.kw_visited.argtypes=[ptr];lib.kw_visited.restype=ctypes.POINTER(ctypes.c_uint8)
        lib.kw_info.argtypes=[ptr,u64p];lib.kw_get_sampler.argtypes=[ptr,u64p]
        lib.kw_restore_progress.argtypes=[ptr,ctypes.c_uint64,u64p];lib.kw_restore_progress.restype=ctypes.c_int
        lib.kw_step.argtypes=[ptr,ctypes.c_uint64,f64];lib.kw_step.restype=ctypes.c_int
        lib.kw_finite.argtypes=[ptr];lib.kw_finite.restype=ctypes.c_int
        lib.kw_contexts.argtypes=[ptr];lib.kw_contexts.restype=ctypes.POINTER(ctypes.c_uint32)
        lib.kw_targets.argtypes=[ptr];lib.kw_targets.restype=ctypes.POINTER(ctypes.c_uint32)
        lib.kw_evaluate.argtypes=[ptr,u32,targets,ctypes.c_int,ctypes.c_int,f64]
        lib.kw_logits.argtypes=[ptr,u32,ctypes.c_int,f32]
        for name in ['kw_destroy','kw_info','kw_get_sampler','kw_evaluate','kw_logits']:
            getattr(lib,name).restype=None
        self.handle=lib.kw_create(str(Path(corpus).resolve()).encode(),*[config[k] for k in ('vocab_size','context_size','embedding_dim','hidden_dim','batch_size','cpu_threads','seed')])
        if not self.handle:raise RuntimeError('C engine allocation failed')
        info=self.info
        self.arrays={name:np.ctypeslib.as_array(lib.kw_array(self.handle,i),shape=(info['parameters'],))
                     for i,name in enumerate(['parameters','moment1','moment2','gradients','initial'])}
        self.visited=np.ctypeslib.as_array(lib.kw_visited(self.handle),shape=(info['paragraphs'],))
    @property
    def info(self):
        a=(ctypes.c_uint64*8)();self.library.kw_info(self.handle,a)
        return dict(zip(['parameters','windows','paragraphs','updates','consumed','visited_paragraphs','corpus_ram_bytes','workspace_bytes'],map(int,a)))
    @property
    def sampler(self):
        a=(ctypes.c_uint64*5)();self.library.kw_get_sampler(self.handle,a);return list(map(int,a))
    def step(self,limit):
        a=np.zeros(5,dtype=np.float64)
        if not self.library.kw_step(self.handle,int(limit),a):return None
        return dict(zip(['loss','gradient_norm','clipping_scale','batch_size','epochs'],a.tolist()))
    def restore_progress(self,updates,sampler):
        self.validate_progress(updates,sampler,self.visited)
        a=(ctypes.c_uint64*5)(*sampler)
        if not self.library.kw_restore_progress(self.handle,updates,a):raise ValueError('Invalid saved sampler/progress')
    def validate_progress(self,updates,sampler,visited):
        if (not isinstance(updates,int) or not 0<=updates<2**31-1 or len(sampler)!=5
            or any(not isinstance(v,int) or not 0<=v<2**64 for v in sampler)):
            raise ValueError('Invalid saved progress types/bounds')
        random,position,multiplier,shift,consumed=sampler;windows=self.info['windows']
        if (random>2**32-1 or position>windows or not 1<=multiplier<windows
            or math.gcd(multiplier,windows)!=1 or shift>=windows
            or (position!=windows and consumed%windows!=position)
            or (position==windows and consumed%windows!=0)
            or visited.dtype!=np.uint8 or visited.shape!=self.visited.shape or np.any(visited>1)):
            raise ValueError('Invalid saved sampler/progress')
    def batch(self,count):
        x=np.ctypeslib.as_array(self.library.kw_contexts(self.handle),shape=(count*self.config['context_size'],)).reshape(count,-1).copy()
        y=np.ctypeslib.as_array(self.library.kw_targets(self.handle),shape=(count,)).copy();return x,y
    def evaluate(self,x,y,space_id):
        x=self._contexts(x);raw=np.asarray(y)
        if raw.ndim!=1 or len(raw)!=len(x) or not np.issubdtype(raw.dtype,np.integer) or np.any(raw<0) or np.any(raw>=self.config['vocab_size']):
            raise ValueError('Invalid target shape/type/range')
        y=np.ascontiguousarray(raw,dtype=np.uint32)
        out=np.zeros(6,dtype=np.float64);self.library.kw_evaluate(self.handle,x,y,len(y),space_id,out);return out
    def logits(self,x):
        x=self._contexts(x);out=np.empty((len(x),self.config['vocab_size']),dtype=np.float32)
        self.library.kw_logits(self.handle,x,len(x),out);return out
    def _contexts(self,x):
        raw=np.asarray(x)
        if (raw.ndim!=2 or raw.shape[1]!=self.config['context_size'] or not 1<=len(raw)<=self.config['batch_size']
            or not np.issubdtype(raw.dtype,np.integer) or np.any(raw<0) or np.any(raw>=self.config['vocab_size'])):
            raise ValueError('Invalid context shape/type/range/capacity')
        return np.ascontiguousarray(raw,dtype=np.uint32)
    def close(self):
        if self.handle:self.library.kw_destroy(self.handle);self.handle=None
