"""Atomic, pickle-free full-state checkpoints for the C trainer."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np

SCHEMA=1

def save(path,engine,identity,state):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    info=engine.info
    metadata={'schema':SCHEMA,'identity':identity,'config':engine.config,'state':dict(state),
              'updates':info['updates'],'consumed':info['consumed'],'sampler':engine.sampler,
              'visited_paragraphs':info['visited_paragraphs']}
    for name in ['parameters','moment1','moment2']:
        array=engine.arrays[name]
        if not np.isfinite(array).all() or (name=='moment2' and np.any(array<0)):
            raise FloatingPointError('Refuse nonfinite/invalid checkpoint '+name)
    with temporary.open('wb') as stream:
        np.savez(stream,metadata=np.frombuffer(json.dumps(metadata,ensure_ascii=False,sort_keys=True).encode(),dtype=np.uint8),
                 parameters=engine.arrays['parameters'],moment1=engine.arrays['moment1'],moment2=engine.arrays['moment2'],
                 visited=engine.visited)
        stream.flush();os.fsync(stream.fileno())
    # Keep a previous complete file even if the newest complete checkpoint is later damaged.
    if path.exists():os.replace(path,path.with_name(path.stem+'.previous'+path.suffix))
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return metadata

def restore(path,engine,identity):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'metadata','parameters','moment1','moment2','visited'}:
            raise ValueError('Unknown checkpoint array schema')
        raw=archive['metadata']
        if raw.dtype!=np.uint8 or raw.ndim!=1 or raw.size>2**20:raise ValueError('Invalid checkpoint metadata')
        metadata=json.loads(raw.tobytes())
        if metadata['schema']!=SCHEMA or metadata['identity']!=identity or metadata['config']!=engine.config:
            raise ValueError('Checkpoint schema/config/source/corpus identity differs')
        if not isinstance(metadata['state'],dict):raise ValueError('Invalid checkpoint run state')
        arrays={name:archive[name] for name in ['parameters','moment1','moment2']}
        visited=archive['visited']
        for name,a in arrays.items():
            if a.dtype!=np.float32 or a.shape!=engine.arrays[name].shape or not np.isfinite(a).all():
                raise ValueError('Invalid saved '+name)
        if np.any(arrays['moment2']<0):raise ValueError('Negative second moment')
        if visited.dtype!=np.uint8 or visited.shape!=engine.visited.shape or np.any(visited>1):
            raise ValueError('Invalid saved visited vector')
        if int(visited.sum())!=metadata['visited_paragraphs'] or metadata['sampler'][4]!=metadata['consumed']:
            raise ValueError('Checkpoint progress counters inconsistent')
        # Validate everything before changing live state.
        engine.validate_progress(metadata['updates'],metadata['sampler'],visited)
        for name,a in arrays.items():np.copyto(engine.arrays[name],a)
        np.copyto(engine.visited,visited)
        engine.restore_progress(metadata['updates'],metadata['sampler'])
    return metadata['state']

def restore_latest(directory,engine,identity):
    errors=[]
    for name in ['latest.npz','latest.previous.npz']:
        path=Path(directory)/name
        try:
            state=restore(path,engine,identity)
        except Exception as error:errors.append({'path':str(path),'error':f'{type(error).__name__}: {error}'})
        else:
            latest=Path(directory)/'latest.npz'
            if name!='latest.npz' and latest.exists():
                os.replace(latest,latest.with_name(f'latest.rejected-{time.time_ns()}.npz'))
            return state,path,errors
    raise ValueError('No valid identity-matched checkpoint: '+json.dumps(errors))
