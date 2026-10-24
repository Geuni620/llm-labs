"""Preserve an already-evaluated saved checkpoint, without loading a model."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import zipfile


def metadata(stream):
    with zipfile.ZipFile(stream) as z:
        if set(z.namelist())!={'metadata.npy','parameters.npy','moment1.npy','moment2.npy','visited.npy'}:
            raise ValueError('Unexpected checkpoint schema')
        data=io.BytesIO(z.read('metadata.npy'))
        if data.read(6)!=b'\x93NUMPY':raise ValueError('Invalid metadata array')
        version=data.read(2);size_bytes=2 if version==b'\x01\x00' else 4 if version in (b'\x02\x00',b'\x03\x00') else 0
        if not size_bytes:raise ValueError('Unknown NumPy header version')
        length=struct.unpack('<H' if size_bytes==2 else '<I',data.read(size_bytes))[0]
        if length>2**20:raise ValueError('Oversize metadata header')
        header=ast.literal_eval(data.read(length).decode('utf-8'))
        if header['descr']!='|u1' or header['fortran_order'] or len(header['shape'])!=1:
            raise ValueError('Unexpected metadata dtype/layout')
        raw=data.read()
        if len(raw)!=header['shape'][0] or len(raw)>2**20:raise ValueError('Invalid metadata length')
        return json.loads(raw)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    choose=p.add_mutually_exclusive_group(required=True)
    choose.add_argument('--min-elapsed',type=float);choose.add_argument('--metric-label')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=a.output.resolve();sidecar=output.with_suffix('.json')
    if output.exists() or sidecar.exists():raise FileExistsError('Preserve previous archive')
    run=a.run.resolve();info=json.loads((run/'run.json').read_text())
    metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
    candidates=[m for m in metrics if m['consumed']>0 and
                (m['actual_elapsed_seconds']>=a.min_elapsed if a.min_elapsed is not None else m['label']==a.metric_label)]
    if not candidates:raise ValueError('Requested actual evaluation milestone not reached')
    wanted=candidates[0]
    if wanted['evaluation_window_sha256']!=info['evaluation_window_sha256']:raise ValueError('Evaluation conditions differ')
    rejected=[]
    for name in ('latest.npz','latest.previous.npz'):
        source=run/name
        try:
            with source.open('rb') as original:
                m=metadata(original)
                if (m['schema']!=1 or m['identity']!=info['identity'] or m['config']!=info['config']
                    or m['consumed']!=wanted['consumed'] or m['updates']!=wanted['updates']
                    or m['sampler'][4]!=m['consumed'] or m['state']['last_validation']!=wanted['validation']
                    or m['state']['last_generation']!=wanted['generation']):
                    raise ValueError('Saved weights do not match the requested actual evaluation')
                original.seek(0);digest=hashlib.sha256();temporary=output.with_name(output.name+'.partial')
                with temporary.open('xb') as copied:
                    while block:=original.read(4*1024*1024):copied.write(block);digest.update(block)
                    copied.flush();os.fsync(copied.fileno())
                # The open source inode stays stable even if latest rotates during copying.
                with temporary.open('rb') as check:
                    if metadata(check)!=m:raise ValueError('Copied metadata differs')
                os.replace(temporary,output)
                record={'kind':'ARCHIVED_ACTUAL_EVALUATED_CHECKPOINT','source_at_copy':str(source),'path':str(output),
                        'bytes':output.stat().st_size,'sha256':digest.hexdigest(),'actual_metric':wanted,
                        'requested_min_elapsed_seconds':a.min_elapsed,'requested_label':a.metric_label,
                        'policy':'Exact saved state for the recorded evaluation; no training, replay, new evaluation or inference.'}
                sidecar.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
                directory=os.open(output.parent,os.O_RDONLY)
                try:os.fsync(directory)
                finally:os.close(directory)
                print(json.dumps({k:record[k] for k in ['path','bytes','sha256']},ensure_ascii=False));return
        except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:
            rejected.append({'source':str(source),'error':f'{type(error).__name__}: {error}'})
    raise RuntimeError('No saved state matches this evaluated milestone; refusing a misleading archive: '+json.dumps(rejected))


if __name__=='__main__':main()
