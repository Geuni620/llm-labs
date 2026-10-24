"""Independent Torch reference in a process that never loads the C/OpenMP ABI."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'gpu-benchmark'))
import verify_mps as oracle
from corpus_fixture import parameter_layout

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();directory=a.directory
    config=json.loads((directory/'config.json').read_text())
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    initial=np.load(directory/'initial.npy',allow_pickle=False)
    reference=oracle.build_reference(torch,config,initial,parameter_layout(config))
    for i in range(1,5):
        x=np.load(directory/f'x{i}.npy',allow_pickle=False);y=np.load(directory/f'y{i}.npy',allow_pickle=False)
        expected=oracle.reference_update(torch,reference,x,y,1)
        np.savez(directory/f'expected{i}.npz',**{k:v.detach().cpu().numpy() for k,v in expected.items()})
    with torch.no_grad():
        logits=reference[0](torch.from_numpy(x.astype(np.int64)))
        losses=torch.nn.functional.cross_entropy(logits,torch.from_numpy(y.astype(np.int64)),reduction='none')
    np.savez(directory/'evaluation.npz',logits=logits.numpy(),losses=losses.numpy())

if __name__=='__main__':main()
