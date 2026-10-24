"""Separate post-run Kiwi restoration; never import into the C trainer."""
import argparse
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent;WIKI=BASE.parent
sys.path.insert(0,str(BASE))
from summarize import assert_not_running,identity

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--control',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous decoded examples')
    result=json.loads(a.result.read_text());run=Path(result['control']['run_dir']);assert_not_running(run,a.control)
    from kiwipiepy import Kiwi
    from importlib.metadata import version
    sys.path.insert(0,str(WIKI))
    from step5_tokenize import detokenize
    kiwi=Kiwi(num_workers=1,model_type='cong-global')
    curves=result['validation_curve']
    rows=[]
    for index,point in enumerate(curves):
        for generation in point['generation']:
            raw=' '.join(generation['generated_tokens'])
            try:
                surface=detokenize(raw,kiwi) if raw else ''
                full=detokenize(' '.join(generation['prompt_tokens']+generation['generated_tokens']),kiwi)
                error=None
            except Exception as e:surface=None;full=None;error=f'{type(e).__name__}: {e}'
            rows.append({'curve_index':index,'label':point['label'],'at':point['at'],
                         'actual_elapsed_seconds':point['actual_elapsed_seconds'],
                         'target_predictions':point['target_predictions'],'updates':point['updates'],
                         'epochs_equivalent':point['epochs_equivalent'],
                         **generation,'generated_surface':surface,'prompt_and_generated_surface':full,'restoration_error':error})
    output={'kind':'ACTUAL_POST_RUN_GENERATION_EXAMPLES','result_identity':identity(a.result),
            'restorer':{'kiwipiepy':version('kiwipiepy'),'model_type':'cong-global','workers':1},
            'scope':'Surface restoration of all saved checkpoint samples, including reached 3h and epoch-1 states; no training or evaluation rerun. Raw form/POS tokens and errors retained.','examples':rows}
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'examples':len(rows),'restoration_errors':sum(r['restoration_error'] is not None for r in rows)}))

if __name__=='__main__':main()
