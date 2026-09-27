"""Average checkpoints from one training lineage; no additional optimizer updates."""
import argparse,json
from pathlib import Path
import torch
from common import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--weights',type=float,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if len({x.resolve().parent for x in a.checkpoints})!=1:
        p.error('This helper only averages checkpoints from one run directory.')
    checkpoints=[torch.load(x,map_location='cpu',weights_only=True) for x in a.checkpoints]
    weights=a.weights or [1.]*len(checkpoints)
    if len(weights)!=len(checkpoints) or min(weights)<0 or sum(weights)<=0:p.error('Invalid weights')
    weights=[w/sum(weights) for w in weights]
    reference=checkpoints[-1]
    for c in checkpoints:
        assert c['config']==reference['config'] and c['seed']==reference['seed']
        assert c['implementation']==reference['implementation']
        assert c['schedule_steps']==reference['schedule_steps']
        assert c['train_tokens']/c['steps']==reference['train_tokens']/reference['steps']
    model={k:sum(w*c['model'][k].float() for w,c in zip(weights,checkpoints)) for k in reference['model']}
    result={**reference,'model':model,'steps':max(c['steps'] for c in checkpoints),
            'train_tokens':max(c['train_tokens'] for c in checkpoints),
            'averaging':{'same_training_lineage':True,'weights':weights,
                         'parents':[{'path':str(p),'sha256':sha(p)} for p in a.checkpoints]}}
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,a.output)
    print(json.dumps({'output':str(a.output),'training_steps':result['steps'],
                      'training_targets':result['train_tokens'],'weights':weights}),flush=True)


if __name__=='__main__':main()
