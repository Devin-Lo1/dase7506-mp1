"""Select three global copying weights by observed suffix-match length.

Only validation aggregate likelihoods select the three scalars. No validation
tokens, predictions, or per-position weights are saved in inference assets.
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from common import load_data, windows, sha
from context_cache import suffix_probabilities
from hybrid import Hybrid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter();torch.set_num_threads(4)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    c['config'].pop('copy_order_weights',None)
    c['config']['copy_weight']=0.
    model=Hybrid(c['config']).eval();model.load_state_dict(c['model'])
    weights=torch.tensor([0.,.1,.2,.3,.4,.5,.6,.7,.8,.9],dtype=torch.float64)
    totals=torch.zeros(3,len(weights),dtype=torch.float64)
    counts=torch.zeros(3,dtype=torch.int64);inactive_nll=0.
    tokens,nbytes=load_data()['validation']
    with torch.no_grad():
        for idx,(x,y) in enumerate(windows(tokens)):
            valid=y!=-100;target=y.clamp_min(0).unsqueeze(-1)
            base=model.predict_log_probs(x).exp().gather(-1,target).squeeze(-1)[valid].double()
            copied,active,order=suffix_probabilities(x,return_order=True)
            copied=copied.gather(-1,target).squeeze(-1)[valid].double()
            order=order.squeeze(-1)[valid].clamp_max(4)
            inactive_nll-=base[order<2].clamp_min(1e-30).log().sum().item()
            for group in range(3):
                selected=order==group+2;counts[group]+=selected.sum()
                prob=(1-weights[:,None])*base[selected]+weights[:,None]*copied[selected]
                totals[group]-=prob.clamp_min(1e-30).log().sum(-1)
            if (idx+1)%10==0:print('validation batches',idx+1,flush=True)
    best_indices=totals.argmin(-1)
    chosen=weights[best_indices].tolist()
    table=[0.,0.,chosen[0],chosen[1],chosen[2],chosen[2],chosen[2]]
    c['config']['copy_order_weights']=table
    model.config.update(c['config'])
    x=torch.tensor([[5,6,7,5,6,7,5,6,8,5,6,9],[5,6,7,8,5,6,9,5,6,7,8,5]])
    changed=x.clone();changed[:,7:]=(changed[:,7:]+20)%2048
    with torch.no_grad():
        first=model.predict_log_probs(x)
        torch.testing.assert_close(first[:,:7],model.predict_log_probs(changed)[:,:7],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(first.logsumexp(-1),torch.zeros(2,12),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(first[:1],model.predict_log_probs(x[:1]),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(first,model.predict_log_probs(x),atol=1e-6,rtol=1e-6)
    best_nll=inactive_nll+sum(totals[g,best_indices[g]].item() for g in range(3))
    constant=(totals.sum(0)+inactive_nll)/math.log(2)/nbytes
    report={'best':{'copy_order_weights':table,'bpb':best_nll/math.log(2)/nbytes},
            'constant_weight_results':[{'weight':w,'bpb':b} for w,b in zip(weights.tolist(),constant.tolist())],
            'group_target_counts':counts.tolist(),'groups':['2','3','4+'],
            'search_seconds':time.perf_counter()-started,'selected_on':'validation',
            'parent_checkpoint':str(a.checkpoint),'parent_sha256':sha(a.checkpoint),
            'contract_checks':'passed','source_sha256':sha(Path(__file__))}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save(c,a.output_dir/'checkpoint.pt')
    (a.output_dir/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
