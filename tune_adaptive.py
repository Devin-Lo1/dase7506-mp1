"""Validation search over global temperature and confidence-gated count weights."""
import argparse,itertools,json,math,time
from pathlib import Path
import torch
from common import load_data,windows,sha
from hybrid import Hybrid
from context_cache import cache_probabilities,suffix_probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter();torch.set_num_threads(4)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=Hybrid(c['config']).eval();model.load_state_dict(c['model'])
    temperatures=[1.0,1.1,1.2,1.3]
    policies=[(w,w,1.) for w in [0.,0.1,0.2,0.3,0.4]]
    policies+=list(itertools.product([0.,0.1,0.2],[0.3,0.5,0.7,0.9],[0.5,1.,2.]))
    grid=list(itertools.product(temperatures,policies));totals=[0.]*len(grid)
    data=load_data();tokens,nbytes=data['validation']
    with torch.no_grad():
        for idx,(x,y) in enumerate(windows(tokens)):
            valid=y!=-100;target=y.clamp_min(0).unsqueeze(-1)
            features=model.neural.features(x);logits=model.neural.head(features).float()
            neural={t:(logits/t).softmax(-1).gather(-1,target).squeeze(-1)[valid].double() for t in temperatures}
            counts=model.count_probabilities(x)
            count=counts.gather(-1,target).squeeze(-1)[valid].double()
            confidence=counts.amax(-1)[valid].double()
            cached=cache_probabilities(features,x,c['config'].get('cache_theta',20.))
            cache=cached.gather(-1,target).squeeze(-1)[valid].double()
            mask=torch.ones_like(valid);mask[:,0]=False
            cw=c['config'].get('cache_weight',0.)*mask[valid]
            copied,enabled,order=suffix_probabilities(x,return_order=True)
            copy=copied.gather(-1,target).squeeze(-1)[valid].double()
            if c['config'].get('copy_order_weights'):
                table=torch.tensor(c['config']['copy_order_weights'])
                cpw=(table[order.long().clamp_max(len(table)-1)]*enabled).squeeze(-1)[valid]
            else:
                cpw=c['config'].get('copy_weight',0.)*enabled.squeeze(-1)[valid]
            for j,(t,(low,high,power)) in enumerate(grid):
                weight=low+(high-low)*confidence.pow(power)
                base=(1-weight)*neural[t]+weight*count
                base=(1-cw)*base+cw*cache
                base=(1-cpw)*base+cpw*copy
                totals[j]-=base.clamp_min(1e-30).log().sum().item()
            if (idx+1)%10==0:print('validation batches',idx+1,flush=True)
    results=[]
    for (t,(low,high,power)),nll in zip(grid,totals):
        results.append({'temperature':t,'count_weight_low':low,'count_weight_high':high,
                        'count_confidence_power':power,'bpb':nll/math.log(2)/nbytes})
    results.sort(key=lambda r:r['bpb']);best=results[0]
    c['config'].update({k:v for k,v in best.items() if k!='bpb'})
    c['config'].update(adaptive_count=True,mixture_weight=1.)
    model.config.update(c['config'])
    x=tokens[:48].reshape(2,24);changed=x.clone();changed[:,12:]=(changed[:,12:]+19)%2048
    with torch.no_grad():
        first=model.predict_log_probs(x)
        torch.testing.assert_close(first[:,:12],model.predict_log_probs(changed)[:,:12],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(first.logsumexp(-1),torch.zeros(2,24),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(first[:1],model.predict_log_probs(x[:1]),atol=1e-5,rtol=1e-5)
    a.output_dir.mkdir(parents=True,exist_ok=True);torch.save(c,a.output_dir/'checkpoint.pt')
    report={'best':best,'results':results,'search_seconds':time.perf_counter()-started,
            'parent_checkpoint':str(a.checkpoint),'parent_sha256':sha(a.checkpoint),
            'contract_checks':'passed','selected_on':'validation'}
    (a.output_dir/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':main()
