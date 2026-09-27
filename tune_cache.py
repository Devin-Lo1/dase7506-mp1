"""Validation-only grid for train-count/causal-cache interpolation and temperature."""
import argparse,itertools,json,math,time
from pathlib import Path
import torch
from common import load_data,windows,PROTOCOL,sha
from hybrid import Hybrid
from context_cache import cache_probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--neural',type=Path,required=True)
    p.add_argument('--counts',type=Path,default=Path('runs/ngram/train-kn12-pruned.pt'))
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    original=torch.load(a.neural,map_location='cpu',weights_only=True)
    counts=torch.load(a.counts,map_location='cpu',weights_only=True)
    config={'vocab':2048,'context':256,'neural_config':original['config'],
            'neural_implementation':original['implementation'],
            'ngram_shapes':{k:{'shape':list(v.shape),'dtype':str(v.dtype).split('.')[-1]} for k,v in counts['arrays'].items()},
            'mixture_weight':0.3,'discount':0.5,'max_order':counts['metadata']['max_order'],
            'cache_weight':0.1,'cache_theta':20.,'temperature':1.0}
    model=Hybrid(config).eval()
    model.load_state_dict({**{'neural.'+k:v for k,v in original['model'].items()},**counts['arrays']})
    # Check the actual composite predictor, including the within-window cache.
    x=torch.randint(2048,(2,24));changed=x.clone();changed[:,12:]=(changed[:,12:]+31)%2048
    with torch.no_grad():
        first=model.predict_log_probs(x)
        other=model.predict_log_probs(changed)
        torch.testing.assert_close(first[:,:12],other[:,:12],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(first.logsumexp(-1),torch.zeros(2,24),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(first[:1],model.predict_log_probs(x[:1]),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(first,model.predict_log_probs(x),atol=1e-6,rtol=1e-6)
    grid=list(itertools.product([0.9,1.,1.1],[0.,0.1,0.2,0.3,0.4],[0.,0.05,0.1,0.15,0.2],[10.,20.,40.]))
    totals=torch.zeros(len(grid),dtype=torch.float64)
    temps,weights,cacheweights,thetas=map(list,zip(*grid))
    tempsindex=torch.tensor([[0.9,1.,1.1].index(t) for t in temps])
    thetaindex=torch.tensor([[10.,20.,40.].index(t) for t in thetas])
    weights=torch.tensor(weights,dtype=torch.float64)[:,None]
    cacheweights=torch.tensor(cacheweights,dtype=torch.float64)[:,None]
    data=load_data();tokens,nbytes=data['validation']
    with torch.no_grad():
        for idx,(x,y) in enumerate(windows(tokens)):
            valid=y!=-100;target=y.clamp_min(0).unsqueeze(-1)
            features=model.neural.features(x);logits=model.neural.head(features).float()
            neural=torch.stack([(logits/t).softmax(-1).gather(-1,target).squeeze(-1)[valid] for t in [0.9,1.,1.1]]).double()
            count=model.count_probabilities(x).gather(-1,target).squeeze(-1)[valid].double()
            cache=torch.stack([cache_probabilities(features,x,t).gather(-1,target).squeeze(-1)[valid] for t in [10.,20.,40.]]).double()
            firstpos=torch.zeros_like(valid);firstpos[:,0]=True
            active=(~firstpos[valid]).double()[None,:]
            cw=cacheweights*active
            base=(1-weights)*neural[tempsindex]+weights*count[None,:]
            pred=(1-cw)*base+cw*cache[thetaindex]
            totals-=pred.clamp_min(1e-30).log().sum(-1)
            if (idx+1)%10==0:print('validation batches',idx+1,flush=True)
    results=[]
    for params,nll in zip(grid,totals.tolist()):
        results.append(dict(zip(['temperature','mixture_weight','cache_weight','cache_theta'],params))|{'bpb':nll/math.log(2)/nbytes})
    results.sort(key=lambda r:r['bpb']);best=results[0]
    config.update({k:v for k,v in best.items() if k!='bpb'})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint={'protocol':PROTOCOL,'implementation':'hybrid','config':config,'model':model.state_dict(),
                'seed':original['seed'],'train_tokens':original['train_tokens'],
                'steps':original.get('steps'),'schedule_steps':original.get('schedule_steps'),
                'neural_averaging':original.get('averaging'),
                'neural_min_lr_ratio':original.get('min_lr_ratio',0.1),
                'neural_parent_sha256':sha(a.neural),'count_metadata':counts['metadata']}
    torch.save(checkpoint,a.output_dir/'checkpoint.pt')
    summary={'neural_checkpoint':str(a.neural),'neural_training_targets':original['train_tokens'],
             'additional_count_training_targets':counts['metadata']['training_targets'],
             'best':best,'results':results,'search_seconds':time.perf_counter()-started,
             'contract_checks':'passed','selected_on':'validation','neural_parent_sha256':sha(a.neural),
             'count_asset_sha256':sha(a.counts),'hybrid_source_sha256':sha(Path('hybrid.py')),
             'cache_source_sha256':sha(Path('context_cache.py'))}
    (a.output_dir/'search.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':main()
