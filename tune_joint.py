"""Coordinate search of existing global mixture settings on validation only.

Target probabilities exist only during this search; none are saved or deployed.
The predictor remains the unchanged causal Hybrid implementation.
"""
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
    if a.output_dir.exists():raise ValueError('Use a new output directory')
    a.output_dir.mkdir(parents=True)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=Hybrid(c['config']).eval();model.load_state_dict(c['model']);cfg=c['config']
    temperatures=sorted(set([round(1+i*.025,3) for i in range(15)]+[cfg['temperature']]))
    discounts=sorted(set([.25,.5,.75,1.]+[cfg['discount']]))
    thetas=sorted(set([1.,5.,10.,20.,40.]+[cfg['cache_theta']]))
    chunks={('neural',t):[] for t in temperatures}
    chunks.update({(kind,d):[] for d in discounts for kind in ['count','confidence']})
    chunks.update({('cache',t):[] for t in thetas})
    chunks.update({(kind,0):[] for kind in ['copy','order','cache_mask']})
    data=load_data();tokens,nbytes=data['validation']
    with torch.no_grad():
        for i,(x,y) in enumerate(windows(tokens)):
            valid=y!=-100;target=y.clamp_min(0).unsqueeze(-1)
            def target_p(p):return p.gather(-1,target).squeeze(-1)[valid].clone()
            h=model.neural.features(x);logits=model.neural.head(h).float()
            for t in temperatures:chunks['neural',t].append(target_p((logits/t).softmax(-1)))
            for d in discounts:
                model.config['discount']=d;prob=model.count_probabilities(x)
                chunks['count',d].append(target_p(prob))
                chunks['confidence',d].append(prob.amax(-1)[valid])
            for t in thetas:chunks['cache',t].append(target_p(cache_probabilities(h,x,t)))
            copied,enabled,order=suffix_probabilities(x,return_order=True)
            chunks['copy',0].append(target_p(copied))
            chunks['order',0].append((order.squeeze(-1)*enabled.squeeze(-1))[valid])
            mask=torch.ones_like(valid);mask[:,0]=False;chunks['cache_mask',0].append(mask[valid])
            if (i+1)%10==0:print('validation batches',i+1,flush=True)
    arrays={k:torch.cat(v).double() for k,v in chunks.items()};del chunks,logits,h,prob,data
    order=arrays['order',0].long().clamp_max(4)
    original=cfg.get('copy_order_weights',[0,0,0,0,0])
    state={'temperature':cfg['temperature'],'discount':cfg['discount'],
           'low':cfg['count_weight_low'],'high':cfg['count_weight_high'],
           'power':cfg['count_confidence_power'],'cache_weight':cfg['cache_weight'],
           'theta':cfg['cache_theta'],'copy2':original[2],'copy3':original[3],'copy4':original[4]}
    def score(s):
        weight=s['low']+(s['high']-s['low'])*arrays['confidence',s['discount']].pow(s['power'])
        prob=(1-weight)*arrays['neural',s['temperature']]+weight*arrays['count',s['discount']]
        cw=s['cache_weight']*arrays['cache_mask',0]
        prob=(1-cw)*prob+cw*arrays['cache',s['theta']]
        table=torch.tensor([0.,0.,s['copy2'],s['copy3'],s['copy4']],dtype=torch.float64)
        w=table[order];prob=(1-w)*prob+w*arrays['copy',0]
        return -prob.clamp_min(1e-30).log().sum().item()/math.log(2)/nbytes
    history=[];initial=score(state)
    domains={'temperature':temperatures,'discount':discounts,'cache_weight':[0,.025,.05,.075,.1,.15,.2],
             'theta':thetas,'low':[0,.05,.1,.15,.2,.25,.3],
             'high':[.1,.2,.3,.4,.5,.7,.9],'power':[.25,.5,1.,2.,3.,4.],
             'copy2':[i*.05 for i in range(9)],'copy3':[i*.05 for i in range(11)],
             'copy4':[i*.05 for i in range(13)]}
    best=initial
    for iteration in range(3):
        before=best
        for key,values in domains.items():
            options=[(best,state)]
            for value in values:
                trial={**state,key:value}
                if trial['low']>trial['high']:continue
                loss=score(trial);history.append({'iteration':iteration,'parameter':key,'value':value,'bpb':loss})
                options.append((loss,trial))
            best,state=min(options,key=lambda row:row[0])
        print(json.dumps({'iteration':iteration,'bpb':best,'settings':state}),flush=True)
        if before-best<1e-8:break
    cfg.update(temperature=state['temperature'],discount=state['discount'],
               count_weight_low=state['low'],count_weight_high=state['high'],
               count_confidence_power=state['power'],cache_weight=state['cache_weight'],
               cache_theta=state['theta'],copy_order_weights=[0.,0.,state['copy2'],state['copy3']]+[state['copy4']]*3)
    model.config.update(cfg)
    with torch.no_grad():
        x=tokens[:48].reshape(2,24);changed=x.clone();changed[:,12:]=(changed[:,12:]+19)%2048
        first=model.predict_log_probs(x)
        torch.testing.assert_close(first[:,:12],model.predict_log_probs(changed)[:,:12],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(first.logsumexp(-1),torch.zeros(2,24),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(first[:1],model.predict_log_probs(x[:1]),atol=1e-5,rtol=1e-5)
    torch.save(c,a.output_dir/'checkpoint.pt')
    report={'best':{'bpb':best,**state},'initial_bpb':initial,'results':history,
            'search_seconds':time.perf_counter()-started,'selected_on':'validation','contract_checks':'passed',
            'parent_checkpoint':str(a.checkpoint),'parent_sha256':sha(a.checkpoint),'source_sha256':sha(__file__)}
    (a.output_dir/'search.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['best']),flush=True)


if __name__=='__main__':main()
