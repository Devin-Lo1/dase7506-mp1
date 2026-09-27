"""Stateless CPU neural/count-model interpolation; all assets live in checkpoint.
The count expert uses only input prefixes and fixed train-derived statistics.
"""
import numpy as np
import torch
from torch import nn
from student import StudentGPT
from context_cache import cache_probabilities, suffix_probabilities


class Hybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context=256
        self.config=dict(config)
        if config.get('neural_implementation','student')=='student_hash':
            from student_hash import HashGPT
            self.neural=HashGPT(config['neural_config'])
        elif config.get('neural_implementation','student')=='student_local':
            from student_local import LocalGPT
            self.neural=LocalGPT(config['neural_config'])
        else:
            if config.get('neural_implementation','student')!='student':
                raise ValueError('Unsupported neural implementation')
            self.neural=StudentGPT(config['neural_config'])
        for name,description in config['ngram_shapes'].items():
            self.register_buffer(name,torch.zeros(description['shape'],dtype=getattr(torch,description['dtype'])))

    def forward(self, ids):
        return self.neural(ids)

    def count_probabilities(self, ids):
        """Expand backoff from highest order, avoiding repeated dense scaling."""
        if ids.device.type!='cpu':
            raise ValueError('The compact count expert is implemented for CPU evaluation.')
        x=ids.detach().numpy().astype(np.int64,copy=False)
        batch,length=x.shape;n=batch*length
        p=np.zeros((n,2048),dtype=np.float32)
        remaining=np.ones(n,dtype=np.float32)
        discount=self.config.get('discount',0.75)
        positions=np.tile(np.arange(length),batch)
        for order in range(min(self.config.get('max_order',5),length+1),1,-1):
            history=order-1;query=np.zeros_like(x)
            for j in range(history):
                base=2048 if order<=5 else 2053
                query[:,history-1:]=query[:,history-1:]*base+x[:,j:length-history+1+j]
                if order>5:query[:,history-1:] &= 0x7fffffffffffffff
            keys=getattr(self,f'n{order}_keys').numpy()
            at=np.searchsorted(keys,query.ravel());safe=np.minimum(at,len(keys)-1)
            valid=(at<len(keys))&(keys[safe]==query.ravel())&(positions>=history-1)
            ptr=getattr(self,f'n{order}_ptr').numpy();starts=ptr[safe]
            sizes=np.where(valid,ptr[safe+1]-starts,0)
            rows=np.repeat(np.arange(n),sizes)
            if len(rows)==0:continue
            offsets=np.cumsum(sizes)-sizes
            edges=np.arange(len(rows))-np.repeat(offsets,sizes)+np.repeat(starts,sizes)
            counts=getattr(self,f'n{order}_counts').numpy()[edges]
            totals=getattr(self,f'n{order}_totals').numpy()[safe]
            mass=np.maximum(counts-discount,0)/totals[rows]
            used=np.bincount(rows,weights=mass,minlength=n).astype(np.float32)
            tokens=getattr(self,f'n{order}_tokens').numpy()[edges].astype(np.int64)
            p[rows,tokens]+=remaining[rows]*mass
            remaining*=np.maximum(1-used,0)
        p+=remaining[:,None]*self.unigram.numpy()[None,:]
        p/=p.sum(-1,keepdims=True)
        return torch.from_numpy(p.reshape(batch,length,2048))

    def _count_probabilities_reference(self, ids):
        """Original low-to-high recurrence retained as a numerical reference."""
        if ids.device.type!='cpu':
            raise ValueError('The compact count expert is implemented for CPU evaluation.')
        x=ids.detach().numpy().astype(np.int64,copy=False)
        batch,length=x.shape
        n=batch*length
        p=np.broadcast_to(self.unigram.numpy(),(n,2048)).copy()
        discount=self.config.get('discount',0.75)
        for order in range(2,self.config.get('max_order',5)+1):
            history=order-1
            if length<history: break
            query=np.zeros_like(x)
            for j in range(history):
                base=2048 if order<=5 else 2053
                query[:,history-1:]=query[:,history-1:]*base+x[:,j:length-history+1+j]
                if order>5:query[:,history-1:] &= 0x7fffffffffffffff
            query=query.ravel()
            keys=getattr(self,f'n{order}_keys').numpy()
            at=np.searchsorted(keys,query)
            safe=np.minimum(at,len(keys)-1)
            valid=(at<len(keys))&(keys[safe]==query)
            valid &= np.tile(np.arange(length)>=history-1,batch)
            ptr=getattr(self,f'n{order}_ptr').numpy()
            starts=ptr[safe]
            sizes=np.where(valid,ptr[safe+1]-starts,0)
            rows=np.repeat(np.arange(n),sizes)
            if len(rows)==0: continue
            offsets=np.cumsum(sizes)-sizes
            edges=np.arange(len(rows))-np.repeat(offsets,sizes)+np.repeat(starts,sizes)
            counts=getattr(self,f'n{order}_counts').numpy()[edges]
            totals=getattr(self,f'n{order}_totals').numpy()[safe]
            mass=np.maximum(counts-discount,0)/totals[rows]
            used=np.bincount(rows,weights=mass,minlength=n).astype(np.float32)
            p*=np.maximum(1-used,0)[:,None]
            tokens=getattr(self,f'n{order}_tokens').numpy()[edges].astype(np.int64)
            p[rows,tokens]+=mass
        p/=p.sum(-1,keepdims=True)
        return torch.from_numpy(p.reshape(batch,length,2048))

    def probabilities_from_features(self, features, ids):
        weight=self.config.get('mixture_weight',0.3)
        neural=(self.neural.head(features).float()/self.config.get('temperature',1.0)).softmax(-1)
        probabilities=neural
        if weight:
            counts=self.count_probabilities(ids)
            if self.config.get('adaptive_count',False):
                confidence=counts.amax(-1,keepdim=True).pow(self.config.get('count_confidence_power',1.0))
                low=self.config['count_weight_low'];high=self.config['count_weight_high']
                weight=low+(high-low)*confidence
            probabilities=(1-weight)*neural+weight*counts
        cache_weight=self.config.get('cache_weight',0.0)
        if cache_weight:
            cached=cache_probabilities(features,ids,self.config.get('cache_theta',20.0))
            mixed=(1-cache_weight)*probabilities+cache_weight*cached
            mixed[:,0]=probabilities[:,0]
            probabilities=mixed
        copy_weight=self.config.get('copy_weight',0.0)
        if copy_weight or self.config.get('copy_order_weights'):
            copied,confidence,order=suffix_probabilities(ids,return_order=True)
            if self.config.get('copy_order_weights'):
                table=torch.tensor(self.config['copy_order_weights'],device=ids.device)
                weight=table[order.long().clamp_max(len(table)-1)]*confidence
            else:
                weight=copy_weight*confidence
            probabilities=(1-weight)*probabilities+weight*copied
        return probabilities

    def predict_log_probs(self, ids):
        features=self.neural.features(ids)
        logp=self.probabilities_from_features(features,ids).clamp_min(1e-30).log()
        return logp-torch.logsumexp(logp,dim=-1,keepdim=True)


def build_model(config):
    return Hybrid(config)
