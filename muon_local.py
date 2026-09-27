"""Single-device batched Muon with auxiliary AdamW.
Algorithm adapted from Keller Jordan's Muon (MIT, see MUON_LICENSE.txt):
https://github.com/KellerJordan/Muon/blob/master/muon.py
Hidden matrices use Newton-Schulz updates; embeddings and vectors use AdamW.
"""
from collections import defaultdict
import torch


def orthogonalize(updates, steps=5):
    x=updates.to(torch.bfloat16)
    transposed=x.shape[-2]>x.shape[-1]
    if transposed:x=x.mT
    x=x/(x.norm(dim=(-2,-1),keepdim=True)+1e-7)
    for _ in range(steps):
        gram=x@x.mT
        polynomial=-4.7750*gram+2.0315*(gram@gram)
        x=3.4445*x+polynomial@x
    if transposed:x=x.mT
    return x*max(1,updates.shape[-2]/updates.shape[-1])**0.5


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, groups):
        super().__init__(groups,dict(lr=0.002,weight_decay=0.1,use_muon=False,
                                    betas=(0.9,0.95),momentum=0.95,eps=1e-8,lr_scale=1.0))

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            if group['use_muon']:
                by_shape=defaultdict(list)
                for p in group['params']:
                    if p.grad is not None:by_shape[tuple(p.shape)].append(p)
                for params in by_shape.values():
                    updates=[]
                    for p in params:
                        state=self.state[p]
                        if 'momentum' not in state:state['momentum']=torch.zeros_like(p)
                        momentum=state['momentum']
                        momentum.lerp_(p.grad,1-group['momentum'])
                        updates.append(p.grad.lerp(momentum,group['momentum']))
                    update=orthogonalize(torch.stack(updates)).float()
                    for p,u in zip(params,update):
                        p.mul_(1-group['lr']*group['weight_decay'])
                        p.add_(u,alpha=-group['lr'])
            else:
                b1,b2=group['betas']
                for p in group['params']:
                    if p.grad is None:continue
                    state=self.state[p]
                    if not state:
                        state.update(step=0,mean=torch.zeros_like(p),variance=torch.zeros_like(p))
                    state['step']+=1
                    state['mean'].lerp_(p.grad,1-b1)
                    state['variance'].lerp_(p.grad.square(),1-b2)
                    corrected_mean=state['mean']/(1-b1**state['step'])
                    corrected_variance=state['variance']/(1-b2**state['step'])
                    p.mul_(1-group['lr']*group['weight_decay'])
                    p.addcdiv_(corrected_mean,corrected_variance.sqrt().add_(group['eps']),value=-group['lr'])
        return loss
