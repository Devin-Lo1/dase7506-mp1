"""Continuous cache restricted to observed tokens inside each individual window.
Grave et al.: https://arxiv.org/abs/1612.04426 .
At position t, only keys h_j with j<t contribute, with values ids[j+1].
Thus every cached value is observed at or before t; nothing crosses calls.
"""
import torch
from torch.nn import functional as F


def cache_probabilities(features, ids, theta=20.0):
    batch,length=ids.shape
    output=features.new_zeros(batch,length,2048,dtype=torch.float32)
    if length<2:return output
    normalized=F.normalize(features.float(),dim=-1)
    logits=torch.bmm(normalized,normalized[:,:-1].transpose(1,2))*theta
    t=torch.arange(length,device=ids.device)[:,None]
    j=torch.arange(length-1,device=ids.device)[None,:]
    logits=logits.masked_fill(j>=t,-1e9)
    attention=logits.softmax(-1)
    attention[:,0,:]=0
    output.scatter_add_(2,ids[:,None,1:].expand(-1,length,-1),attention)
    return output


def _suffix_probabilities_torch(ids, max_order=6, min_order=2, return_order=False):
    """Copy continuations of the longest previously observed matching suffix.

    The confidence mask is one only after a >=min_order-token context matches.
    The source continuation ids[j+1] is used only when j<t, so it is observed.
    """
    batch,length=ids.shape
    out=torch.zeros(batch,length,2048,device=ids.device)
    if length<2:
        zero=torch.zeros(batch,length,1,device=ids.device)
        return (out,zero,zero) if return_order else (out,zero)
    equal=ids[:,:,None]==ids[:,None,:]
    match=equal
    causal=torch.ones(length,length,dtype=torch.bool,device=ids.device).tril(-1)
    weights=torch.zeros(batch,length,length,device=ids.device)
    longest=torch.zeros(batch,length,1,device=ids.device)
    for order in range(1,min(max_order,length)+1):
        if order>1:match=equal & F.pad(match[:,:-1,:-1],(1,0,1,0),value=False)
        valid=match & causal
        count=valid.sum(-1,keepdim=True)
        present=count>0
        weights=torch.where(present,valid.float()/count.clamp_min(1),weights)
        longest=torch.where(present,float(order),longest)
    out.scatter_add_(2,ids[:,None,1:].expand(-1,length,-1),weights[:,:,:-1])
    result=(out,(longest>=min_order).float())
    return (*result,longest) if return_order else result


def suffix_probabilities(ids, max_order=6, min_order=2, return_order=False):
    """CPU boolean matching avoids dense floating-point attention temporaries."""
    if ids.device.type != 'cpu':
        return _suffix_probabilities_torch(ids,max_order,min_order,return_order)
    import numpy as np
    x=ids.detach().numpy()
    batch,length=x.shape
    out=np.zeros((batch,length,2048),dtype=np.float32)
    longest=np.zeros((batch,length),dtype=np.int8)
    if length<2:
        zero=torch.from_numpy(longest[:,:,None].astype(np.float32))
        result=(torch.from_numpy(out),zero)
        return (*result,zero) if return_order else result
    equal=x[:,:,None]==x[:,None,:]
    match=equal.copy()
    causal=np.tri(length,k=-1,dtype=bool)
    selected=np.zeros_like(equal)
    for order in range(1,min(max_order,length)+1):
        if order>1:
            shifted=np.zeros_like(match)
            shifted[:,1:,1:]=match[:,:-1,:-1]
            match=equal & shifted
        valid=match & causal
        present=valid.any(-1)
        selected=np.where(present[:,:,None],valid,selected)
        longest=np.where(present,order,longest)
    confidence=longest>=min_order
    selected &= confidence[:,:,None]
    counts=selected.sum(-1,dtype=np.int32)
    b,t,j=np.nonzero(selected)
    values=(1.0/counts[b,t]).astype(np.float32)
    np.add.at(out,(b,t,x[b,j+1]),values)
    result=(torch.from_numpy(out),torch.from_numpy(confidence[:,:,None].astype(np.float32)))
    return (*result,torch.from_numpy(longest[:,:,None].astype(np.int64))) if return_order else result
