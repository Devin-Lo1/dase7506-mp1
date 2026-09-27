"""Student Transformer with causal depthwise Q/K/V convolutions.

Inspired by Primer (https://arxiv.org/abs/2109.08668). This experiment keeps
SwiGLU and uses identity-initialized length-three convolutions; it is not a
reproduction of the complete Primer recipe. No pretrained parameters are used.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from student import Block, StudentGPT


class LocalBlock(Block):
    def __init__(self,config):
        super().__init__(config)
        width=config['width'];self.kernel=config.get('local_kernel',3)
        if self.kernel<1:raise ValueError('local_kernel must be positive')
        self.local_qkv=nn.Conv1d(3*width,3*width,self.kernel,groups=3*width,bias=False)
        with torch.no_grad():
            self.local_qkv.weight.zero_()
            self.local_qkv.weight[:,:,-1]=1.

    def forward(self,x,rotary=None):
        batch,length,width=x.shape
        projected=self.qkv(self.norm1(x))
        if projected.device.type=='cpu':
            # Explicit shifts avoid slow grouped Conv1d dispatch on this CPU.
            weights=self.local_qkv.weight[:,0]
            mixed=projected*weights[:,-1]
            for offset in range(1,min(self.kernel,length)):
                if torch.is_grad_enabled():
                    mixed=mixed+F.pad(projected[:,:-offset]*weights[:,-1-offset],(0,0,offset,0))
                else:
                    mixed[:,offset:].addcmul_(projected[:,:-offset],weights[:,-1-offset])
            projected=mixed
        else:
            projected=self.local_qkv(F.pad(projected.transpose(1,2),(self.kernel-1,0))).transpose(1,2)
        q,k,v=projected.reshape(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        if rotary is not None:
            q=self.rotate(q,*rotary);k=self.rotate(k,*rotary)
        attended=F.scaled_dot_product_attention(q,k,v,is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.)
        x=x+self.dropout(self.proj(attended.transpose(1,2).reshape(batch,length,width)))
        gate,up=self.gate_up(self.norm2(x)).chunk(2,dim=-1)
        return x+self.dropout(self.down(F.silu(gate)*up))


class LocalGPT(StudentGPT):
    def __init__(self,config):
        super().__init__(config)
        self.blocks=nn.ModuleList([LocalBlock(config) for _ in range(config['depth'])])
        self.blocks.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight,std=.02/math.sqrt(2*config['depth']))
            nn.init.normal_(block.down.weight,std=.02/math.sqrt(2*config['depth']))


def build_model(config):return LocalGPT(config)
