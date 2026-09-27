"""Local Transformer with learned hashed prefix n-gram input features.

Motivated by neural n-gram augmentation (https://arxiv.org/abs/2207.06366).
This uses direct token-ID hashes, not that paper's latent vector quantization.
All tables are random-initialized and learned only through the training loss.
"""
import torch
from torch import nn
from torch.nn import functional as F
from student_local import LocalGPT


class HashGPT(LocalGPT):
    def __init__(self,config):
        super().__init__(config)
        self.hash_buckets=config.get('hash_buckets',32768)
        dimension=config.get('hash_width',32)
        if self.hash_buckets<1 or self.hash_buckets & (self.hash_buckets-1) or dimension<1:
            raise ValueError('Hash buckets must be a positive power of two; width must be positive')
        self.hash_tables=nn.ModuleList([nn.Embedding(self.hash_buckets,dimension) for _ in range(2)])
        self.hash_projection=nn.Linear(2*dimension,config['width'],bias=False)
        for table in self.hash_tables:nn.init.normal_(table.weight,std=.02)
        nn.init.normal_(self.hash_projection.weight,std=.02)

    def prefix_hashes(self,ids):
        # These coefficients and 2048-token IDs fit within signed int32.
        values=ids.to(torch.int32)
        previous=F.pad(values[:,:-1],(1,0))
        previous2=F.pad(values[:,:-2],(2,0))[:,:values.shape[1]]
        bigram=(previous*17489+values*32531).bitwise_and(self.hash_buckets-1)
        trigram=(previous2*17489+previous*27127+values*32531).bitwise_and(self.hash_buckets-1)
        return bigram.long(),trigram.long()

    def features(self,ids):
        length=ids.shape[1]
        if length>self.context:raise ValueError('Input exceeds the fixed context')
        hashes=self.prefix_hashes(ids)
        positions=torch.arange(length,device=ids.device)[None,:,None]
        encoded=[table(key)*(positions>=order) for order,(table,key) in enumerate(zip(self.hash_tables,hashes),1)]
        x=self.embedding_dropout(self.token(ids)+self.hash_projection(torch.cat(encoded,dim=-1)))
        rotary=None
        if self.pos is not None:x=x+self.pos(torch.arange(length,device=ids.device))
        else:rotary=(self.rotary_cos[:length],self.rotary_sin[:length])
        for block in self.blocks:x=block(x,rotary)
        return self.norm(x)


def build_model(config):return HashGPT(config)
