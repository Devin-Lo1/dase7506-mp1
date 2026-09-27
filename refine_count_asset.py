"""Retain singleton trigram continuation counts, with lossless integer packing.

Only training text is read. Context IDs and CSR pointers use the smallest
signed integer type that exactly represents every stored value.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from tokenizers import Tokenizer
from common import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('runs/ngram/train-kn12-pruned.pt'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter()
    data=torch.load(a.base,map_location='cpu',weights_only=True)
    manifest=json.loads(Path('data/manifest.json').read_text())
    for name in ['wikitext_train.txt','tokenizer.json']:
        assert sha(Path('data')/name)==manifest['sha256'][name]
    tokenizer=Tokenizer.from_file('data/tokenizer.json')
    ids=np.asarray(tokenizer.encode(Path('data/wikitext_train.txt').read_text()).ids,dtype=np.int64)
    length=len(ids)-3
    keys=ids[:length].copy()
    for offset in range(1,4):keys=keys*2048+ids[offset:offset+length]
    fourgrams=np.unique(keys)
    keys,counts=np.unique(fourgrams%(2048**3),return_counts=True)
    contexts,starts=np.unique(keys//2048,return_index=True)
    totals=np.add.reduceat(counts,starts)
    arrays=data['arrays']
    for name,value in [('keys',contexts),('ptr',np.r_[starts,len(keys)]),
                       ('tokens',(keys%2048).astype(np.int16)),
                       ('counts',counts.astype(np.float32)),('totals',totals.astype(np.float32))]:
        arrays['n3_'+name]=torch.from_numpy(value)
    for name,value in list(arrays.items()):
        if value.dtype==torch.int64:
            if int(value.max())<=32767:arrays[name]=value.to(torch.int16)
            elif int(value.max())<=2147483647:arrays[name]=value.to(torch.int32)
    data['metadata'].update(singleton_trigram_continuations=True,
                            packing='lossless signed integer range packing',
                            refinement_seconds=time.perf_counter()-started,
                            refinement_training_targets=len(ids)-1,
                            refinement_source_sha256=sha(Path(__file__)),
                            refinement_parent_sha256=sha(a.base),
                            uncompressed_tensor_bytes=sum(v.numel()*v.element_size() for v in arrays.values()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,a.output)
    a.output.with_suffix('.json').write_text(json.dumps(data['metadata'],indent=2)+'\n')
    print(json.dumps(data['metadata'],indent=2),flush=True)


if __name__=='__main__':main()
