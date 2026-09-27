"""Train-only compact n-grams with Kneser-Ney lower-order continuation counts."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer


def build(output, min_count=2):
    started=time.perf_counter()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'data/manifest.json').read_text())
    for name in ['wikitext_train.txt','tokenizer.json']:
        assert hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()==manifest['sha256'][name]
    tok=Tokenizer.from_file(str(root/'data/tokenizer.json'))
    ids=np.asarray(tok.encode((root/'data/wikitext_train.txt').read_text()).ids,dtype=np.int64)
    unique={}
    for order in range(2,6):
        key=ids[:len(ids)-order+1].copy()
        for j in range(1,order): key=key*2048+ids[j:len(ids)-order+1+j]
        unique[order]=np.unique(key,return_counts=True)
    unigram=np.bincount(unique[2][0]%2048,minlength=2048).astype(np.float32)+0.01
    unigram/=unigram.sum()
    arrays={'unigram':torch.from_numpy(unigram)}
    for order in range(2,6):
        if order<5:
            keys,counts=np.unique(unique[order+1][0]%(2048**order),return_counts=True)
        else:
            keys,counts=unique[order]
        prefix=keys//2048
        contexts,first=np.unique(prefix,return_index=True)
        totals=np.add.reduceat(counts,first)
        keep=counts >= (1 if order==2 else min_count)
        retained=keys[keep]
        retained_counts=counts[keep]
        sizes=np.bincount(np.searchsorted(contexts,retained//2048),minlength=len(contexts))
        active=sizes>0
        contexts,totals,sizes=contexts[active],totals[active],sizes[active]
        ptr=np.r_[0,np.cumsum(sizes)].astype(np.int64)
        for key,val in [('keys',contexts),('ptr',ptr),('tokens',(retained%2048).astype(np.int16)),
                        ('counts',retained_counts.astype(np.float32)),('totals',totals.astype(np.float32))]:
            arrays[f'n{order}_{key}']=torch.from_numpy(val)
        print('order',order,'contexts',len(contexts),'edges',len(retained),flush=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    metadata={'training_tokens':len(ids),'training_targets':len(ids)-1,'max_order':5,
              'minimum_count_high_orders':min_count,'source_sha256':manifest['sha256']['wikitext_train.txt'],
              'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seconds':time.perf_counter()-started,
              'uncompressed_tensor_bytes':sum(v.numel()*v.element_size() for v in arrays.values())}
    torch.save({'arrays':arrays,'metadata':metadata},output)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/ngram/train-kn5.pt')
    p.add_argument('--min-count',type=int,default=2)
    a=p.parse_args();build(a.output,a.min_count)
