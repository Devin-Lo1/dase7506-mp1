"""Assemble the frozen recipe from a trained neural model and train-only counts.
No validation/test fitting, checkpoint selection or hyperparameter search occurs.
"""
import argparse,json
from pathlib import Path
import torch
from common import PROTOCOL,sha
from hybrid import Hybrid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--neural',type=Path,required=True)
    p.add_argument('--counts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise ValueError('Output exists; preserve previous artifacts')
    neural=torch.load(a.neural,map_location='cpu',weights_only=True)
    counts=torch.load(a.counts,map_location='cpu',weights_only=True)
    config=json.loads(Path('configs/final_hybrid.json').read_text())
    assert neural['steps']==1200 and neural['implementation']=='student_local'
    assert neural['config']==config['neural_config']
    assert neural['train_tokens']==157286400
    expected={k:(tuple(v['shape']),getattr(torch,v['dtype'])) for k,v in config['ngram_shapes'].items()}
    assert counts['arrays'].keys()==expected.keys()
    for k,v in counts['arrays'].items():assert (tuple(v.shape),v.dtype)==expected[k],k
    model=Hybrid(config).eval()
    model.load_state_dict({**{'neural.'+k:v for k,v in neural['model'].items()},**counts['arrays']})
    result={'protocol':PROTOCOL,'implementation':'hybrid','config':config,'model':model.state_dict(),
            'seed':neural['seed'],'steps':1200,'train_tokens':neural['train_tokens'],
            'schedule_steps':neural['schedule_steps'],'neural_averaging':None,'neural_min_lr_ratio':0.,
            'neural_parent_sha256':sha(a.neural),'count_metadata':counts['metadata'],
            'reproduction_note':'Frozen global settings; no test-based tuning.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,a.output)
    print('Saved',a.output,'SHA256',sha(a.output))


if __name__=='__main__':main()
