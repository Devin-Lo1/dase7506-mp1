"""Check package files, frozen checkpoint identity and the loaded causal contract."""
import argparse,hashlib,json
from pathlib import Path
import torch
from common import make_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    freeze=json.loads(Path('FREEZE.json').read_text())
    assert sha(a.checkpoint)==freeze['checkpoint_sha256'],'Wrong checkpoint'
    for f,digest in freeze['runtime_sha256'].items():
        if Path(f).exists():assert sha(f)==digest,f
    manifest=json.loads(Path('PACKAGE_MANIFEST.json').read_text())
    for f in ['common.py','evaluate.py','model.py','configs/baseline.json','tests/test_contract.py',
              'data/manifest.json','data/tokenizer.json','data/wikitext_train.txt','data/wikitext_validation.txt','data/wikitext_test.txt']:
        assert sha(f)==manifest['code/'+f],f
    torch.set_num_threads(4);torch.manual_seed(1047)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert c['steps']==1200
    model,_=make_model(c['implementation'],c['config'],torch.device('cpu'));model.load_state_dict(c['model']);model.eval()
    x=torch.randint(2048,(2,48));other=x.clone();other[:,24:]=(other[:,24:]+17)%2048
    with torch.no_grad():
        p=model.predict_log_probs(x)
        torch.testing.assert_close(p[:,:24],model.predict_log_probs(other)[:,:24],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(p[:1],model.predict_log_probs(x[:1]),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(p,model.predict_log_probs(x),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(p.logsumexp(-1),torch.zeros(2,48),atol=1e-6,rtol=1e-6)
        assert torch.isfinite(p).all()
    print('PASS: exact frozen checkpoint; unchanged benchmark; causal, normalized, stateless predictions.')


if __name__=='__main__':main()
