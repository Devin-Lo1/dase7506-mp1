"""POSIX development resource check using the unchanged full-validation scorer.

Run after training has stopped to avoid CPU contention. This is not final
full-test certification. Maximum RSS includes interpreter/loading.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=Path('runs/baseline/checkpoint.pt'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/resource-validation'))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--split', choices=['validation','test'], default='validation',
                   help='Use test only after freezing the method and checkpoint.')
    args = p.parse_args()
    if sys.platform not in ('darwin','linux'):
        p.error('This wrapper supports macOS and Linux getrusage RSS units.')
    if args.repeats < 1:
        p.error('repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {'baseline': [], 'candidate': []}
    for repeat in range(args.repeats):
        for name, checkpoint in [('baseline',args.baseline),('candidate',args.candidate)]:
            output = args.output_dir / f'{name}-{repeat+1}.json'
            wrapper = ('import atexit,resource,runpy,sys; '
                       'atexit.register(lambda: print("MP1_MAX_RSS="+str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),file=sys.stderr)); '
                       'script=sys.argv.pop(1); sys.argv[0]=script; '
                       'runpy.run_path(script,run_name="__main__")')
            cmd = [sys.executable,'-c',wrapper,'evaluate.py',
                   '--checkpoint',str(checkpoint),'--device','cpu','--precision','fp32',
                   '--threads','4','--split',args.split,'--output',str(output)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            output.with_suffix('.time.txt').write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(run.stderr + run.stdout)
            result = json.loads(output.read_text())
            rss = re.search(r'MP1_MAX_RSS=(\d+)', run.stderr)
            if rss is None:
                raise RuntimeError('Could not parse maximum RSS')
            row = {'seconds': result['seconds'], 'bpb': result['bpb'],
                   'peak_rss_bytes': int(rss.group(1))*(1 if sys.platform=='darwin' else 1024), 'command': cmd}
            records[name].append(row)
            print(json.dumps({'name':name,'repeat':repeat+1,**row}),flush=True)
    ratio = statistics.median(r['seconds'] for r in records['candidate']) / statistics.median(
        r['seconds'] for r in records['baseline'])
    checkpoint = torch.load(args.candidate,map_location='cpu',weights_only=True)
    assets = [args.candidate,Path('student.py'),Path('common.py'),Path('evaluate.py')]
    neural_implementation=checkpoint['config'].get('neural_implementation',checkpoint['implementation'])
    if neural_implementation in ('student_local','student_hash'):
        assets.append(Path('student_local.py'))
    if neural_implementation=='student_hash':assets.append(Path('student_hash.py'))
    if checkpoint['implementation'] in ('hybrid','knn_model'):
        assets.extend([Path('hybrid.py'),Path('context_cache.py')])
    if checkpoint['implementation']=='knn_model':
        assets.append(Path('knn_model.py'))
        directory=Path(checkpoint['config']['asset_dir'])
        assets.extend(directory/name for name in checkpoint['config']['asset_hashes'])
    summary = {'split':args.split,'precision':'fp32','threads':4,'runs':records,
               'median_time_ratio':ratio,
               'candidate_peak_rss_gib':max(r['peak_rss_bytes'] for r in records['candidate'])/1024**3,
               'model_assets_mib':sum(f.stat().st_size for f in assets)/1024**2,
               'model_assets':[str(f) for f in assets],
               'note':('Development validation check only; final full-test measurement remains required.'
                       if args.split=='validation' else 'Frozen full-test predictor resource measurement.')}
    summary['development_limits_pass'] = (ratio <= 5 and summary['candidate_peak_rss_gib'] <= 4
                                          and summary['model_assets_mib'] <= 64)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
