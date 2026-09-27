"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--schedule-steps', type=int, default=None,
                   help='Optional full cosine horizon, for a matched truncated ablation.')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accumulation-steps', type=int, default=1)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--checkpoint-at',type=int,nargs='*',default=[75,150],
                   help='Extra checkpoint steps for equal-target controls, when reached.')
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--min-lr-ratio', type=float, default=0.1,
                   help='Final cosine learning-rate floor relative to peak.')
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--decay-matrices-only', action='store_true')
    p.add_argument('--optimizer', choices=['adamw','muon'], default='adamw')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    schedule_steps = args.schedule_steps or args.steps
    if schedule_steps < args.steps:
        p.error('Schedule horizon must be at least the number of training steps.')
    if not 0. <= args.min_lr_ratio <= 1.:
        p.error('min-lr-ratio must be between zero and one.')
    if args.steps < 1 or args.batch_size < 1 or args.accumulation_steps < 1 or args.log_every < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            p.error('MPS training requires an available Apple GPU.')
        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision('highest')
        device, precision = torch.device('mps'), ('bf16' if args.precision == 'bf16' else 'fp32')
    else:
        device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    parameters = model.parameters()
    if args.decay_matrices_only:
        parameters = [{'params':[p for p in model.parameters() if p.ndim>=2], 'weight_decay':args.weight_decay},
                      {'params':[p for p in model.parameters() if p.ndim<2], 'weight_decay':0.0}]
    if args.optimizer == 'muon':
        from muon_local import MuonAdamW
        matrices=[p for name,p in model.named_parameters() if name.startswith('blocks.') and p.ndim==2]
        matrix_ids={id(p) for p in matrices}
        auxiliary=[p for p in model.parameters() if id(p) not in matrix_ids]
        optimizer=MuonAdamW([
            {'params':matrices,'use_muon':True,'lr_scale':10.0,'weight_decay':args.weight_decay/10},
            {'params':[p for p in auxiliary if p.ndim>=2],'weight_decay':args.weight_decay},
            {'params':[p for p in auxiliary if p.ndim<2],'weight_decay':0.0}])
    else:
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best_bpb, best_step = float('inf'), None

    def validate():
        if device.type != 'mps':
            return score(model,*data['validation'],device,'fp32')
        # The fixed scorer uses float64 reductions, unavailable on MPS.
        # Evaluate a separate CPU copy; never move optimizer parameters around.
        torch.mps.synchronize()
        cpu_model, _ = make_model(args.implementation, config, torch.device('cpu'))
        cpu_model.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()})
        return score(cpu_model,*data['validation'],torch.device('cpu'),'fp32')

    def save_checkpoint(path, completed_steps):
        torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                    'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'seed':args.seed,'train_tokens':completed_steps*args.batch_size*256*args.accumulation_steps,
                    'steps':completed_steps,'schedule_steps':schedule_steps,
                    'min_lr_ratio':args.min_lr_ratio},path)

    for step in range(args.steps):
        learning_rate = args.learning_rate * min(1.,(step+1)/max(1,args.warmup_steps)) * (args.min_lr_ratio+(1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*step/schedule_steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate*group.get('lr_scale',1.0)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.
        for microstep in range(args.accumulation_steps):
            starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
            batch = tokens[starts[:,None]+torch.arange(257,device=device)]
            with autocast(device, precision):
                loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
            (loss/args.accumulation_steps).backward()
            accumulated_loss += loss.detach().item()/args.accumulation_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%args.log_every == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':accumulated_loss,'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if step+1 in args.checkpoint_at and not (args.eval_every>0 and (step+1)%args.eval_every==0):
            save_checkpoint(args.run_dir/f'checkpoint-step-{step+1}.pt',step+1)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            validation_started = time.perf_counter()
            intermediate = validate()
            intermediate.pop('window_nll_nats')
            validation_history.append({'step':step+1,**intermediate})
            save_checkpoint(args.run_dir/f'checkpoint-step-{step+1}.pt', step+1)
            if intermediate['bpb'] < best_bpb:
                best_bpb, best_step = intermediate['bpb'], step+1
                save_checkpoint(args.run_dir/'best.pt', step+1)
            intermediate_validation_seconds += time.perf_counter()-validation_started
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    if device.type == 'mps':
        torch.mps.synchronize()
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = validate()
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    save_checkpoint(checkpoint, args.steps)
    if validation['bpb'] < best_bpb:
        best_bpb, best_step = validation['bpb'], args.steps
        save_checkpoint(args.run_dir/'best.pt', args.steps)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'steps':args.steps,'batch_size':args.batch_size,'schedule_steps':schedule_steps,
              'accumulation_steps':args.accumulation_steps,
              'effective_batch_size':args.batch_size*args.accumulation_steps,
              'extra_checkpoint_steps':[s for s in args.checkpoint_at if 1<=s<=args.steps],
              'learning_rate':args.learning_rate,'min_lr_ratio':args.min_lr_ratio,'warmup_steps':args.warmup_steps,
              'weight_decay':args.weight_decay,'decay_matrices_only':args.decay_matrices_only,
              'optimizer':args.optimizer,
              'best_validation_bpb':best_bpb,'best_step':best_step,
              'trainer_sha256':sha(Path(__file__)),
              'train_tokens':args.steps*args.batch_size*256*args.accumulation_steps,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    if device.type == 'mps':
        result.update(device_name='Apple GPU (MPS)', validation_device='cpu',
                      mps_current_allocated_gb=torch.mps.current_allocated_memory()/1e9)
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
