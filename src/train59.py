# -*- coding: utf-8 -*-
"""第 59 步：训练近年强基线（SwinIR-3D / 三维条件扩散）。与 EDSR-3D、SRGAN-3D 同数据、同折、同 seed、同批大小、同优化器族、无增广、无地质。
SwinIR-3D：L1，30 000 步，AdamW 2e-4、wd 1e-4、恒定学习率（同 EDSR-3D）；bf16 混合精度
  （第一版用 fp16，CQ 折第 3520 步起前向溢出、此后全为 NaN，故改 bf16——动态范围同 fp32；SC 折首版未出现 NaN，保留 fp16 结果。配方其余相同）。
扩散：ε 预测 MSE，40 000 步（扩散模型通常需要更多步数），AdamW 1e-4、wd 0，梯度裁剪 1.0，权重 EMA 0.999（推理用 EMA）。
用法：python train59.py --model swin|diff --fold CQ --seed 101 --gpu 0 --out runs/CQ_swin"""
import sys, json, time, argparse, signal, random, copy
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from dataset import PairDataset
from models59 import SwinIR3D, DiffSR3D
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument('--model', required=True); p.add_argument('--fold', required=True); p.add_argument('--seed', type=int, required=True)
p.add_argument('--gpu', type=int, default=0); p.add_argument('--out', required=True); p.add_argument('--steps', type=int, default=0)
p.add_argument('--bs', type=int, default=4); p.add_argument('--workers', type=int, default=4); p.add_argument('--resume', default='auto')
cfg = p.parse_args()
if not cfg.steps: cfg.steps = 30000 if cfg.model == 'swin' else 40000
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = torch.device('cuda:%d' % cfg.gpu); torch.cuda.set_device(dev); out = Path(cfg.out); (out / 'ckpt').mkdir(parents=True, exist_ok=True)
fm = fold_meta(cfg.fold)
ds = PairDataset(str(DATA_ROOT), fm['train'], unit_split='train', c_src='blk')
g = torch.Generator(); g.manual_seed(cfg.seed)
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers, drop_last=True, pin_memory=True, persistent_workers=True, generator=g)
if cfg.model == 'swin':
    model = SwinIR3D().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); ema = None
else:
    model = DiffSR3D().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    ema = copy.deepcopy(model).eval(); [q.requires_grad_(False) for q in ema.parameters()]
AMP = torch.bfloat16 if cfg.model == 'swin' else torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=(cfg.model != 'swin')); STOP = {'f': False}; BAD = {'n': 0}
signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__('f', True))
step = 0; cks = sorted((out / 'ckpt').glob('step*.pt'))
if cfg.resume == 'auto' and cks:
    d = torch.load(cks[-1], map_location=dev, weights_only=False)
    model.load_state_dict(d['model']); opt.load_state_dict(d['opt']); step = d['step']
    if ema is not None: ema.load_state_dict(d['ema'])
    print('从 %s 续跑' % cks[-1].name, flush=True)
meta = dict(fold=cfg.fold, train=fm['train'], test=fm['test'], mode=cfg.model, cfg=vars(cfg), params=sum(q.numel() for q in model.parameters()))
(out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print('%s 参数 %.2f M，%d 步' % (cfg.model, meta['params'] / 1e6, cfg.steps), flush=True)
logf = open(out / 'log.jsonl', 'a'); t0 = time.time(); it = iter(dl)


def save(s_):
    tmp = out / 'ckpt' / ('step%07d.pt.tmp' % s_)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), ema=(ema.state_dict() if ema is not None else None), step=s_, cfg=vars(cfg)), tmp)
    tmp.rename(out / 'ckpt' / ('step%07d.pt' % s_))
    for old in sorted((out / 'ckpt').glob('step*.pt'))[:-2]: old.unlink()


while step < cfg.steps and not STOP['f']:
    try: b = next(it)
    except StopIteration: it = iter(dl); b = next(it)
    lr_ = b['lr'].to(dev, non_blocking=True); hr_ = b['hr'].to(dev, non_blocking=True).float()
    with torch.amp.autocast('cuda', dtype=AMP):
        loss = F.l1_loss(model(lr_).float(), hr_) if cfg.model == 'swin' else model.loss(lr_, hr_)
    if not torch.isfinite(loss):                       # 非有限值保护：跳过该步；连续 50 步则报错退出（不产出 final.pt）
        BAD['n'] += 1; opt.zero_grad(set_to_none=True)
        if BAD['n'] >= 50: print('连续 50 步 loss 非有限，第 %d 步退出' % step, flush=True); sys.exit(3)
        continue
    BAD['n'] = 0
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    if cfg.model == 'diff': scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); step += 1
    if ema is not None:
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
    if step % 20 == 0:
        logf.write(json.dumps(dict(step=step, loss=float(loss), elapsed_min=(time.time() - t0) / 60, mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9)) + '\n'); logf.flush()
        (out / 'heartbeat.json').write_text(json.dumps(dict(step=step, loss=float(loss))))
    if step % 500 == 0: save(step)
save(step)
if step >= cfg.steps:
    (out / 'ckpt' / ('step%07d.pt' % step)).rename(out / 'ckpt' / 'final.pt')
print('结束于第 %d 步（%.1f 分钟）' % (step, (time.time() - t0) / 60), flush=True)
