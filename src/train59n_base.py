# -*- coding: utf-8 -*-
"""Degradation-aware fine-tuning of the EDSR-3D and SRGAN-3D baselines for whole-plug scans.

The recipe is the one used for the mean path in train59n.py (stage 1): start from the final baseline weights
and add noise with the whole-plug noise spectrum to the coarse input,
  N(f) = median over the training-lithology pairs of (P_whole-plug coarse - P_miniplug coarse), set to 0 below 0.15 cycles/voxel,
  amplitude a ~ U(0, 1.3) per sample; the target stays the clean fine scan;
6000 steps, batch size 4, a new AdamW optimizer with learning rate 1e-4 and weight decay 1e-4, no augmentation,
mixed precision, and the same seed per fold as the m59n run.
Each baseline keeps its own loss: EDSR-3D pure L1; SRGAN-3D L1 + 0.005 * hinge adversarial loss
(past its warm-up), with the discriminator and its optimizer restored from the baseline checkpoint.

Usage: python train59n_base.py --kind edsr --fold CQ --seed 101 --gpu 0 \
           --init_ckpt runs/CQ_edsr/ckpt/step0030000.pt --out runs/CQ_edsr59n
"""
import sys, json, time, argparse, signal, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from paths import DATA_ROOT, fold_meta  # noqa: E402
from dataset import PairDataset
from train import _mk_winit
from train_edsr import EDSR3D
from train_srgan import PatchD3D
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument('--kind', required=True, choices=['edsr', 'srgan']); p.add_argument('--fold', required=True)
p.add_argument('--seed', type=int, required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--out', required=True)
p.add_argument('--init_ckpt', required=True)
p.add_argument('--steps', type=int, default=6000); p.add_argument('--bs', type=int, default=4); p.add_argument('--workers', type=int, default=4)
p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--lam', type=float, default=5e-3)
p.add_argument('--noise_amp', type=float, default=1.3); p.add_argument('--dom', default=str(DATA_ROOT) + '/plugs/dom59.json')
cfg = p.parse_args()
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = torch.device('cuda:%d' % cfg.gpu); torch.cuda.set_device(dev)
out = Path(cfg.out); (out / 'ckpt').mkdir(parents=True, exist_ok=True)
STOP = {'f': False}; signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__('f', True)); signal.signal(signal.SIGINT, lambda *a: STOP.__setitem__('f', True))
fm = fold_meta(cfg.fold)            # same fold split as every other model
ds = PairDataset(str(DATA_ROOT), fm['train'], unit_split='train', aug=False, c_src='blk')
g_ = torch.Generator(); g_.manual_seed(cfg.seed)
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers, drop_last=True, pin_memory=True,
                persistent_workers=True, generator=g_, worker_init_fn=_mk_winit(cfg.seed))
scaler = torch.amp.GradScaler('cuda')
d0 = torch.load(cfg.init_ckpt, map_location=dev, weights_only=False)
G = EDSR3D().to(dev); G.load_state_dict(d0['model']); G.train()
optG = torch.optim.AdamW(G.parameters(), lr=cfg.lr, weight_decay=1e-4)
nets = dict(model=G, opt=optG)
if cfg.kind == 'srgan':
    D = PatchD3D().to(dev); D.load_state_dict(d0['D'])
    optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.99)); optD.load_state_dict(d0['optD'])
    nets.update(D=D, optD=optD)
del d0
# ---------- whole-plug noise (identical to train59n.py) ----------
_DOM = json.load(open(cfg.dom)); _n = 36; _w1 = np.hanning(_n); _W3 = _w1[:, None, None] * _w1[None, :, None] * _w1[None, None, :]
_f = np.fft.fftfreq(_n); _FR = np.sqrt(_f[:, None, None] ** 2 + _f[None, :, None] ** 2 + _f[None, None, :] ** 2)
_EDG = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.87]); _CEN = (_EDG[:-1] + _EDG[1:]) / 2
_TRG = [g for g in fm['train'] if g in _DOM]
NF = np.median([np.array(_DOM[g]['pl']) - np.array(_DOM[g]['ps']) for g in _TRG], 0); NF[_CEN < 0.15] = 0; NF = np.clip(NF, 0, None)
AMP = torch.from_numpy(np.interp(_FR, np.r_[0, _CEN], np.r_[0.0, np.sqrt(NF / ((_W3 ** 2).sum() / _W3.sum()))]).astype(np.float32)).to(dev)
print('noise spectrum from training samples', _TRG, np.round(NF * 1e3, 2), flush=True)


def add_noise(x):
    n = torch.fft.ifftn(torch.fft.fftn(torch.randn_like(x), dim=(-3, -2, -1)) * AMP, dim=(-3, -2, -1)).real
    return x + cfg.noise_amp * torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * n


step = 0
cks = sorted((out / 'ckpt').glob('step*.pt'))
if cks:
    d = torch.load(cks[-1], map_location=dev, weights_only=False)
    for k, v in nets.items(): v.load_state_dict(d[k])
    step = d['step']; print('resuming from %s' % cks[-1].name, flush=True)
meta = dict(fold=cfg.fold, train=fm['train'], test=fm['test'], kind=cfg.kind, cfg=vars(cfg),
            arch='degradation-aware fine-tuning of a baseline (whole-plug noise, recipe of m59n)', NF=NF.tolist(),
            params=sum(q.numel() for q in G.parameters()))
(out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print('%s-3D fine-tuning  fold %s  %.2fM parameters' % (cfg.kind.upper(), cfg.fold, meta['params'] / 1e6), flush=True)


def save(s_, final=False):
    tmp = out / 'ckpt' / 'tmp.pt'
    torch.save(dict({k: v.state_dict() for k, v in nets.items()}, step=s_, cfg=vars(cfg)), tmp)
    tmp.rename(out / 'ckpt' / ('final.pt' if final else 'step%07d.pt' % s_))
    for old in sorted((out / 'ckpt').glob('step*.pt'))[:-3]: old.unlink()


logf = open(out / 'log.jsonl', 'a'); t0 = time.time(); it = iter(dl)
while step < cfg.steps and not STOP['f']:
    try: b = next(it)
    except StopIteration: it = iter(dl); b = next(it)
    lr_ = b['lr'].to(dev, non_blocking=True).float(); hr_ = b['hr'].to(dev, non_blocking=True).float()
    lr_ = add_noise(lr_)
    with torch.amp.autocast('cuda'):
        fake = G(lr_).float()
    P = {}
    if cfg.kind == 'srgan':                                   # discriminator in full precision, as in the original training
        ld = F.relu(1 - D(hr_)).mean() + F.relu(1 + D(fake.detach())).mean()
        optD.zero_grad(set_to_none=True); ld.backward(); optD.step(); P['d_loss'] = float(ld.detach())
    l1 = F.l1_loss(fake, hr_); loss = l1
    if cfg.kind == 'srgan':
        lg = -D(fake).mean(); loss = l1 + cfg.lam * lg; P['g_adv'] = float(lg.detach())
    optG.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(optG); scaler.update()
    step += 1
    if step % 20 == 0:
        P.update(step=step, l1=float(l1.detach()), loss=float(loss.detach()), elapsed_min=(time.time() - t0) / 60,
                 mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9)
        logf.write(json.dumps(P, ensure_ascii=False) + '\n'); logf.flush()
        (out / 'heartbeat.json').write_text(json.dumps(dict(step=step, loss=P['loss'])))
    if step % 500 == 0: save(step)
save(step)
if step >= cfg.steps: save(step, final=True)
print('finished at step %d (%.1f min)' % (step, (time.time() - t0) / 60), flush=True)
