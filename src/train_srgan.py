# -*- coding: utf-8 -*-
"""外部基线：SRGAN-3D —— 生成式超分的 GAN 派标准基线。
生成器与 EDSR-3D **完全相同**（从 train_edsr 导入），两个基线的差别只在于有无对抗训练。
判别器：3D PatchGAN（112³ → 7³ 逐块判别），谱归一化，hinge 损失。
训练（ESRGAN 标准做法）：前 WARM 步纯 L1 预热，之后 L1 + LAMBDA·对抗项。
与本项目模型及 EDSR-3D 同数据、同折、同 seed、同 30000 步、同批大小、同 G 优化器
（AdamW 2e-4、wd 1e-4、恒定学习率、混合精度、无增广）。不使用任何地质变量。
用法：python train_srgan.py --fold CQ --seed 101 --gpu 0 --out runs/CQ_srgan"""
import sys, json, time, argparse, signal, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from dataset import PairDataset
from train_edsr import EDSR3D
from torch.utils.data import DataLoader

SN = nn.utils.spectral_norm


class PatchD3D(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        s.f = nn.Sequential(
            SN(nn.Conv3d(1, c, 4, 2, 1)), nn.LeakyReLU(0.2, True),          # 56³
            SN(nn.Conv3d(c, 2 * c, 4, 2, 1)), nn.LeakyReLU(0.2, True),      # 28³
            SN(nn.Conv3d(2 * c, 4 * c, 4, 2, 1)), nn.LeakyReLU(0.2, True),  # 14³
            SN(nn.Conv3d(4 * c, 8 * c, 4, 2, 1)), nn.LeakyReLU(0.2, True),  # 7³
            SN(nn.Conv3d(8 * c, 1, 3, 1, 1)))

    def forward(s, x):
        return s.f(x)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fold', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gpu', type=int, default=0); p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=30000); p.add_argument('--bs', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--lr_d', type=float, default=1e-4)
    p.add_argument('--warm', type=int, default=5000); p.add_argument('--lam', type=float, default=5e-3)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--resume', default='')
    cfg = p.parse_args()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    dev = 'cuda:%d' % cfg.gpu; out = Path(cfg.out); (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    fm = fold_meta(cfg.fold)
    ds = PairDataset(str(DATA_ROOT), fm['train'], unit_split='train', c_src='blk')
    g = torch.Generator(); g.manual_seed(cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers, drop_last=True,
                    pin_memory=True, persistent_workers=True, generator=g)
    G = EDSR3D().to(dev); D = PatchD3D().to(dev)
    optG = torch.optim.AdamW(G.parameters(), lr=cfg.lr, weight_decay=1e-4)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.99))
    scaler = torch.amp.GradScaler('cuda')
    STOP = {'f': False}
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__('f', True))
    step = 0
    cks = sorted((out / 'ckpt').glob('step*.pt'))
    if cfg.resume == 'auto' and cks:
        d = torch.load(cks[-1], map_location=dev, weights_only=False)
        G.load_state_dict(d['model']); D.load_state_dict(d['D'])
        optG.load_state_dict(d['opt']); optD.load_state_dict(d['optD']); step = d['step']
        print('从 %s 续跑' % cks[-1].name, flush=True)
    meta = dict(fold=cfg.fold, train=fm['train'], test=fm['test'], mode='srgan3d', cfg=vars(cfg),
                params=sum(q.numel() for q in G.parameters()), params_D=sum(q.numel() for q in D.parameters()))
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print('SRGAN-3D  G %.2f M  D %.2f M' % (meta['params'] / 1e6, meta['params_D'] / 1e6), flush=True)
    logf = open(out / 'log.jsonl', 'a'); t0 = time.time(); it = iter(dl)

    def save(s_):
        tmp = out / 'ckpt' / ('step%07d.pt.tmp' % s_)
        torch.save(dict(model=G.state_dict(), D=D.state_dict(), opt=optG.state_dict(), optD=optD.state_dict(),
                        step=s_, cfg=vars(cfg)), tmp)
        tmp.rename(out / 'ckpt' / ('step%07d.pt' % s_))

    while step < cfg.steps and not STOP['f']:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        lr_ = b['lr'].to(dev, non_blocking=True); hr_ = b['hr'].to(dev, non_blocking=True).float()
        adv = step >= cfg.warm
        with torch.amp.autocast('cuda'):
            fake = G(lr_).float()
        ld = torch.zeros((), device=dev); lg = torch.zeros((), device=dev)
        if adv:                                     # 判别器（全精度，谱归一化在半精度下不稳）
            ld = F.relu(1 - D(hr_)).mean() + F.relu(1 + D(fake.detach())).mean()
            optD.zero_grad(set_to_none=True); ld.backward(); optD.step()
        l1 = F.l1_loss(fake, hr_)
        if adv:
            lg = -D(fake).mean()
        loss = l1 + (cfg.lam * lg if adv else 0.0)
        optG.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(optG); scaler.update()
        step += 1
        if step % 20 == 0:
            logf.write(json.dumps(dict(step=step, l1=float(l1.detach()), d_loss=float(ld.detach()),
                                       g_adv=float(lg.detach()), elapsed_min=(time.time() - t0) / 60,
                                       mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9)) + '\n'); logf.flush()
            (out / 'heartbeat.json').write_text(json.dumps(dict(step=step, loss=float(l1.detach()))))
        if step % 500 == 0:
            save(step)
    save(step)
    print('已存最终检查点 step%07d.pt（共 %d 步）' % (step, step), flush=True)
