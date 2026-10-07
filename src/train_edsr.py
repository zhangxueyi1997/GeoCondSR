# -*- coding: utf-8 -*-
"""外部基线：EDSR-3D（Lim et al. 2017 的三维版），纯 L1 训练 —— 数字岩心超分文献中最常用的 CNN 基线族。
与本项目模型**同数据、同折、同 seed、同步数、同批大小、同优化器**：
  AdamW lr 2e-4、wd 1e-4、恒定学习率、30000 步、bs 4、混合精度、无增广。
结构：LR 36³（含外围上下文）上 16 个残差块 → 取中心 16³ 特征 → 三线性上采样到 112³ →
     两层 HR 卷积，作为残差叠加在三线性上采样的 LR 上（VDSR 式全局残差）。
不使用任何地质变量。评测时与本项目模型一样做 8 轮硬 H 投影（数据一致性），保证可比。
用法：python train_edsr.py --fold CQ --seed 101 --gpu 0 --out runs/CQ_edsr"""
import sys, json, time, argparse, signal, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from dataset import PairDataset
from torch.utils.data import DataLoader


class RB(nn.Module):
    def __init__(s, c):
        super().__init__(); s.a = nn.Conv3d(c, c, 3, padding=1); s.b = nn.Conv3d(c, c, 3, padding=1)

    def forward(s, x):
        return x + 0.1 * s.b(F.relu(s.a(x)))


class EDSR3D(nn.Module):
    def __init__(s, c=64, nb=16, hc=32, hr=112, k=16):
        super().__init__()
        s.hr, s.k = hr, k
        s.head = nn.Conv3d(1, c, 3, padding=1)
        s.body = nn.Sequential(*[RB(c) for _ in range(nb)], nn.Conv3d(c, c, 3, padding=1))
        s.up = nn.Conv3d(c, hc, 3, padding=1)
        s.tail = nn.Sequential(nn.Conv3d(hc, hc, 3, padding=1), nn.ReLU(), nn.Conv3d(hc, 1, 3, padding=1))

    def forward(s, lr):
        x = s.head(lr); x = x + s.body(x)
        off = (lr.shape[-1] - s.k) // 2; sl = slice(off, off + s.k)
        x = x[..., sl, sl, sl]
        base = F.interpolate(lr[..., sl, sl, sl], size=(s.hr,) * 3, mode='trilinear', align_corners=False)
        h = F.interpolate(s.up(x), size=(s.hr,) * 3, mode='trilinear', align_corners=False)
        return base + s.tail(F.relu(h))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fold', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gpu', type=int, default=0); p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=30000); p.add_argument('--bs', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', default='')
    cfg = p.parse_args()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    dev = 'cuda:%d' % cfg.gpu; out = Path(cfg.out); (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    fm = fold_meta(cfg.fold)   # 与本项目模型同一折划分
    ds = PairDataset(str(DATA_ROOT), fm['train'], unit_split='train', c_src='blk')
    g = torch.Generator(); g.manual_seed(cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers, drop_last=True,
                    pin_memory=True, persistent_workers=True, generator=g)
    model = EDSR3D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    STOP = {'f': False}
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__('f', True))
    step = 0
    cks = sorted((out / 'ckpt').glob('step*.pt'))
    if cfg.resume == 'auto' and cks:
        d = torch.load(cks[-1], map_location=dev, weights_only=False)
        model.load_state_dict(d['model']); opt.load_state_dict(d['opt']); step = d['step']
        print('从 %s 续跑' % cks[-1].name, flush=True)
    meta = dict(fold=cfg.fold, train=fm['train'], test=fm['test'], mode='edsr3d', cfg=vars(cfg),
                params=sum(q.numel() for q in model.parameters()))
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print('EDSR-3D 参数 %.2f M' % (meta['params'] / 1e6), flush=True)
    logf = open(out / 'log.jsonl', 'a'); t0 = time.time(); it = iter(dl)

    def save(tag_step):
        tmp = out / 'ckpt' / ('step%07d.pt.tmp' % tag_step)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=tag_step, cfg=vars(cfg)), tmp)
        tmp.rename(out / 'ckpt' / ('step%07d.pt' % tag_step))

    while step < cfg.steps and not STOP['f']:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        lr_ = b['lr'].to(dev, non_blocking=True); hr_ = b['hr'].to(dev, non_blocking=True)
        with torch.amp.autocast('cuda'):
            loss = F.l1_loss(model(lr_).float(), hr_.float())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1
        if step % 20 == 0:
            el = (time.time() - t0) / 60
            logf.write(json.dumps(dict(step=step, l1=float(loss), elapsed_min=el,
                                       mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9)) + '\n'); logf.flush()
            (out / 'heartbeat.json').write_text(json.dumps(dict(step=step, loss=float(loss))))
        if step % 500 == 0:
            save(step)
    save(step)
    print('已存最终检查点 step%07d.pt（共 %d 步）' % (step, step), flush=True)
