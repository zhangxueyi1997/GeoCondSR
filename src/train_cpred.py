# -*- coding: utf-8 -*-
"""第 55 步：训练「粗扫 → 地质变量 ĉ」预测器（每折只用训练岩性）。
标签 = 细扫实测 c_blk（IGV, f_pore, mu_inter, f_dense），仅训练时使用；推理只需粗扫。
报告：训练岩性留出块与测试岩性上逐变量 R²（测试岩性只报告，不参与任何选择）。
用法：python train_cpred.py --fold CQ --seed 101 --gpu 0 --out runs/CQ_cpred"""
import sys, json, argparse, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from dataset import PairDataset
from models54 import CPred
from train import _mk_winit
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument('--fold', required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--out', required=True); p.add_argument('--steps', type=int, default=6000); p.add_argument('--bs', type=int, default=16)
p.add_argument('--lr', type=float, default=1e-3)
cfg = p.parse_args()
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
dev = torch.device('cuda:%d' % cfg.gpu); out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
ROOT = str(DATA_ROOT); fm = fold_meta(cfg.fold)
ds = PairDataset(ROOT, fm['train'], unit_split='train', aug=True, c_src='blk')
kk = np.random.default_rng(1).choice(len(ds), min(600, len(ds)), replace=False)
C = np.stack([ds[int(i)]['c'] for i in kk]).astype(np.float64); mu, sd = C.mean(0), C.std(0) + 1e-6
m = CPred(mu, sd).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
g_ = torch.Generator(); g_.manual_seed(cfg.seed)
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=4, drop_last=True, generator=g_, worker_init_fn=_mk_winit(cfg.seed), persistent_workers=True)
it = iter(dl); t0 = time.time(); step = 0
while step < cfg.steps:
    try: b = next(it)
    except StopIteration: it = iter(dl); b = next(it)
    lr_ = b['lr'].to(dev); c = b['c'].to(dev).float()
    loss = F.mse_loss(m.e(lr_).float(), (c - m.mu) / m.sd)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
    if step % 500 == 0: print('step %d  标准化 MSE %.4f' % (step, float(loss)), flush=True)


def r2(split_ds, n=300):
    idx = np.random.default_rng(0).choice(len(split_ds), min(n, len(split_ds)), replace=False)
    P, T = [], []
    with torch.no_grad():
        for j in idx:
            bb = split_ds[int(j)]; P.append(m(torch.from_numpy(bb['lr'])[None].to(dev))[0].cpu().numpy()); T.append(bb['c'])
    P, T = np.array(P), np.array(T)
    return [float(1 - ((P[:, k] - T[:, k]) ** 2).sum() / max(((T[:, k] - T[:, k].mean()) ** 2).sum(), 1e-12)) for k in range(T.shape[1])]


m.eval()
r_held = r2(PairDataset(ROOT, fm['train'], unit_split='heldout', c_src='blk'))
r_test = r2(PairDataset(ROOT, fm['test'], c_src='blk'))
torch.save(dict(model=m.state_dict(), r2_heldout=r_held, r2_test=r_test, cfg=vars(cfg)), out / 'final.pt')
json.dump(dict(r2_heldout=r_held, r2_test=r_test, names=['IGV', 'f_pore', 'mu_inter', 'f_dense']), open(out / 'r2.json', 'w'), ensure_ascii=False)
print('完成 %s：R² 训练岩性留出 %s | 测试岩性 %s（%.1f 分钟）' % (cfg.fold, np.round(r_held, 3).tolist(), np.round(r_test, 3).tolist(), (time.time() - t0) / 60), flush=True)
