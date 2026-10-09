# -*- coding: utf-8 -*-
"""整柱应用图的示例切片（只推理）：CQ-1（G01 A 柱，重庆折）前 8 个窗口，按 plugsr59.py 的“标定”流程（近表面空气 + 杯状校正）处理后，
用本折模型超分：三线性、EDSR-3D、SRGAN-3D、本文退化感知微调版（_m59n/_n59n，定稿流程）。保存中心切片与孔隙率。"""
import sys, json, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, Generator, N_C
from models54 import load_mean
from dataset import PairDataset, load_H
from train_edsr import EDSR3D
B_SUF = os.environ.get('B_SUF', '')   # baseline weights: empty = original checkpoints; 59n = degradation-aware fine-tuned
dev = 'cuda:0'; ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT); PD = str(DATA_ROOT) + '/plugs/'; FOLD = 'CQ'; PID = 'CQ-1'
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev); N = 112


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


meta = fold_meta(FOLD)
dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
kk = np.random.default_rng(1).choice(len(dtr), min(300, len(dtr)), replace=False)
cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)
M = load_mean('%s/%s_m59n/ckpt/final.pt' % (RUNS, FOLD), dev).eval()
Gn = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
Gn.load_state_dict(torch.load('%s/%s_n59n/ckpt/final.pt' % (RUNS, FOLD), map_location=dev, weights_only=False)['model'])
base = {}
for nm, suf in (('EDSR-3D', '_edsr'), ('SRGAN-3D', '_srgan')):
    m = EDSR3D().to(dev).eval(); m.load_state_dict(torch.load('%s/%s%s%s/ckpt/%s' % (RUNS, FOLD, suf, B_SUF, 'final.pt' if B_SUF else 'step0030000.pt'), map_location=dev, weights_only=False)['model']); base[nm] = m
d = np.load(PD + '%s.npz' % PID); W = d['win'][:8]
c = np.load(PD + 'cup/%s.npz' % PID); a0, D0, an = float(c['air']), float(c['D']), float(c['air_near']); Dn = a0 + D0 - an
oy, ox = np.mgrid[-18:18, -18:18]
G = np.stack([np.polyval(c['coef'], (np.hypot(y + oy - c['cy'], x + ox - c['cx']) / c['R']) ** 2) for _, y, x in c['pos'][:8]])
Gn_ = (G * D0 + a0 - an) / Dn
Wc = (((W.astype(np.float32) * D0 + a0 - an) / Dn) / Gn_[:, None]).astype(np.float32)
OUT = {'win_raw_mid': W[:, 18].astype(np.float32), 'win_cal_mid': Wc[:, 18], 'pos': c['pos'][:8]}
with torch.no_grad():
    lr = torch.from_numpy(Wc)[:, None].to(dev); B = lr.shape[0]; k = 16; off = 10; sl = slice(off, off + k)
    O = {'三线性': F.interpolate(lr[..., sl, sl, sl], size=(N,) * 3, mode='trilinear', align_corners=False)}
    for nm, m in base.items(): O[nm] = proj(m(lr), lr)
    mu, feat = M(lr, ret_feat=True); torch.manual_seed(7); O['本文'] = proj(Gn(mu, cconst.expand(B, -1), a_feat=feat)[2], lr)
    for nm, o in O.items():
        OUT[nm] = o[:, 0, N // 2].cpu().numpy().astype(np.float32); OUT[nm + '|phi'] = (o[:, 0] < 0.5).float().mean(dim=(1, 2, 3)).cpu().numpy()
        print(nm, np.round(OUT[nm + '|phi'], 4), flush=True)
np.savez_compressed(str(OUT_ROOT) + '/plugs59/wc_vis59%s.npz' % (('_b' + B_SUF) if B_SUF else ''), **OUT)
print('DONE')