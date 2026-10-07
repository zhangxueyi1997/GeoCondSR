# -*- coding: utf-8 -*-
"""微结构统计曲线（只推理）：每个测试折终评块序前 60 块（rng 0，与终评同一批），方法：细扫真值、三线性、EDSR-3D、SRGAN-3D、本文（只用粗扫，_n54b）。
孔隙相 = u < 0.5（112³、2 μm）。统计：
  S2(r)：孔隙相两点概率，沿 z、y、x 三向平均，r = 0..40 体素；
  弦长分布：三个轴向上孔隙相连续段长度的计数（1..112 体素）；
  孔隙团：6 连通孔隙团的等效球直径，按体积加权的直方图（对数分箱）。
输出 outputs/eval59/micro59.json（逐折、逐方法汇总）。"""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
from scipy import ndimage as nd
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, Generator, N_C
from models54 import load_mean
from dataset import PairDataset, load_H
from train_edsr import EDSR3D
dev = 'cuda:0'; ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT)
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev); NB = 60; RMAX = 40
DB = np.logspace(np.log10(2.0), np.log10(120.0), 25)


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


def s2(m):
    m = m.astype(np.float32); out = np.zeros(RMAX + 1)
    for ax in range(3):
        n = m.shape[ax]
        for r in range(RMAX + 1):
            a = np.take(m, range(0, n - r), axis=ax); b = np.take(m, range(r, n), axis=ax); out[r] += (a * b).mean()
    return out / 3


def chords(m):
    h = np.zeros(113)
    for ax in range(3):
        v = np.moveaxis(m, ax, -1).reshape(-1, m.shape[ax]).astype(np.int8)
        p = np.pad(v, ((0, 0), (1, 1))); d = np.diff(p, axis=1)
        st = np.argwhere(d == 1); en = np.argwhere(d == -1)
        L = en[:, 1] - st[:, 1]; h += np.bincount(L, minlength=113)[:113]
    return h


def clusters(m):
    lab, n = nd.label(m); cnt = np.bincount(lab.ravel())[1:]
    d = 2 * (3 * cnt * 8.0 / (4 * np.pi)) ** (1 / 3)
    return np.histogram(d, bins=DB, weights=cnt)[0]


RES = {}
t0 = time.time()
with torch.no_grad():
    for FOLD in ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']:
        meta = fold_meta(FOLD)
        ds = PairDataset(ROOT, meta['test'], c_src='blk')
        J = np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)[:NB]
        M = load_mean('%s/%s_m54b/ckpt/final.pt' % (RUNS, FOLD), dev).eval()
        G = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
        G.load_state_dict(torch.load('%s/%s_n54b/ckpt/final.pt' % (RUNS, FOLD), map_location=dev, weights_only=False)['model'])
        base = {}
        for nm, suf in (('EDSR-3D', '_edsr'), ('SRGAN-3D', '_srgan')):
            m = EDSR3D().to(dev).eval(); m.load_state_dict(torch.load('%s/%s%s/ckpt/step0030000.pt' % (RUNS, FOLD, suf), map_location=dev, weights_only=False)['model']); base[nm] = m
        dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
        kk = np.random.default_rng(1).choice(len(dtr), 300, replace=False)
        cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)
        acc = {}
        for q, j in enumerate(J):
            b = ds[int(j)]; lr = torch.from_numpy(b['lr'])[None].to(dev); sl = slice(10, 26)
            V = {'细扫': b['hr'][0], '三线性': F.interpolate(lr[..., sl, sl, sl], size=(112,) * 3, mode='trilinear', align_corners=False)[0, 0].cpu().numpy()}
            for nm, m in base.items(): V[nm] = proj(m(lr), lr)[0, 0].cpu().numpy()
            mu, feat = M(lr, ret_feat=True); torch.manual_seed(7 + q); V['本文'] = proj(G(mu, cconst, a_feat=feat)[2], lr)[0, 0].cpu().numpy()
            for nm, v in V.items():
                mk = v < 0.5; a = acc.setdefault(nm, dict(phi=[], s2=np.zeros(RMAX + 1), ch=np.zeros(113), cl=np.zeros(len(DB) - 1)))
                a['phi'].append(float(mk.mean())); a['s2'] += s2(mk); a['ch'] += chords(mk); a['cl'] += clusters(mk)
        RES[FOLD] = {nm: dict(phi=a['phi'], s2=(a['s2'] / NB).tolist(), chords=a['ch'].tolist(), clusters=a['cl'].tolist()) for nm, a in acc.items()}
        RES['bins_um'] = DB.tolist()
        print(FOLD, '完成', ' '.join('%s φ %.4f' % (nm, np.mean(a['phi'])) for nm, a in acc.items()), '%.0f s' % (time.time() - t0), flush=True)
json.dump(RES, open(str(OUT_ROOT) + '/eval59/micro59.json', 'w'), ensure_ascii=False)
print('DONE', flush=True)