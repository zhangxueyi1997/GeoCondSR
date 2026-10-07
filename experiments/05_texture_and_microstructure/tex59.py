# -*- coding: utf-8 -*-
"""纹理保真统计（只推理）：每个测试折终评块序前 60 块（与 micro59 同一批）。方法：细扫、三线性、EDSR-3D、SRGAN-3D、本文（只用粗扫，_n54b）。
  细节带 d = v − lowpass(v, 31 μm)：能量 mean(d²)；
  全体积径向功率谱（去均值、Hann 窗），0–0.5 周/体素分 40 箱，逐块求平均；
  云南折第 1 块：细节带三维功率沿 kz 求和（112²），用于展示周期性伪影。
输出 outputs/eval59/tex59.npz。"""
import sys, json, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, Generator, N_C, lowpass, CUT1_UM
from models54 import load_mean
from dataset import PairDataset, load_H
from train_edsr import EDSR3D
dev = 'cuda:0'; ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT)
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev); NB = 60
fr = torch.fft.fftfreq(112, device=dev); KX, KY, KZ = torch.meshgrid(fr, fr, fr, indexing='ij'); KR = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
EDG = torch.linspace(0, 0.5, 41, device=dev); IDX = torch.bucketize(KR.flatten(), EDG) - 1
w1 = torch.hann_window(112, periodic=False, device=dev); W3 = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


def rpsd(v):
    p = (torch.fft.fftn((v - v.mean()) * W3).abs() ** 2).flatten()
    ok = (IDX >= 0) & (IDX < 40)
    s = torch.zeros(40, device=dev).index_add_(0, IDX[ok], p[ok]); c = torch.zeros(40, device=dev).index_add_(0, IDX[ok], torch.ones_like(p[ok]))
    return (s / c.clamp(min=1)).cpu().numpy()


OUT = {}
with torch.no_grad():
    for FOLD in ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']:
        meta = json.load(open('%s/%s_proj/meta.json' % (RUNS, FOLD)))
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
        E = {}; S = {}
        for q, j in enumerate(J):
            b = ds[int(j)]; lr = torch.from_numpy(b['lr'])[None].to(dev); sl = slice(10, 26)
            V = {'细扫': torch.from_numpy(b['hr'][0]).to(dev), '三线性': F.interpolate(lr[..., sl, sl, sl], size=(112,) * 3, mode='trilinear', align_corners=False)[0, 0]}
            for nm, m in base.items(): V[nm] = proj(m(lr), lr)[0, 0]
            mu, feat = M(lr, ret_feat=True); torch.manual_seed(7 + q); V['本文'] = proj(G(mu, cconst, a_feat=feat)[2], lr)[0, 0]
            for nm, v in V.items():
                d = v - lowpass(v[None, None], CUT1_UM)[0, 0]
                E.setdefault(nm, []).append(float((d ** 2).mean())); S.setdefault(nm, []).append(rpsd(v))
                if FOLD == 'YN' and q == 0:
                    OUT['spec|%s' % nm] = torch.fft.fftshift(torch.fft.fftn(d).abs() ** 2).sum(0).cpu().numpy().astype(np.float32)
        for nm in E:
            OUT['E|%s|%s' % (FOLD, nm)] = np.array(E[nm]); OUT['P|%s|%s' % (FOLD, nm)] = np.mean(S[nm], 0)
        print(FOLD, ' '.join('%s 细节能量比 %.3f' % (nm, np.mean(np.array(E[nm]) / np.array(E['细扫']))) for nm in E), flush=True)
OUT['kbins'] = ((EDG[:-1] + EDG[1:]) / 2).cpu().numpy()
np.savez_compressed(str(OUT_ROOT) + '/eval59/tex59.npz', **OUT)
print('DONE', flush=True)