# -*- coding: utf-8 -*-
"""三维孔隙结构对比图的体数据（只推理，不训练）。块的选取与 eval56/vis_paper.py 相同：每个测试折终评块序第 1 块（rng 0，未经挑选）。
方法：三线性、EDSR-3D、SRGAN-3D、本文无地质（_n54b）、本文粗扫+稀疏细扫（_g54b）、真值；均做 8 轮硬投影（三线性除外），噪声种子 7。
输出 float16 的 112³ 体，存 outputs/eval59/vis3d59.npz。"""
import sys, json, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, Generator, N_C
from models54 import load_mean, load_cpred
from dataset import PairDataset, load_H
from train_edsr import EDSR3D
dev = 'cuda:0'; ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT)
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev)


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


def load_g(f, arm, M):
    G = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
    G.load_state_dict(torch.load('%s/%s%s/ckpt/final.pt' % (RUNS, f, arm), map_location=dev, weights_only=False)['model']); return G


def gen(M, G, lr, c):
    mu, feat = M(lr, ret_feat=True); torch.manual_seed(7); return G(mu, c, a_feat=feat)[2]


OUT = {}
with torch.no_grad():
    for FOLD in ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']:
        meta = json.load(open('%s/%s_proj/meta.json' % (RUNS, FOLD)))
        ds = PairDataset(ROOT, meta['test'], c_src='blk')
        j = int(np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)[0])
        b = ds[j]; g, f = ds.items[j]; cell = np.load(f)['cell'][:3].astype(int)
        lr = torch.from_numpy(b['lr'])[None].to(dev)
        M = load_mean('%s/%s_m54b/ckpt/final.pt' % (RUNS, FOLD), dev).eval()
        Gn, Gb = load_g(FOLD, '_n54b', M), load_g(FOLD, '_g54b', M); CP = load_cpred('%s/%s_cpred/final.pt' % (RUNS, FOLD), dev)
        cf = np.load('%s/cfield2/%s.npz' % (ROOT, g)); js = [k for k, (gg, _) in enumerate(ds.items) if gg == g]
        cl = np.array([np.load(ds.items[k][1])['cell'][:3].astype(int) for k in js]); cbs = cf['c_blk'][tuple(cl.T)].astype(np.float64)
        ok = cf['valid'][tuple(cl.T)] & np.isfinite(cbs).all(1)
        chs = np.concatenate([CP(torch.from_numpy(np.stack([ds[k]['lr'] for k in js[s:s + 32]])).to(dev)).cpu().numpy() for s in range(0, len(js), 32)]).astype(np.float64)
        cl, cbs, chs = cl[ok], cbs[ok], chs[ok]
        far = np.where(np.abs(cl - cell).max(1) >= 3)[0]; far = np.random.default_rng(5).choice(far, min(10, len(far)), replace=False)
        ch = CP(lr)[0].cpu().numpy().astype(np.float64); ccal = torch.from_numpy((ch + (cbs[far] - chs[far]).mean(0)).astype(np.float32))[None].to(dev)
        dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
        kk = np.random.default_rng(1).choice(len(dtr), 300, replace=False)
        cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)
        k = 16; o = (36 - k) // 2; sl = slice(o, o + k)
        V = {'三线性': F.interpolate(lr[..., sl, sl, sl], size=(112,) * 3, mode='trilinear', align_corners=False)}
        for nm, suf in (('EDSR-3D', '_edsr'), ('SRGAN-3D', '_srgan')):
            m = EDSR3D().to(dev).eval(); m.load_state_dict(torch.load('%s/%s%s/ckpt/step0030000.pt' % (RUNS, FOLD, suf), map_location=dev, weights_only=False)['model']); V[nm] = proj(m(lr), lr)
        V['本文·无地质'] = proj(gen(M, Gn, lr, cconst), lr)
        V['本文·粗扫+稀疏细扫'] = proj(gen(M, Gb, lr, ccal), lr)
        V['真值'] = torch.from_numpy(b['hr'][0]).to(dev)[None, None]
        OUT['%s|粗扫' % FOLD] = b['lr'][0][sl, sl, sl].astype(np.float16)
        for nm, v in V.items():
            vv = v[0, 0].float().cpu().numpy()
            OUT['%s|%s' % (FOLD, nm)] = vv.astype(np.float16)
            print(FOLD, g, nm, 'phi(<0.5) %.4f' % (vv < 0.5).mean(), flush=True)
        OUT['%s|meta' % FOLD] = np.array([j, *cell]); OUT['%s|gid' % FOLD] = np.array(g)
np.savez_compressed(str(OUT_ROOT) + '/eval59/vis3d59.npz', **OUT)
print('DONE')