# -*- coding: utf-8 -*-
"""架构图与地质控制图的数据（只推理，不训练）。重庆折（CQ），有地质版模型 _g54b（训练时用实测 c）。
(1) 图 2 的 G01 同一块（配对文件 00174.npz）：粗扫、细扫、均值 mu、残差 r、投影输出 xhat 的三个可见面与中心层，c 取 G01 的样品整体状态。
(2) 地质控制：同一粗扫输入、同一噪声（种子 7），c 依次换成 14 个样品的整体状态（cfield2 全部有效格点均值），记录投影后输出的中心层、孔隙率、骨架窗以下平均灰度。
(3) 同样的替换在重庆折终评块序前 40 块上重复，记录各块输出孔隙率，看控制是否稳定。"""
import sys, json, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, Generator, N_C
from models54 import load_mean
from dataset import PairDataset, load_H
dev = 'cuda:0'; ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT); FOLD = 'CQ'
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev)
ALL = ['G01', 'G19', 'G05', 'G06', 'G07', 'G09', 'G11', 'G03', 'G16', 'G17', 'G02', 'G12', 'G13', 'G15']


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


def faces(v):
    v = np.asarray(v, np.float32)
    return dict(front=v[:, 0, :], top=v[-1, :, :], side=v[:, :, -1], mid=v[v.shape[0] // 2])


C = {}
for g in ALL:
    cf = np.load('%s/cfield2/%s.npz' % (ROOT, g)); cb = cf['c_blk'].astype(np.float64)
    ok = cf['valid'] & np.isfinite(cb).all(-1); C[g] = cb[ok].mean(0)
meta = json.load(open('%s/%s_proj/meta.json' % (RUNS, FOLD)))
ds = PairDataset(ROOT, meta['test'], c_src='blk')
j0 = [k for k, (g, f) in enumerate(ds.items) if g == 'G01' and str(f).endswith('00174.npz')][0]
M = load_mean('%s/%s_m54b/ckpt/final.pt' % (RUNS, FOLD), dev).eval()
G = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
G.load_state_dict(torch.load('%s/%s_g54b/ckpt/final.pt' % (RUNS, FOLD), map_location=dev, weights_only=False)['model'])
OUT = {'c_names': np.array(ALL), 'c_vals': np.stack([C[g] for g in ALL])}
with torch.no_grad():
    b = ds[j0]; lr = torch.from_numpy(b['lr'])[None].to(dev)
    mu, feat = M(lr, ret_feat=True)
    cg = torch.from_numpy(C['G01'].astype(np.float32))[None].to(dev)
    torch.manual_seed(7); out = G(mu, cg, a_feat=feat)[2]; xh = proj(out, lr)
    for nm, v in (('lr', b['lr'][0]), ('hr', b['hr'][0]), ('mu', mu[0, 0].cpu().numpy()), ('r', (out - mu)[0, 0].cpu().numpy()), ('xhat', xh[0, 0].cpu().numpy())):
        for fk, fv in faces(v).items(): OUT['arch|%s|%s' % (nm, fk)] = fv.astype(np.float16)
    print('arch 块', j0, 'phi hr %.4f xhat %.4f mu %.4f' % ((b['hr'][0] < 0.5).mean(), (xh < 0.5).float().mean().item(), (mu < 0.5).float().mean().item()), flush=True)
    for g in ALL:
        cc = torch.from_numpy(C[g].astype(np.float32))[None].to(dev)
        torch.manual_seed(7); o = proj(G(mu, cc, a_feat=feat)[2], lr)[0, 0]
        low = o[o < 0.52]
        OUT['ctrl|%s|mid' % g] = o[56].cpu().numpy().astype(np.float16)
        OUT['ctrl|%s|stat' % g] = np.array([(o < 0.5).float().mean().item(), low.mean().item() if low.numel() > 50 else np.nan])
        print('控制', g, 'c', np.round(C[g], 3), '输出孔隙率 %.4f  低灰度均值 %.3f' % tuple(OUT['ctrl|%s|stat' % g]), flush=True)
    OUT['ctrl|hr_phi'] = np.array([(b['hr'][0] < 0.5).mean()])
    # (3) 前 40 个终评块
    J = np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)[:40]
    P = np.zeros((len(J), len(ALL))); T = np.zeros(len(J))
    for s in range(0, len(J), 8):
        jj = J[s:s + 8]; lb = torch.from_numpy(np.stack([ds[int(j)]['lr'] for j in jj])).to(dev)
        T[s:s + len(jj)] = [(ds[int(j)]['hr'][0] < 0.5).mean() for j in jj]
        mu_, fe_ = M(lb, ret_feat=True)
        for q, g in enumerate(ALL):
            cc = torch.from_numpy(C[g].astype(np.float32))[None].to(dev).expand(len(jj), -1)
            torch.manual_seed(7); o = proj(G(mu_, cc, a_feat=fe_)[2], lb)
            P[s:s + len(jj), q] = (o[:, 0] < 0.5).float().mean(dim=(1, 2, 3)).cpu().numpy()
    OUT['sweep|phi'] = P; OUT['sweep|true'] = T; OUT['sweep|gid'] = np.array([ds.items[int(j)][0] for j in J])
    print('前 40 块：真值孔隙率均值 %.4f；各 c 下输出孔隙率均值' % T.mean(), dict(zip(ALL, np.round(P.mean(0), 4))), flush=True)
np.savez_compressed(str(OUT_ROOT) + '/eval59/geo_ctrl59.npz', **OUT)
print('DONE')