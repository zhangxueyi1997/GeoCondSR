# -*- coding: utf-8 -*-
"""第 57b 步诊断（只评测）：整体水平哪几项推准了才有用：推理全程只用粗扫。粗扫逐块 ĉ（_cpred）管「哪块高哪块低」，
1 mm 粗扫窗推出的「整体水平」替换第 56 步里的「远处 10 块细扫标定」：ĉ_cal = ĉ − 该样品 ĉ 均值 + 粗扫整体水平。
模型沿用 _g54b（训练用细扫 c_blk 作标签），对照 _n54b；另测 _g54c 喂同一 ĉ_cal。指标函数与第 45–56 步逐字相同。
用法：python eval57.py <折> <gpu> [块数]"""
import sys, glob, json, os, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import GenesisSR, degrade, PORE_CUT, lowpass, CUT1_UM, Generator, N_C
from models54 import MeanPath, load_mean, load_cpred
from dataset import PairDataset, load_H
from train_edsr import EDSR3D
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

dev = 'cuda:%s' % sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 200; NV = 150 if NB == 200 else 20
ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT); OUT = os.environ.get('EVAL_OUT', str(OUT_ROOT) + '/eval/')
NEW_SUF, OLD_SUF, OLD_STEP = os.environ.get('NEW_SUF', '_rot2'), os.environ.get('OLD_SUF', '_proj'), int(os.environ.get('OLD_STEP', '30000'))
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev); S6 = ndi.generate_binary_structure(3, 1)
LO, HI = -0.3, 3.1; DR = HI - LO; WS = [1, 2, 3, 4, 6, 8]
N = 112; f1 = torch.fft.fftfreq(N, device=dev)
HF = torch.sqrt(f1[:, None, None] ** 2 + f1[None, :, None] ** 2 + f1[None, None, :] ** 2) > 0.05


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


def stripe(v):
    P = torch.fft.fftn(v.float() - v.float().mean()).abs() ** 2; q = P[HF]
    return float(torch.topk(q, 20).values.sum() / q.sum())


# ---------- 指标（与第 45/49 步冻结脚本逐字相同） ----------
def conn(m):
    if not m.any(): return 0.0
    lab, _ = ndi.label(m, structure=S6); return float(np.bincount(lab.ravel())[1:].max() / m.sum())
def ssa(m):
    s = 0
    for ax in range(3):
        a = np.moveaxis(m, ax, 0); s += int(np.count_nonzero(a[1:] != a[:-1]))
    return s / m.size
def euler(m):
    b = m; n0 = b.sum()
    n1 = (b[1:] & b[:-1]).sum() + (b[:, 1:] & b[:, :-1]).sum() + (b[:, :, 1:] & b[:, :, :-1]).sum()
    f = lambda a: (a[1:, 1:] & a[1:, :-1] & a[:-1, 1:] & a[:-1, :-1])
    n2 = f(b).sum() + f(b.transpose(1, 0, 2)).sum() + f(b.transpose(2, 0, 1)).sum()
    n3 = (b[1:, 1:, 1:] & b[1:, 1:, :-1] & b[1:, :-1, 1:] & b[1:, :-1, :-1] &
          b[:-1, 1:, 1:] & b[:-1, 1:, :-1] & b[:-1, :-1, 1:] & b[:-1, :-1, :-1]).sum()
    return float(n0 - n1 + n2 - n3)
def chord(m):
    tot, n = 0, 0
    for ax in range(3):
        a = np.moveaxis(m, ax, -1).reshape(-1, m.shape[ax]).astype(np.int8)
        d = np.diff(np.pad(a, ((0, 0), (1, 1))), axis=1)
        L = np.argwhere(d == -1)[:, 1] - np.argwhere(d == 1)[:, 1]
        tot += int(L.sum()); n += len(L)
    return tot / max(n, 1)
def s2(m, R=30):
    f = m.astype(np.float32); Fq = np.fft.rfftn(f)
    ac = np.fft.irfftn(Fq * np.conj(Fq), s=f.shape) / f.size
    return (ac[:R, 0, 0] + ac[0, :R, 0] + ac[0, 0, :R]) / 3
def frag(m, n=4):
    d = m.shape[0] // n; v = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = m[i*d:(i+1)*d, j*d:(j+1)*d, k*d:(k+1)*d]
                v.append(0.0 if not s.any() else ndi.label(s, structure=S6)[1] / s.size * 1000.0)
    return float(np.mean(v))
def feats(v):
    m = v < PORE_CUT
    return dict(m=m, phi=float(m.mean()), ssa=ssa(m), chi=euler(m), chord=chord(m),
                s2=s2(m), conn=conn(m), frag=frag(m), vc=np.clip(v, LO, HI))
def compare(p, t):
    inter = np.count_nonzero(p['m'] & t['m']); den = p['m'].sum() + t['m'].sum()
    return {'PSNR': peak_signal_noise_ratio(t['vc'], p['vc'], data_range=DR),
            'SSIM': structural_similarity(t['vc'], p['vc'], data_range=DR, win_size=7),
            'Dice': (2 * inter / den) if den else 1.0,
            '孔隙度误差': abs(p['phi'] - t['phi']),
            '比表面相对误差': abs(p['ssa'] - t['ssa']) / max(t['ssa'], 1e-9),
            'chi_abs': abs(p['chi'] - t['chi']),
            '弦长相对误差': abs(p['chord'] - t['chord']) / max(t['chord'], 1e-9),
            'S2距离': float(np.abs(p['s2'] - t['s2']).mean()),
            '碎裂度误差': abs(np.log((p['frag'] + 0.01) / (t['frag'] + 0.01))),
            'conn_p': p['conn']}
def summarize(rows, TF):
    tchi = np.array([t['chi'] for t in TF]); tconn = np.array([t['conn'] for t in TF])
    out = {k: float(np.mean([r[k] for r in rows])) for k in
           ['PSNR', 'SSIM', 'Dice', '孔隙度误差', '比表面相对误差', '弦长相对误差', 'S2距离', '碎裂度误差']}
    out['欧拉数误差'] = float(np.mean([r['chi_abs'] for r in rows])) / max(tchi.std(), 1e-9)
    P = np.array([r['conn_p'] for r in rows])
    out['连通度MAE'] = float(np.abs(P - tconn).mean()); out['连通跟踪'] = float(np.corrcoef(P, tconn)[0, 1])
    return out



def load_new(f, arm):
    M = load_mean('%s/%s_m54b/ckpt/final.pt' % (RUNS, f), dev).eval()
    G = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
    G.load_state_dict(torch.load('%s/%s%s/ckpt/final.pt' % (RUNS, f, arm), map_location=dev, weights_only=False)['model']); return M, G


def gen(M, G, lr, c):
    mu, feat = M(lr, ret_feat=True); torch.manual_seed(7); return G(mu, c, a_feat=feat)[2]






FOLD = sys.argv[1]; OUT = str(OUT_ROOT) + '/eval57b/'
meta = fold_meta(FOLD)
CP = load_cpred('%s/%s_cpred/final.pt' % (RUNS, FOLD), dev)
# ---------- 粗扫推「整体水平」（只用粗扫；细扫只作训练岩性的标签） ----------
# 1 mm 窗（72³）的 3 个灰度统计 [分位2, 分位10, 标准差] → 0.75 mm 邻域的细扫 c（4 维），岭回归只用训练岩性拟合；
# 测试样品的整体水平 = 该样品全部 1 mm 窗预测值的平均。数据来自第 56 步 ana56_data.npz（行格式见 ana56.py）。
A56 = np.load(str(OUT_ROOT) + '/eval56/ana56_data.npz'); K3 = [0, 1, 6]
def rows(g): r = A56[g + '_rows']; return r[:, 3 + 18:3 + 27][:, K3], r[:, 43:47], r[:, 39:43]      # X72(k3), c3, c_blk
Xtr = np.concatenate([rows(g)[0] for g in meta['train']]); Ytr = np.concatenate([rows(g)[1] for g in meta['train']])
okr = np.isfinite(Xtr).all(1) & np.isfinite(Ytr).all(1); Xtr, Ytr = Xtr[okr], Ytr[okr]
mx, sx, my = Xtr.mean(0), Xtr.std(0) + 1e-9, Ytr.mean(0); Z = (Xtr - mx) / sx
Wl = np.linalg.solve(Z.T @ Z + np.eye(3), Z.T @ (Ytr - my))
LV = {}
for g in meta['test']:
    X, _, cb = rows(g); ok = np.isfinite(X).all(1)
    LV[g] = (((X[ok] - mx) / sx) @ Wl + my).mean(0)
    print(FOLD, g, '粗扫推的整体水平', np.round(LV[g], 4), '| 细扫真实整体水平', np.round(np.nanmean(cb, 0), 4), flush=True)
dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
kk = np.random.default_rng(1).choice(len(dtr), min(300, len(dtr)), replace=False)
cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)   # 与第 55/56 步同一常数
M, Gb = load_new(FOLD, '_g54b'); Gn = load_new(FOLD, '_n54b')[1]
ds = PairDataset(ROOT, meta['test'], c_src='blk')
ii = np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)[:NB]     # 与第 54/55/56 步同一批 200 块
cf = {g: dict(np.load('%s/cfield2/%s.npz' % (ROOT, g))) for g in meta['test']}
T = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None].to(dev)
U = {}
with torch.no_grad():
    for g in meta['test']:
        js = [j for j, (gg, _) in enumerate(ds.items) if gg == g]
        cl = np.array([np.load(ds.items[j][1])['cell'][:3].astype(int) for j in js])
        cb = cf[g]['c_blk'][tuple(cl.T)].astype(np.float64); ok = cf[g]['valid'][tuple(cl.T)] & np.isfinite(cb).all(1)
        ch = np.concatenate([CP(torch.from_numpy(np.stack([ds[j]['lr'] for j in js[s:s + 32]])).to(dev)).cpu().numpy() for s in range(0, len(js), 32)]).astype(np.float64)
        U[g] = dict(cell=cl[ok], cb=cb[ok], ch=ch[ok], chmean=ch.mean(0))          # chmean：该样品全部配对单元的 ĉ 平均（只用粗扫）
TL = {g: np.nanmean(rows(g)[2], 0) for g in meta['test']}     # 细扫真实整体水平（诊断用：看哪几项推准了才有用）
ARMS = ['G·块实测c', 'A·整体水平4项全真', 'B·孔隙率+孔隙灰度真', 'C·IGV+高密相真', 'D·整体水平全用粗扫', '无地质(训练)']
R = {k: [] for k in ARMS}; CV = {k: [] for k in ARMS[:5]}; TF = []; rng = np.random.default_rng(5)
with torch.no_grad():
    for n_, j in enumerate(ii):
        b = ds[int(j)]; g, f = ds.items[int(j)]; cell = np.load(f)['cell'][:3].astype(int); u = U[g]
        lr = torch.from_numpy(b['lr'])[None].to(dev); tf = feats(b['hr'][0]); TF.append(tf)
        far = np.where(np.abs(u['cell'] - cell).max(1) >= 3)[0]; far = rng.choice(far, min(10, len(far)), replace=False)
        ch = CP(lr)[0].cpu().numpy().astype(np.float64); d0 = ch - u['chmean']; t, l = TL[g], LV[g]      # 维度顺序 IGV, f_pore, mu_inter, f_dense
        C = {'G·块实测c': np.asarray(b['c'], np.float64), 'A·整体水平4项全真': d0 + t,
             'B·孔隙率+孔隙灰度真': d0 + np.r_[l[0], t[1], t[2], l[3]], 'C·IGV+高密相真': d0 + np.r_[t[0], l[1], l[2], t[3]],
             'D·整体水平全用粗扫': d0 + l}
        for k, c in C.items():
            R[k].append(compare(feats(proj(gen(M, Gb, lr, T(c)), lr)[0, 0].float().cpu().numpy()), tf)); CV[k].append(c)
        R['无地质(训练)'].append(compare(feats(proj(gen(M, Gn, lr, cconst), lr)[0, 0].float().cpu().numpy()), tf))
        if n_ % 50 == 0: print(FOLD, n_, flush=True)
res = {k: summarize(R[k], TF) for k in ARMS}
Tb = np.array(CV['G·块实测c'])
res['c与块实测R2'] = {k: [float(1 - ((np.array(CV[k])[:, q] - Tb[:, q]) ** 2).sum() / max(((Tb[:, q] - Tb[:, q].mean()) ** 2).sum(), 1e-12)) for q in range(4)] for k in CV}
res['整体水平'] = {g: dict(粗扫推=LV[g].tolist(), 细扫真实=np.nanmean(rows(g)[2], 0).tolist()) for g in meta['test']}
json.dump(res, open(OUT + ('eval57b_%s.json' if NB == 200 else 'smoke57b_%s.json') % FOLD, 'w'), ensure_ascii=False, indent=1)
print('DONE', FOLD, flush=True)
