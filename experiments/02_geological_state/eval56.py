# -*- coding: utf-8 -*-
"""第 56 步判决实验（只评测、不训练）：第 54 步的地质增益，来自「块级」还是「样品级」信息？
同一个训练好的 _g54b（训练时用块实测 c_blk），同一批测试块、同一噪声，只换推理时喂的 c：
  块实测c        目标块自身细扫统计（上限，循环）
  样品均值c      该测试样品全部有效格 c_blk 的均值（不含块级起伏；工程上 = 这块岩心有一次细扫）
  远处10块细扫   与目标块相距 ≥3 格（>750 μm，块级相关已≈0）的 10 块 c_blk 均值（工程上 = 稀疏细扫）
  粗扫ĉ          只用粗扫预测（_cpred）
  粗扫ĉ+远处标定 ĉ + 远处 10 块的 (c_blk − ĉ) 均值（粗扫管块级起伏，远处细扫管整体标定）
  训练常数c      训练折均值（看 _g54b 自身对 c 的依赖）
对照：无地质（训练）_n54b。指标函数与第 45/49/51/54/55 步逐字相同。
用法：python eval56.py <折> <gpu>"""
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


FOLD = sys.argv[1]; OUT = str(OUT_ROOT) + '/eval56/'
if NB == 200 and os.path.exists(OUT + 'eval56_%s.json' % FOLD): print('已有结果，跳过', FOLD, flush=True); sys.exit(0)   # 提前跑过则后续排队的重复启动直接跳过
meta = json.load(open('%s/%s_proj/meta.json' % (RUNS, FOLD)))
CP = load_cpred('%s/%s_cpred/final.pt' % (RUNS, FOLD), dev)
dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
kk = np.random.default_rng(1).choice(len(dtr), min(300, len(dtr)), replace=False)
cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)   # 与第 55 步同一常数
M, Gb = load_new(FOLD, '_g54b'); Gn = load_new(FOLD, '_n54b')[1]
ds = PairDataset(ROOT, meta['test'], c_src='blk')
ii = np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)[:NB]     # 与第 54/55 步同一批 200 块（NB<200 为冒烟）
cf = {g: dict(np.load('%s/cfield2/%s.npz' % (ROOT, g))) for g in meta['test']}
T = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None].to(dev)
# 每个测试样品：全部配对单元的格坐标、c_blk、ĉ（远处标定块从这里抽）
U = {}
with torch.no_grad():
    for g in meta['test']:
        js = [j for j, (gg, _) in enumerate(ds.items) if gg == g]
        cl = np.array([np.load(ds.items[j][1])['cell'][:3].astype(int) for j in js])
        cb = cf[g]['c_blk'][tuple(cl.T)].astype(np.float64); ok = cf[g]['valid'][tuple(cl.T)] & np.isfinite(cb).all(1)
        ch = np.concatenate([CP(torch.from_numpy(np.stack([ds[j]['lr'] for j in js[s:s + 32]])).to(dev)).cpu().numpy() for s in range(0, len(js), 32)])
        v = cf[g]['c_blk'][cf[g]['valid'] & np.isfinite(cf[g]['c_blk']).all(-1)].astype(np.float64)
        U[g] = dict(js=js, cell=cl[ok], cb=cb[ok], ch=ch[ok].astype(np.float64), smean=v.mean(0))
ARMS = ['G·块实测c', 'G·样品均值c', 'G·远处10块细扫', 'G·粗扫ĉ', 'G·粗扫ĉ+远处标定', 'G·训练常数c', '无地质(训练)']
R = {k: [] for k in ARMS}; CV = {k: [] for k in ARMS[:5]}; TF = []; rng = np.random.default_rng(5)
with torch.no_grad():
    for n_, j in enumerate(ii):
        b = ds[int(j)]; g, f = ds.items[int(j)]; cell = np.load(f)['cell'][:3].astype(int); u = U[g]
        lr = torch.from_numpy(b['lr'])[None].to(dev); tf = feats(b['hr'][0]); TF.append(tf)
        far = np.where(np.abs(u['cell'] - cell).max(1) >= 3)[0]; far = rng.choice(far, min(10, len(far)), replace=False)
        ch = CP(lr)[0].cpu().numpy().astype(np.float64)
        C = {'G·块实测c': np.asarray(b['c'], np.float64), 'G·样品均值c': u['smean'], 'G·远处10块细扫': u['cb'][far].mean(0),
             'G·粗扫ĉ': ch, 'G·粗扫ĉ+远处标定': ch + (u['cb'][far] - u['ch'][far]).mean(0)}
        for k, c in C.items():
            R[k].append(compare(feats(proj(gen(M, Gb, lr, T(c)), lr)[0, 0].float().cpu().numpy()), tf)); CV[k].append(c)
        R['G·训练常数c'].append(compare(feats(proj(gen(M, Gb, lr, cconst), lr)[0, 0].float().cpu().numpy()), tf))
        R['无地质(训练)'].append(compare(feats(proj(gen(M, Gn, lr, cconst), lr)[0, 0].float().cpu().numpy()), tf))
        if n_ % 50 == 0: print(FOLD, n_, flush=True)
res = {k: summarize(R[k], TF) for k in ARMS}
Tb = np.array(CV['G·块实测c'])
res['c与块实测R2'] = {k: [float(1 - ((np.array(CV[k])[:, q] - Tb[:, q]) ** 2).sum() / max(((Tb[:, q] - Tb[:, q].mean()) ** 2).sum(), 1e-12)) for q in range(4)] for k in CV}
json.dump(res, open(OUT + ('eval56_%s.json' if NB == 200 else 'smoke56_%s.json') % FOLD, 'w'), ensure_ascii=False, indent=1)
print('DONE', FOLD, flush=True)
