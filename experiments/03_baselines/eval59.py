# -*- coding: utf-8 -*-
"""第 59 步终评：近年强基线加入同一评测。
方法：三线性 / EDSR-3D / SRGAN-3D / SwinIR-3D / 扩散（EMA 权重，DDIM 50 步）/ 本文·仅粗扫（_n54b + 训练常数 c）/
      本文·粗扫+稀疏细扫（_g54b，ĉ + 远处 10 块细扫标定；与第 56 步「G·粗扫ĉ+远处标定」逐字相同）。
同一批 200 块（rng 0）、噪声种子 7；除三线性外全部做 8 次硬投影（同第 55 步）。指标函数与第 45–57 步逐字相同。
另存：每块每方法的孔隙掩膜（packbits，供 LBM）与首块中间切片（作图）。
用法：python eval59.py <折> <gpu> [块数]；环境变量 SKIP59=1：不含两个新基线（先行算渗流用），输出前缀 eval59a"""
import sys, json, os, time, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, PORE_CUT, lowpass, CUT1_UM, Generator, N_C
from models54 import load_mean, load_cpred
from models59 import SwinIR3D, DiffSR3D
from dataset import PairDataset, load_H
from train_edsr import EDSR3D
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


dev = 'cuda:%s' % sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 200
ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT); OUT = str(OUT_ROOT) + '/eval59/out/'; os.makedirs(OUT, exist_ok=True)
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev); S6 = ndi.generate_binary_structure(3, 1)
LO, HI = -0.3, 3.1; DR = HI - LO
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


def pos(v, t):
    """位置指标：细节带（<31 μm）逐点相关、7³ 局部能量相关（第 53 步定义）。"""
    d = v - lowpass(v[None, None], CUT1_UM)[0, 0]; dt = t - lowpass(t[None, None], CUT1_UM)[0, 0]
    E = lambda a: F.avg_pool3d(a[None, None] ** 2, 7, stride=1, padding=3, count_include_pad=False)[0, 0]
    cc = lambda a, b: float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])
    return cc(d, dt), cc(E(d), E(dt))


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


FOLD = sys.argv[1]; SKIP = os.environ.get('SKIP59', '0') == '1'; PFX = 'eval59a' if SKIP else 'eval59'
if NB == 200 and os.path.exists(OUT + PFX + '_%s.json' % FOLD): print('已有结果，跳过', FOLD, flush=True); sys.exit(0)
meta = json.load(open('%s/%s_proj/meta.json' % (RUNS, FOLD)))
CP = load_cpred('%s/%s_cpred/final.pt' % (RUNS, FOLD), dev)
dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
kk = np.random.default_rng(1).choice(len(dtr), min(300, len(dtr)), replace=False)
cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)   # 与第 55/56 步同一常数
M, Gb = load_new(FOLD, '_g54b'); Gn = load_new(FOLD, '_n54b')[1]
base = {}
for nm, suf in (('EDSR-3D', '_edsr'), ('SRGAN-3D', '_srgan')):
    m = EDSR3D().to(dev).eval(); m.load_state_dict(torch.load('%s/%s%s/ckpt/step0030000.pt' % (RUNS, FOLD, suf), map_location=dev, weights_only=False)['model']); base[nm] = m
def ck59(n):   # 正式评测只用 final.pt；冒烟（块数<200）可用最新中间检查点
    import glob
    p = '%s/%s_%s/ckpt/final.pt' % (RUNS, FOLD, n)
    if NB < 200 and not os.path.exists(p): p = sorted(glob.glob('%s/%s_%s/ckpt/step*.pt' % (RUNS, FOLD, n)))[-1]
    print('载入', p, flush=True); return torch.load(p, map_location=dev, weights_only=False)
if not SKIP:
    SW = SwinIR3D().to(dev).eval(); SW.load_state_dict(ck59('swin')['model'])
    DF = DiffSR3D().to(dev).eval(); DF.load_state_dict(ck59('diff')['ema'])
ds = PairDataset(ROOT, meta['test'], c_src='blk')
ii = np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)[:NB]     # 与第 54–57 步同一批 200 块
cf = {g: dict(np.load('%s/cfield2/%s.npz' % (ROOT, g))) for g in meta['test']}
T = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None].to(dev)
U = {}
with torch.no_grad():
    for g in meta['test']:
        js = [j for j, (gg, _) in enumerate(ds.items) if gg == g]
        cl = np.array([np.load(ds.items[j][1])['cell'][:3].astype(int) for j in js])
        cb = cf[g]['c_blk'][tuple(cl.T)].astype(np.float64); ok = cf[g]['valid'][tuple(cl.T)] & np.isfinite(cb).all(1)
        ch = np.concatenate([CP(torch.from_numpy(np.stack([ds[j]['lr'] for j in js[s:s + 32]])).to(dev)).cpu().numpy() for s in range(0, len(js), 32)])
        U[g] = dict(cell=cl[ok], cb=cb[ok], ch=ch[ok].astype(np.float64))
MM = ['三线性', 'EDSR-3D', 'SRGAN-3D'] + ([] if SKIP else ['SwinIR-3D', '扩散']) + ['本文·仅粗扫', '本文·粗扫+稀疏细扫']
R = {k: [] for k in MM}; ST = {k: [] for k in MM}; PO = {k: [] for k in MM}; TF = []; MASK = {k: [] for k in MM + ['细扫']}
VIS = {}; TIM = {k: 0.0 for k in MM}; rng = np.random.default_rng(5)
with torch.no_grad():
    for n_, j in enumerate(ii):
        b = ds[int(j)]; g, f = ds.items[int(j)]; cell = np.load(f)['cell'][:3].astype(int); u = U[g]
        lr = torch.from_numpy(b['lr'])[None].to(dev); t = torch.from_numpy(b['hr'][0]).to(dev); tf = feats(b['hr'][0]); TF.append(tf)
        far = np.where(np.abs(u['cell'] - cell).max(1) >= 3)[0]; far = rng.choice(far, min(10, len(far)), replace=False)
        ch = CP(lr)[0].cpu().numpy().astype(np.float64); ccal = ch + (u['cb'][far] - u['ch'][far]).mean(0)
        k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); O = {}
        def run(nm, fn):
            torch.cuda.synchronize(); t0 = time.time(); O[nm] = fn(); torch.cuda.synchronize(); TIM[nm] += time.time() - t0
        run('三线性', lambda: F.interpolate(lr[..., sl, sl, sl], size=(N,) * 3, mode='trilinear', align_corners=False))
        for nm, m in base.items(): run(nm, lambda m=m: proj(m(lr), lr))
        if not SKIP: run('SwinIR-3D', lambda: proj(SW(lr), lr))
        def dsamp():
            with torch.autocast('cuda'): x = DF.sample(lr, steps=50, seed=7)      # 与训练同一混合精度（fp16）
            return proj(x.float(), lr)
        if not SKIP: run('扩散', dsamp)
        run('本文·仅粗扫', lambda: proj(gen(M, Gn, lr, cconst), lr))
        run('本文·粗扫+稀疏细扫', lambda: proj(gen(M, Gb, lr, T(ccal)), lr))
        for nm, o in O.items():
            v = o[0, 0].float(); pf = feats(v.cpu().numpy()); R[nm].append(compare(pf, tf)); ST[nm].append(stripe(v)); PO[nm].append(pos(v, t))
            MASK[nm].append(np.packbits(pf['m']))
            if n_ < 4: VIS.setdefault(nm, []).append(v[N // 2].cpu().numpy().astype(np.float16))
        MASK['细扫'].append(np.packbits(tf['m']))
        if n_ < 4:
            VIS.setdefault('细扫', []).append(b['hr'][0][N // 2].astype(np.float16)); VIS.setdefault('粗扫', []).append(b['lr'][0][18].astype(np.float16))
        if n_ % 25 == 0: print(FOLD, n_, {k_: round(v_ / (n_ + 1), 2) for k_, v_ in TIM.items()}, flush=True)
res = {nm: dict(summarize(R[nm], TF), 条纹指数=float(np.mean(ST[nm])), 细节逐点相关=float(np.mean([p[0] for p in PO[nm]])),
                细节位置对准=float(np.mean([p[1] for p in PO[nm]])), 每块用时秒=TIM[nm] / len(ii)) for nm in MM}
res['真值条纹指数'] = float(np.mean([stripe(torch.from_numpy(ds[int(j)]['hr'][0]).to(dev)) for j in ii[:40]]))
res['逐块'] = {nm: [{k_: float(v_) for k_, v_ in r.items()} for r in R[nm]] for nm in MM}
res['逐块'].update(真值=[dict(phi=t_['phi'], conn=t_['conn'], chi=t_['chi']) for t_ in TF])
tag = PFX + ('_%s' if NB == 200 else '_smoke_%s')
np.savez_compressed(OUT + (tag % FOLD) + '_masks.npz', idx=ii, **{k_: np.stack(v_) for k_, v_ in MASK.items()})
np.savez_compressed(OUT + (tag % FOLD) + '_vis.npz', **{k_: np.stack(v_) for k_, v_ in VIS.items()})
json.dump(res, open(OUT + (tag % FOLD) + '.json', 'w'), ensure_ascii=False, indent=1)
print('DONE', FOLD, flush=True)
