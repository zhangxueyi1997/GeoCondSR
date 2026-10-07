# -*- coding: utf-8 -*-
"""第 55 步终评（地质变量来源：C_MODE=pred 粗扫预测 ĉ / ann 周围区域 / blk 目标块，诊断用）。基于第 54 步终评（单折，GPU）——指标函数与第 45/49/51 步逐字相同。
原第 51 步说明：。测试岩性 200 块，与第 45/49 步同一批块（rng 0）、同一指标面板（函数逐字复制）。
方法：三线性 / 旧模型 _proj（w=1）/ 新配方 _rot2（w=1）/ 新配方 w*（w 只在训练岩性留出块上按连通跟踪选，同第 48 步）
      / EDSR-3D / SRGAN-3D（各自 3 万步检查点存在才评）。另记条纹指数与地质特异性（真c vs 配错c）。
用法：python eval51.py <折> <gpu>"""
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



def load_old(run, step):
    sd = torch.load('%s/%s/ckpt/step%07d.pt' % (RUNS, run, step), map_location=dev, weights_only=False)['model']
    g = GenesisSR(mode='given', level=True, afeat=any(k.startswith('g.fa4') for k in sd)).to(dev).eval(); g.load_state_dict(sd, strict=False); return g


def load_new(f, arm):
    M = load_mean('%s/%s%s/ckpt/final.pt' % (RUNS, f, os.environ.get('M_SUF', '_m54')), dev).eval()
    if arm is None: return M, None
    G = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
    G.load_state_dict(torch.load('%s/%s%s/ckpt/final.pt' % (RUNS, f, arm), map_location=dev, weights_only=False)['model']); return M, G


def gen(M, G, lr, c):
    mu, feat = M(lr, ret_feat=True); torch.manual_seed(7); return G(mu, c, a_feat=feat)[2]


def pos(v, t):
    """位置指标：细节带（<31 μm）逐点相关、7³ 局部能量相关（第 53 步定义）。"""
    d = v - lowpass(v[None, None], CUT1_UM)[0, 0]; dt = t - lowpass(t[None, None], CUT1_UM)[0, 0]
    E = lambda a: F.avg_pool3d(a[None, None] ** 2, 7, stride=1, padding=3, count_include_pad=False)[0, 0]
    cc = lambda a, b: float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])
    return cc(d, dt), cc(E(d), E(dt))


FOLD = sys.argv[1]
meta = json.load(open('%s/%s_proj/meta.json' % (RUNS, FOLD)))
C_MODE = os.environ.get('C_MODE', 'blk'); CSRC = {'ann': 'ann', 'nbr': 'nbr_' + FOLD}.get(C_MODE, 'blk')   # nbr = 周围回归预测（按本折拟合）
CP = load_cpred('%s/%s%s/final.pt' % (RUNS, FOLD, os.environ.get('CPRED_SUF', '_cpred')), dev) if C_MODE == 'pred' else None


def getc(b):
    # 本方法的地质输入：pred = 只用粗扫推出的 ĉ；ann/blk = 对应支撑的实测值
    if CP is not None:
        return CP(torch.from_numpy(b['lr'])[None].to(dev)).float()
    return torch.from_numpy(np.asarray(b['c'], np.float32))[None].to(dev)


dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
kk = np.random.default_rng(1).choice(len(dtr), min(300, len(dtr)), replace=False)
cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)   # 无地质组训练用的常数
dtr_m = PairDataset(ROOT, meta['train'], unit_split='train', c_src=CSRC)
with torch.no_grad():
    cmean = torch.cat([getc(dtr_m[int(i)]) for i in kk]).mean(0, keepdim=True)                            # 本方法 c 的训练折均值（引导强度基准）
g_old = load_old(FOLD + '_rot2', 10000)
ARM_G = os.environ.get('ARM_G', '_g54')   # 冒烟测试时可换成小样运行名
M, Gg = load_new(FOLD, ARM_G)
ARM_N = os.environ.get('ARM_N', '_n54')
HAVE_N = os.path.exists('%s/%s%s/ckpt/final.pt' % (RUNS, FOLD, ARM_N))
Gn = load_new(FOLD, ARM_N)[1] if HAVE_N else None
PREV_G, PREV_M = os.environ.get('PREV_G', ''), os.environ.get('PREV_M', '_m54')   # 可选：上一轮有地质版作对照列
if PREV_G:
    Mp = load_mean('%s/%s%s/ckpt/final.pt' % (RUNS, FOLD, PREV_M), dev).eval()
    Gp = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=Mp.ch).to(dev).eval()
    Gp.load_state_dict(torch.load('%s/%s%s/ckpt/final.pt' % (RUNS, FOLD, PREV_G), map_location=dev, weights_only=False)['model'])
# ---------- w* 只在训练岩性留出块上选（同第 48 步：连通跟踪） ----------
dv = PairDataset(ROOT, meta['train'], unit_split='heldout', c_src=CSRC)
iv = np.random.default_rng(0).choice(len(dv), min(NV, len(dv)), replace=False)
TV, PV = [], {w: [] for w in WS}
with torch.no_grad():
    for j in iv:
        b = dv[int(j)]; lr = torch.from_numpy(b['lr'])[None].to(dev); c = getc(b)
        TV.append(conn(b['hr'][0] < PORE_CUT)); oc, om = gen(M, Gg, lr, c), gen(M, Gg, lr, cmean)
        for w in WS: PV[w].append(conn(proj(om + w * (oc - om), lr)[0, 0].float().cpu().numpy() < PORE_CUT))
val = {w: float(np.corrcoef(PV[w], TV)[0, 1]) for w in WS}; wstar = max(WS, key=lambda w: val[w])
print(FOLD, '留出块选 w：', {w: round(v, 4) for w, v in val.items()}, 'w* =', wstar, flush=True)
# ---------- 测试岩性 ----------
ds = PairDataset(ROOT, meta['test'], c_src=CSRC); ds_blk = PairDataset(ROOT, meta['test'], c_src='blk')
ii = np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)[:NB]; blocks = [ds[int(j)] for j in ii]; blocks_blk = [ds_blk[int(j)] for j in ii]
perm = np.random.default_rng(3).permutation(len(blocks)); TF = [feats(b['hr'][0]) for b in blocks]
base = {}
for nm, suf in (('EDSR-3D', '_edsr'), ('SRGAN-3D', '_srgan')):
    m = EDSR3D().to(dev).eval(); m.load_state_dict(torch.load('%s/%s%s/ckpt/step0030000.pt' % (RUNS, FOLD, suf), map_location=dev, weights_only=False)['model']); base[nm] = m
MM = ['三线性', 'EDSR-3D', 'SRGAN-3D', '上一版_rot2', '仅均值通路', '新·有地质 w=1', '新·有地质 w*'] + (['新·无地质'] if HAVE_N else []) + (['参照·c_blk版'] if PREV_G else [])
R = {k: [] for k in MM}; ST = {k: [] for k in MM}; PO = {k: [] for k in MM}
GEO = {k: [] for k in ('w1_true', 'w1_wrong', 'ws_true', 'ws_wrong')}; CPR = []
with torch.no_grad():
    for i, (b, tf) in enumerate(zip(blocks, TF)):
        lr = torch.from_numpy(b['lr'])[None].to(dev); c = getc(b); cw = getc(blocks[perm[i]]); t = torch.from_numpy(b['hr'][0]).to(dev)
        cb = torch.from_numpy(np.asarray(blocks_blk[i]['c'], np.float32))[None].to(dev)      # 目标块实测 c（参照模型用）
        CPR.append((c[0].cpu().numpy(), cb[0].cpu().numpy()))
        k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k)
        O = {'三线性': F.interpolate(lr[..., sl, sl, sl], size=(N,) * 3, mode='trilinear', align_corners=False)}
        for nm, m in base.items(): O[nm] = proj(m(lr), lr)
        torch.manual_seed(7); O['上一版_rot2'] = proj(g_old(lr, c_true=cb)['hr'], lr)
        O['仅均值通路'] = proj(M(lr), lr)
        oc, om, ow = gen(M, Gg, lr, c), gen(M, Gg, lr, cmean), gen(M, Gg, lr, cw)
        O['新·有地质 w=1'] = proj(oc, lr); O['新·有地质 w*'] = proj(om + wstar * (oc - om), lr)
        if HAVE_N: O['新·无地质'] = proj(gen(M, Gn, lr, cconst), lr)
        if PREV_G: O['参照·c_blk版'] = proj(gen(Mp, Gp, lr, cb), lr)
        for nm, o in O.items():
            v = o[0, 0].float(); R[nm].append(compare(feats(v.cpu().numpy()), tf)); ST[nm].append(stripe(v)); PO[nm].append(pos(v, t))
        for tag, w in (('w1', 1), ('ws', wstar)):
            for cc_, o in (('true', oc), ('wrong', ow)):
                pf = feats(proj(om + w * (o - om), lr)[0, 0].float().cpu().numpy())
                GEO['%s_%s' % (tag, cc_)].append(dict(frag=abs(np.log((pf['frag'] + .01) / (tf['frag'] + .01))), phi=abs(pf['phi'] - tf['phi']), conn=pf['conn']))
        if i % 50 == 0: print(FOLD, i, flush=True)
TCONN = np.array([t['conn'] for t in TF])
res = {nm: dict(summarize(R[nm], TF), 条纹指数=float(np.mean(ST[nm])), 细节逐点相关=float(np.mean([p[0] for p in PO[nm]])),
                细节位置对准=float(np.mean([p[1] for p in PO[nm]]))) for nm in MM}
res['真值条纹指数'] = float(np.mean([stripe(torch.from_numpy(b['hr'][0]).to(dev)) for b in blocks[:40]]))
res['地质'] = {k: dict(frag=float(np.mean([r['frag'] for r in v])), phi=float(np.mean([r['phi'] for r in v])),
                     conn_trk=float(np.corrcoef([r['conn'] for r in v], TCONN)[0, 1])) for k, v in GEO.items()}
res['wstar'] = wstar; res['val'] = val; res['有无地质对照'] = HAVE_N; res['C_MODE'] = C_MODE
P_, T_ = np.array([a for a, _ in CPR]), np.array([b_ for _, b_ in CPR])
res['c与目标块实测c的R2'] = [float(1 - ((P_[:, k] - T_[:, k]) ** 2).sum() / max(((T_[:, k] - T_[:, k].mean()) ** 2).sum(), 1e-12)) for k in range(T_.shape[1])]
json.dump(res, open(OUT + ('eval55_%s.json' % FOLD if NB == 200 else 'smoke_%s.json' % FOLD), 'w'), ensure_ascii=False, indent=1)
print('DONE', FOLD, flush=True)
