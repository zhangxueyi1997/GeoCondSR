# -*- coding: utf-8 -*-
"""第 59 步对照：噪声敏感性（有真值）。在测试小柱粗扫上叠加与大圆柱同谱的噪声，看各方法 2 μm 孔隙率如何变化、谱减校正能恢复多少。
噪声谱 N(f)：本折训练岩性配对组 (P大 − P小) 的中位数（与 plugsr59 同一口径，f < 0.15 周/体素置 0）；噪声 = 白噪声经 sqrt(N) 整形（每块独立，种子固定）。
输入三种：干净 / 加噪 / 加噪后谱减校正（校正与 plugsr59.harmonize 同一函数）。每折取与终评同一批的前 100 块。
用法：python noise59.py <折> <gpu>"""
import sys, json, os, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, PORE_CUT, Generator, N_C
from models54 import load_mean
from dataset import PairDataset, load_H
from train_edsr import EDSR3D

FOLD, dev = sys.argv[1], 'cuda:%s' % sys.argv[2]; MS, GS = os.environ.get('M_SUF', '_m54b'), os.environ.get('G_SUF', '_n54b')
ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT); PD = str(DATA_ROOT) + '/plugs/'; OUT = str(OUT_ROOT) + '/plugs59/'
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev); N = 112; BS = 4; NB = 100


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


meta = fold_meta(FOLD)
dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
kk = np.random.default_rng(1).choice(len(dtr), min(300, len(dtr)), replace=False)
cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)
M = load_mean('%s/%s%s/ckpt/final.pt' % (RUNS, FOLD, MS), dev).eval()
Gn = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
Gn.load_state_dict(torch.load('%s/%s%s/ckpt/final.pt' % (RUNS, FOLD, GS), map_location=dev, weights_only=False)['model'])
base = {}
for nm, suf in (('EDSR-3D', '_edsr'), ('SRGAN-3D', '_srgan')):
    m = EDSR3D().to(dev).eval(); m.load_state_dict(torch.load('%s/%s%s/ckpt/step0030000.pt' % (RUNS, FOLD, suf), map_location=dev, weights_only=False)['model']); base[nm] = m
MM = ['粗扫阈值', '三线性', 'EDSR-3D', 'SRGAN-3D', '本文·仅粗扫']

DOM = json.load(open(PD + 'dom59.json')); n36 = 36; w1 = np.hanning(n36); W3 = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
f36 = np.fft.fftfreq(n36); FR = np.sqrt(f36[:, None, None] ** 2 + f36[None, :, None] ** 2 + f36[None, None, :] ** 2)
EDG = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.87]); CEN = (EDG[:-1] + EDG[1:]) / 2
f72 = np.fft.fftfreq(2 * n36); FR2 = np.sqrt(f72[:, None, None] ** 2 + f72[None, :, None] ** 2 + f72[None, None, :] ** 2)
TRG = [g for g in meta['train'] if g in DOM]
NF = np.median([np.array(DOM[g]['pl']) - np.array(DOM[g]['ps']) for g in TRG], 0); NF[CEN < 0.15] = 0; NF = np.clip(NF, 0, None)


def psd(W):
    out = []
    for v in W:
        v = v.astype(np.float32); p = np.abs(np.fft.fftn((v - v.mean()) * W3)) ** 2 / W3.sum()
        out.append([p[(FR >= EDG[i]) & (FR < EDG[i + 1])].mean() for i in range(len(EDG) - 1)])
    return np.mean(out, 0)


def harmonize(W):
    Pl = psd(W); Hb = np.sqrt(np.clip((Pl - NF) / Pl, 0.02, 1.0)); Hb[CEN < 0.1] = 1.0
    H = np.interp(FR2, np.r_[0, CEN], np.r_[1.0, Hb]); out = []
    for v in W:
        v = v.astype(np.float32); m = v.mean(); e = np.pad(v - m, ((0, n36), (0, n36), (0, n36)), mode='symmetric')
        out.append((np.real(np.fft.ifftn(np.fft.fftn(e) * H))[:n36, :n36, :n36] + m).astype(np.float32))
    return np.stack(out)


# 噪声整形：白噪声（单位方差）在 psd() 口径下各频带功率 = ΣW3²/ΣW3；按带插值幅度 sqrt(NF / 该值)
A = np.interp(FR, np.r_[0, CEN], np.r_[0.0, np.sqrt(NF / ((W3 ** 2).sum() / W3.sum()))])
def add_noise(W, seed):
    g = np.random.default_rng(seed)
    return np.stack([v + np.real(np.fft.ifftn(np.fft.fftn(g.standard_normal(v.shape)) * A)).astype(np.float32) for v in W])


@torch.no_grad()
def run(W):
    out = {k: [] for k in MM}
    for s in range(0, len(W), BS):
        lr = torch.from_numpy(np.asarray(W[s:s + BS], np.float32))[:, None].to(dev); B = lr.shape[0]
        k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k)
        O = {'粗扫阈值': lr[..., sl, sl, sl], '三线性': F.interpolate(lr[..., sl, sl, sl], size=(N,) * 3, mode='trilinear', align_corners=False)}
        for nm, m in base.items(): O[nm] = proj(m(lr), lr)
        mu, feat = M(lr, ret_feat=True); torch.manual_seed(7 + s); O['本文·仅粗扫'] = proj(Gn(mu, cconst.expand(B, -1), a_feat=feat)[2], lr)
        for nm, o in O.items(): out[nm] += (o[:, 0] < PORE_CUT).float().mean(dim=(1, 2, 3)).cpu().tolist()
    return out


ds = PairDataset(ROOT, meta['test'], c_src='blk')
ii = np.random.default_rng(0).choice(len(ds), min(200, len(ds)), replace=False)[:NB]
W = np.stack([ds[int(j)]['lr'][0] for j in ii]); tru = [float((ds[int(j)]['hr'][0] < PORE_CUT).mean()) for j in ii]
Wn = add_noise(W, 59); Wh = harmonize(Wn)
res = dict(fold=FOLD, NF=NF.tolist(), 细扫真值=tru, psd_clean=psd(W).tolist(), psd_noisy=psd(Wn).tolist(), psd_harm=psd(Wh).tolist())
for tag, X in (('干净', W), ('加噪', Wn), ('加噪+校正', Wh)):
    res[tag] = run(X); print(FOLD, tag, ' '.join('%s %.4f' % (k, np.mean(v)) for k, v in res[tag].items()), '| 真值 %.4f' % np.mean(tru), flush=True)
json.dump(res, open(OUT + ('noise59_%s.json' % FOLD if GS == '_n54b' else 'noise59_%s%s.json' % (FOLD, GS)), 'w'), ensure_ascii=False, indent=0)
print('DONE', FOLD, flush=True)
