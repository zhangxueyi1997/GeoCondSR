# -*- coding: utf-8 -*-
"""第 59 步：40 根岩心柱整柱粗扫（13.93 μm）超分，检验柱级 2 μm 可分辨孔隙度与实验室氦孔隙度的一致性。
每根柱 300 个 36³ 窗口（本地 extract_plugs.py 抽取，归一化与训练粗扫同一口径），每窗输出中心 224 μm 的 112³。
模型：该柱岩性作为测试折的那一折（训练未见该岩性）；山西（SX，只作训练）与福建（FJ，无细扫）不属任何测试折，6 折模型都跑，报告时单列。
方法：粗扫直接阈值（中心 16³，u<0.5）/ 三线性 / EDSR-3D / SRGAN-3D / 本文·仅粗扫（_n54b + 训练常数 c）；8 次硬投影同终评。
对照（域差异）：同一折的测试小柱粗扫（配对块，rng 0 取 300 块），同样方法 + 细扫真值。
用法：python plugsr59.py <折> <gpu>；环境变量 M_SUF / G_SUF 选模型（默认 _m54b / _n54b；抗噪微调版 _m59n / _n59n）"""
import sys, json, os, glob, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, PORE_CUT, Generator, N_C
from models54 import load_mean
from dataset import PairDataset, load_H
from train_edsr import EDSR3D

FOLD, dev = sys.argv[1], 'cuda:%s' % sys.argv[2]; MS, GS = os.environ.get('M_SUF', '_m54b'), os.environ.get('G_SUF', '_n54b')
B_SUF = os.environ.get('B_SUF', '')   # baseline weights: empty = original checkpoints; 59n = degradation-aware fine-tuned (edsr59n, srgan59n)
CUP = os.environ.get('CUP', '1') == '1'                    # 默认做成像条件标定（第 59 步定稿）
ONLY = os.environ.get('ONLY', '')                          # 逗号分隔的柱号（调试用）
ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT); PD = str(DATA_ROOT) + '/plugs/'; OUT = str(OUT_ROOT) + '/plugs59/'; os.makedirs(OUT, exist_ok=True)
TAB = json.load(open(PD + 'plug_table.json', encoding='utf-8'))
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev); N = 112; BS = 4


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


meta = fold_meta(FOLD)
dtr = PairDataset(ROOT, meta['train'], unit_split='train', c_src='blk')
kk = np.random.default_rng(1).choice(len(dtr), min(300, len(dtr)), replace=False)
cconst = torch.from_numpy(np.stack([dtr[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)   # 与终评同一常数
M = load_mean('%s/%s%s/ckpt/final.pt' % (RUNS, FOLD, MS), dev).eval()
Gn = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
Gn.load_state_dict(torch.load('%s/%s%s/ckpt/final.pt' % (RUNS, FOLD, GS), map_location=dev, weights_only=False)['model'])
base = {}
for nm, suf in (('EDSR-3D', '_edsr'), ('SRGAN-3D', '_srgan')):
    m = EDSR3D().to(dev).eval(); m.load_state_dict(torch.load('%s/%s%s%s/ckpt/%s' % (RUNS, FOLD, suf, B_SUF, 'final.pt' if B_SUF else 'step0030000.pt'), map_location=dev, weights_only=False)['model']); base[nm] = m
MM = ['粗扫阈值', '三线性', 'EDSR-3D', 'SRGAN-3D', '本文·仅粗扫']
# ---------- 域校正（谱减）：大圆柱粗扫比小柱粗扫多一层与岩性无关的高频噪声。噪声谱 N(f) = 本折训练岩性配对组 (P大 − P小) 的中位数
# （只用两种粗扫，不用细扫）；每根柱 H(f) = sqrt(clip((P大 − N)/P大, 0.02, 1))，f < 0.1 周/体素不动（N 在 f < 0.15 置 0：该段以结构为主）；窗口镜像延拓到 72³ 频域滤波后裁回。
DOM = json.load(open(PD + 'dom59.json')); n36 = 36; w1 = np.hanning(n36); W3 = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
f36 = np.fft.fftfreq(n36); FR = np.sqrt(f36[:, None, None] ** 2 + f36[None, :, None] ** 2 + f36[None, None, :] ** 2)
EDG = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.87]); CEN = (EDG[:-1] + EDG[1:]) / 2
f72 = np.fft.fftfreq(2 * n36); FR2 = np.sqrt(f72[:, None, None] ** 2 + f72[None, :, None] ** 2 + f72[None, None, :] ** 2)
TRG = [g for g in meta['train'] if g in DOM]
NF = np.median([np.array(DOM[g]['pl']) - np.array(DOM[g]['ps']) for g in TRG], 0); NF[CEN < 0.15] = 0; NF = np.clip(NF, 0, None)
print(FOLD, '噪声谱来自训练组', TRG, np.round(NF * 1e3, 2), flush=True)


def psd(W):
    out = []
    for v in W:
        v = v.astype(np.float32); p = np.abs(np.fft.fftn((v - v.mean()) * W3)) ** 2 / W3.sum()
        out.append([p[(FR >= EDG[i]) & (FR < EDG[i + 1])].mean() for i in range(len(EDG) - 1)])
    return np.mean(out, 0)


def harmonize(W, sc=1.0):                         # sc：噪声谱的单位换算系数（灰度重新标定后）
    Pl = psd(W); Hb = np.sqrt(np.clip((Pl - NF * sc) / Pl, 0.02, 1.0)); Hb[CEN < 0.1] = 1.0
    H = np.interp(FR2, np.r_[0, CEN], np.r_[1.0, Hb]); out = []
    for v in W:
        v = v.astype(np.float32); m = v.mean(); e = np.pad(v - m, ((0, n36), (0, n36), (0, n36)), mode='symmetric')
        out.append((np.real(np.fft.ifftn(np.fft.fftn(e) * H))[:n36, :n36, :n36] + m).astype(np.float32))
    return np.stack(out), Hb


@torch.no_grad()
def run(W):
    """W: (n,36,36,36) float → {方法: 每窗 2 μm 孔隙率}（中心 224 μm）。"""
    out = {k: [] for k in MM}
    for s in range(0, len(W), BS):
        lr = torch.from_numpy(np.asarray(W[s:s + BS], np.float32))[:, None].to(dev); B = lr.shape[0]
        k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k)
        O = {'粗扫阈值': lr[..., sl, sl, sl], '三线性': F.interpolate(lr[..., sl, sl, sl], size=(N,) * 3, mode='trilinear', align_corners=False)}
        for nm, m in base.items(): O[nm] = proj(m(lr), lr)
        mu, feat = M(lr, ret_feat=True); torch.manual_seed(7 + s); O['本文·仅粗扫'] = proj(Gn(mu, cconst.expand(B, -1), a_feat=feat)[2], lr)
        for nm, o in O.items(): out[nm] += (o[:, 0] < PORE_CUT).float().mean(dim=(1, 2, 3)).cpu().tolist()
    return out


res = {'fold': FOLD, 'test': meta['test'], 'plugs': {}, 'small': {}}
own = lambda code: code == FOLD or code in ('SX', 'FJ')
for pid, t in sorted(TAB.items()):
    if not own(t['code']) or (ONLY and pid not in ONLY.split(',')): continue
    f = PD + '%s.npz' % pid
    if not os.path.exists(f): print('缺', pid, flush=True); continue
    d = np.load(f); W = d['win']
    r = run(W); Wh, Hb = harmonize(W); rh = run(Wh)
    res['plugs'][pid] = dict(t, phi_coarse_all=float(d['phi_coarse']), p2=float(d['p2']), p10=float(d['p10']), std=float(d['std']),
                             win_p2=np.percentile(W.reshape(len(W), -1).astype(np.float32), 2, axis=1).mean().item(),
                             win_std=float(W.astype(np.float32).std(axis=(1, 2, 3)).mean()), H=Hb.tolist(), raw=r, harm=rh)
    print(FOLD, pid, t['group'], t['ab'], '原始', ' '.join('%s %.4f' % (k, np.mean(v)) for k, v in r.items()), '| 氦 %.2f' % t['phi_he'], flush=True)
    print(FOLD, pid, t['group'], t['ab'], '校正', ' '.join('%s %.4f' % (k, np.mean(v)) for k, v in rh.items()), flush=True)
    if CUP:                                            # 成像条件标定：近表面空气 + 杯状伪影（射束硬化）径向校正（参数由本地 cup_plugs.py、airnear_plugs.py 得到）
        c = np.load(PD + 'cup/%s.npz' % pid); a0, D0, an = float(c['air']), float(c['D']), float(c['air_near']); Dn = a0 + D0 - an
        oy, ox = np.mgrid[-18:18, -18:18]
        G = np.stack([np.polyval(c['coef'], (np.hypot(y + oy - c['cy'], x + ox - c['cx']) / c['R']) ** 2) for _, y, x in c['pos']])
        Gn_ = (G * D0 + a0 - an) / Dn                                                   # 骨架剖面换到新灰度单位
        Wc = (((W.astype(np.float32) * D0 + a0 - an) / Dn) / Gn_[:, None]).astype(np.float32); rc = run(Wc)
        Wch, _ = harmonize(Wc, (D0 / Dn) ** 2); rch = run(Wch)
        res['plugs'][pid].update(cal=rc, cal_harm=rch, air_near=an, contrast_gain=Dn / D0)
        print(FOLD, pid, t['group'], t['ab'], '标定', ' '.join('%s %.4f' % (k, np.mean(v)) for k, v in rc.items()), flush=True)
        print(FOLD, pid, t['group'], t['ab'], '标定+谱减', ' '.join('%s %.4f' % (k, np.mean(v)) for k, v in rch.items()), flush=True)
# 对照：本折测试小柱的粗扫（与大圆柱同一套方法）+ 细扫真值
for g in meta['test']:
    ds = PairDataset(ROOT, [g], c_src='blk'); ii = np.random.default_rng(0).choice(len(ds), min(300, len(ds)), replace=False)
    W = np.stack([ds[int(j)]['lr'][0] for j in ii]); r = run(W)
    r['细扫真值'] = [float((ds[int(j)]['hr'][0] < PORE_CUT).mean()) for j in ii]
    res['small'][g] = dict(r, win_p2=np.percentile(W.reshape(len(W), -1), 2, axis=1).mean().item(), win_std=float(W.std(axis=(1, 2, 3)).mean()))
    print(FOLD, '小柱', g, ' '.join('%s %.4f' % (k, np.mean(v)) for k, v in r.items()), flush=True)
res['model'] = [MS, GS, B_SUF]
json.dump(res, open(OUT + 'plugsr59_%s%s%s%s%s.json' % (FOLD, GS, '_cup' if CUP else '', '_only' if ONLY else '', ('_b' + B_SUF) if B_SUF else ''), 'w'), ensure_ascii=False, indent=0)
print('DONE', FOLD, flush=True)
