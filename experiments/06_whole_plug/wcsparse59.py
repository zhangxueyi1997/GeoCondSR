# -*- coding: utf-8 -*-
"""整柱“粗扫 + 稀疏细扫”端到端检验（预登记 prereg_wcsparse_20261002.md）。只推理。
窗口与成像条件标定同 plugsr59.py 的 cal 条件；模型 _m59n + _g59n；c_i = ĉ_i − mean(ĉ) + level，level 取该组小柱细扫 10 个随机格点的 c_blk 均值。
变体：sparse（主）、sparse_all（level 取全部细扫格点，参照）、coarse（c = ĉ_i，只用粗扫）。噪声种子 7 + 批序号，与 plugsr59 一致。
用法：python wcsparse59.py <折> <gpu>"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models import degrade, Generator, N_C, PORE_CUT
from models54 import load_mean, load_cpred
from dataset import load_H
FOLD, dev = sys.argv[1], 'cuda:%s' % sys.argv[2]
ROOT = str(DATA_ROOT); RUNS = str(RUNS_ROOT); PD = str(DATA_ROOT) + '/plugs/'
TAB = json.load(open(PD + 'plug_table.json', encoding='utf-8'))
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev); N = 112; BS = 4


def proj(hr, lr, n=8):
    k = 16; off = (lr.shape[-1] - k) // 2; sl = slice(off, off + k); lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:], mode='trilinear', align_corners=False)
    return hr


M = load_mean('%s/%s_m59n/ckpt/final.pt' % (RUNS, FOLD), dev).eval()
G = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev).eval()
G.load_state_dict(torch.load('%s/%s_g59n/ckpt/final.pt' % (RUNS, FOLD), map_location=dev, weights_only=False)['model'])
CP = load_cpred('%s/%s_cpred/final.pt' % (RUNS, FOLD), dev)


@torch.no_grad()
def run(W, C):
    out = []
    for s in range(0, len(W), BS):
        lr = torch.from_numpy(np.asarray(W[s:s + BS], np.float32))[:, None].to(dev)
        c = torch.from_numpy(np.asarray(C[s:s + BS], np.float32)).to(dev)
        mu, feat = M(lr, ret_feat=True); torch.manual_seed(7 + s)
        o = proj(G(mu, c, a_feat=feat)[2], lr)
        out += (o[:, 0] < PORE_CUT).float().mean(dim=(1, 2, 3)).cpu().tolist()
    return out


res = {'fold': FOLD, 'model': ['_m59n', '_g59n'], 'plugs': {}}
for pid, t in sorted(TAB.items()):
    g = t['group']
    if t['code'] != FOLD or not os.path.exists('%s/cfield2/%s.npz' % (ROOT, g)): continue
    d = np.load(PD + '%s.npz' % pid); W = d['win']
    c = np.load(PD + 'cup/%s.npz' % pid); a0, D0, an = float(c['air']), float(c['D']), float(c['air_near']); Dn = a0 + D0 - an
    oy, ox = np.mgrid[-18:18, -18:18]
    Gp = np.stack([np.polyval(c['coef'], (np.hypot(y + oy - c['cy'], x + ox - c['cx']) / c['R']) ** 2) for _, y, x in c['pos']])
    Wc = (((W.astype(np.float32) * D0 + a0 - an) / Dn) / ((Gp * D0 + a0 - an) / Dn)[:, None]).astype(np.float32)
    with torch.no_grad():
        ch = np.concatenate([CP(torch.from_numpy(Wc[s:s + 32])[:, None].to(dev)).cpu().numpy() for s in range(0, len(Wc), 32)]).astype(np.float64)
    cf = np.load('%s/cfield2/%s.npz' % (ROOT, g)); cb = cf['c_blk'][cf['valid'] & np.isfinite(cf['c_blk']).all(-1)].astype(np.float64)
    idx = np.random.default_rng(5).choice(len(cb), 10, replace=False); lev10, levall = cb[idx].mean(0), cb.mean(0)
    V = {'sparse': ch - ch.mean(0) + lev10, 'sparse_all': ch - ch.mean(0) + levall, 'coarse': ch}
    phi = {k: run(Wc, v) for k, v in V.items()}
    res['plugs'][pid] = dict(t, lev10=lev10.tolist(), levall=levall.tolist(), chat_mean=ch.mean(0).tolist(), chat_std=ch.std(0).tolist(), phi=phi)
    print(FOLD, pid, g, t['ab'], ' '.join('%s %.4f' % (k, np.mean(v)) for k, v in phi.items()), '| 氦 %.2f' % t['phi_he'], '| level10 fp %.4f all %.4f ĉ %.4f' % (lev10[1], levall[1], ch.mean(0)[1]), flush=True)
os.makedirs(str(OUT_ROOT) + '/plugs59', exist_ok=True)
json.dump(res, open(str(OUT_ROOT) + '/plugs59/wcsparse59_%s.json' % FOLD, 'w'), ensure_ascii=False, indent=0)
print('DONE', flush=True)
