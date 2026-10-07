# -*- coding: utf-8 -*-
"""第 56 步诊断（只读、CPU）：粗扫预测 ĉ 在测试岩性上错在哪——整体偏移还是块级起伏？
以及「远处少量细扫做标定」能补回多少。标定块与目标块至少相距 3 格（>750 μm，块级相关已≈0），算「周围」而非紧邻。"""
import sys, json, numpy as np, torch
from pathlib import Path
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from models54 import load_cpred
torch.set_num_threads(4)
R = Path(str(DATA_ROOT)); NM = ['IGV', 'f_pore', 'mu_inter', 'f_dense']
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
cf = {p.stem: dict(np.load(p)) for p in sorted((R / 'cfield2').glob('*.npz'))}
r2 = lambda p, t: np.array([1 - ((p[:, k] - t[:, k]) ** 2).sum() / max(((t[:, k] - t[:, k].mean()) ** 2).sum(), 1e-12) for k in range(4)])
rng = np.random.default_rng(0); out = {}
for F in FOLDS:
    meta = fold_meta(F); m = load_cpred(str(RUNS_ROOT) + '/%s_cpred/final.pt' % F, 'cpu')
    G = {}
    for g in meta['test']:
        cells, L = [], []
        for f in sorted((R / 'pairs' / g).glob('*.npz')):
            z = np.load(f); cells.append(z['cell'][:3].astype(int)); L.append(z['lr'].astype(np.float32))
        cells = np.array(cells); d = cf[g]
        with torch.no_grad():
            ch = np.concatenate([m(torch.from_numpy(np.stack(L[i:i + 50]))[:, None]).numpy() for i in range(0, len(L), 50)])
        ix = tuple(cells.T); cb = d['c_blk'][ix].astype(float); ok = d['valid'][ix] & np.isfinite(cb).all(1) & np.isfinite(ch).all(1)
        G[g] = dict(cell=cells[ok], ch=ch[ok].astype(float), c=cb[ok], nbr=d['c_nbr_' + F][ix][ok].astype(float), ann=d['c_ann'][ix][ok].astype(float))
    C = np.concatenate([G[g]['c'] for g in G]); CH = np.concatenate([G[g]['ch'] for g in G])
    mse = ((CH - C) ** 2).mean(0); bias2 = np.concatenate([np.repeat((G[g]['ch'] - G[g]['c']).mean(0, keepdims=True) ** 2, len(G[g]['c']), 0) for g in G]).mean(0)
    wcorr = np.array([np.corrcoef(np.concatenate([G[g]['ch'][:, k] - G[g]['ch'][:, k].mean() for g in G]),
                                  np.concatenate([G[g]['c'][:, k] - G[g]['c'][:, k].mean() for g in G]))[0, 1] for k in range(4)])
    res = dict(n=len(C), 粗扫预测=r2(CH, C), 偏移占误差比例=bias2 / mse, 样品内相关_粗扫=wcorr,
               周围回归=r2(np.concatenate([G[g]['nbr'] for g in G]), C))
    for K in (1, 3, 10):
        P1, P2 = [], []
        for g in G:
            cl, ch, c = G[g]['cell'], G[g]['ch'], G[g]['c']
            for i in range(len(c)):
                far = np.where(np.abs(cl - cl[i]).max(1) >= 3)[0]; j = rng.choice(far, min(K, len(far)), replace=False)
                P1.append(ch[i] + (c[j] - ch[j]).mean(0)); P2.append(c[j].mean(0))
        res['远处%d块细扫标定+粗扫' % K] = r2(np.array(P1), C); res['只用远处%d块细扫均值' % K] = r2(np.array(P2), C)
    out[F] = {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}
    print('== %s（测试岩性 %s，%d 块）R² 顺序 %s' % (F, meta['test'], len(C), NM), flush=True)
    for k, v in out[F].items():
        if k != 'n': print('   %-22s %s' % (k, v), flush=True)
print('\n== 6 折均值')
for k in out['CQ']:
    if k != 'n': print('   %-22s %s' % (k, np.round(np.mean([out[F][k] for F in FOLDS], 0), 3).tolist()))
print('   （中位数）')
for k in out['CQ']:
    if k != 'n': print('   %-22s %s' % (k, np.round(np.median([out[F][k] for F in FOLDS], 0), 3).tolist()))
json.dump(out, open(str(OUT_ROOT) + '/eval56/chk.json', 'w'), ensure_ascii=False, indent=1)
print('DONE')
