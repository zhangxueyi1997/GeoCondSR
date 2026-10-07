# -*- coding: utf-8 -*-
"""局外人诊断：c_blk 的方差在哪一层（岩性/样品/块），块与块之间空间相关多长。只读本地 cfield2 原始文件。"""
import sys, glob, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../preprocessing')); from paths import DATA_ROOT, OUT_ROOT  # noqa: E402
from admit_real import CODE
NM = ['IGV', 'f_pore', 'mu_inter', 'f_dense']
D = {}
for f in sorted(glob.glob(str(DATA_ROOT / 'cfield2' / '*.npz'))):
    g = f[-7:-4]; d = np.load(f)
    ok = d['valid'] & np.isfinite(d['c_blk']).all(-1) & np.isfinite(d['c_ann']).all(-1)
    D[g] = dict(blk=d['c_blk'].astype(float), ann=d['c_ann'].astype(float), cell=d['c_cell'].astype(float), ok=ok, code=CODE[g])
# A. 方差分解
allv = np.concatenate([D[g]['blk'][D[g]['ok']] for g in D]); gm = allv.mean(0); tot = allv.var(0)
codes = sorted({D[g]['code'] for g in D})
bl = np.zeros(4); bs = np.zeros(4); wi = np.zeros(4); n = len(allv)
for c in codes:
    gs = [g for g in D if D[g]['code'] == c]; vc = np.concatenate([D[g]['blk'][D[g]['ok']] for g in gs]); mc = vc.mean(0)
    bl += len(vc) * (mc - gm) ** 2
    for g in gs:
        v = D[g]['blk'][D[g]['ok']]; ms = v.mean(0); bs += len(v) * (ms - mc) ** 2; wi += ((v - ms) ** 2).sum(0)
print('A. c_blk 方差分解（占总方差 %）：', NM)
print('   岩性之间        ', np.round(100 * bl / n / tot, 1))
print('   同岩性样品之间  ', np.round(100 * bs / n / tot, 1))
print('   同样品块与块之间', np.round(100 * wi / n / tot, 1))
print('   各岩性均值（标准化到总标准差）：')
for c in codes:
    vc = np.concatenate([D[g]['blk'][D[g]['ok']] for g in D if D[g]['code'] == c])
    print('     %-4s n=%4d 均值 %s  岩性内标准差/总标准差 %s' % (c, len(vc), np.round((vc.mean(0) - gm) / np.sqrt(tot), 2), np.round(vc.std(0) / np.sqrt(tot), 2)))
# B. 样品内去均值后，块与块的空间相关（按方向、按滞后）
print('\nB. 同样品内（去样品均值）c_blk 与相距 k 格的 c_blk 的相关；1 格 = 252 μm 中心距，块不重叠')
for ax, an in [(0, 'z(深度)'), (1, 'y'), (2, 'x')]:
    for k in (1, 2, 3, 4):
        A_, B_ = [], []
        for g in D:
            v, ok = D[g]['blk'], D[g]['ok']; r = v - v[ok].mean(0)
            s0 = [slice(None)] * 3; s1 = [slice(None)] * 3; s0[ax] = slice(0, -k); s1[ax] = slice(k, None)
            m = ok[tuple(s0)] & ok[tuple(s1)]
            A_.append(r[tuple(s0)][m]); B_.append(r[tuple(s1)][m])
        A_, B_ = np.concatenate(A_), np.concatenate(B_)
        print('   %-7s 滞后 %d：%s  (对数 %d)' % (an, k, np.round([np.corrcoef(A_[:, j], B_[:, j])[0, 1] for j in range(4)], 2), len(A_)))
# C. 样品内：目标块与自身环带、整格的相关（去样品均值）
print('\nC. 同样品内（去样品均值）c_blk 与 本格环带 c_ann / 整格 c_cell 的相关')
for key in ('ann', 'cell'):
    A_, B_ = [], []
    for g in D:
        ok = D[g]['ok']; a = D[g]['blk'][ok]; b = D[g][key][ok]
        A_.append(a - a.mean(0)); B_.append(b - b.mean(0))
    A_, B_ = np.concatenate(A_), np.concatenate(B_)
    print('   %-4s %s' % (key, np.round([np.corrcoef(A_[:, j], B_[:, j])[0, 1] for j in range(4)], 2)))
print('   分岩性（c_ann）：')
for c in codes:
    A_, B_ = [], []
    for g in D:
        if D[g]['code'] != c: continue
        ok = D[g]['ok']; a = D[g]['blk'][ok]; b = D[g]['ann'][ok]; A_.append(a - a.mean(0)); B_.append(b - b.mean(0))
    A_, B_ = np.concatenate(A_), np.concatenate(B_)
    print('     %-4s %s' % (c, np.round([np.corrcoef(A_[:, j], B_[:, j])[0, 1] for j in range(4)], 2)))
