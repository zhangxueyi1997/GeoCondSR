# -*- coding: utf-8 -*-
"""样品级：整块样品的粗扫统计量，能不能在「没见过的岩性」上猜出该样品细扫 c 的整体水平？（留一岩性，14 个样品）"""
import json, numpy as np
from pathlib import Path
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT  # noqa: E402
R = Path(str(DATA_ROOT)); NM = ['IGV', 'f_pore', 'mu_inter', 'f_dense']
CODE = {'G01': 'CQ', 'G19': 'CQ', 'G02': 'GZ', 'G03': 'YN', 'G16': 'YN', 'G17': 'YN', 'G05': 'SC', 'G06': 'SC', 'G07': 'SC', 'G09': 'SC', 'G11': 'SC', 'G12': 'SD', 'G13': 'SHX', 'G15': 'SX'}
S, C = {}, {}
for g in CODE:
    L = np.stack([np.load(f)['lr'].astype(np.float32) for f in sorted((R / 'pairs' / g).glob('*.npz'))]); v = L.ravel()
    q = np.percentile(v, [2, 10, 50, 90, 98])
    d = np.load(R / 'cfield2' / (g + '.npz')); ok = d['valid'] & np.isfinite(d['c_blk']).all(-1)
    S[g] = dict(均值=v.mean(), 标准差=v.std(), 暗部占比=(v < 0.5).mean(), 分位2=q[0], 分位10=q[1], 中位=q[2], 分位90=q[3], 分位98=q[4])
    C[g] = d['c_blk'][ok].astype(np.float64).mean(0)
G = list(CODE); Y = np.array([C[g] for g in G]); keys = list(S[G[0]])
print('每个样品粗扫统计（全部 200 个 36³ 窗口）与细扫 c 均值：')
for g in G: print('  %s %-4s %s | c %s' % (g, CODE[g], ' '.join('%s %.3f' % (k, S[g][k]) for k in keys), np.round(C[g], 3)))
print('\n单个粗扫统计量 → 样品 c 均值：14 样品相关 / 留一岩性预测的 R²（跨样品）')
for k in keys:
    x = np.array([S[g][k] for g in G]); P = np.zeros_like(Y)
    for lit in set(CODE.values()):
        te = np.array([CODE[g] == lit for g in G]); A = np.c_[x[~te], np.ones((~te).sum())]
        w = np.linalg.lstsq(A, Y[~te], rcond=None)[0]; P[te] = np.c_[x[te], np.ones(te.sum())] @ w
    r = [np.corrcoef(x, Y[:, j])[0, 1] for j in range(4)]; r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    print('  %-6s 相关 %s  留一岩性 R² %s' % (k, np.round(r, 2), np.round(r2, 2)))
print('DONE')
