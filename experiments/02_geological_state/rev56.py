# -*- coding: utf-8 -*-
import sys, glob, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../preprocessing')); from paths import DATA_ROOT, OUT_ROOT  # noqa: E402
from admit_real import CODE
NM = ['IGV', 'f_pore', 'mu_inter', 'f_dense']; D = {}
for f in sorted(glob.glob(str(DATA_ROOT / 'cfield2' / '*.npz'))):
    g = f[-7:-4]; d = np.load(f); ok = d['valid'] & np.isfinite(d['c_blk']).all(-1)
    D[g] = dict(b=d['c_blk'].astype(float), ok=ok, norm=d['norm'], code=CODE[g])
print('1. 各样品细扫的判读窗口 [lo, hi]（骨架 = lo..hi；IGV = 1 − 骨架占比）与样品均值 c')
SM = []
for g in D:
    m = D[g]['b'][D[g]['ok']].mean(0); SM.append(m)
    print('   %s %-4s lo %.2f hi %.2f   均值 c %s' % (g, D[g]['code'], D[g]['norm'][2], D[g]['norm'][3], np.round(m, 3)))
SM = np.array(SM); LO = np.array([D[g]['norm'][2] for g in D]); HI = np.array([D[g]['norm'][3] for g in D])
print('   样品均值 c 与 hi 的相关（14 个样品）：', np.round([np.corrcoef(HI, SM[:, k])[0, 1] for k in range(4)], 2))
print('   lo 被截到 0.52 的样品数：%d/14' % int((LO <= 0.5201).sum()))
print('\n2. 支撑尺度：把 m×m×m 个相邻块平均（边长约 m×252 μm），样品内标准差 / 样品之间标准差')
sb = SM.std(0)
for mm in (1, 2, 3, 4):
    W = []
    for g in D:
        b, ok = D[g]['b'], D[g]['ok']; nz, ny, nx = ok.shape; vals = []
        for z in range(0, nz - mm + 1, mm):
            for y in range(0, ny - mm + 1, mm):
                for x in range(0, nx - mm + 1, mm):
                    o = ok[z:z + mm, y:y + mm, x:x + mm]
                    if o.all(): vals.append(b[z:z + mm, y:y + mm, x:x + mm][o].mean(0))
        if len(vals) > 3: vals = np.array(vals); W.append(((vals - vals.mean(0)) ** 2).mean(0))
    w = np.sqrt(np.mean(W, 0))
    print('   m=%d（约 %4d μm）样品内/样品间 = %s' % (mm, mm * 252, np.round(w / sb, 2)))
