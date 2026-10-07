# -*- coding: utf-8 -*-
"""修正掩膜后重测「细扫区以外的粗扫」：去掉空气层（中心区域逐层中位 <0.6）、平面上只留列中位 >0.7 且向内腐蚀 10 体素的岩心内部。"""
import sys, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
sys.argv = ['x']; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ana56 as A
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, OUT_ROOT  # noqa: E402
R = Path(str(DATA_ROOT)); d = np.load(str(OUT_ROOT) + '/eval56/ana56_data.npz'); GS = A.GS
Fo, info = [], []
for g in GS:
    c = np.load(R / 'cfield' / (g + '.npz')); V = np.load(R / 'cache' / str(c['lr_cache'])).astype(np.float32); V = (V - c['norm'][0]) / c['norm'][1]
    z0 = int(c['lr_z0']); nz = np.load(R / 'cfield2' / (g + '.npz'))['valid'].shape[0]; fz0, fz1 = z0, z0 + (nz - 1) * 18 + 36
    ny, nx = V.shape[1:]; cm = np.median(V[:, ny // 2 - 60:ny // 2 + 60, nx // 2 - 60:nx // 2 + 60].reshape(V.shape[0], -1), 1)
    rock_z = cm > 0.6; rock_z = ndi.binary_erosion(rock_z, iterations=5)
    col = ndi.binary_erosion(np.median(V[rock_z][::4], 0) > 0.7, iterations=10)
    zz = np.where(rock_z & ((np.arange(V.shape[0]) < fz0) | (np.arange(V.shape[0]) >= fz1)))[0]
    v = V[zz][:, col][::2, ::2].ravel() if len(zz) else np.array([])
    Fo.append(A.cfeat(v)); info.append((g, len(zz), int(col.sum())))
Fo = np.array(Fo)
for (g, n, m), f, ff in zip(info, Fo, d['Ff']): print('%s 细扫区外岩心层数 %4d（%.1f mm） 平面列 %6d | out p2 %.2f p10 %.2f std %.2f | field p2 %.2f p10 %.2f std %.2f' % (g, n, n * 0.014, m, f[0], f[1], f[6], ff[0], ff[1], ff[6]))
cs = np.array([A.CODE[g] for g in GS]); Ys = np.array([d[g + '_rows'][:, 39:43][:, A.CI].mean(0) for g in GS])
Hs = np.array([np.nanmean(d[g + '_H'][:, 3:9], 0) for g in GS])
ok = np.isfinite(Fo).all(1); print('有效样品', ok.sum())
if ok.all():
    print('样品级 嵌套选特征 留一岩性 R²：细扫区以外（修正）%s ｜ 细扫区 %s' % (np.round(A.nested(Fo, Ys, cs), 2), np.round(A.nested(d['Ff'], Ys, cs), 2)))
    print('统一阈值描述量 [孔隙率, 高密相, 孔隙灰度, 孔径, 比表面, 粒径]：细扫区以外（修正）%s' % np.round(A.nested(Fo, Hs, cs), 2))
    print('                                                  细扫区           %s' % np.round(A.nested(d['Ff'], Hs, cs), 2))
    print('细扫区 与 细扫区以外 粗扫统计的样品间相关 [p2, p10, p50, std]：', np.round([np.corrcoef(Fo[:, k], d['Ff'][:, k])[0, 1] for k in (0, 1, 2, 6)], 2))
