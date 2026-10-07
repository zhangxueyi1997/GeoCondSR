# -*- coding: utf-8 -*-
"""微结构统计曲线（4.1 节），数字岩心超分论文的常用评价。数据：data/eval/micro59.json（服务器 eval59/micro59.py：
6 个测试产地各取终评块序前 60 块，共 360 块；孔隙相 u<0.5）。
(a) 两点概率函数 S2(r)（360 块平均）；(b) 归一化自协方差 (S2 − E[φ²]) / (E[φ] − E[φ²])，去掉孔隙量差别后比较孔隙尺度；
(c) 孔隙弦长分布（三个轴向合并，按条数归一化）；(d) 孔隙团等效直径分布（按体积加权，纵轴为该直径段贡献的孔隙率）。"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

R = json.load(open(FIGDATA / 'eval/micro59.json', encoding='utf-8'))
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; VOX = 2.0; NV = 112 ** 3
M = [('细扫', 'Fine scan', '-'), ('三线性', 'Trilinear', '-'), ('EDSR-3D', 'EDSR-3D', '-'), ('SRGAN-3D', 'SRGAN-3D', '-'), ('本文', 'Ours (coarse only)', '-')]
AGG = {}
for k, *_ in M:
    phi = np.concatenate([R[f][k]['phi'] for f in FOLDS]); nb = [len(R[f][k]['phi']) for f in FOLDS]
    s2 = sum(np.array(R[f][k]['s2']) * n for f, n in zip(FOLDS, nb)) / sum(nb)
    ch = sum(np.array(R[f][k]['chords']) for f in FOLDS); cl = sum(np.array(R[f][k]['clusters']) for f in FOLDS)
    AGG[k] = dict(phi=phi, s2=s2, ch=ch, cl=cl, n=sum(nb))
bins = np.array(R['bins_um']); r = np.arange(len(AGG['细扫']['s2'])) * VOX; L = np.arange(113) * VOX

fig = plt.figure(figsize=(T.DOUBLE * T.MM, 56 * T.MM))
gs = GridSpec(1, 5, figure=fig, left=0.05, right=0.99, top=0.9, bottom=0.2, wspace=0.5, width_ratios=[1, 1, 1, 1, 0.95])
axs = [fig.add_subplot(gs[0, i]) for i in range(5)]
STAT = {}
for k, name, ls in M:
    a = AGG[k]; col = T.C[k]; lw = 1.2 if k in ('细扫', '本文') else 0.9
    axs[0].plot(r, 100 * a['s2'], color=col, lw=lw, ls=ls, label=name)
    Ep, Ep2 = a['phi'].mean(), (a['phi'] ** 2).mean()
    axs[1].plot(r, (a['s2'] - Ep2) / (Ep - Ep2), color=col, lw=lw, ls=ls)
    p = a['ch'][1:] / a['ch'][1:].sum()
    axs[2].plot(L[1:], p, color=col, lw=lw, ls=ls)
    cen = np.sqrt(bins[:-1] * bins[1:])
    axs[3].plot(cen, 100 * a['cl'] / (a['n'] * NV), color=col, lw=lw, ls=ls, marker='o', ms=1.8)
    mc = (L[1:] * a['ch'][1:]).sum() / a['ch'][1:].sum()
    f_ = (a['s2'] - Ep2) / (Ep - Ep2); rc = np.interp(np.exp(-1), f_[::-1], r[::-1])
    sp = 100 * a['cl'][cen < 5].sum() / a['cl'].sum(); STAT[k] = (mc, rc, sp)
    print('%-10s φ %.4f  平均弦长 %.1f μm  自协方差降到 1/e 的距离 %.1f μm  <5μm 团占孔隙 %.1f%%' % (k, Ep, mc, rc, sp))
axs[0].set_xlabel('Lag r (μm)'); axs[0].set_ylabel(r'Two-point probability $S_2(r)$ (%)'); axs[0].set_xlim(0, 80)
axs[0].legend(loc='upper right', fontsize=5.8, handlelength=1.4)
axs[1].set_xlabel('Lag r (μm)'); axs[1].set_ylabel('Normalized autocovariance'); axs[1].set_xlim(0, 80); axs[1].axhline(np.exp(-1), color='#BDBDBD', lw=0.5, ls=(0, (2, 2)))
axs[1].text(78, np.exp(-1) + 0.02, '1/e', fontsize=5.5, color='#8A8A8A', ha='right')
axs[2].set_xscale('log'); axs[2].set_yscale('log'); axs[2].set_xlabel('Pore chord length (μm)'); axs[2].set_ylabel('Probability'); axs[2].set_xlim(2, 200)
axs[3].set_xscale('log'); axs[3].set_xlabel('Cluster diameter (μm)'); axs[3].set_ylabel('Porosity per size class (%)')
ax = axs[4]; MM = [('EDSR-3D', 'EDSR'), ('SRGAN-3D', 'SRGAN'), ('本文', 'Ours')]; w = 0.26
for q, (desc, dn) in enumerate([(0, 'mean\nchord'), (1, 'correl.\nlength'), (2, 'pores\n< 5 μm')]):
    for j, (m, lab) in enumerate(MM):
        e = 100 * abs(STAT[m][desc] - STAT['细扫'][desc]) / STAT['细扫'][desc]
        ax.bar(q + (j - 1) * w, e, w * 0.95, color=T.C[m], zorder=2)
        if m == '本文': ax.text(q + (j - 1) * w, e + 2, '%d' % round(e), ha='center', va='bottom', fontsize=5.3, color=T.C['本文'], fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels(['mean\nchord', 'correl.\nlength', 'pores\n< 5 μm'], fontsize=5.6)
ax.set_ylabel('Relative error vs. fine scan (%)'); ax.set_ylim(0, 130)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=T.C[m], label=l) for m, l in MM], fontsize=5.5, loc='upper right', handlelength=1.0)
for ax, lab in zip(axs, 'abcde'): T.label(ax, lab, dx=-0.22)
T.save(fig, H / 'fig_micro')