# -*- coding: utf-8 -*-
"""纹理保真图（4.1 节，替换原伪影图；只比较最终模型与基线）。
(a) 云南折第 1 块（与图 visual 同一块）31 μm 以下细节带的 7³ 局部能量，同一色标（data/vis_paper.npz）；
(b) 同一块细节带三维功率谱沿 kz 求和，对数色标（data/eval/tex59.npz）；
(c) 径向功率谱（6 个测试产地各 60 块平均）；(d) 细节能量与细扫之比（各产地先按总能量求比，柱 = 6 产地均值）；
(e) 条纹指数（data/eval/eval55c，6 折，每折 200 块），虚线为细扫。"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LogNorm

V = np.load(FIGDATA / 'vis_paper.npz'); X = np.load(FIGDATA / 'eval/tex59.npz')
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
EV = {f: json.load(open(FIGDATA / 'eval/eval55c' / ('eval55_%s.json' % f), encoding='utf-8')) for f in FOLDS}
IM = [('真值', '细扫', 'Fine scan', '细扫'), ('EDSR-3D', 'EDSR-3D', 'EDSR-3D', 'EDSR-3D'), ('SRGAN-3D', 'SRGAN-3D', 'SRGAN-3D', 'SRGAN-3D'), ('本文·无地质', '本文', 'GeoCondSR', '本文')]
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 96 * T.MM))
outer = GridSpec(1, 2, figure=fig, left=0.035, right=0.99, top=0.93, bottom=0.1, wspace=0.12, width_ratios=[1.05, 1])
gi = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0, 0], wspace=0.05, hspace=0.18)
emax = np.percentile(V['YN|能量|真值'], 99.5)
smax = X['spec|细扫'].max(); snorm = LogNorm(smax * 1e-4, smax)
for j, (kv, kx, t, ck) in enumerate(IM):
    ax = fig.add_subplot(gi[0, j]); ax.imshow(V['YN|能量|%s' % kv], cmap='magma', vmin=0, vmax=emax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=6.4, pad=2.5)
    r = V['YN|能量|%s' % kv].mean() / V['YN|能量|真值'].mean()
    ax.text(0.04, 0.04, 'energy ×%.2f' % r, transform=ax.transAxes, color='white', fontsize=5.4)
    if j == 0: ax.set_ylabel('Local detail energy', fontsize=6.2); T.label(ax, 'a', dx=-0.08, dy=1.05)
    ax = fig.add_subplot(gi[1, j]); ax.imshow(X['spec|%s' % kx], cmap='viridis', norm=snorm, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    if j == 0: ax.set_ylabel(r'Detail spectrum ($k_x$–$k_y$)', fontsize=6.2); T.label(ax, 'b', dx=-0.08, dy=1.05)
gr = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 1], wspace=0.5, hspace=0.55, height_ratios=[1.1, 1])
# (c) 径向功率谱
ax = fig.add_subplot(gr[0, :]); k = X['kbins'] / 2.0
for kx, t, ck in (('三线性', 'Trilinear', '三线性'), ('EDSR-3D', 'EDSR-3D', 'EDSR-3D'), ('SRGAN-3D', 'SRGAN-3D', 'SRGAN-3D'), ('本文', 'GeoCondSR', '本文'), ('细扫', 'Fine scan', '细扫')):
    P = np.mean([X['P|%s|%s' % (f, kx)] for f in FOLDS], 0)
    ax.plot(k[1:], P[1:], color=T.C[ck], lw=1.3 if ck in ('本文', '细扫') else 0.9, ls='--' if ck == '细扫' else '-', label=t, zorder=4 if ck == '细扫' else 3)
ax.axvline(1 / 31, color='#8A8A8A', lw=0.5, ls=(0, (2, 2))); ax.text(1 / 31 * 1.07, 0.93, '31 μm', transform=ax.get_xaxis_transform(), fontsize=5.5, color='#6E6E6E')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'Spatial frequency (μm$^{-1}$)'); ax.set_ylabel('Power')
ax.legend(fontsize=5.6, loc='lower left', handlelength=1.6, ncol=2, columnspacing=0.8)
T.label(ax, 'c', dx=-0.1)
# (d) 细节能量比
ax = fig.add_subplot(gr[1, 0]); MS = [('三线性', 'Tri.'), ('EDSR-3D', 'EDSR'), ('SRGAN-3D', 'SRGAN'), ('本文', 'GeoCondSR')]
for i, (m, lab) in enumerate(MS):
    v = np.array([X['E|%s|%s' % (f, m)].sum() / X['E|%s|细扫' % f].sum() for f in FOLDS])
    ax.bar(i, v.mean(), 0.66, color=T.C[m], zorder=2); ax.scatter(i + np.linspace(-0.18, 0.18, 6), v, s=4, color='#333333', alpha=0.6, lw=0, zorder=3)
    print('细节能量比', m, np.round(v, 3), '均值 %.3f' % v.mean())
ax.axhline(1, color='#1A1A1A', lw=0.6, ls='--'); ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in MS], fontsize=5.8, rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Detail energy / fine scan', fontsize=6.2); ax.set_ylim(0, 1.7)
T.label(ax, 'd', dx=-0.34)
# (e) 条纹指数
ax = fig.add_subplot(gr[1, 1]); MS2 = [('三线性', '三线性', 'Tri.'), ('EDSR-3D', 'EDSR-3D', 'EDSR'), ('SRGAN-3D', 'SRGAN-3D', 'SRGAN'), ('新·无地质', '本文', 'GeoCondSR')]
for i, (m, ck, lab) in enumerate(MS2):
    v = np.array([EV[f][m]['条纹指数'] for f in FOLDS])
    ax.bar(i, v.mean(), 0.66, color=T.C[ck], zorder=2); ax.scatter(i + np.linspace(-0.18, 0.18, 6), v, s=4, color='#333333', alpha=0.6, lw=0, zorder=3)
    print('条纹指数', m, '%.3f' % v.mean())
ft = np.mean([EV[f]['真值条纹指数'] for f in FOLDS]) if not isinstance(EV['CQ']['真值条纹指数'], dict) else np.nan
ax.axhline(0.025, color='#1A1A1A', lw=0.6, ls='--'); ax.text(3.45, 0.027, 'fine scan', fontsize=5.3, ha='right', va='bottom')
ax.set_xticks(range(4)); ax.set_xticklabels([l for *_, l in MS2], fontsize=5.8, rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Stripe index', fontsize=6.2)
T.label(ax, 'e', dx=-0.34)
T.save(fig, H / 'fig_texture')