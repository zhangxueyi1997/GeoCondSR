# -*- coding: utf-8 -*-
"""视觉对比（4.1 节），超分论文通行的“整图 + 放大”版式。6 个测试产地各一块（终评块序第 1 块，未经挑选；data/vis_paper.npz）。
第 1 列为整块 224 μm 细扫中心层，红框为放大区域（统一取块中心 112 μm，不按内容挑选）；其余各列为同一区域的放大：
粗扫（16² 中的中心 8²，最近邻）、三线性、EDSR-3D、SRGAN-3D、本文（只用粗扫）、本文（加稀疏细扫）、细扫。灰度窗 [0, 1.6]。"""
import sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

d = np.load(FIGDATA / 'vis_paper.npz')
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
PN = {'CQ': 'Chongqing', 'SC': 'Sichuan', 'YN': 'Yunnan', 'GZ': 'Guizhou', 'SD': 'Shandong', 'SHX': 'Shaanxi'}
COLS = [('粗扫', 'Coarse 14 μm', '#6E6E6E'), ('三线性', 'Trilinear', '#6E6E6E'), ('EDSR-3D', 'EDSR-3D', T.C['EDSR-3D']), ('SRGAN-3D', 'SRGAN-3D', T.C['SRGAN-3D']),
        ('本文·无地质', 'Ours', T.C['本文']), ('本文·粗扫+稀疏细扫', 'Ours + sparse fine', T.C['本文+稀疏细扫']), ('真值', 'Fine scan 2 μm', '#1A1A1A')]
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 142 * T.MM))
gs = GridSpec(len(FOLDS), 9, figure=fig, left=0.035, right=0.995, top=0.955, bottom=0.01, wspace=0.05, hspace=0.07,
              width_ratios=[1.15, 0.12] + [1] * 7)


def frame(ax, col='#808080', lw=0.4):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color(col); s.set_linewidth(lw)


for i, f in enumerate(FOLDS):
    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(d['%s|真值' % f], cmap='gray', vmin=0, vmax=1.6, interpolation='nearest', extent=(0, 224, 224, 0))
    ax.add_patch(Rectangle((56, 56), 112, 112, fill=False, ec='#E03A2F', lw=0.9)); frame(ax)
    ax.set_ylabel('%s (%s)' % (PN[f], str(d['%s|gid' % f])), fontsize=6.3, labelpad=2)
    if i == 0: ax.set_title('Fine scan, 224 μm', fontsize=6.3, pad=2.5)
    if i == len(FOLDS) - 1:
        ax.plot([160, 210], [212, 212], color='white', lw=1.3, solid_capstyle='butt'); ax.text(185, 205, '50 μm', color='white', fontsize=5.3, ha='center', va='bottom')
    for j, (k, t, col) in enumerate(COLS):
        ax = fig.add_subplot(gs[i, j + 2]); a = d['%s|%s' % (f, k)]
        a = a[4:12, 4:12] if a.shape[0] == 16 else a[28:84, 28:84]
        ax.imshow(a, cmap='gray', vmin=0, vmax=1.6, interpolation='nearest', extent=(0, 112, 112, 0)); frame(ax, '#E03A2F', 0.6)
        if i == 0: ax.set_title(t, fontsize=6.3, pad=2.5, color=col, fontweight='bold' if k.startswith('本文') else 'normal')
        if i == len(FOLDS) - 1 and j == len(COLS) - 1:
            ax.plot([80, 100], [104, 104], color='white', lw=1.3, solid_capstyle='butt'); ax.text(90, 100, '20 μm', color='white', fontsize=5.3, ha='center', va='bottom')
fig.text(0.61, 0.985, 'Enlarged: central 112 μm of each block (red box)', ha='center', va='top', fontsize=6.3, color='#E03A2F')
T.save(fig, H / 'fig05_visual_sci')