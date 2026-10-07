# -*- coding: utf-8 -*-
"""地质控制图（4.3 节）：同一粗扫输入、同一噪声，只替换地质状态 c（14 个样品的整体状态）。数据：data/geo_ctrl59.npz（服务器 eval59/geo_ctrl59.py，重庆折有地质版 _g54b）。
(a) G01 目标块（图 2 同一块）中心层：粗扫、5 种状态下的输出、细扫。 (b) 两个极端状态的输出之差。
(c) 重庆折前 40 个终评块：替换状态后输出孔隙率的变化（相对各块本样品自身状态）与状态 f_p 之差，按块所属样品（G01 / G19）分两组，点为均值，竖线为四分位距。"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

Z = np.load(FIGDATA / 'geo_ctrl59.npz'); NAMES = list(Z['c_names']); C = Z['c_vals']
P, G = Z['sweep|phi'], Z['sweep|gid']
FP, UP = r'$f_\mathrm{p}$', r'$\bar{u}_\mathrm{p}$'
SHOW = ['G11', 'G12', 'G19', 'G01', 'G06']
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 100 * T.MM))
gt = GridSpec(1, 7, figure=fig, left=0.012, right=0.988, top=0.90, bottom=0.535, wspace=0.05)
gb = GridSpec(1, 3, figure=fig, left=0.06, right=0.935, top=0.44, bottom=0.09, wspace=0.46, width_ratios=[0.8, 1.25, 1.0])


def img(ax, a, title, sub=None, col='#222222', cmap='gray', lo=0, hi=1.6):
    ax.imshow(np.asarray(a, np.float32), cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest', extent=(0, 224, 224, 0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.4); s.set_color('#808080')
    ax.set_title(title, fontsize=6.3, pad=2.5, color=col)
    if sub: ax.text(0.5, -0.05, sub, transform=ax.transAxes, ha='center', va='top', fontsize=5.6, color='#333333')


ax = fig.add_subplot(gt[0, 0]); img(ax, Z['arch|lr|mid'][10:26, 10:26], 'Coarse input', '14 μm, same for all')
T.label(ax, 'a', dx=0.0, dy=1.13)
for k, g in enumerate(SHOW):
    q = NAMES.index(g); ph = Z['ctrl|%s|stat' % g][0]
    ax = fig.add_subplot(gt[0, k + 1])
    img(ax, Z['ctrl|%s|mid' % g], 'state of %s' % g, FP + ' %.1f%%, ' % (100 * C[q, 1]) + UP + ' %.2f' % C[q, 2] + chr(10) + r'output $\phi$ = %.1f%%' % (100 * ph),
        col='#C0392B' if g == 'G01' else '#222222')
ax = fig.add_subplot(gt[0, 6]); img(ax, Z['arch|hr|mid'], 'Fine scan', '2 μm' + chr(10) + r'$\phi$ = %.1f%%' % (100 * float(Z['ctrl|hr_phi'][0])))
fig.text(0.5, 0.955, 'Same coarse block (G01) and same noise; only the geological state c is replaced', ha='center', fontsize=6.5, color='#444444')

# (b) 差图
ax = fig.add_subplot(gb[0, 0])
d = Z['ctrl|G06|mid'].astype(np.float32) - Z['ctrl|G11|mid'].astype(np.float32)
im = ax.imshow(d, cmap='RdBu_r', vmin=-0.5, vmax=0.5, interpolation='nearest', extent=(0, 224, 224, 0))
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.4); s.set_color('#808080')
ax.set_title('G06 state − G11 state', fontsize=6.3, pad=2.5)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.4)
cb.set_label(r'$\Delta$ grey', fontsize=5.8, labelpad=1)
T.label(ax, 'b', dx=-0.05, dy=1.03)

# (c) 替换状态的剂量-响应
ax = fig.add_subplot(gb[0, 1])
res = []
for own, col, mk in (('G01', '#1A1A1A', 'o'), ('G19', '#0072B2', 's')):
    rows = np.where(G == own)[0]; q0 = NAMES.index(own)
    dphi = 100 * (P[rows] - P[rows, q0][:, None]); dfp = 100 * (C[:, 1] - C[q0, 1])
    o = np.argsort(dfp)
    ax.vlines(dfp[o], np.percentile(dphi, 25, 0)[o], np.percentile(dphi, 75, 0)[o], color=col, lw=0.6, alpha=0.6)
    ax.plot(dfp[o], dphi.mean(0)[o], marker=mk, ms=3, lw=0.8, color=col, label='blocks of %s (n = %d)' % (own, len(rows)))
    res += [spearmanr(C[:, 1], P[i])[0] for i in rows]
ax.axhline(0, color='#BDBDBD', lw=0.5); ax.axvline(0, color='#BDBDBD', lw=0.5)
ax.set_xlabel('Change in state ' + FP + ' (percentage points)'); ax.set_ylabel(r'Change in output porosity (pp)')
ax.legend(loc='upper left', fontsize=5.8, handlelength=1.5)
ax.text(0.97, 0.05, r'per-block Spearman $\rho$ (' + FP + r', $\phi$)' + chr(10) + 'median %.2f, n = %d' % (np.median(res), len(res)), transform=ax.transAxes, ha='right', va='bottom', fontsize=5.8)
T.label(ax, 'c', dx=-0.16)

# (d) 各状态下 40 块的平均输出孔隙率
ax = fig.add_subplot(gb[0, 2])
m = 100 * P.mean(0); sc = ax.scatter(100 * C[:, 1], m, c=C[:, 2], cmap='viridis_r', s=16, edgecolor='white', lw=0.4, zorder=3, vmin=0.1, vmax=0.47)
for g in ('G11', 'G19', 'G01', 'G06', 'G13'):
    q = NAMES.index(g); ax.annotate(g, (100 * C[q, 1], m[q]), xytext=((-14, -2) if g == 'G06' else (3, -3 if g != 'G13' else 3)), textcoords='offset points', fontsize=5.5)
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.4)
cb.set_label('state ' + UP, fontsize=5.8, labelpad=1)
rho = spearmanr(C[:, 1], m)[0]
ax.set_xlabel('State ' + FP + ' (%)'); ax.set_ylabel('Mean output porosity (%)')
ax.text(0.04, 0.96, r'Spearman $\rho$ = %.2f' % rho + chr(10) + '14 states, 40 blocks', transform=ax.transAxes, va='top', fontsize=5.8)
T.label(ax, 'd', dx=-0.2)
print('逐块 Spearman 中位 %.3f（%d 块）；均值曲线 Spearman %.3f；输出孔隙率 %.2f%%–%.2f%%；逐块极差中位 %.2f pp' % (np.median(res), len(res), rho, m.min(), m.max(), 100 * np.median(P.max(1) - P.min(1))))
T.save(fig, H / 'fig_geoctrl')