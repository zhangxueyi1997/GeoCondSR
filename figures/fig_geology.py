# -*- coding: utf-8 -*-
"""地质状态参数图（3.2 节）。
(a) G01 与 G19 细扫灰度直方图（results/hist_cache.npz 的 T：7 个横截面、0.88R 圆盘内），骨架窗、孔隙阈值与参数的取值区间；
    三角为样品级 ū_p（cfield2 全部有效格点均值，与正文数字一致）。
(b) 两样品各一个 224 μm 目标块的中心切片：14 μm 粗扫与 2 μm 细扫，灰度窗 [0, 1.6]。G01 取自 data/fig2_G01.npz（图 2 同一块），
    G19 取自 data/vis_paper.npz 的 CQ 块（图 5 同一块）。
(c) 14 个样品在 (f_p, ū_p) 平面上的分布：小点为逐块值（每样品随机 120 块），大点为样品均值。
(d) 可分辨比例 f_p / 氦孔隙度 与 ū_p。
(e) 样品间标准差（样品均值的总体标准差）与样品内块间标准差（各样品的中位数）。"""
import csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

NL = chr(10)
FP, UP, FD = r'$f_\mathrm{p}$', r'$\bar{u}_\mathrm{p}$', r'$f_\mathrm{d}$'
CF = (DATA_ROOT / 'cfield2')
G = ['G01', 'G19', 'G05', 'G06', 'G07', 'G09', 'G11', 'G03', 'G16', 'G17', 'G02', 'G12', 'G13', 'G15']
PROV = {'G01': 'CQ', 'G19': 'CQ', 'G05': 'SC', 'G06': 'SC', 'G07': 'SC', 'G09': 'SC', 'G11': 'SC', 'G03': 'YN', 'G16': 'YN',
        'G17': 'YN', 'G02': 'GZ', 'G12': 'SD', 'G13': 'SHX', 'G15': 'SX'}
PNAME = {'CQ': 'Chongqing', 'SC': 'Sichuan', 'YN': 'Yunnan', 'GZ': 'Guizhou', 'SD': 'Shandong', 'SHX': 'Shaanxi', 'SX': 'Shanxi'}
PCOL = {'CQ': '#0072B2', 'SC': '#D55E00', 'YN': '#009E73', 'GZ': '#CC79A7', 'SD': '#E6A100', 'SHX': '#56B4E9', 'SX': '#8C8C8C'}
C01, C19 = '#1A1A1A', '#0072B2'

# ---------- 数据 ----------
BLK, MEAN, UHI = {}, {}, {}
for g in G:
    d = np.load(CF / ('%s.npz' % g))
    v = d['c_blk'][d['valid'] & np.isfinite(d['c_blk']).all(-1)].astype(float)  # 列：f_out, f_p, ū_p, f_d
    BLK[g], MEAN[g], UHI[g] = v, v.mean(0), float(d['norm'][3])
LAB = {r['组号']: r for r in csv.DictReader(open(H / 'tab01_samples.csv', encoding='utf-8-sig'))}
HE = {g: float(LAB[g]['实验室孔隙度_B']) for g in G}
hc = np.load(str(FIGDATA / 'hist_cache.npz'))
edges = np.linspace(-0.5, 2.5, 601); cen = (edges[:-1] + edges[1:]) / 2; bw = edges[1] - edges[0]
z1 = np.load(FIGDATA / 'fig2_G01.npz'); vp = np.load(FIGDATA / 'vis_paper.npz')
IMG = {('G01', 'lr'): z1['lr'][18, 10:26, 10:26], ('G01', 'hr'): z1['hr'][56],
       ('G19', 'lr'): vp['CQ|粗扫'], ('G19', 'hr'): vp['CQ|真值']}
assert str(vp['CQ|gid']) == 'G19'

A = np.array([MEAN[g] for g in G]); SDW = np.array([BLK[g].std(0, ddof=1) for g in G])
between = A.std(0); within = np.median(SDW, 0)
ratio = np.array([100 * MEAN[g][1] / HE[g] for g in G]); rho = spearmanr(A[:, 2], ratio)[0]
print('样品级 f_p %.1f–%.1f%%，ū_p %.2f–%.2f' % (100 * A[:, 1].min(), 100 * A[:, 1].max(), A[:, 2].min(), A[:, 2].max()))
print('样品间 SD', np.round(between, 3), ' 样品内 SD 中位', np.round(within, 3))
print('f_p/He 与 ū_p 的 Spearman %.2f（n=%d）' % (rho, len(G)))
for g in ('G01', 'G19'):
    print(g, 'He %.2f  f_p %.2f%%  ū_p %.3f  u_hi %.3f' % (HE[g], 100 * MEAN[g][1], MEAN[g][2], UHI[g]))

# ---------- 作图 ----------
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 122 * T.MM))
top = GridSpec(1, 2, figure=fig, left=0.07, right=0.985, top=0.955, bottom=0.575, width_ratios=[2.3, 1], wspace=0.08)
bot = GridSpec(1, 3, figure=fig, left=0.07, right=0.985, top=0.44, bottom=0.085, width_ratios=[1.25, 1, 0.8], wspace=0.40)

# (a) 直方图
ax = fig.add_subplot(top[0, 0])
uhi = 0.5 * (UHI['G01'] + UHI['G19'])
ax.axvspan(0.52, uhi, color='#EFEFEF', lw=0, zorder=0)
ax.axvline(0.5, color=T.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
for g, col in (('G01', C01), ('G19', C19)):
    hs = np.convolve(hc[g + '_T'], np.ones(5) / 5, mode='same')  # 5 箱滑动平均，去掉 16 位整数灰度造成的分箱锯齿
    ax.plot(cen, hs / bw, color=col, lw=0.9, zorder=3,
            label='%s  (He %.1f%%,  ' % (g, HE[g]) + FP + ' %.1f%%)' % (100 * MEAN[g][1]))
ax.set_yscale('log'); ax.set_xlim(-0.25, 2.05); ax.set_ylim(3e-3, 12)
ax.set_xlabel(r'Normalized grey value $u$  (air = 0, framework peak = 1)'); ax.set_ylabel('Probability density')
for g, col in (('G01', C01), ('G19', C19)):
    ax.plot(MEAN[g][2], 4.2e-3, marker='^', ms=4.5, color=col, mec='white', mew=0.4, zorder=5, clip_on=False)
ax.text(0.5 * (MEAN['G01'][2] + MEAN['G19'][2]), 7.5e-3, UP, fontsize=6.5, color='#404040', ha='center', va='bottom')
kw = dict(fontsize=6.5, ha='center', va='top', color='#404040')
ax.text(0.12, 8.5, FP + r': $u<0.5$' + NL + '(resolvable pores)', **kw)
ax.text(1.09, 8.5, 'framework window' + NL + r'$[u_\mathrm{lo}, u_\mathrm{hi}]$', **kw)
ax.text(1.87, 8.5, FD + r': $u\geq u_\mathrm{hi}$' + NL + '(dense minerals)', **kw)
ax.legend(loc='lower right', handlelength=1.6)
T.label(ax, 'a', dx=-0.06)

# (b) 图像 2×2
sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=top[0, 1], wspace=0.05, hspace=0.07)
for i, kind in enumerate(('lr', 'hr')):
    for j, g in enumerate(('G01', 'G19')):
        a = fig.add_subplot(sub[i, j]); im = IMG[(g, kind)]
        a.imshow(im, cmap='gray', vmin=0, vmax=1.6, interpolation='nearest', extent=(0, 224, 224, 0))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(True); s.set_linewidth(0.4); s.set_color('#808080')
        if i == 0: a.set_title(g, fontsize=7, pad=2, color=C01 if g == 'G01' else C19, fontweight='bold')
        if j == 0: a.set_ylabel('14 μm coarse' if kind == 'lr' else '2 μm fine', fontsize=6.5, labelpad=2)
        if i == 1 and j == 1:
            a.plot([160, 210], [212, 212], color='white', lw=1.4, solid_capstyle='butt')
            a.text(185, 205, '50 μm', color='white', fontsize=5.5, ha='center', va='bottom')
        if i == 0 and j == 0: T.label(a, 'b', dx=-0.14)

# (c) (f_p, ū_p) 平面
ax = fig.add_subplot(bot[0, 0]); rng = np.random.default_rng(0)
for g in G:
    v = BLK[g]; k = rng.choice(len(v), min(120, len(v)), replace=False)
    ax.scatter(100 * v[k, 1], v[k, 2], s=1.2, color=PCOL[PROV[g]], alpha=0.25, lw=0, zorder=1, rasterized=True)
seen = set()
for g in G:
    p = PROV[g]; lab = PNAME[p] if p not in seen else None; seen.add(p)
    ax.scatter(100 * MEAN[g][1], MEAN[g][2], s=22, color=PCOL[p], edgecolor='white', lw=0.6, zorder=3, label=lab)
OFF = {'G01': (6, -2), 'G19': (6, 0), 'G11': (5, 3), 'G12': (5, -5)}
for g, (dx, dy) in OFF.items():
    ax.annotate(g, (100 * MEAN[g][1], MEAN[g][2]), xytext=(dx, dy), textcoords='offset points', fontsize=6, va='center')
ax.set_xlim(-0.5, 25); ax.set_ylim(0.0, 0.62)
ax.set_xlabel('Resolvable porosity ' + FP + ' (%)'); ax.set_ylabel('Mean pore grey value ' + UP)
ax.legend(loc='upper right', ncol=2, columnspacing=0.8, handletextpad=0.2, markerscale=0.8, fontsize=6)
T.label(ax, 'c', dx=-0.13)

# (d) 可分辨比例
ax = fig.add_subplot(bot[0, 1])
for g, r in zip(G, ratio):
    ax.scatter(MEAN[g][2], r, s=20, color=PCOL[PROV[g]], edgecolor='white', lw=0.6, zorder=3)
OFF = {'G01': (-4, 5, 'right'), 'G19': (5, -3, 'left'), 'G13': (-5, 0, 'right'), 'G12': (-5, -1, 'right')}  # G13 见图注
for g, (dx, dy, ha) in OFF.items():
    ax.annotate(g, (MEAN[g][2], 100 * MEAN[g][1] / HE[g]), xytext=(dx, dy), textcoords='offset points', fontsize=6, ha=ha, va='center')
ax.set_xlim(0.08, 0.5); ax.set_ylim(0, 1.2)
ax.set_xlabel('Mean pore grey value ' + UP); ax.set_ylabel(FP + ' / helium porosity')
ax.text(0.04, 0.04, r'Spearman $\rho$ = ' + ('%.2f' % rho).replace('-', '−') + NL + r'$n$ = %d' % len(G), transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5)
T.label(ax, 'd', dx=-0.2)

# (e) 样品间 / 样品内
ax = fig.add_subplot(bot[0, 2]); x = np.arange(3); w = 0.36; idx = [1, 2, 3]
ax.bar(x - w / 2, between[idx], w, color='#4D4D4D', label='Between samples')
ax.bar(x + w / 2, within[idx], w, color='#BDBDBD', label='Within sample' + NL + '(block to block)')
ax.set_xticks(x); ax.set_xticklabels([FP, UP, FD])
ax.set_ylabel('Standard deviation'); ax.set_ylim(0, 0.16); ax.set_xlim(-0.6, 2.6)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=6, handlelength=1.0)
T.label(ax, 'e', dx=-0.28)

T.save(fig, H / 'fig_geology')