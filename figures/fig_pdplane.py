# -*- coding: utf-8 -*-
"""失真–结构平面（5.2 节），仿 Blau & Michaeli (2018)、ESRGAN 图 2 的感知–失真平面。
横轴 PSNR（6 折均值）；纵轴结构误差指数 = 7 项结构误差（孔隙率、比表面、弦长、两点函数、碎裂度、欧拉数、连通误差）
相对三线性插值之比的几何平均（每折先算，再对 6 折取几何平均），越低越好。
数据：data/eval/eval55c（三线性、EDSR-3D、SRGAN-3D、均值通路、本文无地质）与 eval59/eval59a（本文加稀疏细扫，同一批块）。
若 eval59 终评 JSON（含 SwinIR-3D、扩散）存在则一并画出。"""
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

FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
E = FIGDATA / 'eval'
A = {f: json.load(open(E / 'eval55c' / ('eval55_%s.json' % f), encoding='utf-8')) for f in FOLDS}
B = {f: json.load(open(E / 'eval59' / ('eval59a_%s.json' % f), encoding='utf-8')) for f in FOLDS}
STR = ['孔隙度误差', '比表面相对误差', '弦长相对误差', 'S2距离', '碎裂度误差', '欧拉数误差', '连通度MAE']
# (显示名, 颜色键, 取数函数, 参数量 M, 标记)
M = [('Trilinear', '三线性', lambda f: A[f]['三线性'], 0.0, 's'),
     ('EDSR-3D', 'EDSR-3D', lambda f: A[f]['EDSR-3D'], 3.74, 'o'),
     ('SRGAN-3D', 'SRGAN-3D', lambda f: A[f]['SRGAN-3D'], 3.74, 'D'),
     ('Mean path only', '均值通路', lambda f: A[f]['仅均值通路'], 3.76, 'v'),
     ('Ours (coarse only)', '本文', lambda f: A[f]['新·无地质'], 4.97, 'o'),
     ('Ours + sparse fine', '本文+稀疏细扫', lambda f: B[f]['本文·粗扫+稀疏细扫'], 4.97, 'P')]
F59 = E / 'eval59' / 'eval59_CQ.json'
if F59.exists():
    C = {f: json.load(open(E / 'eval59' / ('eval59_%s.json' % f), encoding='utf-8')) for f in FOLDS}
    M += [('SwinIR-3D', 'SwinIR-3D', lambda f: C[f]['SwinIR-3D'], 2.68, '^'), ('Diffusion 3D', '扩散', lambda f: C[f]['扩散'], 12.12, 'X')]


def sidx(r, f):
    t = A[f]['三线性']; return float(np.exp(np.mean([np.log(r[k] / t[k]) for k in STR])))


PTS = {}
for nm, ck, get, par, mk in M:
    ps = np.array([get(f)['PSNR'] for f in FOLDS]); si = np.array([sidx(get(f), f) for f in FOLDS])
    PTS[nm] = (ps, si); print('%-20s PSNR %.2f  结构指数 %.3f（逐折 %s）' % (nm, ps.mean(), np.exp(np.log(si).mean()), np.round(si, 2)))

fig = plt.figure(figsize=(T.DOUBLE * T.MM, 70 * T.MM))
gs = GridSpec(1, 2, figure=fig, left=0.07, right=0.985, top=0.93, bottom=0.15, wspace=0.28, width_ratios=[1.45, 1])
ax = fig.add_subplot(gs[0, 0])
for nm, ck, get, par, mk in M:
    ps, si = PTS[nm]; col = T.C[ck]
    ax.scatter(ps, si, s=9, marker=mk, color=col, alpha=0.35, lw=0, zorder=2)
    ax.scatter(ps.mean(), np.exp(np.log(si).mean()), s=46, marker=mk, color=col, edgecolor='white', lw=0.6, zorder=4,
               label='%s  (%s)' % (nm, '%.2f M' % par if par else 'no parameters'))
ax.set_yscale('log'); ax.set_ylim(0.12, 1.4)
ax.set_yticks([0.2, 0.3, 0.5, 1.0]); ax.set_yticklabels(['0.2', '0.3', '0.5', '1.0']); ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel('PSNR (dB)  →  less pixel distortion'); ax.set_ylabel('Structure error index (vs. trilinear)  ↓ better')
ax.legend(loc='upper left', fontsize=5.8, handletextpad=0.3, borderaxespad=0.2, markerscale=0.8)
ax.annotate('', xy=(22.9, 0.14), xytext=(22.2, 0.2), arrowprops=dict(arrowstyle='-|>', color='#9A9A9A', lw=0.6, mutation_scale=6))
ax.text(22.95, 0.135, 'ideal', fontsize=5.8, color='#7A7A7A', ha='left', va='center')
T.label(ax, 'a', dx=-0.09)

# (b) 逐折：本文相对 EDSR-3D / SRGAN-3D
ax = fig.add_subplot(gs[0, 1])
ax.axhspan(0.05, 1, xmin=0, xmax=1, color='#F7ECEA', lw=0, zorder=0)
ps0, si0 = PTS['Ours (coarse only)']
for base, col in (('EDSR-3D', T.C['EDSR-3D']), ('SRGAN-3D', T.C['SRGAN-3D'])) + ((('Diffusion 3D', T.C['扩散']),) if 'Diffusion 3D' in PTS else ()):
    pb, sb = PTS[base]; dp, rs = ps0 - pb, si0 / sb
    for f, x, y in zip(FOLDS, dp, rs):
        ax.scatter(x, y, marker=T.FOLD_MK[f], s=16, color=col, edgecolor='white', lw=0.4, zorder=3)
    ax.scatter([], [], marker='o', s=16, color=col, label='Ours vs. ' + base)
    print('相对 %s：ΔPSNR %s，结构指数比 %s' % (base, np.round(dp, 2), np.round(rs, 2)))
ax.axhline(1, color='#9A9A9A', lw=0.6); ax.axvline(0, color='#9A9A9A', lw=0.6)
ax.set_yscale('log'); ax.set_ylim(0.15, 1.6); ax.set_xlim(-4, 1.5)
ax.set_yticks([0.2, 0.3, 0.5, 1.0]); ax.set_yticklabels(['0.2', '0.3', '0.5', '1.0']); ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel(r'$\Delta$PSNR, ours − baseline (dB)'); ax.set_ylabel('Structure error index ratio')
ax.legend(loc='upper left', fontsize=5.8, handletextpad=0.3, borderaxespad=0.2)
from matplotlib.lines import Line2D
PN = {'CQ': 'Chongqing', 'SC': 'Sichuan', 'YN': 'Yunnan', 'GZ': 'Guizhou', 'SD': 'Shandong', 'SHX': 'Shaanxi'}
leg1 = ax.get_legend(); ax.add_artist(leg1)
ax.legend(handles=[Line2D([], [], ls='none', marker=T.FOLD_MK[f], ms=3.5, color='#6E6E6E', label=PN[f]) for f in FOLDS], loc='lower right', ncol=2, fontsize=5.4,
          handletextpad=0.2, columnspacing=0.6, borderaxespad=0.2, title='test region', title_fontsize=5.4)
T.label(ax, 'b', dx=-0.16)
T.save(fig, H / 'fig_pdplane')