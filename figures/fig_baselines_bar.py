# -*- coding: utf-8 -*-
"""基线比较（4.1 节）：8 项结构指标的小倍数柱状图。柱 = 6 折均值（同表 2），点 = 各测试产地。
本文相对四个深度基线中最好者的变化只打印到终端（正文与表 2 引用），图上不标注。
数据：data/eval/eval55c（三线性、EDSR-3D、SRGAN-3D、本文无地质）与 eval59/eval59a（本文加稀疏细扫，同一批块）。像素指标见表 2 与图 pdplane。"""
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

FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; E = FIGDATA / 'eval'
A = {f: json.load(open(E / 'eval55c' / ('eval55_%s.json' % f), encoding='utf-8')) for f in FOLDS}
B = {f: json.load(open(E / 'eval59' / ('eval59a_%s.json' % f), encoding='utf-8')) for f in FOLDS}
C = {f: json.load(open(E / 'eval59' / ('eval59_%s.json' % f), encoding='utf-8')) for f in FOLDS}
M = [('三线性', 'Trilinear', lambda f: A[f]['三线性']), ('EDSR-3D', 'EDSR-3D', lambda f: A[f]['EDSR-3D']), ('SRGAN-3D', 'SRGAN-3D', lambda f: A[f]['SRGAN-3D']), ('SwinIR-3D', 'SwinIR-3D', lambda f: C[f]['SwinIR-3D']), ('扩散', 'Diffusion 3D', lambda f: C[f]['扩散']),
     ('本文', T.EN['本文'], lambda f: A[f]['新·无地质']), ('本文+稀疏细扫', T.EN['本文+稀疏细扫'], lambda f: B[f]['本文·粗扫+稀疏细扫'])]
K = [('孔隙度误差', 'Porosity error', -1), ('弦长相对误差', 'Chord-length error', -1), ('S2距离', 'Two-point function', -1), ('连通度MAE', 'Connectivity error', -1),
     ('碎裂度误差', 'Fragmentation error', -1), ('连通跟踪', 'Connectivity tracking', 1), ('比表面相对误差', 'Specific-surface error', -1), ('欧拉数误差', 'Euler-number error', -1)]
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 84 * T.MM))
gs = GridSpec(2, 4, figure=fig, left=0.065, right=0.99, top=0.88, bottom=0.08, wspace=0.42, hspace=0.45)
x = np.arange(len(M))
for i, (key, title, sgn) in enumerate(K):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    V = np.array([[get(f)[key] for f in FOLDS] for _, _, get in M])      # (方法, 折)
    mean = V.mean(1)
    ax.bar(x, mean, 0.72, color=[T.C[k] for k, _, _ in M], edgecolor='none', zorder=2)
    rng = np.random.default_rng(i)
    for j in range(len(M)):
        ax.scatter(x[j] + rng.uniform(-0.2, 0.2, len(FOLDS)), V[j], s=3.5, color='#333333', alpha=0.55, lw=0, zorder=3)
    bestb = (min if sgn < 0 else max)(mean[1:5]); ch = (mean[5] - bestb) / bestb
    top = max(V.max(), mean.max())
    ax.set_title(title, fontsize=6.3, pad=3); ax.set_xticks(x)
    ax.set_xticklabels([]); ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, top * 1.08); ax.tick_params(axis='y', labelsize=5.8)
    if i % 4 == 0: ax.set_ylabel('Error' if sgn < 0 else 'Score', fontsize=6.3)
    T.label(ax, 'abcdefgh'[i], dx=-0.12)
    print('%-12s 均值 %s  本文相对最好基线 %+.1f%%' % (key, np.round(mean, 4), 100 * ch))
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=T.C[k], label=n) for k, n, _ in M], loc='upper center', ncol=7, fontsize=6.0, bbox_to_anchor=(0.5, 0.995), handlelength=1.2, columnspacing=1.6)
T.save(fig, H / 'fig_baselines_bar')