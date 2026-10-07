# -*- coding: utf-8 -*-
"""地质增益图（4.2 节，合并原“载体”与“信息分级”两图）。纵向为 c 的来源，横向为相对本文无地质版的误差降低（%）：
柱 = 5 项结构误差（孔隙率、欧拉数、弦长、两点函数、碎裂度）降低幅度的平均，点 = 各单项；右侧为 11 项指标的计分（≥5/6 折）。
数据与计算同 tab04_fig07_fig08_tiers.py（eval56、eval57、eval57b、eval55c、eval55a；对照为同一批块的无地质版）。"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; E = FIGDATA / 'eval'; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

F = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
L = lambda d, pat: {f: json.load(open(E / d / (pat % f), encoding='utf-8')) for f in F}
R56, R57, R57b = L('eval56', 'eval56_%s.json'), L('eval57', 'eval57_%s.json'), L('eval57b', 'eval57b_%s.json')
R55c, R55a = L('eval55c', 'eval55_%s.json'), L('eval55a', 'eval55_%s.json')
MET = [('PSNR', 1), ('SSIM', 1), ('Dice', 1), ('孔隙度误差', -1), ('比表面相对误差', -1), ('欧拉数误差', -1),
       ('弦长相对误差', -1), ('S2距离', -1), ('连通跟踪', 1), ('连通度MAE', -1), ('碎裂度误差', -1)]
K = ['孔隙度误差', '欧拉数误差', '弦长相对误差', 'S2距离', '碎裂度误差']
KN = ['porosity', 'Euler', 'chord', 'S2', 'fragment.']
mean = lambda R, a, k: np.mean([R[f][a][k] for f in F])


def rule(R, a, b):
    wa = wb = 0
    for k, s in MET:
        w = sum((R[f][a][k] - R[f][b][k]) * s > 0 for f in F); wa += w >= 5; wb += (6 - w) >= 5
    return wa, wb


def red(R, a, b):
    return np.array([100 * (1 - mean(R, a, k) / mean(R, b, k)) for k in K])


PA = [('Coarse, block-wise', R56, 'G·粗扫ĉ', '无地质(训练)', 'coarse'),
      ('Coarse + coarse-scan level', R57, 'G·粗扫ĉ+粗扫整体水平', '无地质(训练)', 'coarse'),
      ('Coarse (trained on ĉ)', R55c, '新·有地质 w=1', '新·无地质', 'coarse'),
      ('Coarse + sparse fine (0.11 mm³)', R56, 'G·粗扫ĉ+远处标定', '无地质(训练)', 'ours'),
      ('Coarse + true whole-rock state', R57b, 'A·整体水平4项全真', '无地质(训练)', 'ref'),
      ('Target-block fine (upper bound)', R56, 'G·块实测c', '无地质(训练)', 'ref')]
PB = [('Whole-rock state, all four', R57b, 'A·整体水平4项全真', '无地质(训练)', 'ref'),
      ('  only f_p, ū_p true', R57b, 'B·孔隙率+孔隙灰度真', '无地质(训练)', 'ours'),
      ('  only f_out, f_d true', R57b, 'C·IGV+高密相真', '无地质(训练)', 'coarse'),
      ('Sample mean, no block variation', R56, 'G·样品均值c', '无地质(训练)', 'coarse'),
      ('Surrounding fine scan (trained)', R55a, '新·有地质 w=1', '新·无地质', 'ref'),
      ('Target-block fine (upper bound)', R56, 'G·块实测c', '无地质(训练)', 'ref')]
COL = {'coarse': '#9DB4CC', 'ours': T.C['本文'], 'ref': '#BDBDBD'}
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 76 * T.MM))
gs = GridSpec(1, 2, figure=fig, left=0.19, right=0.985, top=0.88, bottom=0.25, wspace=0.8)
for p, (rows, title) in enumerate(((PA, 'How much high-resolution data is needed'), (PB, 'Which part of the state carries the gain'))):
    ax = fig.add_subplot(gs[0, p]); n = len(rows)
    for i, (lab, R, a, b, kind) in enumerate(rows):
        y = n - 1 - i; r = red(R, a, b); m = r.mean(); w, l = rule(R, a, b)
        ax.barh(y, m, 0.62, color=COL[kind], zorder=2)
        ax.scatter(r, np.full(len(r), y) + np.linspace(-0.2, 0.2, len(r)), s=5, color='#333333', zorder=3, lw=0)
        ax.text(22.3, y, '%d:%d' % (w, l), va='center', ha='left', fontsize=5.8, color='#444444', clip_on=False)
        print('%-48s 平均降低 %.1f%%  单项 %s  计分 %d:%d' % (lab, m, np.round(r, 1), w, l))
    labs = [lab.replace('f_p', '$f_\\mathrm{p}$').replace('ū_p', '$\\bar{u}_\\mathrm{p}$').replace('f_out', '$f_\\mathrm{out}$').replace('f_d', '$f_\\mathrm{d}$') for lab, *_ in rows]
    ax.set_yticks(range(n)); ax.set_yticklabels(labs[::-1], fontsize=6)
    for tl, (_, _, _, _, kind) in zip(ax.get_yticklabels(), rows[::-1]):
        if kind == 'ours' and p == 0: tl.set_color(T.C['本文']); tl.set_fontweight('bold')
    ax.axvline(0, color='#6E6E6E', lw=0.6); ax.set_xlim(-2, 25.5)
    ax.text(22.3, n - 0.45, 'score', fontsize=5.6, color='#6E6E6E', ha='left', va='bottom', clip_on=False)
    ax.set_xlabel('Error reduction vs. model without geology (%)'); ax.set_title(title, fontsize=6.6, pad=4)
    T.label(ax, 'ab'[p], dx=-0.02 if p else -0.02, dy=1.02)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COL['coarse'], label='coarse scan only / partial state'), Patch(color=COL['ours'], label='deployable with sparse fine scan (a) / key parameters (b)'),
                    Patch(color=COL['ref'], label='reference: needs dense fine-scan data')], loc='lower center', ncol=3, fontsize=5.8, bbox_to_anchor=(0.55, 0.0), handlelength=1.1)
T.save(fig, H / 'fig_geogain')