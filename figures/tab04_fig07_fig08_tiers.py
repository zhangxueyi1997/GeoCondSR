# -*- coding: utf-8 -*-
"""表 4、图 7、图 8：地质信息的载体与信息分级。
推理替换（同一有地质模型 _g54b、同一批块与噪声）：data/eval/eval56、eval57、eval57b；
训练对照（各自训练）：eval55c（粗扫推地质 _g54c）、eval55n（周围回归 _g54n）、eval55a（周围一圈细扫 _g54a）。
对照组均为本文无地质版（_n54b，同一批块），计分规则见 3.9 节。误差变化 = 该来源的 6 折均值 / 无地质 6 折均值 − 1。"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; E = FIGDATA / 'eval'
import gse_style as S
S.FIGDIR = OUT_ROOT / 'figures'; S.FIGDIR.mkdir(parents=True, exist_ok=True); S.apply_style()
import matplotlib.pyplot as plt

F = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
L = lambda d, pat: {f: json.load(open(E / d / (pat % f), encoding='utf-8')) for f in F}
R56, R57, R57b = L('eval56', 'eval56_%s.json'), L('eval57', 'eval57_%s.json'), L('eval57b', 'eval57b_%s.json')
R55c, R55n, R55a = L('eval55c', 'eval55_%s.json'), L('eval55n', 'eval55_%s.json'), L('eval55a', 'eval55_%s.json')
MET = [('PSNR', 1), ('SSIM', 1), ('Dice', 1), ('孔隙度误差', -1), ('比表面相对误差', -1), ('欧拉数误差', -1),
       ('弦长相对误差', -1), ('S2距离', -1), ('连通跟踪', 1), ('连通度MAE', -1), ('碎裂度误差', -1)]
# (标签, 数据集, 臂名, 对照臂名, 高分辨信息)
ARMS = [
    ('粗扫逐块', R56, 'G·粗扫ĉ', '无地质(训练)', '无'),
    ('粗扫逐块 + 粗扫整体水平', R57, 'G·粗扫ĉ+粗扫整体水平', '无地质(训练)', '无'),
    ('粗扫（训练时即用 ĉ）', R55c, '新·有地质 w=1', '新·无地质', '无'),
    ('粗扫逐块 + 稀疏细扫校准', R56, 'G·粗扫ĉ+远处标定', '无地质(训练)', '≈0.11 mm³'),
    ('粗扫逐块 + 真实整体状态', R57b, 'A·整体水平4项全真', '无地质(训练)', '全样品统计'),
    ('  其中仅 f_p、ū_p 为真值', R57b, 'B·孔隙率+孔隙灰度真', '无地质(训练)', '全样品统计'),
    ('  其中仅 f_out、f_d 为真值', R57b, 'C·IGV+高密相真', '无地质(训练)', '全样品统计'),
    ('样品整体均值（常数）', R56, 'G·样品均值c', '无地质(训练)', '全样品统计'),
    ('目标块周围细扫（训练）', R55a, '新·有地质 w=1', '新·无地质', '目标块周围'),
    ('目标块实测（上限）', R56, 'G·块实测c', '无地质(训练)', '目标块'),
]


def rule(R, a, b):
    wa = wb = 0
    for k, s in MET:
        w = sum((R[f][a][k] - R[f][b][k]) * s > 0 for f in F); wa += w >= 5; wb += (6 - w) >= 5
    return wa, wb


mean = lambda R, a, k: np.mean([R[f][a][k] for f in F])
K = [('孔隙度误差', '孔隙率'), ('欧拉数误差', '欧拉数'), ('弦长相对误差', '弦长'), ('S2距离', '两点函数'), ('碎裂度误差', '碎裂度'), ('连通度MAE', '连通')]
print('**表 4** 地质状态的信息来源与超分结果（对本文无地质版，6 折）\n')
print('| 推理时 c 的来源 | 需要的高分辨数据 | 计分 | ' + ' | '.join(n + '误差变化' for _, n in K) + ' |')
print('|---|---|---|' + '---|' * len(K))
nogeo = {k: mean(R56, '无地质(训练)', k) for k, _ in K}
print('| 无（本文无地质版） | 无 | — | ' + ' | '.join('%.4f' % nogeo[k] if k in ('孔隙度误差', 'S2距离', '连通度MAE') else '%.3f' % nogeo[k] for k, _ in K) + ' |')
ROWS = []
for lab, R, a, b, hr in ARMS:
    wa, wb = rule(R, a, b); ch = [100 * (mean(R, a, k) / mean(R, b, k) - 1) for k, _ in K]
    ROWS.append((lab, hr, wa, wb, ch)); print('| %s | %s | %d:%d | ' % (lab, hr, wa, wb) + ' | '.join('%+.0f%%' % c for c in ch) + ' |')
print('\n注：第一行给无地质版的误差绝对值，其余各行为相对它的变化；负值为误差减小。')
# 无地质版在 eval55c 与 eval56 中是否同一数值（同块同噪声，应一致）
print('核对：无地质孔隙率误差 eval55c %.5f vs eval56 %.5f' % (mean(R55c, '新·无地质', '孔隙度误差'), mean(R56, '无地质(训练)', '孔隙度误差')))

# ---- 图 7：机理（哪种信息承载增益）----
SEL7 = ['目标块实测（上限）', '目标块周围细扫（训练）', '粗扫逐块 + 真实整体状态', '  其中仅 f_p、ū_p 为真值', '  其中仅 f_out、f_d 为真值', '样品整体均值（常数）']
EN7 = ['Target-block\nmeasurement', 'Surrounding fine\nscan (trained)', 'Coarse + true\nsample state', 'True $f_p$, $\\bar{u}_p$\nonly', 'True $f_{out}$, $f_d$\nonly', 'Sample mean\n(constant)']
K7 = [('孔隙度误差', 'Porosity'), ('欧拉数误差', 'Euler number'), ('弦长相对误差', 'Chord length'), ('碎裂度误差', 'Fragmentation')]
C7 = ['#1f6fb4', '#6a9fd4', '#e8890c', '#2e7d32']
fig, ax = S.figure(S.DOUBLE, height_mm=78, constrained_layout=True)
rows = {r[0]: r for r in ROWS}; x = np.arange(len(SEL7)); w = 0.19
for i, (k, nm) in enumerate(K7):
    ki = [kk for kk, _ in K].index(k)
    ax.bar(x + (i - 1.5) * w, [rows[s][4][ki] for s in SEL7], w, color=C7[i], label=nm)
for j, s in enumerate(SEL7):
    ax.text(x[j], 3, '%d:%d' % (rows[s][2], rows[s][3]), ha='center', va='bottom', fontsize=S.PT_NOTE)
ax.axhline(0, color='k', lw=0.7); ax.set_xticks(x); ax.set_xticklabels(EN7)
ax.set_ylabel('Error change vs. no geology (%)'); ax.set_ylim(-26, 8); ax.legend(ncol=4, loc='lower center')
S.save(fig, 'fig07_carrier')

# ---- 图 8：信息分级（工程视角）----
SEL8 = ['粗扫逐块', '粗扫逐块 + 粗扫整体水平', '粗扫（训练时即用 ĉ）', '粗扫逐块 + 稀疏细扫校准', '粗扫逐块 + 真实整体状态', '目标块实测（上限）']
EN8 = ['Coarse\nblock-wise', 'Coarse +\ncoarse level', 'Coarse\n(trained)', 'Coarse +\nsparse fine', 'Coarse +\ntrue state', 'Target\nblock']
fig, axs = S.figure(S.DOUBLE, height_mm=74, ncols=2, constrained_layout=True, gridspec_kw=dict(width_ratios=[2.6, 1]))
ax = axs[0]; x = np.arange(len(SEL8))
for i, (k, nm) in enumerate(K7):
    ki = [kk for kk, _ in K].index(k)
    ax.plot(x, [rows[s][4][ki] for s in SEL8], 'o-', color=C7[i], label=nm, ms=4)
ax.axhline(0, color='k', lw=0.7); ax.axvspan(-0.4, 2.4, color='#eeeeee', zorder=0)
ax.text(1, 6.0, 'coarse scan only', ha='center', fontsize=S.PT_NOTE); ax.text(4, 6.0, '+ high-resolution information', ha='center', fontsize=S.PT_NOTE)
ax.set_xticks(x); ax.set_xticklabels(EN8); ax.set_ylabel('Error change vs. no geology (%)'); ax.set_ylim(-22, 9); ax.set_xlim(-0.4, 5.4)
ax.legend(ncol=2, loc='lower left')
ax2 = axs[1]; sc = [rows[s][2] - rows[s][3] for s in SEL8]
ax2.barh(x, [rows[s][2] for s in SEL8], color='#c8102e', height=0.55, label='metrics won (≥5/6 folds)')
ax2.barh(x, [-rows[s][3] for s in SEL8], color='#9e9e9e', height=0.55, label='metrics lost')
ax2.set_yticks(x); ax2.set_yticklabels([e.replace('\n', ' ') for e in EN8]); ax2.invert_yaxis(); ax2.axvline(0, color='k', lw=0.7)
ax2.set_xlim(-2, 6.5); ax2.set_xticks([-1, 0, 2, 4, 6]); ax2.set_xticklabels(['1', '0', '2', '4', '6']); ax2.set_xlabel('Metrics lost | won (of 11)')
for a_, t in zip(axs, 'ab'): S.panel_label(a_, '(%s)' % t, dx=-0.08 if a_ is ax else -0.9)
S.save(fig, 'fig08_tiers')
