# -*- coding: utf-8 -*-
"""整体状态可预测性图（SCI 版式，数据与计算同 fig09_level.py）与表 5：岩石整体状态（样品级 f_p、ū_p）的可预测性。
真值：各样品全部有效格的 c_blk 平均（data/ana56_data.npz）。
稀疏细扫校准：同 eval57 的来源 4，逐目标块模拟（每块随机 10 个相距 ≥3 格的块，data/lvl57_data.npz 的配对单元 ĉ 与 c_blk）。
仅粗扫：data/lvl57_pred.npy（ana57c.py 的 V0–V4 与嵌套选择，均为留一产地外推）。
氦孔隙度：B 柱孔隙度（paper1_scale_loss/src/p1/physical.py），样品级留一产地线性回归（ln 空间），单独或加粗扫 1 mm 窗第 2 百分位数。
误差口径：f_p 取 |ln(估计/真值)|，ū_p 取绝对差；中位数取 13 个测试样品。"""
import sys, re, json
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; D = FIGDATA
sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CODE = {'G01': 'CQ', 'G19': 'CQ', 'G02': 'GZ', 'G03': 'YN', 'G16': 'YN', 'G17': 'YN', 'G05': 'SC', 'G06': 'SC', 'G07': 'SC',
        'G09': 'SC', 'G11': 'SC', 'G12': 'SD', 'G13': 'SHX', 'G15': 'SX'}
GS = list(CODE); FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; TEST = [g for g in GS if CODE[g] in FOLDS]; EPS = 1e-3
A = np.load(D / 'ana56_data.npz'); L = np.load(D / 'lvl57_data.npz'); PRED = np.load(D / 'lvl57_pred.npy', allow_pickle=True).item()
TL = {g: np.nanmean(A[g + '_rows'][:, 39:43], 0) for g in GS}
src = (REPO_ROOT / 'configs' / 'sample_properties.py').read_text(encoding='utf-8')
PHI = {m[0]: float(m[1]) for m in re.findall(r'\("([A-Z]+-\d+)", "[^"]+", [\d.]+, [\d.]+, [\d.]+, ([\d.]+)', src)}
PAIR = dict(re.findall(r'\("([A-Z]+-\d+)", "([A-Z]+-\d+)"\)', src)); PAIR.update({b: a for a, b in list(PAIR.items())})
AOF = {'G01': 'CQ-1', 'G19': 'CQ-5', 'G02': 'GZ-1', 'G03': 'YN-3', 'G16': 'YN-1', 'G17': 'YN-5', 'G05': 'SC-11', 'G06': 'SC-13', 'G07': 'SC-15', 'G09': 'SC-5', 'G11': 'SC-9', 'G12': 'SD-1', 'G13': 'SX-8', 'G15': 'SX-5'}
HE = {g: PHI[PAIR[AOF[g]]] for g in GS}                                           # 小柱所属的 B 柱
P2 = {g: np.nanmean(A[g + '_rows'][:, 21]) for g in GS}
e_fp = lambda est, t: abs(np.log(max(est, EPS) / max(t, EPS))); e_up = lambda est, t: abs(est - t)
# 稀疏细扫校准：逐目标块模拟
rng = np.random.default_rng(0); SF = {}
for g in TEST:
    cl, cb, ch, ok = L['unit_cell_' + g], L['unit_cb_' + g], L['unit_ch_' + g], L['unit_ok_' + g]; chm = ch.mean(0); est = []
    for i in np.where(ok)[0]:
        far = np.where(ok & (np.abs(cl - cl[i]).max(1) >= 3))[0]; j = rng.choice(far, min(10, len(far)), replace=False)
        est.append(chm + (cb[j] - ch[j]).mean(0))
    SF[g] = np.array(est)
# 氦孔隙度回归（样品级，留一产地）
Y = np.array([TL[g] for g in GS]); Yl = Y.copy(); Yl[:, [0, 1, 3]] = np.log(np.maximum(Y[:, [0, 1, 3]], EPS))
def he_pred(keys):
    P = {}
    for Fo in FOLDS:
        tr = [g for g in GS if CODE[g] != Fo]; te = [g for g in GS if CODE[g] == Fo]
        f = lambda g: [np.log(HE[g]) if k == 'he' else P2[g] for k in keys]
        W = np.linalg.lstsq(np.c_[[f(g) for g in tr], np.ones(len(tr))], Yl[[GS.index(g) for g in tr]], rcond=None)[0]
        for g in te: p = np.r_[f(g), 1.0] @ W; p[[0, 1, 3]] = np.exp(p[[0, 1, 3]]); P[g] = p
    return P
HEP, HECP = he_pred(['he']), he_pred(['he', 'p2'])
NEST = PRED[[k for k in PRED if k not in ('V0', 'V1', 'V2', 'V3', 'V4')][0]]
METH = [('稀疏细扫校准（约 0.11 mm³）', None), ('粗扫：现方法 V0', PRED['V0']), ('粗扫：对数目标 V1', PRED['V1']), ('粗扫：18 维统计 V2', PRED['V2']),
        ('粗扫：整根小柱 V3', PRED['V3']), ('粗扫：1 mm 窗直推 V4', PRED['V4']), ('粗扫：嵌套选择', NEST), ('氦孔隙度', HEP), ('氦孔隙度 + 粗扫', HECP)]
print('**表 5** 岩石整体状态的预测误差（13 个测试样品的中位数，留一产地）\n')
print('| 信息来源 | $f_p$ 误差 $|\\ln(\\hat{f}/f)|$ | $\\bar{u}_p$ 误差 | 最差样品 $f_p$ | 是否达到判据 |'); print('|---|---|---|---|---|')
SUM = {}
for nm, P in METH:
    if P is None:
        efp = [np.sqrt(np.mean([e_fp(e[1], TL[g][1]) ** 2 for e in SF[g]])) for g in TEST]; eup = [np.sqrt(np.mean([e_up(e[2], TL[g][2]) ** 2 for e in SF[g]])) for g in TEST]
    else:
        efp = [e_fp(P[g][1], TL[g][1]) for g in TEST]; eup = [e_up(P[g][2], TL[g][2]) for g in TEST]
    SUM[nm] = (np.median(efp), np.median(eup), max(efp))
bfp, bup = SUM[METH[0][0]][:2]
for nm, _ in METH:
    a, b, w = SUM[nm]; print('| %s | %.3f | %.3f | %.2f | %s |' % (nm, a, b, w, '基准' if nm == METH[0][0] else ('是' if a <= bfp and b <= bup else '否')))
# ---- 图（SCI 版式）----
CS, CC, CH = T.C['本文'], '#6F8FB3', '#D9A23F'
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 66 * T.MM))
gs = GridSpec(1, 3, figure=fig, left=0.065, right=0.99, top=0.9, bottom=0.16, wspace=0.42, width_ratios=[1, 1, 1.15])
tt = np.array([TL[g][1] for g in TEST]); tu = np.array([TL[g][2] for g in TEST])
sfm = np.array([SF[g][:, 1].mean() for g in TEST]); sfs = np.array([SF[g][:, 1].std() for g in TEST])
sum_ = np.array([SF[g][:, 2].mean() for g in TEST]); sus = np.array([SF[g][:, 2].std() for g in TEST])
for p, (t, sm, ss, k, lab, lg) in enumerate(((tt, sfm, sfs, 1, r'Resolvable porosity $f_\mathrm{p}$', True), (tu, sum_, sus, 2, r'Mean pore grey value $\bar{u}_\mathrm{p}$', False))):
    ax = fig.add_subplot(gs[0, p])
    lo, hi = (0.004, 0.2) if lg else (0.08, 0.5)
    xx = np.linspace(lo, hi, 50); ax.plot(xx, xx, color='#333333', lw=0.6)
    if lg:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.fill_between(xx, xx * np.exp(-bfp), xx * np.exp(bfp), color=CS, alpha=0.10, lw=0)
    else:
        ax.fill_between(xx, xx - bup, xx + bup, color=CS, alpha=0.10, lw=0)
    ax.scatter(t, [NEST[g][k] for g in TEST], s=12, color=CC, marker='s', label='Coarse scan only', zorder=3, edgecolor='white', lw=0.3)
    ax.scatter(t, [HECP[g][k] for g in TEST], s=14, color=CH, marker='^', label='Helium + coarse', zorder=3, edgecolor='white', lw=0.3)
    ax.errorbar(t, sm, yerr=ss, fmt='o', ms=3.2, color=CS, lw=0.7, label='Sparse fine scan (0.11 mm³)', zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel('Measured (fine scan)'); ax.set_ylabel('Estimated'); ax.set_title(lab, fontsize=6.6, pad=3)
    if p == 0:
        for g in ('G01', 'G19', 'G12', 'G09'):
            i = TEST.index(g); ax.annotate(g, (tt[i], NEST[g][1]), xytext={'G12': (-16, 3), 'G09': (4, 2), 'G19': (5, -7), 'G01': (-16, 4)}[g], textcoords='offset points', fontsize=5.8)
        ax.legend(loc='lower right', fontsize=5.6, handletextpad=0.2, borderaxespad=0.2)
    T.label(ax, 'ab'[p], dx=-0.18)
ax = fig.add_subplot(gs[0, 2])
names = ['Sparse fine scan', 'Coarse V0', 'Coarse V1', 'Coarse V2', 'Coarse V3', 'Coarse V4', 'Coarse, nested', 'Helium', 'Helium + coarse']
r1 = [SUM[nm][0] / bfp for nm, _ in METH]; r2 = [SUM[nm][1] / bup for nm, _ in METH]; y = np.arange(len(METH))
cols = [CS] + [CC] * 6 + [CH] * 2
ax.barh(y - 0.19, r1, 0.36, color=cols, zorder=2); ax.barh(y + 0.19, r2, 0.36, color=cols, alpha=0.55, zorder=2)
ax.axvline(1, color='#333333', lw=0.6, ls=(0, (3, 2)))
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6); ax.invert_yaxis()
ax.get_yticklabels()[0].set_color(CS); ax.get_yticklabels()[0].set_fontweight('bold')
ax.set_xlabel('Median error / sparse-fine error'); ax.set_title(r'Error of whole-rock state (solid: $f_\mathrm{p}$, light: $\bar{u}_\mathrm{p}$)', fontsize=6.3, pad=3)
T.label(ax, 'c', dx=-0.33)
T.save(fig, H / 'fig_level_sci')
