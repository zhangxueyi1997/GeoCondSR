# -*- coding: utf-8 -*-
"""渗流模拟图（4.3 节，SCI 版式）。数据与口径同 fig10_lbm.py：data/eval/eval59/lbm59_<折>.json（同一批 200 块、同一分割）。
(a) 贯通判别一致率（6 折），(b) 漏判率（细扫贯通而重建不贯通；不含贵州折），(c) 共同块 |Δlog10 k| 中位数（细扫贯通块 ≥5 的折）；
柱 = 各折均值，点 = 各折，数字 = 本文（只用粗扫）相对最好基线。(d) 共同块上的渗透率散点。"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; D = FIGDATA / 'eval/eval59'; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
L = {f: json.load(open(D / ('lbm59_%s.json' % f), encoding='utf-8')) for f in FOLDS}
MS = [('三线性', 'Trilinear', '三线性'), ('EDSR-3D', 'EDSR-3D', 'EDSR-3D'), ('SRGAN-3D', 'SRGAN-3D', 'SRGAN-3D'),
      ('本文·仅粗扫', 'Ours', '本文'), ('本文·粗扫+稀疏细扫', 'Ours + sparse fine', '本文+稀疏细扫')]
FP = [f for f in FOLDS if len(L[f]['细扫']['kz']) >= 5]
FM = [f for f in FOLDS if np.array(L[f]['细扫']['span']).sum() > 0]
CMP = [m for m in MS if m[0] != '三线性']
dl = {m[0]: {} for m in CMP}; SC = []
for f in FP:
    kt = np.array(L[f]['细扫']['kz']); com = kt > 0
    for m, *_ in CMP: com &= np.array(L[f][m]['kz']) > 0
    for m, *_ in CMP:
        kp = np.array(L[f][m]['kz'])[com]; k0 = kt[com]
        dl[m][f] = float(np.median(np.abs(np.log10(kp) - np.log10(k0)))) if com.sum() >= 3 else np.nan
        if m in ('EDSR-3D', 'SRGAN-3D', '本文·仅粗扫'): SC.append((m, k0, kp))


def vals(m, key):
    if key == 'agree': return [float((np.array(L[f][m]['span']) == np.array(L[f]['细扫']['span'])).mean()) for f in FOLDS]
    if key == 'miss':
        out = []
        for f in FP:
            Tt = np.array(L[f]['细扫']['span']); P = np.array(L[f][m]['span']); out.append(float((Tt & ~P).sum() / max(Tt.sum(), 1)))
        return out
    return [dl[m][f] for f in FP]


MS = [('三线性', 'Trilinear', '三线性'), ('EDSR-3D', 'EDSR-3D', 'EDSR-3D'), ('SRGAN-3D', 'SRGAN-3D', 'SRGAN-3D'), ('SwinIR-3D', 'SwinIR-3D', 'SwinIR-3D'),
      ('扩散', 'Diffusion 3D', '扩散'), ('本文·仅粗扫', 'Ours', '本文'), ('本文·粗扫+稀疏细扫', 'Ours + sparse fine', '本文+稀疏细扫')]
CMP = [m for m in MS if m[0] != '三线性']
dl = {m[0]: {} for m in CMP}
for f in FP:
    kt = np.array(L[f]['细扫']['kz']); com = kt > 0
    for m, *_ in CMP: com &= np.array(L[f][m]['kz']) > 0
    for m, *_ in CMP:
        kp = np.array(L[f][m]['kz'])[com]; k0 = kt[com]
        dl[m][f] = float(np.median(np.abs(np.log10(kp) - np.log10(k0)))) if com.sum() >= 3 else np.nan


def vals(m, key):
    Tt = {f: np.array(L[f]['细扫']['span']) for f in FOLDS}
    if key == 'agree': return [float((np.array(L[f][m]['span']) == Tt[f]).mean()) for f in FOLDS]
    if key == 'miss': return [float((Tt[f] & ~np.array(L[f][m]['span'])).sum() / max(Tt[f].sum(), 1)) for f in FP]
    if key == 'fa': return [float((~Tt[f] & np.array(L[f][m]['span'])).sum() / max((~Tt[f]).sum(), 1)) for f in FOLDS]
    return [dl[m][f] for f in FP]


from scipy.stats import wilcoxon
PN = {'CQ': 'Chongqing', 'SC': 'Sichuan', 'YN': 'Yunnan', 'SHX': 'Shaanxi', 'SD': 'Shandong'}
EB = {m: [] for m, *_ in CMP}; FB = []
for f in FP:
    kt = np.array(L[f]['细扫']['kz']); com = kt > 0
    for m, *_ in CMP: com &= np.array(L[f][m]['kz']) > 0
    for m, *_ in CMP: EB[m] += list(np.abs(np.log10(np.array(L[f][m]['kz'])[com]) - np.log10(kt[com])))
    FB += [f] * int(com.sum())
EB = {m: np.array(v) for m, v in EB.items()}; FB = np.array(FB)
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 66 * T.MM))
gs = GridSpec(1, 4, figure=fig, left=0.055, right=0.99, top=0.8, bottom=0.12, wspace=0.42, width_ratios=[0.9, 1.05, 1.35, 1.05])
# (a) 贯通判别一致率
ax = fig.add_subplot(gs[0, 0])
for i_, (m, lab, ck) in enumerate(MS):
    v = np.array(vals(m, 'agree')); ax.bar(i_, v.mean(), 0.72, color=T.C[ck], zorder=2)
    ax.scatter(i_ + np.linspace(-0.2, 0.2, len(v)), v, s=3.5, color='#333333', alpha=0.6, lw=0, zorder=3)
    if m == '本文·仅粗扫': ax.text(i_, 1.02, '%.3f' % v.mean(), ha='center', va='bottom', fontsize=6, color=T.C['本文'], fontweight='bold')
ax.set_ylim(0.5, 1.1); ax.set_xticks([]); ax.set_title('Percolation agreement (↑)', fontsize=6.3, pad=3); T.label(ax, 'a', dx=-0.2)
# (b) 漏判–误判平面
ax = fig.add_subplot(gs[0, 1])
for m, lab, ck in MS:
    mi, fa = np.mean(vals(m, 'miss')), np.mean(vals(m, 'fa'))
    ax.scatter(100 * fa, 100 * mi, s=34 if ck.startswith('本文') else 26, color=T.C[ck], edgecolor='white', lw=0.5, zorder=3)
ax.set_xlabel('False percolation (%)'); ax.set_ylabel('Missed percolating blocks (%)'); ax.set_xlim(-1, 12); ax.set_ylim(0, 100)
ax.annotate('ideal', xy=(0, 0), xytext=(3.2, 12), fontsize=5.8, color='#7A7A7A', arrowprops=dict(arrowstyle='-|>', color='#9A9A9A', lw=0.5, mutation_scale=5))
ax.set_title('Two kinds of percolation error', fontsize=6.3, pad=3); T.label(ax, 'b', dx=-0.2)
# (c) 逐产地共同块 |Δlog10 k|
ax = fig.add_subplot(gs[0, 2]); PF = [f for f in FP if (FB == f).sum() >= 3]; w = 0.13
for q, f in enumerate(PF):
    v = [np.median(EB[m][FB == f]) for m, *_ in CMP]; b = int(np.argmin(v))
    for k, ((m, lab, ck), y) in enumerate(zip(CMP, v)):
        xx = q + (k - (len(CMP) - 1) / 2) * w; ax.bar(xx, y, w * 0.92, color=T.C[ck], zorder=2)
        if k == b: ax.scatter(xx, y + 0.035, marker='*', s=18, color=T.C[ck], lw=0, zorder=4)
    ax.text(q, -0.075, 'n = %d' % (FB == f).sum(), ha='center', fontsize=5.4, color='#6E6E6E', transform=ax.get_xaxis_transform())
ax.set_xticks(range(len(PF))); ax.set_xticklabels([PN[f] for f in PF], fontsize=6)
ax.set_ylabel(r'Median $|\Delta\log_{10}k|$'); ax.set_ylim(0, 0.78); ax.tick_params(axis='x', pad=8)
ax.set_title('Permeability error by region (star: lowest)', fontsize=6.3, pad=3); T.label(ax, 'c', dx=-0.12)
# (d) 全部共同块的分布
ax = fig.add_subplot(gs[0, 3])
bp = ax.boxplot([EB[m] for m, *_ in CMP], widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color='#1A1A1A', lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
for patch, (m, lab, ck) in zip(bp['boxes'], CMP): patch.set_facecolor(T.C[ck]); patch.set_edgecolor('#4D4D4D'); patch.set_linewidth(0.5)
ax.set_xticks([]); ax.set_ylabel(r'$|\Delta\log_{10}k|$ per block'); ax.set_ylim(0, 2.45)
pv = [wilcoxon(EB['本文·仅粗扫'], EB[b]).pvalue for b in ('扩散', 'EDSR-3D')]
ax.set_title('All %d common blocks' % len(FB), fontsize=6.3, pad=3)
ax.text(0.97, 0.97, 'ours vs. diffusion: p = %.2f' % pv[0] + chr(10) + 'ours vs. EDSR-3D: p = %.2f' % pv[1] + chr(10) + '(paired Wilcoxon)', transform=ax.transAxes, ha='right', va='top', fontsize=5.4, color='#444444')
T.label(ax, 'd', dx=-0.2)
print('合并中位数', {m: round(float(np.median(EB[m])), 3) for m, *_ in CMP}, '；p', np.round(pv, 3))
fig.legend(handles=[Patch(color=T.C[ck], label=lab) for _, lab, ck in MS], loc='upper center', ncol=7, fontsize=6.0, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.2)
T.save(fig, H / 'fig_lbm_sci')