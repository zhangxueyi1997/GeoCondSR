# -*- coding: utf-8 -*-
"""图 11：整柱粗扫的应用。
(a) 整柱与小柱粗扫的径向功率谱之比（同组 14 对，A 柱；数据为 plugs/dom59.json，由 experiments/06_whole_plug/dom59.py 生成）；
(b) 岩心内外的径向灰度剖面（远场空气与骨架众数归一化；data/plugs/fig11_radial.npz）；
(c) 有真值的噪声对照：测试小柱配对块（每折前 100 块）加整柱同谱噪声，2 μm 孔隙率（data/plugs/out/noise59_<折>[_n59n].json）；
(d) 整柱超分孔隙率与氦孔隙度（data/plugs/agg_plug59<模型><后缀>.json，agg_plug59.py 输出）。"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; D = FIGDATA / 'plugs'
import gse_style as S
S.FIGDIR = OUT_ROOT / 'figures'; S.FIGDIR.mkdir(parents=True, exist_ok=True); S.apply_style()
import matplotlib.pyplot as plt

DOM = json.load(open(D / 'dom59.json')); EDG = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.87]); CEN = (EDG[:-1] + EDG[1:]) / 2
RAD = np.load(D / 'fig11_radial.npz')
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
NZ = {f: json.load(open(D / 'out' / ('noise59_%s.json' % f), encoding='utf-8')) for f in FOLDS}
NZn = {f: json.load(open(D / 'out' / ('noise59_%s_n59n.json' % f), encoding='utf-8')) for f in FOLDS if (D / 'out' / ('noise59_%s_n59n.json' % f)).exists()}
F = sorted(D.glob('agg_plug59_n59n_cup.json')); HAVE_D = bool(F)          # (d) 整柱结果齐了才画 2×2，否则一行三块
if HAVE_D: fig, axs = S.figure(S.DOUBLE, height_mm=125, nrows=2, ncols=2, constrained_layout=True); AX = list(axs.ravel())
else: fig, axs = S.figure(S.DOUBLE, height_mm=68, nrows=1, ncols=3, constrained_layout=True); AX = list(axs)
# (a)
ax = AX[0]
for g, v in DOM.items(): ax.plot(CEN, np.array(v['pl']) / np.array(v['ps']), color='#888', lw=0.7, alpha=0.8)
ax.plot(CEN, np.median([np.array(v['pl']) / np.array(v['ps']) for v in DOM.values()], 0), color='#c8102e', lw=1.6, label='Median of 14 pairs')
ax.axhline(1, color='k', lw=0.6, ls=':'); ax.set_yscale('log'); ax.set_xlabel('Spatial frequency (cycles per coarse voxel)'); ax.set_ylabel('Power ratio, whole plug / miniplug')
ax.legend(loc='upper left', fontsize=7.5)
# (b)
ax = AX[1]
for pid, c in (('CQ-1', '#1f6fb4'), ('GZ-1', '#e8890c'), ('SC-13', '#2ca02c')): ax.plot(RAD['rb'], RAD[pid], color=c, lw=1.1, label=pid)
ax.axhline(1, color='k', lw=0.6, ls=':'); ax.axhline(0, color='k', lw=0.6, ls=':'); ax.axvline(1, color='#888', lw=0.6)
ax.set_xlabel('Radius $r/R$'); ax.set_ylabel('Median normalized gray level $u$'); ax.set_xlim(0, 1.6); ax.set_ylim(-0.5, 1.5); ax.legend(loc='lower left', fontsize=7.5)
# (c)
ax = AX[2]; W = 0.18
for i, f in enumerate(FOLDS):
    t = np.mean(NZ[f]['细扫真值'])
    bars = [('本文·仅粗扫', '干净', '#c8102e', 1.0), ('本文·仅粗扫', '加噪', '#c8102e', 0.45), ('EDSR-3D', '加噪', '#1f6fb4', 0.6)]
    for j, (m, tag, c, al) in enumerate(bars):
        ax.bar(i + (j - 1.5) * W, np.mean(NZ[f][tag][m]), W, color=c, alpha=al, lw=0)
    if f in NZn: ax.bar(i + 1.5 * W, np.mean(NZn[f]['加噪']['本文·仅粗扫']), W, color='#7a0019', lw=0)
    ax.plot([i - 2 * W, i + 2 * W], [t, t], color='k', lw=1.2)
ax.set_xticks(range(len(FOLDS))); ax.set_xticklabels(FOLDS); ax.set_ylabel('2-μm porosity')
for lab, c, al in (('Ours, clean', '#c8102e', 1.0), ('Ours, noisy', '#c8102e', 0.45), ('EDSR-3D, noisy', '#1f6fb4', 0.6), ('Noise-aware ours, noisy', '#7a0019', 1.0)):
    ax.bar([np.nan], [np.nan], color=c, alpha=al, label=lab)
ax.plot([], [], color='k', lw=1.2, label='Fine scan'); ax.set_ylim(0, 0.26)
ax.legend(fontsize=6.3, loc='upper center', ncol=2, handlelength=1.2, columnspacing=0.8, handletextpad=0.4)
# (d) 整柱超分孔隙率与氦孔隙度（整柱标定结果齐了才画）
if HAVE_D:
    ax = AX[3]; rows = json.load(open(F[0], encoding='utf-8')); A = [r for r in rows if r['main']]
    for key, lab, c, mk in (('cal|本文·仅粗扫', 'Ours (calibrated, noise-aware)', '#c8102e', 'o'), ('cal|EDSR-3D', 'EDSR-3D (calibrated)', '#1f6fb4', 's'), ('cal|粗扫阈值', '14-μm threshold', '#888', '^')):
        ax.scatter([100 * r['he'] for r in A], [100 * r[key] for r in A], s=14, color=c, marker=mk, label=lab, lw=0)
    x = np.linspace(0, 22, 10); ax.plot(x, x, 'k-', lw=0.7); ax.plot(x, 0.5 * x, 'k:', lw=0.7)
    ax.set_xlabel('Helium porosity (%)'); ax.set_ylabel('Whole-plug 2-μm porosity (%)'); ax.legend(fontsize=7, loc='upper left')
for ax, t in zip(AX, 'abcd'): S.panel_label(ax, '(%s)' % t, dx=-0.16 if HAVE_D else -0.22)
S.save(fig, 'fig11_wholecore')
