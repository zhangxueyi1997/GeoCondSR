# -*- coding: utf-8 -*-
"""图 2：真实双分辨率配对数据。
(a) 小柱 14 μm 粗扫横截面（G01），框出一个 504 μm 上下文格与其中心 224 μm 目标块；
(b) 上下文格（36² 粗体素）；(c) 目标块的粗扫（16²）；(d) 同一目标块的细扫（112²，2.0 μm）；
(e) 14 个配准样品的相干谱 γ²(λ)；(f) 实验室孔隙度（B 柱）与 2 μm 可分辨孔隙度。
数据：data/fig2_G01.npz（服务器 pairs/G01/00174.npz 与同层粗扫切片）、data/spec_G??.npz（registration_2026）、tab01_samples.csv。"""
import csv, sys
from pathlib import Path
import numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent
import gse_style as S
S.FIGDIR = H
S.apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

d = np.load(FIGDATA / 'fig2_G01.npz')
xs, lr, hr = d['xs'], d['lr'], d['hr']; y0, x0 = int(d['y0']), int(d['x0'])
VMIN, VMAX = 0.0, 1.6
fig = plt.figure(figsize=(S.DOUBLE * S.MM, 118 * S.MM), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
ax = [fig.add_subplot(gs[0, i]) for i in range(4)]


def img(a, arr, extent_um):
    a.imshow(arr, cmap='gray', vmin=VMIN, vmax=VMAX, interpolation='nearest', extent=(0, extent_um, extent_um, 0))
    a.set_xticks([]); a.set_yticks([])


def bar(a, L, extent_um, text, yf=0.93):
    x1 = extent_um * 0.93; y = extent_um * yf
    a.plot([x1 - L, x1], [y, y], color='white', lw=2.2, solid_capstyle='butt')
    a.text(x1 - L / 2, y - extent_um * 0.03, text, color='white', ha='center', va='bottom', fontsize=S.PT_NOTE)


# (a) 横截面
n = xs.shape[0]; ext = n * 14.0
img(ax[0], xs, ext)
ax[0].add_patch(Rectangle((x0 * 14, y0 * 14), 36 * 14, 36 * 14, fill=False, ec=S.C_MICRO, lw=1.2))
ax[0].add_patch(Rectangle(((x0 + 10) * 14, (y0 + 10) * 14), 16 * 14, 16 * 14, fill=False, ec=S.C_PLUG, lw=1.0))
bar(ax[0], 1000, ext, '1 mm', yf=0.975); ax[0].set_title('Coarse, 14 μm', fontsize=S.PT_LABEL)
# (b) 上下文格
img(ax[1], lr[18], 36 * 14.0)
ax[1].add_patch(Rectangle((10 * 14, 10 * 14), 16 * 14, 16 * 14, fill=False, ec=S.C_PLUG, lw=1.2))
bar(ax[1], 100, 36 * 14.0, '100 μm'); ax[1].set_title('Context, 504 μm', fontsize=S.PT_LABEL)
for sp in ax[1].spines.values(): sp.set_edgecolor(S.C_MICRO); sp.set_linewidth(1.2)
# (c)(d) 目标块：粗扫 16² 与细扫 112²（细扫第 59 层对应粗扫第 18 层的中心面）
img(ax[2], lr[18, 10:26, 10:26], 224.0); bar(ax[2], 50, 224.0, '50 μm'); ax[2].set_title('Target, coarse', fontsize=S.PT_LABEL)
img(ax[3], hr[59], 224.0); bar(ax[3], 50, 224.0, '50 μm'); ax[3].set_title('Target, fine (2.0 μm)', fontsize=S.PT_LABEL)
for a in ax[2:]:
    for sp in a.spines.values(): sp.set_edgecolor(S.C_PLUG); sp.set_linewidth(1.2)
# (e) 相干谱
ae = fig.add_subplot(gs[1, :2])
for f in sorted((FIGDATA).glob('spec_G*.npz')):
    s = np.load(f); o = np.argsort(s['lam']); ae.plot(s['lam'][o], s['g2'][o], color=S.C_GREY, lw=0.8, alpha=0.8)
ae.axhline(0.5, color='k', lw=0.7, ls='--'); ae.axvline(31.2, color=S.C_PLUG, lw=1.0)
ae.axvline(28.0, color=S.C_MICRO, lw=0.8, ls=':')
ae.text(31.2 * 1.05, 0.06, '31 μm', color=S.C_PLUG, fontsize=S.PT_NOTE)
ae.text(28.0 * 1.04, 0.9, 'Nyquist', color=S.C_MICRO, fontsize=S.PT_NOTE, ha='left')
ae.set_xscale('log'); ae.set_xlim(26, 300); ae.set_xlabel('Wavelength λ (μm)'); ae.set_ylabel('Coherence γ²'); ae.set_ylim(-0.02, 1.02)
ae.set_xticks([30, 50, 100, 200, 300]); ae.set_xticklabels(['30', '50', '100', '200', '300'])
from matplotlib.ticker import NullFormatter
ae.xaxis.set_minor_formatter(NullFormatter())
# (f) 实验室孔隙度 vs 2 μm 可分辨孔隙度
af = fig.add_subplot(gs[1, 2:])
rows = [r for r in csv.DictReader(open(H / 'tab01_samples.csv', encoding='utf-8-sig')) if r['用途'] != '配准失败，未使用']
x = np.array([float(r['实验室孔隙度_B']) for r in rows]); y = np.array([float(r['细扫可分辨孔隙度']) for r in rows])
af.plot([0, 21], [0, 21], color='k', lw=0.7, ls='--'); af.plot([0, 21], [0, 10.5], color=S.C_GREY, lw=0.7, ls=':')
af.scatter(x, y, s=18, color=S.C_MICRO, zorder=3)
OFF = {'G02': (-16, 5), 'G09': (4, -9), 'G15': (4, -9), 'G01': (-18, 5), 'G16': (-17, 4), 'G03': (4, 3)}
for r, xi, yi in zip(rows, x, y): af.annotate(r['组号'], (xi, yi), xytext=OFF.get(r['组号'], (3, 2)), textcoords='offset points', fontsize=S.PT_NOTE)
af.text(12.3, 13.0, '1:1', fontsize=S.PT_NOTE, ha='right'); af.text(20.5, 9.2, '1:2', fontsize=S.PT_NOTE, ha='right', color=S.C_GREY)
af.set_xlim(0, 21); af.set_ylim(0, 14); af.set_xlabel('Laboratory porosity, plug B (%)'); af.set_ylabel('Porosity resolved at 2 μm (%)')
for a, t in zip(ax + [ae, af], 'abcdef'): S.panel_label(a, t, dx=-0.02 if a in ax else -0.12)
S.save(fig, 'fig02_data')
