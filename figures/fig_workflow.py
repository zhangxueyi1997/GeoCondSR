# -*- coding: utf-8 -*-
"""工作流程总图（3.1 节）。五个阶段：成像 → 配准与配对 → 地质状态 → 网络 → 整柱应用。缩略图均为真实数据：
整柱切片（CQ-1，标定后）、G01 配对块（data/fig2_G01.npz）、G01/G19 细扫直方图（hist_cache）、整柱窗口超分结果（data/plugs/wc_vis59.npz）。"""
import sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
import fastio
T.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

NL = chr(10)
FP, UP, FD = r'$f_\mathrm{p}$', r'$\bar{u}_\mathrm{p}$', r'$f_\mathrm{d}$'
# ---------- 缩略图数据 ----------
P = FIGDATA / 'plugs'; c = np.load(P / 'cup/CQ-1.npz')
files = sorted((RAW_ROOT / 'G01' / 'large_ct' / 'raw16').glob('*.tif'))   # without the raw slices, use the copy of this slice in the figure data
img = (fastio.read(files[int(c['pos'][0, 0]) + 18]) if files else np.load(FIGDATA / 'plugs' / 'CQ-1_slice.npy')).astype(np.float32)
a0, D0, an = float(c['air']), float(c['D']), float(c['air_near']); Dn = a0 + D0 - an
yy, xx = np.indices(img.shape); rr = np.hypot(yy - float(c['cy']), xx - float(c['cx'])) / float(c['R'])
gn = (np.polyval(c['coef'], np.clip(rr, 0, 0.88) ** 2) * D0 + a0 - an) / Dn
core = (img - an) / Dn / np.where(rr <= 1.0, gn, 1.0); R = float(c['R']); h = int(R * 1.04); cy, cx = int(c['cy']), int(c['cx'])
core = core[cy - h:cy + h, cx - h:cx + h]
z1 = np.load(FIGDATA / 'fig2_G01.npz'); lr, hr = z1['lr'][18], z1['hr'][56]
hc = np.load(str(FIGDATA / 'hist_cache.npz')); e = np.linspace(-0.5, 2.5, 601); cen = (e[:-1] + e[1:]) / 2
wv = np.load(P / 'wc_vis59.npz'); sr = wv['本文'][0]; wc = wv['win_cal_mid'][0]

fig = plt.figure(figsize=(T.DOUBLE * T.MM, 66 * T.MM))
W = 1.0; x0s = [0.008, 0.207, 0.406, 0.605, 0.804]; bw = 0.188; y0, bh = 0.09, 0.88
HEAD = ['1  Dual-resolution imaging', '2  Registration and pairing', '3  Geological state ' + r'$\mathbf{c}$', '4  Network', '5  Core-scale application']
for k, x0 in enumerate(x0s):
    hi = k == 2
    fig.patches.append(FancyBboxPatch((x0, y0), bw, bh, boxstyle='round,pad=0,rounding_size=0.012', transform=fig.transFigure,
                                      fc='#FBF1EF' if hi else '#F4F6F8', ec='#C0392B' if hi else '#C9CED6', lw=0.8 if hi else 0.5, zorder=-5))
    fig.text(x0 + 0.008, y0 + bh - 0.035, HEAD[k], fontsize=7, fontweight='bold', color='#7B1A14' if hi else '#1F2A36', va='top')
    if k < 4:
        fig.patches.append(FancyArrowPatch((x0 + bw + 0.001, 0.55), (x0s[k + 1] - 0.001, 0.55), transform=fig.transFigure,
                                           arrowstyle='-|>', mutation_scale=7, lw=0.8, color='#6E6E6E'))


def thumb(rect, a, ext=None, circle=False, title=None, col='#808080'):
    ax = fig.add_axes(rect); ax.imshow(a, cmap='gray', vmin=0, vmax=1.6, interpolation='nearest' if a.shape[0] < 40 else 'antialiased')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(not circle); s.set_color(col); s.set_linewidth(0.5)
    if title: ax.set_title(title, fontsize=5.5, pad=1.5, color='#404040')
    return ax


def body(x0, y, txt, **kw):
    fig.text(x0 + 0.008, y, txt, fontsize=5.8, va='top', color='#333333', linespacing=1.35, **kw)


# 1 成像
ax = thumb([x0s[0] + 0.012, 0.47, 0.08, 0.08 * 180 / 66], core, circle=True, title='Core plug, 13.9 μm')
thumb([x0s[0] + 0.105, 0.60, 0.035, 0.035 * 180 / 66], lr[10:26, 10:26], title='14 μm')
thumb([x0s[0] + 0.145, 0.60, 0.035, 0.035 * 180 / 66], hr, title='2 μm')
fig.text(x0s[0] + 0.143, 0.555, 'Miniplug', fontsize=5.5, ha='center', color='#404040')
body(x0s[0], 0.40, 'Core plug Φ25 × 50 mm:' + NL + '  whole-plug scan, 4 sections' + NL + 'Miniplug Φ3 mm:' + NL + '  coarse 14 μm + fine ≈2 μm' + NL + 'Helium porosity (core plug)')

# 2 配准与配对
thumb([x0s[1] + 0.018, 0.53, 0.07, 0.07 * 180 / 66], lr, title='Coarse, 504 μm')
a2 = thumb([x0s[1] + 0.105, 0.53, 0.07, 0.07 * 180 / 66], hr, title='Fine, 224 μm')
body(x0s[1], 0.40, 'Rigid + affine registration,' + NL + '  residual 0.048 coarse voxel' + NL + '2800 pairs from 14 samples' + NL + 'Structure < 31 μm not' + NL + '  recorded by the coarse scan')

# 3 地质状态
ax = fig.add_axes([x0s[2] + 0.02, 0.535, 0.155, 0.28])
ax.axvspan(0.52, 1.67, color='#EBDDDA', lw=0)
for g, col in (('G01', '#1A1A1A'), ('G19', '#0072B2')):
    ax.plot(cen, np.convolve(hc[g + '_T'], np.ones(5) / 5, 'same') / (e[1] - e[0]), color=col, lw=0.7, label=g)
ax.axvline(0.5, color='#6E6E6E', lw=0.4, ls=(0, (2, 1.5)))
ax.set_yscale('log'); ax.set_xlim(-0.2, 2.0); ax.set_ylim(5e-3, 5); ax.set_yticks([]); ax.set_yticks([], minor=True); ax.tick_params(labelsize=5, length=1.5, pad=1)
ax.set_xticks([0, 0.5, 1, 1.5]); ax.set_xlabel('Fine-scan grey value', fontsize=5.5, labelpad=1)
ax.spines['left'].set_visible(False)
ax.text(0.12, 2.2, FP + ', ' + UP, fontsize=5.8, ha='center'); ax.text(1.85, 2.2, FD, fontsize=5.8, ha='center')
ax.legend(fontsize=5, loc='lower right', handlelength=1.0, borderaxespad=0.1)
body(x0s[2], 0.40, 'Pore amount ' + FP + ', pore fineness ' + UP + ',' + NL + '  dense minerals ' + FD + NL + 'Carrier: whole-rock state' + NL + '  (sample-level mean)' + NL + 'Sparse fine scan ≈0.11 mm³' + NL + '  calibrates the state')

# 4 网络
X = x0s[3]
def node(x, y, w, t, fc='#FFFFFF', ec='#8A94A3'):
    fig.patches.append(FancyBboxPatch((x, y), w, 0.075, boxstyle='round,pad=0,rounding_size=0.008', transform=fig.transFigure, fc=fc, ec=ec, lw=0.5))
    fig.text(x + w / 2, y + 0.0375, t, fontsize=5.6, ha='center', va='center')
def arr(p, q, col='#6E6E6E'):
    fig.patches.append(FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle='-|>', mutation_scale=5, lw=0.5, color=col))
node(X + 0.008, 0.735, 0.074, 'Coarse 14 μm')
node(X + 0.112, 0.735, 0.068, 'State ' + r'$\mathbf{c}$', fc='#FBF1EF', ec='#C0392B')
node(X + 0.008, 0.600, 0.074, 'Mean path ' + r'$M$')
node(X + 0.104, 0.600, 0.076, 'Texture ' + r'$G$', fc='#FBF1EF', ec='#C0392B')
node(X + 0.040, 0.465, 0.110, r'$\mu + r$' + '  →  projection ' + r'$P_H$')
arr((X + 0.045, 0.735), (X + 0.045, 0.675)); arr((X + 0.146, 0.735), (X + 0.146, 0.675), '#C0392B')
arr((X + 0.082, 0.6375), (X + 0.104, 0.6375)); arr((X + 0.045, 0.600), (X + 0.070, 0.540)); arr((X + 0.140, 0.600), (X + 0.120, 0.540), '#C0392B')
body(X, 0.40, 'Where: fixed by the coarse scan' + NL + '  (mean path, hard projection)' + NL + 'What it looks like: set by ' + r'$\mathbf{c}$' + NL + '  (SPADE, amplitude head)')

# 5 整柱应用
thumb([x0s[4] + 0.018, 0.53, 0.07, 0.07 * 180 / 66], wc[10:26, 10:26], title='Whole plug, 14 μm')
thumb([x0s[4] + 0.105, 0.53, 0.07, 0.07 * 180 / 66], sr, title='Ours, 2 μm', col='#C0392B')
body(x0s[4], 0.40, 'Calibration: near-surface air,' + NL + '  cupping, degradation-aware' + NL + '  fine-tuning' + NL + 'Checks: helium porosity' + NL + '  (34 plugs), LBM permeability')
for ext, kw in (('.pdf', {}), ('.png', {'dpi': 600})):
    fig.savefig((H / 'fig_workflow').with_suffix(ext), bbox_inches='tight', pad_inches=0.01, **kw)
print('写出 fig_workflow.pdf + png（紧裁）')