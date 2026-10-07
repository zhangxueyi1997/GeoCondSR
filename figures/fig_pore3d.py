# -*- coding: utf-8 -*-
"""三维孔隙结构对比图（4.1 节）。块与图 visual 相同：各测试折终评块序第 1 块（未经挑选）；取其中细扫孔隙率高于 3% 的三折
（重庆 G19、云南 G17、陕西 G13），其余三折的块孔隙率不足 2%，三维图中几乎无可见孔隙。
数据：data/vis3d59_mask.npz（服务器 eval59/vis3d59.py 推理输出的 u<0.5 孔隙掩膜，112³、2 μm）。
渲染：同一过滤（去掉小于 8 个体素的孔隙团，即等效直径 < 5 μm）→ 高斯平滑 σ=0.7 → 斜视光线投射取首个交点，
以灰度梯度作法向做 Lambert 着色并按深度减暗。右列：等效直径不小于 d 的孔隙团体积之和占块体积的比例（不做过滤，横轴由大到小，同压汞曲线习惯）。"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as nd
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, NullLocator, MaxNLocator

Z = np.load(FIGDATA / 'vis3d59_mask.npz')
ROWS = [('CQ', 'Chongqing'), ('YN', 'Yunnan'), ('SHX', 'Shaanxi')]
COLS = [('真值', 'Fine scan (2 μm)', T.C['细扫']), ('EDSR-3D', 'EDSR-3D', T.C['EDSR-3D']), ('SRGAN-3D', 'SRGAN-3D', T.C['SRGAN-3D']),
        ('本文·无地质', 'Ours (coarse only)', T.C['本文']), ('本文·粗扫+稀疏细扫', 'Ours + sparse fine', T.C['本文+稀疏细扫'])]
N = 112; VOX = 2.0; MINV = 8
PORE = np.array([0x3B, 0x6E, 0xA8]) / 255


def mask(f, m):
    return np.unpackbits(Z['%s|%s|mask' % (f, m)])[:N ** 3].reshape(N, N, N).astype(bool)


def eqd(v):
    lab, n = nd.label(v)
    cnt = np.bincount(lab.ravel()); cnt[0] = 0
    d = 2 * (3 * cnt * VOX ** 3 / (4 * np.pi)) ** (1 / 3)     # 等效球直径（μm）
    return lab, cnt, d


# 视线：从 (+x, −y, +z) 方向斜看（体数组轴序 z, y, x）
e = np.array([1.0, -1.25, 0.95]); e /= np.linalg.norm(e); dvec = -e
up = np.array([0, 0, 1.0]); up = up - up.dot(e) * e; up /= np.linalg.norm(up); rt = np.cross(dvec, up)
K = S = 196; cen = np.array([N / 2 - 0.5] * 3)
kk, ii, jj = np.meshgrid(np.arange(K) - K / 2, np.arange(S) - S / 2, np.arange(S) - S / 2, indexing='ij')
P = cen[:, None, None, None] + kk * dvec[:, None, None, None] - ii * up[:, None, None, None] + jj * rt[:, None, None, None]   # (x, y, z)
COORD = np.stack([P[2], P[1], P[0]])   # → (z, y, x)
kk = ii = jj = P = None
L = np.array([0.35, -0.55, 0.76]); L /= np.linalg.norm(L)


def render(v):
    sm = nd.gaussian_filter(v.astype(np.float32), 0.7)
    R = nd.map_coordinates(sm, COORD, order=1, cval=0.0)
    hit = R > 0.5; has = hit.any(0); k0 = np.where(has, hit.argmax(0), 0)
    gz, gy, gx = np.gradient(sm)
    pts = np.array([np.take_along_axis(COORD[a], k0[None], 0)[0] for a in range(3)])
    g = np.stack([nd.map_coordinates(q, pts, order=1) for q in (gx, gy, gz)])        # (x, y, z)
    nrm = -g / (np.linalg.norm(g, axis=0) + 1e-6)
    lam = np.clip(np.tensordot(L, nrm, 1), 0, 1); vw = np.clip(np.tensordot(e, nrm, 1), 0, 1)
    shade = 0.42 + 0.58 * lam + 0.25 * vw ** 12
    depth = (k0 - k0[has].min()) / max(np.ptp(k0[has]), 1) if has.any() else k0 * 0.0
    shade *= 1 - 0.30 * depth
    rgb = PORE[None, None] * shade[..., None] + 0.12 * vw[..., None] ** 12
    img = np.ones((S, S, 3)); img[has] = np.clip(rgb[has], 0, 1)
    return img


def cube_edges():
    cs = np.array([[x, y, z] for x in (0, N - 1) for y in (0, N - 1) for z in (0, N - 1)], float)
    far = np.argmin(cs @ e)
    E = []
    for a in range(8):
        for b in range(a + 1, 8):
            if (cs[a] != cs[b]).sum() == 1:
                pa, pb = [((c - cen) @ rt + S / 2, -(c - cen) @ up + S / 2) for c in (cs[a], cs[b])]
                E.append((pa, pb, far in (a, b)))
    return E


EDG = cube_edges()
fig = plt.figure(figsize=(T.DOUBLE * T.MM, 118 * T.MM))
gs = GridSpec(3, 5, figure=fig, left=0.035, right=0.80, top=0.925, bottom=0.02, wspace=0.04, hspace=0.10)
gr = GridSpec(3, 1, figure=fig, left=0.86, right=0.985, top=0.915, bottom=0.075, hspace=0.32)
AXC = []
for r, (f, pname) in enumerate(ROWS):
    axc = fig.add_subplot(gr[r, 0]); AXC.append(axc); stats = []
    for c, (m, title, col) in enumerate(COLS):
        v = mask(f, m)
        lab, cnt, d = eqd(v)
        keep = cnt >= MINV; vf = keep[lab] & v
        ax = fig.add_subplot(gs[r, c]); ax.imshow(render(vf), interpolation='bilinear')
        for pa, pb, hid in EDG:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color='#A0A0A0', lw=0.35, ls=(0, (2, 1.5)) if hid else '-', zorder=0 if hid else 3)
        ax.set_xlim(8, S - 8); ax.set_ylim(S - 8, 8); ax.axis('off')
        ax.text(0.03, 0.02, r'$\phi$ = %.1f%%' % (100 * v.mean()), transform=ax.transAxes, fontsize=6, va='bottom')
        if r == 0: ax.set_title(title, fontsize=7, pad=3, color=col if c else '#1A1A1A')
        if c == 0:
            ax.text(-0.04, 0.5, '%s (%s)' % (pname, str(Z['%s|gid' % f])), transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6.5)
        if c == 0: T.label(ax, 'abc'[r], dx=0.02, dy=0.97)
        # 累积分布：等效直径 ≥ x 的孔隙团体积占块体积
        dv = d[1:][cnt[1:] > 0]; vv = cnt[1:][cnt[1:] > 0] * 1.0
        o = np.argsort(dv)[::-1]; xs = dv[o]; ys = np.cumsum(vv[o]) / N ** 3 * 100
        axc.step(xs, ys, where='post', color=col, lw=1.1 if c in (0, 3) else 0.8, ls='-' if c != 4 else (0, (3, 1.2)), label=title if r == 0 else None)
        stats.append((m, 100 * v.mean(), 100 * vf.mean()))
    axc.set_xscale('log'); axc.set_xlim(100, 2.5); axc.set_ylim(0, None)
    axc.xaxis.set_major_locator(FixedLocator([3, 10, 30, 100])); axc.xaxis.set_minor_locator(NullLocator())
    axc.set_xticklabels(['3', '10', '30', '100']); axc.yaxis.set_major_locator(MaxNLocator(4))
    axc.tick_params(labelsize=6)
    axc.set_ylabel('Cum. porosity (%)', fontsize=6.5)
    if r == 2: axc.set_xlabel('Equiv. diameter (μm)', fontsize=6.5)
    T.label(axc, 'def'[r], dx=-0.25, dy=0.97)
    print(f, str(Z['%s|gid' % f]), ' '.join('%s %.2f/%.2f' % s for s in stats))
T.save(fig, H / 'fig_pore3d')