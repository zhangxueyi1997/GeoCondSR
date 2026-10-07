# -*- coding: utf-8 -*-
"""网络架构图（三维块体风格，仿 PlotNeuralNet 的特征图画法，自写 matplotlib 实现）。
(a) 总体结构：均值通路 M（蓝）、残差纹理生成器 G（红）、地质状态 c、硬投影 P_H、判别器 D（仅训练）。
    立方体三个可见面贴真实数据：G01 配对块（图 2 同一块）的粗扫 L、均值 mu、残差 r、输出 xhat、细扫 y（服务器 eval59/geo_ctrl59.py）。
    块体高度表示空间网格（36³/28³/56³/112³…），厚度表示通道数，均按对数缩放，只作示意。
(b) 地质调制：SPADE（式 3）与幅度头。 (c) 退化算子与硬投影（式 5、6），H 为实测 9³ 核的中心层。"""
import sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
T.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb

Z = np.load(FIGDATA / 'geo_ctrl59.npz')
K = np.load(str(DATA_ROOT / 'degradation' / 'H_kernel.npz'))['kernel']
NL = chr(10)
W_MM, H_MM = 180, 141
fig = plt.figure(figsize=(W_MM * T.MM, H_MM * T.MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM); ax.set_aspect('equal'); ax.axis('off')
OX, OY = 0.45, 0.32          # 斜投影深度方向
BLUE, BLUE_D, RED, RED_D, GREY = '#BCD4EA', '#3E6FA8', '#F4C2BA', '#C0392B', '#D5D5D5'


def shade(c, t):
    c = np.array(to_rgb(c)); return tuple(np.clip(c + t * (1 - c) if t > 0 else c * (1 + t), 0, 1))


def slab(x, cy, w, h, fc, band=None, z=3, alpha=1.0):
    y = cy - h / 2; dx, dy = OX * h, OY * h
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, fc=shade(fc, -0.22), ec='#4D4D4D', lw=0.3, zorder=z, alpha=alpha))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], closed=True, fc=shade(fc, 0.35), ec='#4D4D4D', lw=0.3, zorder=z, alpha=alpha))
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, fc=fc, ec='#4D4D4D', lw=0.3, zorder=z + 0.1, alpha=alpha))
    if band:
        b = w * 0.38
        ax.add_patch(Polygon([(x + w - b, y), (x + w, y), (x + w, y + h), (x + w - b, y + h)], closed=True, fc=band, ec='none', zorder=z + 0.2, alpha=0.85))
        ax.add_patch(Polygon([(x + w - b, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + w - b + dx, y + h + dy)], closed=True, fc=shade(band, 0.3), ec='none', zorder=z + 0.2, alpha=0.85))
    return dict(l=(x, cy), r=(x + w + dx / 2, cy + dy / 2), t=(x + w / 2 + dx / 2, cy + h / 2 + dy), b=(x + w / 2, y))


def face(img, o, u, v, cmap, lo, hi, z):
    M = np.array([[u[0], v[0], o[0]], [u[1], v[1], o[1]], [0, 0, 1]])
    ax.imshow(np.asarray(img, np.float32), cmap=cmap, vmin=lo, vmax=hi, extent=(0, 1, 0, 1), origin='lower', interpolation='bilinear',
              transform=Affine2D(M) + ax.transData, zorder=z)
    ax.add_patch(Polygon([o, (o[0] + u[0], o[1] + u[1]), (o[0] + u[0] + v[0], o[1] + u[1] + v[1]), (o[0] + v[0], o[1] + v[1])], closed=True, fc='none', ec='#333333', lw=0.4, zorder=z + 0.1))


def tcube(x, cy, s, key, cmap='gray', lo=0, hi=1.6, z=4):
    y = cy - s / 2; dx, dy = OX * s, OY * s
    face(Z['arch|%s|front' % key], (x, y), (s, 0), (0, s), cmap, lo, hi, z)
    face(Z['arch|%s|top' % key], (x, y + s), (s, 0), (dx, dy), cmap, lo, hi, z)
    face(Z['arch|%s|side' % key], (x + s, y), (dx, dy), (0, s), cmap, lo, hi, z)
    return dict(l=(x, cy), r=(x + s + dx / 2, cy + dy / 2), t=(x + s / 2 + dx / 2, y + s + dy), b=(x + s / 2, y))


def ncube(x, cy, s, seed=3, z=4):
    rng = np.random.default_rng(seed); y = cy - s / 2; dx, dy = OX * s, OY * s
    for o, u, v in (((x, y), (s, 0), (0, s)), ((x, y + s), (s, 0), (dx, dy)), ((x + s, y), (dx, dy), (0, s))):
        face(rng.normal(size=(24, 24)), o, u, v, 'gray', -2.5, 2.5, z)
    return dict(l=(x, cy), r=(x + s + dx / 2, cy + dy / 2), t=(x + s / 2 + dx / 2, y + s + dy), b=(x + s / 2, y))


def arrow(p, q, col='#555555', lw=0.6, ls='-', rad=0.0, z=6, ms=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=col, linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle='arc3,rad=%s' % rad, zorder=z))


def line(pts, col='#555555', lw=0.6, ls='-', z=5):
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, ls=ls, zorder=z, solid_capstyle='butt')


def oplus(x, y, r=1.6, sym='+', z=7):
    ax.add_patch(Circle((x, y), r, fc='white', ec='#333333', lw=0.5, zorder=z))
    ax.text(x, y - 0.1, sym, ha='center', va='center', fontsize=7, zorder=z + 0.1)


def txt(x, y, s, fs=6.2, **kw):
    kw.setdefault('ha', 'center'); kw.setdefault('va', 'center'); kw.setdefault('color', '#222222')
    ax.text(x, y, s, fontsize=fs, zorder=9, **kw)


hN = lambda n: 4.5 + 0.12 * n
wC = lambda c: max(0.9, 0.7 * np.log2(c) - 1.4)

# ================= (a) 总体结构 =================
cM, cG = 112, 76
ax.text(1.5, 139, 'a', fontsize=8.5, fontweight='bold', va='top')
L = tcube(2.5, cM, 12, 'lr')
txt(1.5, cM - 10.5, 'Coarse context $L$' + NL + '36³, 14 μm (504 μm)', fs=5.8, ha='left')
txt(27, cM + 16.5, 'Mean path $M$  (where: pores and grains)', fs=6.6, ha='left', color=BLUE_D, fontweight='bold')
slab(25, cM, wC(64), hN(36), BLUE, band=BLUE_D)
arrow((L['r'][0] + 0.3, cM), (25, cM))
for x in [30.5, 34.2, 37.9, 41.6]: slab(x, cM, wC(64), hN(36), shade(BLUE, -0.08), band=BLUE_D)
txt(38.5, cM - 8.0, '16 residual blocks', fs=5.6)
slab(46.5, cM, wC(64), hN(36), BLUE, band=BLUE_D)
line([(26.4, cM + 8.4), (26.4, cM + 11.3), (52.7, cM + 11.3)], BLUE_D, 0.5); arrow((52.7, cM + 11.3), (52.7, cM + 1.8), BLUE_D, 0.5)
oplus(52.7, cM); txt(57, cM + 12.8, 'global skip', fs=5.2, color=BLUE_D)
arrow((49.5, cM), (51.1, cM)); arrow((54.3, cM), (56.5, cM))
slab(56.5, cM, wC(64), hN(16), BLUE, band=BLUE_D); txt(58.5, cM - 6.2, 'centre 16³', fs=5.4)
slab(61.3, cM, wC(32), hN(16), BLUE, band=BLUE_D)
slab(66.3, cM, wC(32), hN(112), shade(BLUE, 0.25)); txt(68, cM - 11.2, '×7 up', fs=5.4)
for x in (71.8, 74.4, 77.0): slab(x, cM, 1.0, hN(112), BLUE, band=BLUE_D)
txt(76, cM - 11.2, '3 conv', fs=5.4)
oplus(90.5, cM); arrow((90.5, cM + 8.9), (90.5, cM + 1.7), '#555555', 0.5); txt(90.5, cM + 10.6, r'$U(L_c)$', fs=5.8)
arrow((86.2, cM), (88.9, cM))
mu = tcube(94, cM, 15, 'mu'); arrow((92.1, cM), (94, cM))
txt(104, cM + 14.6, r'Conditional mean $\mu$, 112³', fs=6)
txt(101.5, cM - 10.4, r'$\mathcal{L}_M=\Vert\mu-y\Vert_1$', fs=5.8, color=BLUE_D)

# G 行
yr = cM - 15.5
txt(56, cG + 18.3, 'Residual texture generator $G$  (what it looks like)', fs=6.6, ha='left', color=RED_D, fontweight='bold')
m1 = tcube(2.5, cG + 5.5, 8.5, 'mu'); n1 = ncube(2.5, cG - 7.5, 8.5)
txt(8.5, cG + 13.6, r'IN($\mu$)', fs=5.8); txt(7.5, cG - 14.4, r'noise $z$ (8 ch)', fs=5.8)
line([(101.5, cM - 12.2), (101.5, yr), (3.6, yr)], '#8A8A8A', 0.5, (0, (2, 1.5))); arrow((3.6, yr), (3.6, cG + 5.5 + 4.25 + 0.2), '#8A8A8A', 0.5, (0, (2, 1.5)))
line([(m1['r'][0] + 0.5, cG + 5.5), (20.5, cG + 5.5), (20.5, cG)], '#555555', 0.5); line([(n1['r'][0] + 0.5, cG - 7.5), (20.5, cG - 7.5), (20.5, cG)], '#555555', 0.5)
arrow((20.5, cG), (27.5, cG)); txt(23.8, cG + 1.8, '↓4', fs=5.4)
g0 = slab(27.5, cG, wC(64), hN(28), RED, band=RED_D)
for x in [32.8, 36.6, 40.4, 44.2]: slab(x, cG, wC(64), hN(28), shade(RED, -0.05), band=RED_D)
arrow((47.6, cG), (50.2, cG)); txt(48.9, cG + 1.8, '↑2', fs=5.2)
for x in (50.2, 53.7, 57.2): slab(x, cG, wC(32), hN(56), shade(RED, -0.05), band=RED_D if x > 51 else None)
arrow((63.6, cG), (66.3, cG)); txt(65.0, cG + 1.8, '↑2', fs=5.2)
slab(66.3, cG, wC(16), hN(112), shade(RED, 0.15)); slab(69.2, cG, wC(16), hN(112), shade(RED, -0.05), band=RED_D)
slab(77.2, cG, wC(16), hN(112), '#F9E0C7', band='#E67E22')
oplus(90.5, cG); arrow((87.3, cG), (88.9, cG))
rr = tcube(94, cG, 15, 'r', cmap='RdBu_r', lo=-0.6, hi=0.6); arrow((92.1, cG), (94, cG))
txt(101.5, cG - 10.4, r'Residual $r=r_\mathrm{m}+r_\mathrm{f}$', fs=6)
def lab(x0, x1, y0, s, ec=RED_D, fc='#FBF1EF'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, 3.4, boxstyle='round,pad=0,rounding_size=0.6', fc=fc, ec=ec, lw=0.5, zorder=6))
    txt((x0 + x1) / 2, y0 + 1.7, s, fs=5.3)
    return (x0 + x1) / 2, y0
bus = 57.5
LB = [lab(32.6, 47.6, cG - hN(28) / 2 - 4.0, 'SPADE block ×4'), lab(52.4, 61.8, cG - hN(56) / 2 - 4.0, 'SPADE ×2'),
      lab(62.6, 77.8, cG - hN(112) / 2 - 4.0, r'SPADE ×1 $\to r_\mathrm{m}$')]
fh = lab(76.8, 88.8, cG - hN(112) / 2 - 8.4, r'fine head $\to r_\mathrm{f}$', ec='#E67E22', fc='#FDF0E3')
# M 的特征注入 G
yb = cG + 14.5
line([(49.5, cM - 5.0), (49.5, yb), (29.5, yb)], BLUE_D, 0.5, (0, (3, 1.5))); arrow((29.5, yb), (29.5, cG + hN(28) / 2 + OY * hN(28) + 0.3), BLUE_D, 0.5, (0, (3, 1.5)))
line([(49.5, yb), (52.0, yb)], BLUE_D, 0.5, (0, (3, 1.5))); arrow((52.0, yb), (52.0, cG + hN(56) / 2 + OY * hN(56) + 0.3), BLUE_D, 0.5, (0, (3, 1.5)))
txt(30.5, yb + 1.6, r'features of $M$ (IN + zero-init 1×1×1 conv)', fs=5.0, color=BLUE_D, ha='left')
# 地质状态 c 与调制总线
cx0, cy0 = 22, 46.5
ax.add_patch(FancyBboxPatch((cx0 - 1.5, cy0 - 4.4), 48.5, 13.2, boxstyle='round,pad=0,rounding_size=1.2', fc='#FBF1EF', ec=RED_D, lw=0.6, zorder=2))
for i, (nm, col) in enumerate([(r'$f_\mathrm{out}$', '#BDBDBD'), (r'$f_\mathrm{p}$', '#C0392B'), (r'$\bar{u}_\mathrm{p}$', '#E67E22'), (r'$f_\mathrm{d}$', '#7F8C8D')]):
    ax.add_patch(Rectangle((cx0 + i * 6.3, cy0 - 3.2), 5.6, 4.2, fc=col, ec='white', lw=0.5, zorder=3, alpha=0.9))
    txt(cx0 + i * 6.3 + 2.8, cy0 - 1.1, nm, fs=6, color='white')
txt(cx0 + 20.5, cy0 + 5.6, r'Geological state $\mathbf{c}$ (whole-rock level)', fs=6.2, color=RED_D, fontweight='bold')
txt(cx0 + 26.5, cy0 - 1.1, 'from the fine-scan' + NL + 'grey histogram', fs=5.0, ha='left', color='#555555')
line([(cx0 + 12.5, cy0 + 8.8), (cx0 + 12.5, bus), (40.1, bus), (fh[0], bus)], RED_D, 0.6)
for (x, y0) in LB: arrow((x, bus), (x, y0), RED_D, 0.5, ms=4.5)
arrow((fh[0], bus), (fh[0], fh[1]), '#E67E22', 0.5, ms=4.5)
txt(48.6, bus + 1.9, r'$\gamma(\mathbf{c}),\ \beta(\mathbf{c})$', fs=5.4, color=RED_D)
txt(fh[0] + 1.0, bus - 1.9, r'$1+g(\mathbf{c})$', fs=5.4, color='#B35A00', ha='left')
# 求和、投影、输出
cs = (cM + cG) / 2
oplus(119, cs)
arrow((115.9, cM + 3.0), (117.9, cs + 1.2), '#555555', 0.6); arrow((115.9, cG + 3.0), (117.9, cs - 1.2), '#555555', 0.6)
ax.add_patch(FancyBboxPatch((124, cs - 7.2), 23.5, 14.4, boxstyle='round,pad=0,rounding_size=1.2', fc='#F2F2F2', ec='#6E6E6E', lw=0.6, zorder=3))
txt(135.75, cs + 4.3, r'Hard projection $P_H$', fs=6.3, fontweight='bold')
txt(135.75, cs + 0.0, r'$x \leftarrow x+U(L_c-D_H(x))$', fs=5.9)
txt(135.75, cs - 4.1, '8 iterations, measured $H$', fs=5.4, color='#555555')
arrow((120.6, cs), (124, cs))
xo = tcube(151.5, cs, 15.5, 'xhat'); arrow((147.5, cs), (151.5, cs))
txt(160.5, cs + 17.8, r'Output $\hat{x}=P_H(\mu+r)$', fs=6.3, fontweight='bold')
txt(160.5, cs + 14.4, '112³, 2.0 μm (224 μm)', fs=5.8)
# 损失
ax.add_patch(FancyBboxPatch((118.5, 115.5), 60, 21, boxstyle='round,pad=0,rounding_size=1.2', fc='white', ec='#BDBDBD', lw=0.5, zorder=2))
txt(120.5, 133.4, r'Generator loss $\mathcal{L}_G$ (on $r$ and $\hat{x}$)', fs=6.1, ha='left', fontweight='bold')
txt(120.5, 125.3, 'direction-binned log spectrum + flatness' + NL + 'local detail energy (7³)' + NL + 'porosity, Euler number, thickness proxy' + NL + r'degradation consistency, adversarial (×0.1)', fs=5.3, ha='left', color='#333333', linespacing=1.3)
# 判别器
ax.add_patch(FancyBboxPatch((118.5, 43.5), 60, 34.5, boxstyle='round,pad=0,rounding_size=1.2', fc='none', ec='#8A8A8A', lw=0.5, ls=(0, (3, 2)), zorder=2))
txt(120.5, 75.8, r'Discriminator $D$ (training only)', fs=6.1, ha='left', fontweight='bold', color='#444444')
xi = tcube(121, 67.0, 6.5, 'xhat'); yi = tcube(121, 55.5, 6.5, 'hr')
txt(124.3, 72.9, r'output $\hat{x}$', fs=5.3); txt(124.3, 50.3, r'fine $y$', fs=5.3)
line([(129.5, 67.0), (131.8, 67.0), (131.8, 55.5), (129.5, 55.5)], '#555555', 0.5); arrow((131.8, 61.2), (134.5, 61.2), '#555555', 0.5)
for i, (c, n) in enumerate([(32, 56), (64, 28), (128, 14), (256, 7)]):
    slab(134.5 + i * 7.2, 61.2, wC(c), hN(n), GREY, band='#7F7F7F')
    txt(134.5 + i * 7.2 + 1.5, 61.2 - hN(n) / 2 - 2.0, '%d³' % n, fs=5.0)
txt(170.5, 63.5, 'real / fake' + NL + 'per 7³ patch', fs=5.3)
txt(158, 51.5, r'+ $\langle\phi(\mathbf{c}), h(x)\rangle$ projection', fs=5.3, color=RED_D)
txt(158, 47.8, 'spectral norm., 4 stride-2 conv', fs=5.0, color='#555555')

# ================= (b) 地质调制 =================
ax.text(1.5, 37.5, 'b', fontsize=8.5, fontweight='bold', va='top')
ax.add_patch(FancyBboxPatch((3.5, 2.5), 86, 34, boxstyle='round,pad=0,rounding_size=1.2', fc='#FDF7F6', ec='#E3B8B1', lw=0.5, zorder=1))
txt(6, 33.6, 'Geological modulation (Eq. 3) and amplitude head', fs=6.4, ha='left', fontweight='bold', color=RED_D)
def rbox(x, y, w, h, s, fc='white', ec='#6E6E6E', fs=5.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=fc, ec=ec, lw=0.5, zorder=4)); txt(x + w / 2, y + h / 2, s, fs=fs)
slab(7, 22, wC(64), hN(28), RED, band=RED_D); txt(9.5, 15.2, r'feature $h$', fs=5.6)
rbox(17, 19.5, 11, 5, 'GroupNorm' + NL + '(no affine)', fs=5.2)
arrow((13.8, 22), (17, 22)); oplus(33, 22, 1.7, '×'); arrow((28, 22), (31.3, 22))
oplus(41, 22, 1.7, '+'); arrow((34.7, 22), (39.3, 22))
arrow((42.7, 22), (47, 22)); txt(51.5, 22, 'ReLU, conv 3³' + NL + '+ noise', fs=5.4)
rbox(17, 6, 12, 5, r'$\mathbf{c}$ (4)', fc='#FBF1EF', ec=RED_D)
rbox(33, 6, 14, 5, '1×1×1 conv' + NL + '(hidden 64)', fs=5.2)
arrow((29, 8.5), (33, 8.5), RED_D, 0.5)
arrow((40, 11), (33, 20.3), RED_D, 0.5); txt(32.8, 15.3, r'$1+\gamma(\mathbf{c})$', fs=5.6, color=RED_D, ha='right')
arrow((43, 11), (41, 20.3), RED_D, 0.5); txt(44.2, 15.3, r'$\beta(\mathbf{c})$', fs=5.6, color=RED_D, ha='left')
txt(40, 3.9, 'output layer zero-initialized', fs=5.0, color='#555555')
# 幅度头
line([(61, 4.5), (61, 32)], '#E3B8B1', 0.5)
slab(64, 22, wC(16), 7, '#F9E0C7', band='#E67E22'); txt(66, 16.3, r'$r_\mathrm{f}$', fs=5.8)
oplus(75.5, 22, 1.7, '×'); arrow((69.8, 22), (73.8, 22)); arrow((77.2, 22), (81.5, 22)); txt(85, 22, r'$r_\mathrm{f}^{\prime}$', fs=6)
rbox(66, 6, 19, 5, r'$g$: 2-layer FC (32)', fs=5.3)
arrow((75.5, 11), (75.5, 20.3), '#B35A00', 0.5); txt(76.5, 15.3, r'$1+g(\mathbf{c})$', fs=5.6, color='#B35A00', ha='left')
txt(76, 29.6, 'shifts grey values across the' + NL + 'pore threshold (porosity)', fs=5.0, color='#555555')
txt(40, 29.2, 'scales channels (texture statistics),' + NL + 'not voxel positions', fs=5.0, color='#555555')

# ================= (c) 硬投影 =================
ax.text(92, 37.5, 'c', fontsize=8.5, fontweight='bold', va='top')
ax.add_patch(FancyBboxPatch((94.5, 2.5), 84, 34, boxstyle='round,pad=0,rounding_size=1.2', fc='#F7F7F7', ec='#CFCFCF', lw=0.5, zorder=1))
txt(97, 33.6, 'Measured degradation and hard projection (Eqs. 5–6)', fs=6.4, ha='left', fontweight='bold', color='#333333')
slab(98, 19, wC(16), hN(112) * 0.62, '#EDEDED'); txt(97.5, 30.3, r'$x$ (112³)', fs=5.6, ha='left')
rbox(108.5, 16.5, 11, 5, r'$B_7$: 7³ mean', fs=5.3); arrow((104.5, 19), (108.5, 19))
axk = fig.add_axes([123.3 / W_MM, 12.5 / H_MM, 12.5 / W_MM, 12.5 / H_MM])
axk.imshow(K[4], cmap='magma', interpolation='nearest'); axk.set_xticks([]); axk.set_yticks([])
for s_ in axk.spines.values(): s_.set_linewidth(0.4)
txt(129.5, 27.3, r'$H$ (9³, centre slice)', fs=5.2); arrow((119.5, 19), (123.1, 19))
txt(143, 22.3, r'$D_H(x)$', fs=5.8); arrow((136.1, 19), (140.3, 19))
oplus(150, 19, 1.7, '−'); arrow((145.8, 19), (148.3, 19))
txt(150, 28.5, r'$L_c$ (coarse centre 16³)', fs=5.4); arrow((150, 26.8), (150, 20.7))
rbox(155.5, 16.5, 9.5, 5, r'$U$: ×7', fs=5.4); arrow((151.7, 19), (155.5, 19))
oplus(170, 19, 1.7, '+'); arrow((165, 19), (168.3, 19))
line([(100.5, 13.7), (100.5, 8.8 - 3.5), (170, 5.3), (170, 17.3)], '#6E6E6E', 0.5); arrow((170, 16.6), (170, 17.3), '#6E6E6E', 0.5)
arrow((171.7, 19), (176.5, 19)); txt(176, 22, 'repeat ×8', fs=5.2, ha='right', color='#555555')
txt(136, 8.2, 'low frequencies forced to match the coarse scan; fine residual unaffected', fs=5.0, color='#555555')
ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM)
T.save(fig, H / 'fig_arch3d')