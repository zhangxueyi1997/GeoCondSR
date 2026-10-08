# -*- coding: utf-8 -*-
"""整柱应用图（4.7 节）。
(a) CQ-1（G01 A 柱）第 1 段中间切片（标定后灰度：近表面空气 + 杯状校正），方框为该段 50 个 36³ 窗口的位置（0.5 mm）。
(b) 该段第 1 个窗口的中心 224 μm：标定后粗扫与各方法超分的中心切片（服务器 plugs59/wc_vis59.py 输出）。
(c) 34 根测试岩心柱的超分孔隙率与氦孔隙度；(d) 26 根有细扫真值的岩心柱的超分孔隙率与同组小柱细扫；
(e) 本文在各处理阶段的“超分孔隙率/氦孔隙度”中位数与相对细扫真值的相关。数据：data/plugs/out/plugsr59_<折>{_n54b,_n59n}_cup.json。"""
import sys, json, csv
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import sci_style as T
import fastio
T.apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle

P = FIGDATA / 'plugs'; FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
TAB = json.load(open(P / 'plug_table.json', encoding='utf-8'))
FINE = {r['组号']: float(r['细扫可分辨孔隙度']) / 100 for r in csv.DictReader(open(H / 'tab01_samples.csv', encoding='utf-8-sig')) if r['细扫可分辨孔隙度'] not in ('', 'nan')}


def load(gs):
    V = {}
    for f in FOLDS:
        r = json.load(open(P / 'out' / ('plugsr59_%s%s_cup.json' % (f, gs)), encoding='utf-8'))
        V.update({p: v for p, v in r['plugs'].items() if v['code'] == f})
    return V


V59, V54 = load('_n59n'), load('_n54b')
PIDS = sorted(V59); HE = np.array([TAB[p]['phi_he'] / 100 for p in PIDS]); GRP = [TAB[p]['group'] for p in PIDS]
OK = np.array([g in FINE for g in GRP]); FI = np.array([FINE.get(g, np.nan) for g in GRP])


def phi(V, cond, m):
    return np.array([np.mean(V[p][cond][m]) for p in PIDS])


def stat(x, tag):
    e = x[OK] - FI[OK]
    s = dict(bias=e.mean(), mae=np.abs(e).mean(), r_fine=pearsonr(x[OK], FI[OK])[0], r_he=pearsonr(x, HE)[0], rho_he=spearmanr(x, HE)[0], ratio=np.median(x / HE))
    print('%-28s n=%d/%d 偏差 %+.5f MAE %.5f r细扫 %.3f | r氦 %.3f ρ氦 %.3f 比值 %.3f' % (tag, len(x), OK.sum(), s['bias'], s['mae'], s['r_fine'], s['r_he'], s['rho_he'], s['ratio']))
    return s


MAIN = [('粗扫阈值', 'Coarse threshold', '#9A9A9A', 's'), ('EDSR-3D', 'EDSR-3D', T.C['EDSR-3D'], 'o'), ('SRGAN-3D', 'SRGAN-3D', T.C['SRGAN-3D'], 'D'),
        ('本文·仅粗扫', 'GeoCondSR', T.C['本文'], 'o')]
X = {m: phi(V59, 'cal', m) for m, *_ in MAIN}; S = {m: stat(X[m], '标定 ' + m) for m in X}
stat(phi(V59, 'cal', '三线性'), '标定 三线性')
ST = [('No calibration', phi(V54, 'raw', '本文·仅粗扫')), ('+ imaging\ncalibration', phi(V54, 'cal', '本文·仅粗扫')),
      ('+ noise-aware\ntuning', X['本文·仅粗扫'])]
SS = [stat(x, '本文阶段 ' + t.replace(chr(10), ' ')) for t, x in ST]
for m in ('EDSR-3D', 'SRGAN-3D', '本文·仅粗扫'):
    stat(phi(V54, 'cal_harm', m), '标定+谱减（无微调）' + m)
print('细扫与氦（26 根）r %.3f ρ %.3f' % (pearsonr(FI[OK], HE[OK])[0], spearmanr(FI[OK], HE[OK])[0]))

# ---------- 图像 ----------
Z = np.load(P / 'wc_vis59.npz'); c = np.load(P / 'cup/CQ-1.npz'); d = np.load(P / 'CQ-1.npz')
files = sorted((RAW_ROOT / 'G01' / 'large_ct' / 'raw16').glob('*.tif')); z0 = int(c['pos'][0, 0])   # without the raw slices, use the copy of this slice in the figure data
img = (fastio.read(files[z0 + 18]) if files else np.load(FIGDATA / 'plugs' / 'CQ-1_slice.npy')).astype(np.float32)
a0, D0, an = float(c['air']), float(c['D']), float(c['air_near']); Dn = a0 + D0 - an
yy, xx = np.indices(img.shape); rr = np.hypot(yy - float(c['cy']), xx - float(c['cx'])) / float(c['R'])
g = np.polyval(c['coef'], np.clip(rr, 0, 0.88) ** 2); gn = (g * D0 + a0 - an) / Dn
u = (img - an) / Dn / np.where(rr <= 1.0, gn, 1.0)
assert np.allclose(Z['pos'], c['pos'][:8])
R = float(c['R']); cy, cx = float(c['cy']), float(c['cx']); h = int(R * 1.04)

fig = plt.figure(figsize=(T.DOUBLE * T.MM, 182 * T.MM))


def frame(ax, col='#808080', lw=0.4):
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(True); s_.set_color(col); s_.set_linewidth(lw)

top = GridSpec(1, 2, figure=fig, left=0.035, right=0.985, top=0.977, bottom=0.668, width_ratios=[1, 1.55], wspace=0.08)
bot = GridSpec(2, 2, figure=fig, left=0.075, right=0.985, top=0.6, bottom=0.05, wspace=0.3, hspace=0.42)
ax = fig.add_subplot(top[0, 0])
ax.imshow(u[int(cy) - h:int(cy) + h, int(cx) - h:int(cx) + h], cmap='gray', vmin=0, vmax=1.6, interpolation='antialiased')
for k, (_, y, x) in enumerate(c['pos'][:50]):
    ax.add_patch(Rectangle((x - 18 - (cx - h), y - 18 - (cy - h)), 36, 36, fill=False, lw=0.5 if k else 1.1, ec='#E8B04A' if k else '#C0392B'))
ax.plot([2 * h - 80 - 5 / 0.01393, 2 * h - 80], [2 * h - 70] * 2, color='white', lw=1.6, solid_capstyle='butt')
ax.text(2 * h - 80 - 2.5 / 0.01393, 2 * h - 95, '5 mm', color='white', ha='center', va='bottom', fontsize=6.5)
ax.axis('off'); T.label(ax, 'a', dx=0.0, dy=0.985)
ax.text(0.02, 0.02, 'CQ-1 (G01, plug A)\n13.93 μm, calibrated', transform=ax.transAxes, fontsize=6, color='white', va='bottom')

sub = GridSpecFromSubplotSpec(2, 4, subplot_spec=top[0, 1], wspace=0.05, hspace=0.16, height_ratios=[1, 1])
w0 = Z['win_cal_mid'][0]
ax = fig.add_subplot(sub[0, 0]); ax.imshow(w0, cmap='gray', vmin=0, vmax=1.6, interpolation='nearest', extent=(0, 504, 504, 0))
ax.add_patch(Rectangle((140, 140), 224, 224, fill=False, ec='#C0392B', lw=0.8)); frame(ax)
ax.set_title('Coarse window (504 μm)', fontsize=6.5, pad=2); T.label(ax, 'b', dx=-0.08, dy=1.0)
ax = fig.add_subplot(sub[0, 1]); ax.imshow(w0[10:26, 10:26], cmap='gray', vmin=0, vmax=1.6, interpolation='nearest', extent=(0, 224, 224, 0))
frame(ax, '#C0392B', 0.8); ax.set_title('Coarse, 224 μm', fontsize=6.5, pad=2)
for j, (m, t, col) in enumerate([('三线性', 'Trilinear', '#6E6E6E'), ('EDSR-3D', 'EDSR-3D', T.C['EDSR-3D']), ('SRGAN-3D', 'SRGAN-3D', T.C['SRGAN-3D']), ('本文', 'GeoCondSR', T.C['本文'])]):
    ax = fig.add_subplot(sub[1, j]) if j < 4 else None
    ax.imshow(Z[m][0], cmap='gray', vmin=0, vmax=1.6, interpolation='nearest', extent=(0, 224, 224, 0))
    frame(ax); ax.set_title(t, fontsize=6.5, pad=2)
    ax.text(0.04, 0.04, r'$\phi$ %.1f%%' % (100 * float(Z[m + '|phi'][0])), transform=ax.transAxes, fontsize=5.5, color='white', va='bottom')
    if j == 3:
        ax.plot([160, 210], [212, 212], color='white', lw=1.2, solid_capstyle='butt'); ax.text(185, 205, '50 μm', color='white', fontsize=5, ha='center', va='bottom')
axn = fig.add_subplot(sub[0, 2:]); axn.axis('off')

# (c) 与氦孔隙度
ax = fig.add_subplot(bot[0, 0])
xx_ = np.linspace(0, 0.22, 10)
ax.plot(100 * xx_, 100 * xx_, color='#BDBDBD', lw=0.6); ax.plot(100 * xx_, 50 * xx_, color='#BDBDBD', lw=0.6, ls=(0, (3, 2)))
for m, t, col, mk in MAIN:
    ax.scatter(100 * HE, 100 * X[m], s=9 if m != '本文·仅粗扫' else 12, marker=mk, color=col, edgecolor='white', lw=0.3, zorder=3 if m == '本文·仅粗扫' else 2,
               label='%s  (r = %.2f)' % (t, S[m]['r_he']))
ax.text(20.5, 20.8, '1:1', fontsize=6, color=T.GREY, ha='right'); ax.text(21, 9.6, '1:2', fontsize=6, color=T.GREY, ha='right')
ax.set_xlim(0, 21.5); ax.set_ylim(0, 21.5); ax.set_aspect('equal')
ax.set_xlabel('Helium porosity (%)'); ax.set_ylabel('SR resolvable porosity (%)')
ax.legend(loc='upper left', fontsize=5.5, handletextpad=0.2, borderaxespad=0.1)
T.label(ax, 'c', dx=-0.2)

# (d) 与小柱细扫：|偏差| 与平均绝对误差
ax = fig.add_subplot(bot[0, 1])
MB = MAIN + [('三线性', 'Trilinear', '#B7B7B7', 's')]
MB = [MB[4], MB[0], MB[1], MB[2], MB[3]]
xb = np.arange(len(MB)); w = 0.38
for k, (m, t, col, mk) in enumerate(MB):
    st = S[m] if m in S else stat(phi(V59, 'cal', m), '标定 ' + m)
    ax.bar(k - w / 2, 100 * abs(st['bias']), w, color=col, zorder=2)
    ax.bar(k + w / 2, 100 * st['mae'], w, color=col, alpha=0.5, zorder=2)
ax.set_xticks(xb); ax.set_xticklabels(['Tri-\nlinear', 'Coarse\nthresh.', 'EDSR-\n3D', 'SRGAN-\n3D', 'GeoCondSR'], fontsize=5.6)
ax.set_ylabel('Error vs. small-plug fine scan (pp)'); ax.set_ylim(0, 6.2)

T.label(ax, 'd', dx=-0.2)

# (e) 处理阶段
ax = fig.add_subplot(bot[1, 0]); xb = np.arange(3)
ax.bar(xb, [s['ratio'] for s in SS], 0.58, color=['#E3B3AE', '#D07A70', T.C['本文']])
ax.axhline(1.0, color=T.GREY, lw=0.6, label='Upper bound (= 1)'); ax.axhline(0.50, color=T.C['细扫'], lw=0.7, ls=(0, (3, 2)), label='Small-plug fine scan')
ax.legend(loc='upper right', fontsize=5.5, handlelength=1.6, borderaxespad=0.1)
for i, s in enumerate(SS):
    ax.text(i, s['ratio'] + 0.03, 'r = %.2f' % s['r_fine'], ha='center', va='bottom', fontsize=5.5)
ax.set_xticks(xb); ax.set_xticklabels([t for t, _ in ST], fontsize=5.5)
ax.set_ylabel('SR porosity / helium porosity'); ax.set_ylim(0, 1.95); ax.set_xlim(-0.5, 2.5)
T.label(ax, 'e', dx=-0.1)
# (f) 整柱“粗扫 + 稀疏细扫”（预登记 prereg_wcsparse_20261002.md）
WS = {}
for f in FOLDS:
    fp = P / 'out' / ('wcsparse59_%s.json' % f)
    if fp.exists(): WS.update(json.load(open(fp, encoding='utf-8'))['plugs'])
if WS:
    ax = fig.add_subplot(bot[1, 1]); ps = sorted(WS)
    he = np.array([WS[p]['phi_he'] for p in ps]); y0 = np.array([100 * np.mean(V59[p]['cal']['本文·仅粗扫']) for p in ps]); y1 = np.array([100 * np.mean(WS[p]['phi']['sparse']) for p in ps])
    tight = np.array([WS[p]['group'] in ('G02', 'G09', 'G11', 'G12') for p in ps])
    xx_ = np.linspace(0, 22, 10); ax.plot(xx_, xx_, color='#BDBDBD', lw=0.6); ax.plot(xx_, xx_ / 2, color='#BDBDBD', lw=0.6, ls=(0, (3, 2)))
    for a, b, c, t in zip(he, y0, y1, tight):
        ax.annotate('', xy=(a, c), xytext=(a, b), arrowprops=dict(arrowstyle='-|>', lw=0.6 if t else 0.4, color='#7B1A14' if t else '#C9A9A6', mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.scatter(he, y0, s=9, facecolor='white', edgecolor=T.C['本文'], lw=0.6, zorder=3, label='GeoCondSR, coarse only (r = %.2f)' % pearsonr(y0, he)[0])
    ax.scatter(he, y1, s=11, color=T.C['本文+稀疏细扫'], zorder=4, label='+ sparse fine scan (r = %.2f)' % pearsonr(y1, he)[0])
    ax.set_xlim(0, 21.5); ax.set_ylim(0, 15); ax.set_xlabel('Helium porosity (%)'); ax.set_ylabel('SR resolvable porosity (%)')
    ax.legend(loc='upper left', fontsize=5.3, handletextpad=0.2, borderaxespad=0.1)
    T.label(ax, 'f', dx=-0.2)
T.save(fig, H / 'fig_wholecore_app')