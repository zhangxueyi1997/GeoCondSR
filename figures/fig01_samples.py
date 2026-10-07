# -*- coding: utf-8 -*-
"""图 1：样品与实验设备照片。照片来自 paper3_manuscript/资料_实验与设备（用户提供的项目汇报与仪器简介，extract_text.py 导出）。
(a) 岩心柱：进展汇报第 19 页的单根照片，裁去手写标注，统一为同一画幅（14 张自动定框，5 张人工定框）；
(b) 薄板、(c) 小圆柱：进展汇报第 20 页；(d) CT：进展汇报第 21 页；(e) 孔隙度仪：《系列一.岩石孔隙度-吸附-扩散测试仪简介》第 1 页（按 EXIF 方向转正）。"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; SRC = H.parent.parent / '资料_实验与设备' / '图片' / '提取'
import gse_style as S
S.FIGDIR = OUT_ROOT / 'figures'; S.FIGDIR.mkdir(parents=True, exist_ok=True); S.apply_style()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

AUTO = json.load(open(SRC.parent / '_柱体裁切框.json', encoding='utf-8'))
MANUAL = {'2dd33317': (0.49, 0.65, 0.28, 0.71), '6ef7c0f1': (0.48, 0.62, 0.28, 0.66), '8793ec35': (0.46, 0.62, 0.31, 0.75),
          '8d55d5ac': (0.37, 0.54, 0.255, 0.70), '9b4012a1': (0.52, 0.66, 0.33, 0.73)}      # 柱体左、右、上、下（占画面比例）
TW, TH = 200, 360
tiles = []
for name, box in AUTO.items():
    im = Image.open(SRC / name).convert('RGB'); key = name.split('_')[-1].split('.')[0]
    if key in MANUAL:
        x0, x1, y0, y1 = MANUAL[key]; W, Hh = im.size; h = (y1 - y0) * Hh; cx = (x0 + x1) / 2 * W
        box = [cx - 0.31 * h, y0 * Hh - 0.06 * h, cx + 0.31 * h, y0 * Hh - 0.06 * h + 1.12 * h]
    tiles.append(im.crop([int(v) for v in box]).resize((TW, TH), Image.LANCZOS))
g = 8; ncol = 10; nrow = 2
mos = Image.new('RGB', (ncol * TW + (ncol - 1) * g, nrow * TH + (nrow - 1) * g), 'white')
for i, t in enumerate(tiles): mos.paste(t, ((i % ncol) * (TW + g), (i // ncol) * (TH + g)))
pics = {'b': Image.open(SRC / '岩石力学参数预测进展_p20_36b1743c.png').convert('RGB'),
        'c': Image.open(SRC / '岩石力学参数预测进展_p20_90a949a5.png').convert('RGB'),
        'd': Image.open(SRC / '岩石力学参数预测进展_p21_4951ae51.png').convert('RGB'),
        'e': ImageOps.exif_transpose(Image.open(SRC / '系列一.岩石孔隙度-_p01_2e43eb3c.jpg')).convert('RGB')}
if pics['e'].width > pics['e'].height: pics['e'] = pics['e'].rotate(-90, expand=True)       # 仪器照片为竖幅
asp = [pics[k].width / pics[k].height for k in 'bcde']
fig = plt.figure(figsize=(S.DOUBLE / 25.4, 0.1)); W_mm = S.DOUBLE
h1 = W_mm * mos.height / mos.width; h2 = W_mm / sum(asp); fig.set_size_inches(W_mm / 25.4, (h1 + h2 + 11) / 25.4)
gs = GridSpec(2, 4, figure=fig, height_ratios=[h1, h2], width_ratios=asp, hspace=0.10, wspace=0.04, left=0.01, right=0.99, top=0.955, bottom=0.01)
ax = fig.add_subplot(gs[0, :]); ax.imshow(np.asarray(mos)); ax.axis('off'); ax.text(0.0, 1.01, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
for j, k in enumerate('bcde'):
    ax = fig.add_subplot(gs[1, j]); ax.imshow(np.asarray(pics[k])); ax.axis('off')
    ax.text(0.0, 1.02, '(%s)' % k, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
fig.savefig(H / 'fig01_samples.png', dpi=300); fig.savefig(H / 'fig01_samples.pdf', dpi=300)
print('岩心柱', len(tiles), '张；画幅 %.0f × %.0f mm' % (W_mm, h1 + h2 + 8))
