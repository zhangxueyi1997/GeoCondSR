# -*- coding: utf-8 -*-
"""第 59 步（本地）：整柱粗扫的近表面空气电平。远场空气（1.06R 以外）受射束硬化与散射抬高，比紧贴岩心表面的空气高 30%–40% 的对比度；
小柱（3 mm）衰减小，远场即真空气。近表面空气 = 3 层切片上 1.005R–1.04R 环内像素众数的中位数。写入 data/plugs/cup/<柱号>.npz（键 air_near）。"""
import sys, glob
from pathlib import Path
import numpy as np
import fastio, tir
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; P = DATA_ROOT / 'plugs'; ROOT = REPO_ROOT
import csv
DIRS = {r['large_sample_id']: RAW_ROOT / r['group_id'] / 'large_ct' / 'raw16' for r in csv.DictReader(open(REPO_ROOT / 'configs' / 'sample_mapping.csv', encoding='utf-8-sig'))}
DIRS.update({d.name.split('_', 1)[1]: d / 'large_ct' / 'raw16' for d in (RAW_ROOT / 'parallel').glob('PL*')})
for pid, d in DIRS.items():
    z = np.load(P / (pid + '.npz')); air, D, R, cy, cx, n = float(z['air']), float(z['D']), float(z['R']), float(z['cy']), float(z['cx']), int(z['n'])
    fs = sorted(d.glob('*.tif')); near = []
    for q in (0.3, 0.5, 0.7):
        a = fastio.read(fs[int(n * q)]).astype(np.float32); yy, xx = np.indices(a.shape); rr = np.hypot(yy - cy, xx - cx) / R
        near.append(tir.mode_of(a[(rr > 1.005) & (rr < 1.04)]))
    an = float(np.median(near)); c = dict(np.load(P / 'cup' / (pid + '.npz'))); c.update(air_near=an, air=air, D=D)
    np.savez(P / 'cup' / (pid + '.npz'), **c)
    print('%-6s 远场 %.0f 近表面 %.0f 对比度 +%.0f%%' % (pid, air, an, 100 * ((air + D - an) / D - 1)), flush=True)
