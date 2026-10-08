# -*- coding: utf-8 -*-
"""图 11 数据（本地）：(b) 岩心内外径向灰度剖面——3 根代表性 A 柱各 3 层切片，按 r/R 0–1.6 每 0.02 一档取中位数，用远场空气与骨架众数归一化（即原始口径 u）。
输出 data/plugs/fig11_radial.npz。"""
import sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
import fastio  # noqa: E402  (after preprocessing/ is on the path)
H = Path(__file__).resolve().parent; P = FIGDATA / 'plugs'
rb = np.arange(0.0, 1.6, 0.02); out = {'rb': rb + 0.01}
for pid, g in (('CQ-1', 'G01'), ('GZ-1', 'G02'), ('SC-13', 'G06')):
    z = np.load(DATA_ROOT / 'plugs' / (pid + '.npz')); air, D, R, cy, cx, n = [float(z[k]) for k in ('air', 'D', 'R', 'cy', 'cx', 'n')]
    fs = sorted(Path(str(RAW_ROOT) + '/%s/large_ct/raw16' % g).glob('*.tif')); prof = []
    for q in (0.3, 0.5, 0.7):
        u = (fastio.read(fs[int(n * q)]).astype(np.float32) - air) / D; yy, xx = np.indices(u.shape); rr = np.hypot(yy - cy, xx - cx) / R
        prof.append([np.median(u[(rr >= a) & (rr < a + 0.02)]) if ((rr >= a) & (rr < a + 0.02)).any() else np.nan for a in rb])
    out[pid] = np.nanmedian(prof, 0); print(pid, np.round(out[pid][::8], 3), flush=True)
np.savez(P / 'fig11_radial.npz', **out)
