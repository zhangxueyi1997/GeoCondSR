# -*- coding: utf-8 -*-
"""第 59 步（本地）：整柱粗扫的杯状伪影（射束硬化）校正参数。
1) 按 extract_plugs.py 的随机数消耗顺序复现每个窗口的位置（z0, y, x）——CQ-1 在首轮单独抽取（新 rng(0)），其余 39 根在第二轮按顺序（新 rng(0)，跳过 CQ-1）；
   逐根取 2 个窗口从原始切片重读，与已存窗口逐位比较以核验。
2) 径向骨架灰度剖面：在 5 个高度的切片上，按 r/R 分 44 环（0–0.88），每环取归一化灰度 u 的众数（tir.mode_of，只用 u>0.2），
   取 5 层中位数，再以 r² 的 4 次多项式平滑；校正为 u / g(r)（骨架拉平到 1，空气 0 不变）。
输出 data/plugs/cup/<柱号>.npz：pos (300,3)、rb（环中心 r/R）、g_raw、coef（多项式系数，自变量 (r/R)²）。"""
import sys, re, csv, time
from pathlib import Path
import numpy as np
import fastio, tir
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; P = DATA_ROOT / 'plugs'; O = P / 'cup'; O.mkdir(exist_ok=True)
ROOT = REPO_ROOT
PLUGS = []
for r in csv.DictReader(open(REPO_ROOT / 'configs' / 'sample_mapping.csv', encoding='utf-8-sig')):
    PLUGS.append((r['large_sample_id'], RAW_ROOT / r['group_id'] / 'large_ct' / 'raw16'))
for d in sorted((RAW_ROOT / 'parallel').glob('PL*')):
    PLUGS.append((d.name.split('_', 1)[1], d / 'large_ct' / 'raw16'))
NS, NW, L = 6, 50, 36


def positions(pid, rng, shape):
    z = np.load(P / (pid + '.npz')); R, cy, cx, n = float(z['R']), float(z['cy']), float(z['cx']), int(z['n'])
    yy, xx = np.indices(shape); rr = np.hypot(yy - cy, xx - cx)
    cand = np.argwhere(rr[18:-18, 18:-18] <= 0.85 * R - 26) + 18; pos = []
    for z0 in np.linspace(n * 0.15, n * 0.85 - L, NS).astype(int):
        for j in rng.choice(len(cand), NW, replace=False): pos.append((z0, *cand[j]))
    return np.array(pos), z


rng2 = np.random.default_rng(0)
for pid, d in PLUGS:
    t0 = time.time(); files = sorted(d.glob('*.tif')); shape = fastio.read(files[0]).shape
    rng = np.random.default_rng(0) if pid == 'CQ-1' else rng2
    pos, z = positions(pid, rng, shape)
    air, D, R, cy, cx, n = float(z['air']), float(z['D']), float(z['R']), float(z['cy']), float(z['cx']), int(z['n'])
    W = z['win']; ok = []
    for k in (0, 299):                                   # 核验：从原始切片重读
        z0, y, x = pos[k]; v = np.stack([fastio.read(files[zz])[y - 18:y + 18, x - 18:x + 18] for zz in range(z0, z0 + L)]).astype(np.float32)
        ok.append(np.array_equal(((v - air) / D).astype(np.float16), W[k]))
    yy, xx = np.indices(shape); rq = np.hypot(yy - cy, xx - cx) / R
    rb = np.arange(0.01, 0.88, 0.02); prof = []
    for q in (0.2, 0.35, 0.5, 0.65, 0.8):
        u = (fastio.read(files[int(n * q)]).astype(np.float32) - air) / D
        prof.append([tir.mode_of(u[(np.abs(rq - c) < 0.01) & (u > 0.2)]) for c in rb])
    g = np.median(prof, 0); coef = np.polyfit(rb ** 2, g, 4)
    np.savez(O / (pid + '.npz'), pos=pos, rb=rb, g_raw=g, coef=coef, R=R, cy=cy, cx=cx)
    gf = np.polyval(coef, rb ** 2)
    print('%-6s 核验 %s  骨架剖面 r/R=0/0.4/0.8: %.3f %.3f %.3f  拟合残差 %.4f（%.0f s）' % (pid, ok, gf[0], gf[20], gf[40], np.abs(gf - g).max(), time.time() - t0), flush=True)
