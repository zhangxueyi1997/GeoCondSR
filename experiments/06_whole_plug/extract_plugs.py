# -*- coding: utf-8 -*-
"""第 59 步（本地）：从 40 根岩心柱（每组平行对 A、B）的整柱粗扫（13.93 μm）抽取 36³ 窗口，供服务器超分，检验与实验室孔隙度的一致性。
每根：沿高 15%–85% 等距取 6 段、每段连续 36 层；每段在岩心内部（中心距 ≤ 0.85R − 26 px）随机取 50 个窗口，共 300 个。
归一化与训练粗扫同一口径（tir.mode_of）：空气 = 1.06R 以外（角落）像素众数；骨架 = 0.88R 以内、> 0.2·空气 的像素众数；u = (v − 空气)/(骨架 − 空气)。
另存整柱粗扫统计（0.88R 以内的 u：<0.5 占比、分位 2/10/50、标准差）。输出 data/plugs/<柱号>.npz。"""
import sys, re, json, time
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
import fastio, centre, tir  # noqa: E402  (after preprocessing/ is on the path)
H = Path(__file__).resolve().parent; OUT = DATA_ROOT / 'plugs'; OUT.mkdir(parents=True, exist_ok=True)
ROOT = REPO_ROOT; VOX_MM = 0.01393
src = (REPO_ROOT / 'configs' / 'sample_properties.py').read_text(encoding='utf-8')
DIA = {m[0]: float(m[1]) for m in re.findall(r'\("([A-Z]+-\d+)", "[^"]+", [\d.]+, ([\d.]+), [\d.]+, [\d.]+', src)}
PLUGS = []
import csv
for r in csv.DictReader(open(REPO_ROOT / 'configs' / 'sample_mapping.csv', encoding='utf-8-sig')):
    PLUGS.append((r['large_sample_id'], r['group_id'], 'A', RAW_ROOT / r['group_id'] / 'large_ct' / 'raw16'))
for d in sorted((RAW_ROOT / 'parallel').glob('PL*')):
    pid = d.name.split('_', 1)[1]; PLUGS.append((pid, d.name.split('_')[0], 'B', d / 'large_ct' / 'raw16'))
rng = np.random.default_rng(0); NS, NW, L = 6, 50, 36
only = sys.argv[1:] or None
for pid, grp, ab, d in PLUGS:
    if only and pid not in only: continue
    f_out = OUT / ('%s.npz' % pid)
    if f_out.exists(): print('已有', pid, flush=True); continue
    t0 = time.time(); files = sorted(d.glob('*.tif')); n = len(files)
    if n < 100: print('跳过', pid, '切片数', n, flush=True); continue
    R = DIA[pid] / 2 / VOX_MM
    mid = fastio.read(files[n // 2]).astype(np.float32); cy, cx = centre.disc_centre(mid, R)
    yy, xx = np.indices(mid.shape); rr = np.hypot(yy - cy, xx - cx)
    air = float(np.median([tir.mode_of(fastio.read(files[int(n * q)]).astype(np.float32)[rr > 1.06 * R]) for q in (0.3, 0.5, 0.7)]))
    inner = rr <= 0.88 * R
    sv = np.concatenate([fastio.read(files[int(n * q)]).astype(np.float32)[inner] for q in (0.2, 0.35, 0.5, 0.65, 0.8)])
    sv = sv[sv > air * 0.2]; D = tir.mode_of(sv) - air
    W, U = [], []
    cand = np.argwhere(rr[18:-18, 18:-18] <= 0.85 * R - 26) + 18
    for z0 in np.linspace(n * 0.15, n * 0.85 - L, NS).astype(int):
        vol = np.stack([fastio.read(files[z]) for z in range(z0, z0 + L)]).astype(np.float32)
        u = (vol - air) / D; U.append(u[L // 2][inner])
        for j in rng.choice(len(cand), NW, replace=False):
            y, x = cand[j]; W.append(u[:, y - 18:y + 18, x - 18:x + 18].astype(np.float16))
    Uc = np.concatenate(U)
    stats = dict(phi_coarse=float((Uc < 0.5).mean()), p2=float(np.percentile(Uc, 2)), p10=float(np.percentile(Uc, 10)),
                 p50=float(np.percentile(Uc, 50)), std=float(Uc.std()))
    np.savez_compressed(f_out, win=np.stack(W), air=air, D=D, R=R, cy=cy, cx=cx, n=n, group=grp, ab=ab, **stats)
    print('%s %s %s  切片 %d  R %.0f px  空气 %.0f  Δ %.0f  粗扫<0.5 %.4f  分位2 %.3f  std %.3f  （%.0f s）' % (pid, grp, ab, n, R, air, D, stats['phi_coarse'], stats['p2'], stats['std'], time.time() - t0), flush=True)
print('DONE')
