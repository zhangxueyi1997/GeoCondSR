# -*- coding: utf-8 -*-
"""第 59 步：超分结果的渗流评价（LBM，lbm59s：D3Q19 TRT，τ+ = 0.6，镜像周期，已对解析解验证 0.998–1.014）。
读 eval59 / eval59a 存下的孔隙掩膜（同一批 200 块、同一阈值 0.5）：
  1) 贯通判别：每块、三个方向（z/y/x），孔隙是否连通两端（6 连通）——全部 200 块；
  2) 绝对渗透率（z 向）：细扫在 z 向贯通的块（按抽样顺序取前 60 块），细扫与各方法各算一次；方法不贯通记 0。
结果按（折, 体积）缓存到 lbm59_<折>.json，已算过的不重算（新基线训完后只补它们）。k 单位 mD（体素 2 μm）。
用法：python lbm_eval59.py <折> <gpu> <前缀 eval59a|eval59>"""
import sys, os, json, time, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from lbm59s import permeability, spanning


FOLD, dev, PFX = sys.argv[1], 'cuda:%s' % sys.argv[2], sys.argv[3]
OUT = str(OUT_ROOT) + '/eval59/out/'; CF = OUT + 'lbm59_%s.json' % FOLD; N = 112; NK = 60; MD = 4.0 / 9.869233e-4
D = np.load(OUT + '%s_%s_masks.npz' % (PFX, FOLD))
C = json.load(open(CF, encoding='utf-8')) if os.path.exists(CF) else {}
un = lambda row: np.unpackbits(row)[:N ** 3].reshape(N, N, N).astype(bool)
vols = ['细扫'] + [k for k in D.files if k not in ('idx', '细扫')]
if '细扫' not in C:
    T = [un(r) for r in D['细扫']]
    sp = [[bool(spanning(np.moveaxis(p, a, 0)).any()) for a in range(3)] for p in T]
    sel = [i for i, s in enumerate(sp) if s[0]][:NK]
    t0 = time.time(); k = permeability([T[i] for i in sel], dev)
    C['细扫'] = dict(span=sp, sel=sel, kz=[v * MD for v in k], idx=D['idx'].tolist())
    print(FOLD, '细扫：z/y/x 贯通 %s / 200；取 %d 块算 k，用时 %.0f s，%d 步' % (np.sum(sp, 0).tolist(), len(sel), time.time() - t0, permeability.iters), flush=True)
    json.dump(C, open(CF, 'w', encoding='utf-8'), ensure_ascii=False)
sel = C['细扫']['sel']; assert C['细扫']['idx'] == D['idx'].tolist()
for v in vols[1:]:
    if v in C: print(FOLD, v, '已算，跳过', flush=True); continue
    P = [un(r) for r in D[v]]
    sp = [[bool(spanning(np.moveaxis(p, a, 0)).any()) for a in range(3)] for p in P]
    t0 = time.time(); k = permeability([P[i] for i in sel], dev)
    C[v] = dict(span=sp, kz=[x * MD for x in k])
    kt = np.array(C['细扫']['kz']); kp = np.array(C[v]['kz']); ok = (kp > 0) & (kt > 0)
    print(FOLD, v, 'z/y/x 贯通 %s；贯通判别一致率 %.3f；细扫贯通块中也贯通 %d/%d；log10k 误差中位 %.3f（%.0f s）' % (
        np.sum(sp, 0).tolist(), float(np.mean(np.array(sp) == np.array(C['细扫']['span']))), ok.sum(), len(sel),
        float(np.median(np.abs(np.log10(kp[ok]) - np.log10(kt[ok])))) if ok.any() else float('nan'), time.time() - t0), flush=True)
    json.dump(C, open(CF, 'w', encoding='utf-8'), ensure_ascii=False)
print('DONE', FOLD, flush=True)
