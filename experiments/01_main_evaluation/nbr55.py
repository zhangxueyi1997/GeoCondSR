# -*- coding: utf-8 -*-
"""第 55 步路线②：用「周围」的地质变量预测目标块的地质变量（拟合度 + 代替值）。
特征（都不含目标块）：
  · 本格环带 c_ann（格 504 μm 减去中心 224 μm 目标块）；
  · 隔一格邻居（±2 格，±36 粗体素）的 c_cell 均值 —— 格步长 18、边长 36，紧邻格（±1）与目标块重叠，不能用。
目标：c_blk（目标块实测，细扫）。每个留一岩性折单独拟合岭回归，只用训练岩性；测试岩性只报告拟合度。
输出：在 cfield2/{gid}.npz 里新增 c_nbr_<折>（原有字段不动；首次运行先整体备份到 cfield2_bak55/）。"""
import json, shutil, numpy as np
from pathlib import Path
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, OUT_ROOT, fold_meta  # noqa: E402
D = Path(str(DATA_ROOT) + '/cfield2'); BK = Path(str(DATA_ROOT) + '/cfield2_bak55')
if not BK.exists(): shutil.copytree(D, BK); print('已备份到', BK)
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; NAMES = ['IGV', 'f_pore', 'mu_inter', 'f_dense']
data = {f.stem: dict(np.load(BK / f.name)) for f in sorted(D.glob('*.npz'))}      # 一律从备份读，重复运行结果一致
OFF = [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)]


def features(d):
    ann, cell, val = d['c_ann'].astype(np.float64), d['c_cell'].astype(np.float64), d['valid']
    fin = lambda a: np.isfinite(a).all(-1)
    nb = np.full_like(ann, np.nan); nz, ny, nx, _ = ann.shape
    for iz, iy, ix in np.argwhere(val):
        vs = [cell[iz + a, iy + b, ix + c] for a, b, c in OFF
              if 0 <= iz + a < nz and 0 <= iy + b < ny and 0 <= ix + c < nx and val[iz + a, iy + b, ix + c] and fin(cell[iz + a, iy + b, ix + c])]
        if vs: nb[iz, iy, ix] = np.mean(vs, 0)
    miss = ~fin(nb); nb[miss] = ann[miss]                      # 没有隔格邻居时退回只用环带
    X = np.concatenate([ann, nb], -1)
    ok = val & fin(X) & fin(d['c_blk'])
    return X, ok, miss


F = {g: features(d) for g, d in data.items()}
r2 = lambda p, t: [float(1 - ((p[:, k] - t[:, k]) ** 2).sum() / max(((t[:, k] - t[:, k].mean()) ** 2).sum(), 1e-12)) for k in range(t.shape[1])]
out = {g: {} for g in data}; rep = {}
for fold in FOLDS:
    meta = fold_meta(fold); tr, te = meta['train'], meta['test']
    Xtr = np.concatenate([F[g][0][F[g][1]] for g in tr if g in F]); Ytr = np.concatenate([data[g]['c_blk'][F[g][1]].astype(np.float64) for g in tr if g in F])
    mx, sx, my = Xtr.mean(0), Xtr.std(0) + 1e-9, Ytr.mean(0)
    Z = (Xtr - mx) / sx; lam = 1.0
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Ytr - my))
    pred = lambda X: ((X - mx) / sx) @ W + my
    for g in data:
        X, ok, _ = F[g]; P = np.full(data[g]['c_blk'].shape, np.nan, np.float32)
        fx = np.isfinite(X).all(-1); P[fx] = pred(X[fx]).astype(np.float32)
        P[~fx] = my.astype(np.float32)                        # 连环带都没有的格：用训练折均值（不回退到目标块实测）
        out[g]['c_nbr_' + fold] = P
    Xte = np.concatenate([F[g][0][F[g][1]] for g in te]); Yte = np.concatenate([data[g]['c_blk'][F[g][1]].astype(np.float64) for g in te])
    A = np.concatenate([data[g]['c_ann'][F[g][1]].astype(np.float64) for g in te])
    rep[fold] = dict(回归_测试=r2(pred(Xte), Yte), 回归_训练=r2(pred(Xtr), Ytr), 仅环带原值_测试=r2(A, Yte), n_test=int(len(Yte)))
    print('%-4s 测试岩性 R²（IGV, f_pore, mu_inter, f_dense）：周围回归 %s | 直接用环带原值 %s | 训练内 %s  (n=%d)' % (
        fold, np.round(rep[fold]['回归_测试'], 3).tolist(), np.round(rep[fold]['仅环带原值_测试'], 3).tolist(), np.round(rep[fold]['回归_训练'], 3).tolist(), len(Yte)), flush=True)
for g, d in data.items():
    np.savez_compressed(D / (g + '.npz'), **d, **out[g])
json.dump(rep, open(str(OUT_ROOT) + '/eval55/nbr_r2.json', 'w'), ensure_ascii=False, indent=1)
print('6 折均值 测试岩性 R²：周围回归 %s | 直接用环带原值 %s' % (np.round(np.mean([rep[f]['回归_测试'] for f in FOLDS], 0), 3).tolist(),
      np.round(np.mean([rep[f]['仅环带原值_测试'] for f in FOLDS], 0), 3).tolist()))
print('无隔格邻居的有效格占比：%.1f%%' % (100 * np.mean(np.concatenate([F[g][2][data[g]['valid']] for g in data]))))
print('DONE')
