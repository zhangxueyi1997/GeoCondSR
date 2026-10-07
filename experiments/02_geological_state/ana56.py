# -*- coding: utf-8 -*-
"""第 56 步多角度分析（只读、CPU）：「粗扫的周围 → 目标区域地质参数」该怎么做。
角度：① 看多大范围的粗扫（块本身 16³ / 格 36³ / 72³≈1 mm / 144³≈2 mm / 整个细扫区 / 细扫区以外的粗扫）；
     ② 预测多大尺度的地质参数（块 224 μm / 3³ 邻域 ≈0.75 mm / 样品整体）；③ 两尺度组合（局部 + 整体）；
     ④ 样品级结论的稳健性（嵌套选特征 + 置换检验）；⑤ 统一阈值的细扫纹理描述量，哪些是「岩石状态」；⑥ 测试岩石是否在训练范围内。
全部留一岩性（6 折，SX 只作训练），粗扫特征用简单统计量 + 岭回归（少参数，防止 13 个样品被背下来）。"""
import json, numpy as np, multiprocessing as mp
from pathlib import Path
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, OUT_ROOT  # noqa: E402
R = Path(str(DATA_ROOT)); OUT = Path(str(OUT_ROOT) + '/eval56')
CODE = {'G01': 'CQ', 'G19': 'CQ', 'G02': 'GZ', 'G03': 'YN', 'G16': 'YN', 'G17': 'YN', 'G05': 'SC', 'G06': 'SC', 'G07': 'SC',
        'G09': 'SC', 'G11': 'SC', 'G12': 'SD', 'G13': 'SHX', 'G15': 'SX'}
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; GS = list(CODE)
FN = ['p2', 'p10', 'p50', 'p90', 'p98', 'mean', 'std', 'dark', 'bright']; K3 = [0, 1, 6]      # k3 = 分位2、分位10、标准差
CV = ['f_pore', 'mu_inter', 'f_dense']; CI = [1, 2, 3]


def cfeat(v):
    v = v[np.isfinite(v)]
    if v.size < 500: return np.full(9, np.nan)
    return np.r_[np.percentile(v, [2, 10, 50, 90, 98]), v.mean(), v.std(), (v < 0.5).mean(), (v > 1.5).mean()]


def chord(m):
    tot, n = 0, 0
    for ax in range(3):
        a = np.moveaxis(m, ax, -1).reshape(-1, m.shape[ax]).astype(np.int8); d = np.diff(np.pad(a, ((0, 0), (1, 1))), axis=1)
        L = np.argwhere(d == -1)[:, 1] - np.argwhere(d == 1)[:, 1]; tot += int(L.sum()); n += len(L)
    return tot / max(n, 1)


def ssa(m):
    return sum(int(np.count_nonzero(np.moveaxis(m, ax, 0)[1:] != np.moveaxis(m, ax, 0)[:-1])) for ax in range(3)) / m.size


def per_sample(g):
    d = np.load(R / 'cfield' / (g + '.npz')); d2 = np.load(R / 'cfield2' / (g + '.npz'))
    V = np.load(R / 'cache' / str(d['lr_cache'])).astype(np.float32); Vn = (V - d['norm'][0]) / d['norm'][1]; del V
    z0 = int(d['lr_z0']); col = np.median(Vn[::4], axis=0) > 0.4; Vm = np.where(col[None], Vn, np.nan); del Vn
    val = d2['valid'] & np.isfinite(d2['c_blk']).all(-1); nz, ny, nx = val.shape
    fz0, fz1 = z0, z0 + (nz - 1) * 18 + 36
    Ffield = cfeat(Vm[fz0:fz1, :(ny - 1) * 18 + 36, :(nx - 1) * 18 + 36][::2, ::2, ::2].ravel())
    Fout = cfeat(np.concatenate([Vm[:fz0][::2, ::2, ::2].ravel(), Vm[fz1:][::2, ::2, ::2].ravel()]))
    rows = []
    for iz, iy, ix in np.argwhere(val):
        cz, cy, cx = z0 + iz * 18 + 18, iy * 18 + 18, ix * 18 + 18
        w = lambda h, s: cfeat(Vm[max(cz - h, 0):cz + h:s, max(cy - h, 0):cy + h:s, max(cx - h, 0):cx + h:s].ravel())
        sl = (slice(max(iz - 1, 0), iz + 2), slice(max(iy - 1, 0), iy + 2), slice(max(ix - 1, 0), ix + 2))
        c3 = d2['c_blk'][sl][val[sl]].mean(0)
        rows.append(np.r_[iz, iy, ix, w(8, 1), w(18, 1), w(36, 2), w(72, 3), d2['c_blk'][iz, iy, ix], c3])
    # 细扫块（配对单元，200 个）：统一阈值的纹理描述量
    H = []
    for f in sorted((R / 'pairs' / g).glob('*.npz')):
        z = np.load(f); h = z['hr'].astype(np.float32); m = h < 0.5
        H.append([*z['cell'][:3], m.mean(), (h > 1.5).mean(), h[m].mean() if m.sum() > 50 else np.nan,
                  chord(m) if m.any() else 0.0, ssa(m), chord(~m), *cfeat(z['lr'][10:26, 10:26, 10:26].ravel())])
    return g, np.array(rows), Ffield, Fout, np.array(H)


def ridge(Xtr, Ytr, Xte, lam=1.0):
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-9; my = Ytr.mean(0); Z = (Xtr - mx) / sx
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Ytr - my)); return ((Xte - mx) / sx) @ W + my


def lolo(X, Y, code, lam=1.0):
    P = np.full(Y.shape, np.nan)
    for lit in FOLDS:
        te = code == lit; tr = ~te & np.isfinite(X).all(1) & np.isfinite(Y).all(1); P[te] = ridge(X[tr], Y[tr], X[te], lam)
    return P


def score(P, Y, code, gid):
    t = np.isin(code, FOLDS) & np.isfinite(P).all(1) & np.isfinite(Y).all(1); P, Y, gid = P[t], Y[t], gid[t]
    r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    dm = lambda A: np.concatenate([A[gid == g] - A[gid == g].mean(0) for g in np.unique(gid)])
    wc = np.array([np.corrcoef(dm(P)[:, k], dm(Y)[:, k])[0, 1] if dm(P)[:, k].std() > 1e-12 else 0.0 for k in range(Y.shape[1])])
    return r2, wc


def nested(Xs, Ys_, cs):
    """样品级（n=14）：每折只在训练样品内部用留一样品挑最好的单个特征，再预测测试岩性。"""
    P = np.full(Ys_.shape, np.nan)
    for lit in FOLDS:
        te = cs == lit; tr = ~te
        for k in range(Ys_.shape[1]):
            best, bj = -1e18, 0
            for j in range(Xs.shape[1]):
                pp = []
                for i in np.where(tr)[0]:
                    t2 = tr.copy(); t2[i] = False; pp.append(ridge(Xs[t2][:, [j]], Ys_[t2][:, [k]], Xs[[i]][:, [j]], 0.0)[0, 0])
                s_ = -np.mean((np.array(pp) - Ys_[tr, k]) ** 2)
                if s_ > best: best, bj = s_, j
            P[te, k] = ridge(Xs[tr][:, [bj]], Ys_[tr][:, [k]], Xs[te][:, [bj]], 0.0)[:, 0]
    t = np.isin(cs, FOLDS); return 1 - ((P[t] - Ys_[t]) ** 2).sum(0) / ((Ys_[t] - Ys_[t].mean(0)) ** 2).sum(0)


if __name__ == '__main__':
    with mp.Pool(14) as pool: RES = pool.map(per_sample, GS)
    S = {g: dict(rows=r, Ff=ff, Fo=fo, H=h) for g, r, ff, fo, h in RES}
    np.savez_compressed(OUT / 'ana56_data.npz', **{g + '_rows': S[g]['rows'] for g in GS}, **{g + '_H': S[g]['H'] for g in GS},
                        Ff=np.array([S[g]['Ff'] for g in GS]), Fo=np.array([S[g]['Fo'] for g in GS]))
    B = np.concatenate([S[g]['rows'] for g in GS]); gid = np.concatenate([[g] * len(S[g]['rows']) for g in GS]); code = np.array([CODE[g] for g in gid])
    o = 3; X16, X36, X72, X144 = [B[:, o + 9 * i:o + 9 * (i + 1)] for i in range(4)]; o += 36
    Cb = B[:, o:o + 4][:, CI]; C3 = B[:, o + 4:o + 8][:, CI]
    smean = {g: S[g]['rows'][:, 39:43][:, CI].mean(0) for g in GS}; Cs = np.array([smean[g] for g in gid])
    XF = np.array([S[g]['Ff'] for g in gid]); XO = np.array([S[g]['Fo'] for g in gid])
    print('块数 %d；细扫区以外的粗扫 有效样品 %d/14' % (len(B), sum(np.isfinite(S[g]['Fo']).all() for g in GS)), flush=True)
    INP = [('块本身 16³(224μm)', X16[:, K3]), ('格 36³(0.5mm)', X36[:, K3]), ('72³(1mm)', X72[:, K3]), ('144³(2mm)', X144[:, K3]),
           ('整个细扫区', XF[:, K3]), ('细扫区以外', XO[:, K3]),
           ('块本身+整个细扫区', np.c_[X16[:, K3], XF[:, K3]]), ('块本身+细扫区以外', np.c_[X16[:, K3], XO[:, K3]]),
           ('块本身+2mm', np.c_[X16[:, K3], X144[:, K3]])]
    out = {}
    for tn, Y in (('块224μm', Cb), ('邻域0.75mm', C3), ('样品整体', Cs)):
        print('\n== 目标：%s 的细扫参数 %s   【留一岩性，合并 R² ｜ 样品内相关】' % (tn, CV))
        for nm, X in INP:
            r2, wc = score(lolo(X, Y, code), Y, code, gid); out['%s|%s' % (tn, nm)] = dict(r2=r2.tolist(), wc=wc.tolist())
            print('   %-18s R² %s ｜ 样品内相关 %s' % (nm, np.round(r2, 2), np.round(wc, 2) if tn != '样品整体' else '—'), flush=True)
    # ④ 样品级稳健性（n=14）
    Ys = np.array([smean[g] for g in GS]); cs = np.array([CODE[g] for g in GS])
    print('\n== 样品级（14 个样品）稳健性：整个细扫区的粗扫统计 → 样品细扫参数 %s' % CV)
    Xs = np.array([S[g]['Ff'] for g in GS]); r_n = nested(Xs, Ys, cs)
    print('   嵌套选特征（特征只在训练样品内部挑，不看测试）留一岩性 R²：%s' % np.round(r_n, 2), flush=True)
    rng = np.random.default_rng(0); NP = 300; cnt = np.zeros(3)
    for _ in range(NP): cnt += nested(Xs, Ys[rng.permutation(14)], cs) >= r_n
    print('   置换检验 p 值（%d 次打乱样品标签）：%s' % (NP, np.round((cnt + 1) / (NP + 1), 3)), flush=True)
    Xo = np.array([S[g]['Fo'] for g in GS])
    if np.isfinite(Xo).all(): print('   改用细扫区以外的粗扫，嵌套选特征 R²：%s' % np.round(nested(Xo, Ys, cs), 2))
    # ⑤ 统一阈值的细扫纹理描述量
    HN = ['孔隙率(<0.5)', '高密相(>1.5统一)', '孔隙灰度', '孔隙弦长(孔径)', '比表面', '骨架弦长(粒径)']
    Hh = np.concatenate([S[g]['H'] for g in GS]); hg = np.concatenate([[g] * len(S[g]['H']) for g in GS]); hc = np.array([CODE[g] for g in hg])
    Hd, Hx = Hh[:, 3:9], Hh[:, 9:18]
    print('\n== 统一阈值的细扫纹理描述量（块 224 μm，每样品 200 块）')
    Hs = np.array([np.nanmean(Hd[hg == g], 0) for g in GS])
    for k, nm in enumerate(HN):
        y = Hd[:, k]; tot = np.nanvar(y); wi = np.nanmean([np.nanvar(y[hg == g]) for g in GS])
        r_s = nested(Xs, Hs[:, [k]], cs)[0]
        _, wc = score(lolo(Hx[:, K3], Hd[:, [k]], hc), Hd[:, [k]], hc, hg)
        print('   %-16s 样品间占方差 %3.0f%% ｜ 整体粗扫→样品水平 留一岩性 R² %5.2f ｜ 块本身粗扫→块值 样品内相关 %.2f' % (nm, 100 * (1 - wi / tot), r_s, wc[0]), flush=True)
    print('   各样品水平：')
    for i, g in enumerate(GS): print('     %s %-4s %s' % (g, CODE[g], np.round(Hs[i], 4)))
    # ⑥ 范围检查
    print('\n== 测试岩石的样品水平是否落在训练样品范围内（f_pore, mu_inter, 统一阈值高密相, 孔径）')
    sv = np.c_[Ys[:, :2], Hs[:, [1, 3]]]
    for lit in FOLDS:
        te = cs == lit; tr = ~te
        print('   %-4s %s' % (lit, ' | '.join(' '.join('内' if sv[tr, k].min() <= v <= sv[tr, k].max() else '外' for k, v in enumerate(v_)) for v_ in sv[te])))
    json.dump(out, open(OUT / 'ana56.json', 'w'), ensure_ascii=False, indent=1)
    print('DONE')
