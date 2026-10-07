# -*- coding: utf-8 -*-
"""第 57c 步离线分析：改进「粗扫 → 整体水平」。
判据（事先定死）：f_pore 与 mu_inter 两项，测试样品上误差的中位数 ≤ 「10 块细扫校准」误差的中位数 ⇒ 达标，才去跑超分评测。
误差口径：f_pore / IGV / f_dense 用对数比 |ln(推/真)|（致密样品看相对误差）；mu_inter 是灰度，用绝对差。
候选（事先列好）：V0 现方法（1 mm 窗 3 统计 → 邻域 c，线性）；V1 = V0 + 对数目标；V2 = 1 mm 窗 18 维丰富特征 + 对数目标；
V3 = 整根岩心丰富特征，样品级单特征回归（内部留一挑特征）+ 对数目标；V4 = 1 mm 窗丰富特征直接回归「所在样品的整体水平」+ 对数目标。
最终选哪个：每个外层折只在训练岩性上做内层留一岩性，按内层误差挑（测试岩性不参与选择）。"""
import sys, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../preprocessing')); from paths import DATA_ROOT, OUT_ROOT  # noqa: E402
sys.stdout.reconfigure(encoding='utf-8')
D = str(OUT_ROOT) + '/'
A = np.load(D + 'eval56/ana56_data.npz'); L = np.load(D + 'eval57b/lvl57_data.npz')
CODE = {'G01': 'CQ', 'G19': 'CQ', 'G02': 'GZ', 'G03': 'YN', 'G16': 'YN', 'G17': 'YN', 'G05': 'SC', 'G06': 'SC', 'G07': 'SC',
        'G09': 'SC', 'G11': 'SC', 'G12': 'SD', 'G13': 'SHX', 'G15': 'SX'}
GS = list(CODE); FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; TEST = [g for g in GS if CODE[g] in FOLDS]
NM = ['IGV', 'f_pore', 'mu_inter', 'f_dense']; LOGD = [0, 1, 3]; EPS = 1e-3
TL = {g: np.nanmean(A[g + '_rows'][:, 39:43], 0) for g in GS}                      # 真实整体水平（与 eval57b 同口径）
C3 = {g: A[g + '_rows'][:, 43:47] for g in GS}
X9 = {g: A[g + '_rows'][:, 21:30] for g in GS}; X18 = {g: L['win_' + g] for g in GS}; XC = {g: L['core_' + g] for g in GS}


def err(est, g):
    t = TL[g]; e = np.abs(est - t)
    for k in LOGD: e[k] = abs(np.log(max(est[k], EPS) / max(t[k], EPS)))
    return e


# ---------- 精度基准：「10 块细扫校准」（与 eval57 同法：ĉ 全样品均值 + 远处 10 块 (c_blk − ĉ) 均值） ----------
rng = np.random.default_rng(0); BEN = {}
for g in TEST:
    cl, cb, ch, ok = L['unit_cell_' + g], L['unit_cb_' + g], L['unit_ch_' + g], L['unit_ok_' + g]
    chm = ch.mean(0); E = []
    for i in np.where(ok)[0]:
        far = np.where(ok & (np.abs(cl - cl[i]).max(1) >= 3))[0]; j = rng.choice(far, min(10, len(far)), replace=False)
        E.append(err(chm + (cb[j] - ch[j]).mean(0), g))
    BEN[g] = np.sqrt((np.array(E) ** 2).mean(0))                                  # 每样品 RMS 误差
bench = np.median([BEN[g] for g in TEST], 0)


def ridge(X, Y, lam):
    mx, sx = X.mean(0), X.std(0) + 1e-9; my = Y.mean(0); Z = (X - mx) / sx
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - my)); return lambda Xn: ((Xn - mx) / sx) @ W + my


def tolog(Y): Y = Y.copy(); Y[:, LOGD] = np.log(np.maximum(Y[:, LOGD], EPS)); return Y
def fromlog(P): P = P.copy(); P[:, LOGD] = np.exp(P[:, LOGD]); return P


def fit_predict(v, tr, te):
    """用训练样品 tr 拟合候选 v，返回 {测试样品: 整体水平估计}。"""
    if v in ('V0', 'V1', 'V2', 'V4'):
        Xs = X9 if v in ('V0', 'V1') else X18; cols = [0, 1, 6] if v in ('V0', 'V1') else slice(None)
        X = np.concatenate([Xs[g][:, cols] for g in tr])
        Y = np.concatenate([C3[g] if v != 'V4' else np.repeat(TL[g][None], len(C3[g]), 0) for g in tr])
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1); X, Y = X[ok], Y[ok]
        if v != 'V0': Y = tolog(Y)
        f = ridge(X, Y, 1.0 if v in ('V0', 'V1') else 10.0); out = {}
        for g in te:
            Xg = Xs[g][:, cols]; Xg = Xg[np.isfinite(Xg).all(1)]; P = f(Xg)
            out[g] = P.mean(0) if v == 'V0' else fromlog(P).mean(0)
        return out
    # V3：整根岩心 18 维特征，样品级；每个输出维度用训练样品内部留一挑 1 个特征
    Xtr = np.array([XC[g] for g in tr]); Ytr = tolog(np.array([TL[g] for g in tr])); out = {g: np.zeros(4) for g in te}
    for k in range(4):
        best, bj = 1e18, 0
        for j in range(18):
            if not np.isfinite(Xtr[:, j]).all() or Xtr[:, j].std() < 1e-12: continue
            pe = []
            for i in range(len(tr)):
                m = np.ones(len(tr), bool); m[i] = False; A_ = np.c_[Xtr[m, j], np.ones(m.sum())]
                w = np.linalg.lstsq(A_, Ytr[m, k], rcond=None)[0]; pe.append((w[0] * Xtr[i, j] + w[1] - Ytr[i, k]) ** 2)
            if np.mean(pe) < best: best, bj = np.mean(pe), j
        w = np.linalg.lstsq(np.c_[Xtr[:, bj], np.ones(len(tr))], Ytr[:, k], rcond=None)[0]
        for g in te: out[g][k] = w[0] * XC[g][bj] + w[1]
    return {g: fromlog(o[None])[0] for g, o in out.items()}


VS = ['V0', 'V1', 'V2', 'V3', 'V4']; PRED = {v: {} for v in VS + ['嵌套选择']}; CHOSEN = {}
for F in FOLDS:
    te = [g for g in GS if CODE[g] == F]; tr = [g for g in GS if CODE[g] != F]
    for v in VS: PRED[v].update(fit_predict(v, tr, te))
    # 内层：训练岩性里再留一岩性，按 (f_pore 对数误差/基准 + mu_inter 误差/基准) 的中位数挑候选
    inner_l = sorted({CODE[g] for g in tr if CODE[g] in FOLDS}); sc = {}
    for v in VS:
        e = []
        for Fi in inner_l:
            te2 = [g for g in tr if CODE[g] == Fi]; tr2 = [g for g in tr if CODE[g] != Fi]
            for g, est in fit_predict(v, tr2, te2).items(): x = err(est, g); e.append(x[1] / bench[1] + x[2] / bench[2])
        sc[v] = np.median(e)
    CHOSEN[F] = min(sc, key=sc.get); PRED['嵌套选择'].update({g: PRED[CHOSEN[F]][g] for g in te})
    print('折 %-4s 内层挑中 %s   内层得分 %s' % (F, CHOSEN[F], {v: round(s, 2) for v, s in sc.items()}))
print('\n误差口径：IGV/f_pore/f_dense = |ln(推/真)|，mu_inter = |推−真|；数值为 13 个测试样品的中位数')
print('%-22s %s' % ('10块细扫校准（基准）', np.round(bench, 3)))
for v in VS + ['嵌套选择']:
    E = np.array([err(PRED[v][g], g) for g in TEST]); md = np.median(E, 0)
    ok = md[1] <= bench[1] and md[2] <= bench[2]
    print('%-22s %s   最差 f_pore %.2f  mu_inter %.3f   %s' % (v, np.round(md, 3), E[:, 1].max(), E[:, 2].max(), '★达标' if ok else '未达标'))
print('\n逐样品（f_pore 推/真，mu_inter 推/真）')
for g in TEST:
    print('  %s %-4s 真 f_pore %.3f mu %.3f | ' % (g, CODE[g], TL[g][1], TL[g][2]) + ' '.join('%s %.3f/%.3f' % (v, PRED[v][g][1], PRED[v][g][2]) for v in VS + ['嵌套选择'])
          + ' | 基准误差 %.2f/%.3f' % (BEN[g][1], BEN[g][2]))
np.save(D + 'lvl57_pred.npy', {v: {g: PRED[v][g].tolist() for g in TEST} for v in PRED}, allow_pickle=True)
