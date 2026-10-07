# -*- coding: utf-8 -*-
"""第 57c 步数据提取（只读、CPU）：改进「粗扫 → 整体水平」所需的全部数据，存成一个 npz 供离线分析。
① 整根岩心粗扫（去空气层、平面腐蚀 10 体素）的丰富灰度特征；② 每个有效格 72³（≈1 mm）窗的同一组特征（行顺序与 ana56 一致）；
③ 每折测试样品全部配对单元的 ĉ（_cpred，CPU）、c_blk、格坐标——用来模拟「10 块细扫校准」的整体水平误差（精度基准）。
丰富特征 = 分位 [0.5,1,2,5,10,25,50,75,90,98] + 低于 [0.3,0.4,0.5,0.6,0.7,0.8] 的占比 + 均值 + 标准差（18 维）。"""
import sys, json, numpy as np, multiprocessing as mp, torch
from pathlib import Path
from scipy import ndimage as ndi
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
R = Path(str(DATA_ROOT)); RUNS = str(RUNS_ROOT)
CODE = {'G01': 'CQ', 'G19': 'CQ', 'G02': 'GZ', 'G03': 'YN', 'G16': 'YN', 'G17': 'YN', 'G05': 'SC', 'G06': 'SC', 'G07': 'SC',
        'G09': 'SC', 'G11': 'SC', 'G12': 'SD', 'G13': 'SHX', 'G15': 'SX'}
GS = list(CODE); FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
PQ = [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 98]; TH = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def rf(v):
    v = v[np.isfinite(v)]
    if v.size < 500: return np.full(18, np.nan)
    return np.r_[np.percentile(v, PQ), [(v < t).mean() for t in TH], v.mean(), v.std()]


def per_sample(g):
    d = np.load(R / 'cfield' / (g + '.npz')); d2 = np.load(R / 'cfield2' / (g + '.npz'))
    V = np.load(R / 'cache' / str(d['lr_cache'])).astype(np.float32); V = (V - d['norm'][0]) / d['norm'][1]
    ny, nx = V.shape[1:]
    cm = np.median(V[:, ny // 2 - 60:ny // 2 + 60, nx // 2 - 60:nx // 2 + 60].reshape(V.shape[0], -1), 1)
    rz = ndi.binary_erosion(cm > 0.6, iterations=5); col = ndi.binary_erosion(np.median(V[rz][::4], 0) > 0.7, iterations=10)
    Fcore = rf(V[rz][::2][:, col[:, :]][:, ::2].ravel()) if rz.any() else np.full(18, np.nan)
    Vm = np.where(col[None], V, np.nan); del V
    z0 = int(d['lr_z0']); val = d2['valid'] & np.isfinite(d2['c_blk']).all(-1)
    W = []
    for iz, iy, ix in np.argwhere(val):                      # 与 ana56 同一行顺序
        cz, cy, cx = z0 + iz * 18 + 18, iy * 18 + 18, ix * 18 + 18
        W.append(rf(Vm[max(cz - 36, 0):cz + 36:2, max(cy - 36, 0):cy + 36:2, max(cx - 36, 0):cx + 36:2].ravel()))
    return g, Fcore, np.array(W)


if __name__ == '__main__':
    with mp.Pool(14) as pool: RES = pool.map(per_sample, GS)
    out = {}
    for g, fc, w in RES: out['core_' + g] = fc; out['win_' + g] = w
    from models54 import load_cpred
    from dataset import PairDataset
    torch.set_num_threads(8)
    for F in FOLDS:
        meta = json.load(open('%s/%s_proj/meta.json' % (RUNS, F))); m = load_cpred('%s/%s_cpred/final.pt' % (RUNS, F), 'cpu')
        ds = PairDataset(str(R), meta['test'], c_src='blk')
        for g in meta['test']:
            js = [j for j, (gg, _) in enumerate(ds.items) if gg == g]
            cl = np.array([np.load(ds.items[j][1])['cell'][:3].astype(int) for j in js])
            cf = np.load(R / 'cfield2' / (g + '.npz')); cb = cf['c_blk'][tuple(cl.T)].astype(np.float64)
            ok = cf['valid'][tuple(cl.T)] & np.isfinite(cb).all(1)
            with torch.no_grad():
                ch = np.concatenate([m(torch.from_numpy(np.stack([ds[j]['lr'] for j in js[s:s + 50]]))).numpy() for s in range(0, len(js), 50)])
            out['unit_cell_' + g] = cl; out['unit_cb_' + g] = cb; out['unit_ch_' + g] = ch.astype(np.float64); out['unit_ok_' + g] = ok
            print(F, g, '配对单元', len(js), flush=True)
    np.savez_compressed(str(OUT_ROOT) + '/eval57b/lvl57_data.npz', **out)
    print('DONE')
