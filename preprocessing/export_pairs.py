"""导出全分辨率配对训练数据 —— README 待办第 5 步。

只给生成器 G 导
  编码器 E 的监督只需要「LR + ĉ」，而 LR 就在 cache_C 里、ĉ 在 results/cfield 里，
  **一张细扫切片都不用读**。所以本脚本只导 G 需要的高分辨目标，
  数量可以远少于 ĉ 格数（7139），大幅省读盘。

每个训练单元
  lr   36³ 粗体素（0.5 mm）—— 编码器看到的上下文，正好一个 ĉ 格
  hr   HR_LR³ × 7 个体素 —— 以 lr 中心为心的小块真值，重采到 **LR 网格的 7 细分**
       （14/7 = 2.0 μm）。README 待办第 5 步指定的做法：
       **保 LR 原样、把 HR 重采到整数倍网格**，这样真实退化不被插值污染。
  c    该格的 4 个成因变量（来自 results/cfield）

为什么 HR 只取中心一小块
  ×7 的三维生成，输出体是输入体的 343 倍。整块 36³ 对应 252³ = 1600 万体素，
  普通显卡放不下。**E 看全上下文、G 只生成中心一小块**，
  既保住 0.5 mm 的成因条件，又让生成量可控。推理时按格平铺即可。

坐标与重采样
  xmap.Mapper 是仿射的，用有限差分取出它的线性部分 A 与平移 b，
  于是 HR 输出网格 → 细扫坐标是一次仿射，用 order=3 一次插值完成
  （与配准流水线同一个理由：多次线性插值本身就是低通，会把要量的高频抹掉）。

输出 <OUTDIR>/{gid}/{idx:05d}.npz  —— 默认写 D: 盘（E: 是 USB 机械盘，训练读会卡）

用法
    python code/export_pairs.py --pilot G01        # 试点：只做一个样品的少量单元
    python code/export_pairs.py                    # 全量
    python code/export_pairs.py --per 300 --hrlr 16
"""
from __future__ import annotations
import os
import sys, json, time, argparse
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PROJ = Path(os.environ.get('GEOCOND_WORK', str(HERE.parent / 'work')))   # intermediate preprocessing products
RES = PROJ / "results"
REG = RES / "registration"
sys.path.insert(0, str(HERE))
import spectrum, xmap, fastio

V_COARSE = 14.0
UP = 7                      # HR 网格 = LR 网格的 7 细分 → 2.0 μm
CELL = 36                   # 与 cfield 一致


def affine_of(mp, p0):
    """xmap 在 p0 附近的线性部分 A 与平移：fine = A @ placed + b。"""
    a0 = mp(p0)[0]
    A = np.empty((3, 3))
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        A[:, i] = mp(p0 + e)[0] - a0
    return A, a0 - A @ p0


def one_unit(mp, A, b, ctr, hr_n, margin=6):
    """取以 placed 坐标 ctr 为心、hr_n³ 的 HR 体（2.0 μm 网格）。"""
    # HR 输出体素 o（0..hr_n-1）对应的 placed 坐标
    p_of_o = lambda o: ctr + (o - (hr_n - 1) / 2.0) / UP
    corners = np.array([[p_of_o(np.array([i, j, k], float))
                         for i in (0, hr_n - 1)] for j in (0, hr_n - 1)
                        for k in (0, hr_n - 1)]).reshape(-1, 3)
    fc = corners @ A.T + b
    lo = np.floor(fc.min(0)).astype(int) - margin
    hi = np.ceil(fc.max(0)).astype(int) + margin
    if lo[0] < 0 or hi[0] >= len(mp.files):
        return None
    # 读原生细扫小体
    V = np.empty((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), np.float32)
    for i in range(V.shape[0]):
        a = fastio.read(mp.files[lo[0] + i], lo[1], hi[1], lo[2], hi[2])
        if a.shape != V.shape[1:]:
            return None
        V[i] = a
    # 一次仿射把它重采到 HR 网格：out[o] = V[M o + off]
    M = A / UP
    off = A @ (ctr - (hr_n - 1) / 2.0 / UP) + b - lo
    return ndimage.affine_transform(V, M, offset=off,
                                    output_shape=(hr_n,) * 3, order=3,
                                    mode="nearest").astype(np.float32)


def run(gid, outdir, per, hr_lr, k, ntot):
    tag = f"[{k:2d}/{ntot}] {gid}"
    t0 = time.time()
    cf = RES / "cfield" / f"{gid}.npz"
    if not cf.exists():
        print(f"{tag} 没有 ĉ 场，跳过", flush=True); return None
    d = np.load(cf)
    C, valid = d["c"], d["valid"]
    cellsz, stride = int(d["grid"][0]), int(d["grid"][1])
    mp = xmap.Mapper(gid)
    R, I, msk, ref = spectrum.placed(gid, order=3)
    hr_n = hr_lr * UP
    airR, DR, airF, DF = d["norm"][:4]

    idx = np.argwhere(valid)
    rs = np.random.default_rng(0)
    if per and len(idx) > per:
        idx = idx[rs.choice(len(idx), per, replace=False)]
    print(f"{tag} {ref['sample']:<9} 有效格 {int(valid.sum())} → 导出 {len(idx)} 个  "
          f"LR {cellsz}³  HR {hr_n}³ @ {V_COARSE/UP:.2f} μm  ({time.time()-t0:.1f}s)",
          flush=True)

    od = outdir / gid
    od.mkdir(parents=True, exist_ok=True)
    A, b = affine_of(mp, np.array([R.shape[0] / 2, R.shape[1] / 2,
                                   R.shape[2] / 2], float))
    n_ok, nbytes = 0, 0
    for j, (iz, iy, ix) in enumerate(idx):
        z0, y0, x0 = iz * stride, iy * stride, ix * stride
        if z0 + cellsz > R.shape[0]:
            continue
        lr = np.asarray(R[z0:z0 + cellsz, y0:y0 + cellsz, x0:x0 + cellsz],
                        np.float32)
        ctr = np.array([z0 + cellsz / 2.0, y0 + cellsz / 2.0, x0 + cellsz / 2.0])
        try:
            hr = one_unit(mp, A, b, ctr, hr_n)
        except Exception:
            hr = None
        if hr is None:
            continue
        f = od / f"{n_ok:05d}.npz"
        np.savez_compressed(
            f, lr=((lr - airR) / DR).astype(np.float32),
            hr=((hr - airF) / DF).astype(np.float32),
            c=C[iz, iy, ix].astype(np.float32),
            cell=np.array([iz, iy, ix]), gid=gid, up=UP)
        nbytes += f.stat().st_size
        n_ok += 1
        if n_ok % 25 == 0 or j == len(idx) - 1:
            el = time.time() - t0
            print(f"{tag}   {n_ok:4d}/{len(idx)}  {nbytes/1e6:7.1f} MB  "
                  f"已用 {el:6.1f}s  预计本样 {el/max(n_ok,1)*len(idx):6.0f}s",
                  flush=True)
    print(f"{tag} 完成 {n_ok} 个单元，{nbytes/1e6:.0f} MB，"
          f"用时 {time.time()-t0:.0f}s\n", flush=True)
    return dict(gid=gid, sample=ref["sample"], n=n_ok, mb=round(nbytes / 1e6, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get('GEOCOND_WORK', 'work'), 'pairs_npz'))
    ap.add_argument("--per", type=int, default=200, help="每样导出多少单元")
    ap.add_argument("--hrlr", type=int, default=16,
                    help="HR 目标覆盖多少个 LR 体素（HR 边长 = 这个 x7）")
    ap.add_argument("--pilot", default="")
    a = ap.parse_args()
    outdir = Path(a.out)
    sp = json.loads((RES / "splits.json").read_text(encoding="utf-8"))
    gids = [a.pilot] if a.pilot else [r["gid"] for r in sp["samples"] if r["usable"]]
    per = 8 if a.pilot else a.per
    print("=" * 96)
    print(f"导出配对训练数据 → {outdir}")
    print(f"每样 {per} 个单元，LR {CELL}³ 粗体素，HR {a.hrlr*UP}³ @ "
          f"{V_COARSE/UP:.2f} μm（覆盖 {a.hrlr*V_COARSE:.0f} μm）")
    print("=" * 96, flush=True)
    rows, t0 = [], time.time()
    for k, g in enumerate(gids, 1):
        try:
            r = run(g, outdir, per, a.hrlr, k, len(gids))
            if r:
                rows.append(r)
        except KeyboardInterrupt:
            print("\n用户中断，已导出的保留"); sys.exit(1)
        except Exception as e:
            print(f"[{k}/{len(gids)}] {g} 失败：{type(e).__name__}: {e}\n", flush=True)
    if rows:
        n = sum(r["n"] for r in rows); mb = sum(r["mb"] for r in rows)
        print(f"共 {len(rows)} 个样品、{n} 个单元、{mb:.0f} MB，"
              f"用时 {(time.time()-t0)/60:.1f} min")
        print(f"折算全量（14 样 × {a.per} 单元）："
              f"{mb/max(n,1)*a.per*14/1000:.1f} GB，"
              f"{(time.time()-t0)/max(n,1)*a.per*14/60:.0f} min")
