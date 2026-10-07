"""重测成因场 c —— 修正采样噪声与支撑失配（第 30 步）。

为什么要重测
    `cfield.py` 每格只取 **3 张切片**（深度 20/50/80%），每张覆盖整个 0.5×0.5 mm；
    而生成目标是格正中央的 **224 μm** 块。实测几何：
      · 深度：3 张里只有中间那张落在块内；
      · 平面：块仅占切片面积的 (224/504)² = 19.8%。
      ⇒ **采样体素里落在块内的约 1/15 —— 测的基本是邻域，且样本极稀。**

    实测后果（七折 × 200 块）：现有 c 与「同法但在块支撑上重算」的相关仅 **0.70**
    （共享方差 49%）；扣掉 LR 后对终点的偏 R²：现有 c **+0.052 / +0.082**，
    块支撑版 **+0.338 / +0.210**（6.5× / 2.6×）。
    **信息在物理里存在，是测量把它丢了。**

    而第 29 步的公平检验（cond = [ĉ, 实测c]，网络保留全部自由度、白送实测 c）
    显示实测 c **没有被用上** —— 但那是用**这个测不准的 c** 做的。
    ⇒ 「实测无增量」与「我们没测准」尚未分开，本脚本就是为了分开它们。

本脚本的三个改动
    1. **循环倒过来**：`cfield.py` 逐格读切片，同一张切片被反复读取解码
       （噪声审计实测 38 秒/格）。这里按**切片**读一次，一次喂给所有需要它的格子。
    2. **采样加密**：每格沿深度每 2 个粗体素取一张 ⇒ 约 18 张/格（原来 3 张）。
    3. **一次算出三种支撑**，分别回答不同问题：

       | 支撑 | 范围 | 回答 |
       |---|---|---|
       | `cell`  | 整格 504 μm | 去掉采样噪声后，邻域测量够不够用 |
       | `blk`   | 中心 224 μm，与生成目标**同支撑** | 信息上限（含部分循环，仅作诊断） |
       | `ann`   | 格减块 = 块**周围**，不含块自身 | **最干净的地质命题**：邻域成岩状态能否约束这一块 |

       `ann` 是关键：既不循环，又正是「沉积-成岩状态控制微结构」该有的问法。

    统计量用增量计数累积（不存体素）：总数、孔隙数、骨架数、高密数、粒间和与计数。

输出 results/cfield2/{gid}.npz
    c_cell / c_blk / c_ann   (nz, ny, nx, 4)
    n_cell / n_blk / n_ann   每格实际累积的体素数（可查采样是否充分）
    valid, grid, norm

用法
    python code/cfield2.py G01          # 单个样品
    python code/cfield2.py              # 全部
"""
from __future__ import annotations
import os
import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = Path(os.environ.get('GEOCOND_WORK', str(HERE.parent / 'work')))   # intermediate preprocessing products
RES = PROJ / "results"
OUT = RES / "cfield2"
sys.path.insert(0, str(HERE))
import spectrum, tir, fastio, centre, xmap

V_COARSE = 14.0
CELL = 36                    # 0.5 mm / 14 μm
STRIDE = CELL // 2
ZSTEP = 2                    # 沿深度每 2 个粗体素取一张 ⇒ 约 18 张/格（原 3 张）
ZQ = 8                       # 细扫 z 量化步长：让不同格子**共用切片**。
                             # 倾斜校正使同一粗层在不同 (y,x) 映到不同细扫切片，
                             # 不量化时要读 1462 张（几乎整个栈，12.8 GB 走 USB 机械盘，
                             # 实测 1 小时/样品）。量化到 8 像素后约 195 张，降 7.5 倍，
                             # 每格仍能取到约 15 张不同切片 —— 远好于原来的 3 张。
BLK_UM = 224.0               # 生成目标块的边长
PORE_CUT, K_W = 0.50, 2.2
NAMES = ["IGV", "f_pore", "mu_inter", "f_dense"]
SUPPORTS = ("cell", "blk", "ann")


class Acc:
    """一个格、一种支撑的增量累积器。cvals 只需要这几个计数。"""

    __slots__ = ("n", "n_pore", "n_frame", "n_dense", "s_inter", "n_inter")

    def __init__(self):
        self.n = self.n_pore = self.n_frame = self.n_dense = self.n_inter = 0
        self.s_inter = 0.0

    def add(self, u, lo, hi):
        if u.size == 0:
            return
        self.n += u.size
        self.n_pore += int((u < PORE_CUT).sum())
        fr = (u >= lo) & (u < hi)
        self.n_frame += int(fr.sum())
        self.n_dense += int((u >= hi).sum())
        it = u[u < lo]
        self.n_inter += it.size
        self.s_inter += float(it.sum())

    def vals(self):
        if self.n == 0:
            return [np.nan] * 4
        return [1.0 - self.n_frame / self.n,
                self.n_pore / self.n,
                self.s_inter / self.n_inter if self.n_inter > 50 else np.nan,
                self.n_dense / self.n]


def one(gid, k, ntot):
    tag = f"[{k:2d}/{ntot}] {gid}"
    t0 = time.time()
    mp = xmap.Mapper(gid)
    if not mp.ok:
        print(f"{tag} 形状自洽检查未通过，跳过", flush=True)
        return None
    R, I, msk, ref = spectrum.placed(gid, order=3)
    meta = mp.meta
    vf = float(ref["v_true"])
    Pf = int(round(CELL * mp.scale()))          # 格在细扫上的边长（像素）
    Pb = int(round(BLK_UM / vf))                # 块在细扫上的边长（像素）
    nz, ny, nx = R.shape
    gz = max(1, (nz - CELL) // STRIDE + 1)
    gy = max(1, (ny - CELL) // STRIDE + 1)
    gx = max(1, (nx - CELL) // STRIDE + 1)
    zlo_c = (CELL - BLK_UM / V_COARSE) / 2.0    # 块在格内的粗体素起止（深度）
    zhi_c = CELL - zlo_c
    print(f"{tag} {ref['sample']:<9} 网格 {gz}×{gy}×{gx}  格 {Pf}px  块 {Pb}px  "
          f"体素 {vf:.2f}μm  每格约 {CELL//ZSTEP} 张切片", flush=True)

    # 端元与骨架窗：与 cfield.py 完全同一套，保证两份场逐格可比
    radF = meta["D_mm"] * 1000 / vf / 2
    zsF = np.linspace(len(mp.files) * .3, len(mp.files) * .7, 3).astype(int)
    airF = tir.air_level(mp.files, radF, zsF)
    vals = []
    for z in zsF:
        a = fastio.read(mp.files[z]).astype(np.float32)
        cy, cx = centre.disc_centre(a, radF)
        yy, xx = np.indices(a.shape)
        vals.append(a[np.hypot(yy - cy, xx - cx) <= radF * 0.88])
    vv = np.concatenate(vals)
    DF = tir.mode_of(vv) - airF
    if DF <= 0:
        print(f"{tag} 端元异常，跳过", flush=True)
        return None
    wF = tir.hwhm_of(vv) / DF
    lo, hi = max(1.0 - K_W * wF, PORE_CUT + 0.02), 1.0 + K_W * wF
    print(f"{tag}   细扫 Δ {DF:8.1f}  骨架窗 [{lo:.2f}, {hi:.2f}]", flush=True)

    cells = [(iz, iy, ix)
             for iz in range(gz) for iy in range(gy) for ix in range(gx)
             if msk[iy * STRIDE:iy * STRIDE + CELL,
                    ix * STRIDE:ix * STRIDE + CELL].all()]
    acc = {s: [Acc() for _ in cells] for s in SUPPORTS}

    # 按细扫切片分桶：每张切片只读一次，喂给所有需要它的格子
    bucket = defaultdict(list)          # zf -> [(cell_idx, r, c, in_blk_z)]
    for ci, (iz, iy, ix) in enumerate(cells):
        z0, y0, x0 = iz * STRIDE, iy * STRIDE, ix * STRIDE
        seen = set()                    # 量化后同一格可能撞到同一张，去重免得重复计数
        for zc in range(0, CELL, ZSTEP):
            p = mp(np.array([z0 + zc + 0.5, y0 + CELL / 2.0, x0 + CELL / 2.0]))[0]
            zf = int(round(p[0] / ZQ)) * ZQ
            if not (0 <= zf < len(mp.files)) or zf in seen:
                continue
            seen.add(zf)
            bucket[zf].append((ci, int(round(p[1])), int(round(p[2])),
                               zlo_c <= zc < zhi_c))
    nrd = 0
    for zf in sorted(bucket):
        try:
            a = fastio.read(mp.files[zf])
        except Exception:
            continue
        a = (a.astype(np.float32) - airF) / DF
        H, W = a.shape
        nrd += 1
        for ci, rc, cc, in_blk in bucket[zf]:
            r0, r1 = rc - Pf // 2, rc - Pf // 2 + Pf
            c0, c1 = cc - Pf // 2, cc - Pf // 2 + Pf
            if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
                continue
            tile = a[r0:r1, c0:c1]
            acc["cell"][ci].add(tile.ravel(), lo, hi)
            o = (Pf - Pb) // 2
            core = tile[o:o + Pb, o:o + Pb]
            if in_blk:
                # 块 = 中心 Pb×Pb 且深度落在块内；环带 = 该切片扣掉这块中心
                acc["blk"][ci].add(core.ravel(), lo, hi)
                m = np.ones(tile.shape, bool)
                m[o:o + Pb, o:o + Pb] = False
                acc["ann"][ci].add(tile[m], lo, hi)
            else:
                # 深度在块外 ⇒ 整张都属于「块周围」
                acc["ann"][ci].add(tile.ravel(), lo, hi)
        if nrd % 40 == 0:
            el = time.time() - t0
            print(f"{tag}   已读 {nrd}/{len(bucket)} 张切片  {el:5.1f}s  "
                  f"预计还需 {el / nrd * (len(bucket) - nrd):5.0f}s", flush=True)

    C = {s: np.full((gz, gy, gx, len(NAMES)), np.nan, np.float32) for s in SUPPORTS}
    N = {s: np.zeros((gz, gy, gx), np.int64) for s in SUPPORTS}
    valid = np.zeros((gz, gy, gx), bool)
    for ci, (iz, iy, ix) in enumerate(cells):
        if acc["cell"][ci].n == 0:
            continue
        for s in SUPPORTS:
            C[s][iz, iy, ix] = acc[s][ci].vals()
            N[s][iz, iy, ix] = acc[s][ci].n
        valid[iz, iy, ix] = True

    nv = int(valid.sum())
    if nv < 8:
        print(f"{tag}   有效格只有 {nv} 个，跳过", flush=True)
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{gid}.npz", valid=valid, names=np.array(NAMES),
             grid=np.array([CELL, STRIDE, gz, gy, gx]),
             norm=np.array([airF, DF, lo, hi]),
             **{f"c_{s}": C[s] for s in SUPPORTS},
             **{f"n_{s}": N[s] for s in SUPPORTS})
    print(f"{tag} 完成 {nv} 格，读切片 {nrd} 张，用时 {time.time()-t0:.0f}s")
    for s in SUPPORTS:
        mu = np.nanmean(C[s][valid], 0)
        print(f"{tag}   {s:<5} 每格体素 {int(np.median(N[s][valid])):>8d}  "
              + "  ".join(f"{n} {m:.3f}" for n, m in zip(NAMES, mu)), flush=True)
    print("", flush=True)
    return dict(gid=gid, ncell=nv)


def main():
    import json
    sp = json.loads((RES / "splits.json").read_text(encoding="utf-8"))
    gids = [r["gid"] for r in sp["samples"] if r["usable"]]
    if len(sys.argv) > 1:
        gids = [g for g in sys.argv[1:] if g in gids] or sys.argv[1:]
    print(f"重测成因场：{len(gids)} 个样品，每格约 {CELL//ZSTEP} 张切片，"
          f"三种支撑（整格 / 中心块 / 环带）\n", flush=True)
    for k, g in enumerate(gids, 1):
        try:
            one(g, k, len(gids))
        except Exception as e:
            print(f"[{k}] {g} 失败：{type(e).__name__} {e}\n", flush=True)


if __name__ == "__main__":
    main()
