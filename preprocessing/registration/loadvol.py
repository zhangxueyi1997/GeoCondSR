"""流式读取并降采样 CT 体数据。

两个体都太大，不能整体载入（细扫 1601x2006x2006 uint16 = 12.9 GB）。
做法：逐切片读入 -> 裁到试样包围盒 -> 面内块平均 -> z 向块平均，
一次只驻留几张切片。

块平均而不是抽样，是因为要把细扫带到粗扫的体素尺度上，
对应的物理操作是体素积分，抽样会引入混叠。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import fastio
from scipy import ndimage
from skimage.filters import threshold_otsu


def specimen_centre(files, n_probe: int = 7):
    """只定圆心，不找边界。

    找精确边界在这批数据上不可靠：粗扫的径向剖面有**两个台阶**
    （3.36 mm 处亮核->灰边，3.9 mm 处才到空气），半高判据会卡在第一个；
    细扫的四角落在散射晕里，背景电平被高估，半高又偏外。
    实测两者分别给出 3.54 / 4.53 mm，差 28%，而真实边界都在 3.9-4.0 mm。

    所以改为：只求圆心（稳），再按**固定物理尺寸**裁框，两边一致即可。
    """
    idx = np.clip(np.linspace(len(files) * 0.25, len(files) * 0.75,
                              n_probe).astype(int), 0, len(files) - 1)
    cys, cxs = [], []
    cor = 100
    for i in idx:
        a_ = tifffile.imread(str(files[i])).astype(np.float32)
        bg = float(np.median(np.r_[a_[:cor, :cor].ravel(), a_[:cor, -cor:].ravel(),
                                   a_[-cor:, :cor].ravel(), a_[-cor:, -cor:].ravel()]))
        # 电平要取「岩石基质」而不是「亮矿物」：细扫里 p99.5 是亮矿物，
        # 用它当上限会把门限抬到基质之上，只框住零星亮点，圆心随之偏掉
        # （实测细扫圆心因此偏约 190 px，裁框被钳到画幅边缘）。
        core = a_[a_.shape[0] // 4:-a_.shape[0] // 4,
                  a_.shape[1] // 4:-a_.shape[1] // 4]
        body = float(np.median(core))
        p995 = float(np.percentile(a_, 99.5))
        if body - bg < 0.02 * (p995 - bg):
            body = p995          # 粗扫：中心区大半是背景，退回用高分位
        if body - bg < 30:
            continue
        m = ndimage.binary_opening(a_ > bg + 0.35 * (body - bg), np.ones((5, 5)))
        m = ndimage.binary_fill_holes(m)
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        cand = [k + 1 for k in range(n) if sizes[k] > 0.0005 * m.size]
        if not cand:
            continue
        cy0, cx0 = a_.shape[0] / 2, a_.shape[1] / 2
        cents = ndimage.center_of_mass(m, lab, cand)
        keep = cand[int(np.argmin([np.hypot(c[0] - cy0, c[1] - cx0) for c in cents]))]
        cy, cx = ndimage.center_of_mass(m, lab, [keep])[0]
        cys.append(cy); cxs.append(cx)
    if not cys:
        raise RuntimeError("未能定位试样中心")
    return float(np.median(cys)), float(np.median(cxs))


def physical_bbox(files, voxel_um: float, half_mm: float = 2.3):
    """以试样中心为心、边长 2*half_mm 的**方框**（两边取同一 half_mm 即物理一致）。"""
    cy, cx = specimen_centre(files)
    h_px = int(round(half_mm * 1000.0 / voxel_um))
    a0 = tifffile.imread(str(files[len(files) // 2]))
    H, W = a0.shape
    r0 = int(round(cy)) - h_px; c0 = int(round(cx)) - h_px
    n = 2 * h_px
    # 越界则整体平移（不裁小，保证两边像素数按倍率严格对应）
    r0 = max(0, min(r0, H - n)); c0 = max(0, min(c0, W - n))
    if n > H or n > W:
        raise RuntimeError(f"half_mm={half_mm} 超出画幅（需 {n} px，实际 {H}x{W}）")
    return (r0, r0 + n, c0, c0 + n)


def block_mean_2d(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    return a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def load_downsampled(files, factor: float, bbox=None, zrange=None,
                     verbose=False) -> np.ndarray:
    """按 factor 做各向同性块平均（非整数 factor 用最近整数块 + 后续插值补齐）。"""
    if zrange is not None:
        files = files[zrange[0]:zrange[1]]
    if bbox is None:
        bbox = specimen_bbox(files)
    r0, r1, c0, c1 = bbox

    f_int = max(1, int(round(factor)))
    out = []
    buf = []
    for i, p in enumerate(files):
        a = fastio.read(p, r0, r1, c0, c1).astype(np.float32)
        buf.append(block_mean_2d(a, f_int))
        if len(buf) == f_int:
            out.append(np.mean(buf, axis=0))
            buf = []
        if verbose and i % 300 == 0:
            print(f"    ...{i}/{len(files)}", flush=True)
    if buf:
        out.append(np.mean(buf, axis=0))
    vol = np.stack(out).astype(np.float32)

    # 整数块与真实 factor 的残差用样条缩放补齐
    resid = f_int / factor
    if abs(resid - 1.0) > 1e-3:
        vol = ndimage.zoom(vol, resid, order=1)
    return vol, bbox


def load_cropped(files, bbox=None, zrange=None, verbose=False):
    """只裁剪不降采样（用于本身就小的粗扫试样区）。"""
    if zrange is not None:
        files = files[zrange[0]:zrange[1]]
    if bbox is None:
        bbox = specimen_bbox(files)
    r0, r1, c0, c1 = bbox
    out = np.empty((len(files), r1 - r0, c1 - c0), np.float32)
    for i, p in enumerate(files):
        out[i] = fastio.read(p, r0, r1, c0, c1)
        if verbose and i % 200 == 0:
            print(f"    ...{i}/{len(files)}", flush=True)
    return out, bbox


def norm(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, [0.5, 99.5])
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
