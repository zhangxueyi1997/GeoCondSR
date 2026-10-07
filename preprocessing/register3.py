"""配准 v4：用高亮重矿物颗粒做点云配准。

v2/v3 都失败，且失败方式一致：角度响应全平（显著性 1.01 / 1.07 / 1.17）。
换过两种预处理都无效，说明问题不在预处理，而在**灰度相关本身**——
块平均后的细扫与粗扫，灰度分布、噪声结构、伪影都不同，逐体素相关信噪比过低。

改用岩石里的高亮重矿物：它们大、对比度极高、两种分辨率下都稳定存在，
位置构成稀疏点云。把点云栅格化成稀疏二值体再做相位相关，
峰会非常尖锐，且不会被"居中亮圆柱"这种共同外形淹没。

成败判据（事先写死）：
  内点率 > 25% 且峰值显著性 > 2.0  -> 通过
  内点率 15-25%                     -> 边缘
  否则                              -> 失败（即：两个体大概率不覆盖同一段岩石）
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
JOBS = {"G01": ("ZCQ-1", 2.700016), "G04": ("ZSC-1", 2.499891),
        "G03": ("ZYN-3", 2.499891)}
VOX = 14.0


def interior(v):
    mid = v[v.shape[0] // 2]
    bg = np.median(np.r_[mid[:20, :20].ravel(), mid[-20:, -20:].ravel()])
    body = np.percentile(mid, 80)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(
        mid > bg + 0.4 * (body - bg), np.ones((5, 5))))
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    rad = np.sqrt(m.sum() / np.pi) * 0.90
    yy, xx = np.indices(mid.shape)
    return np.hypot(yy - cy, xx - cx) <= rad, (cy, cx)


def bright_grains(v, msk, q=99.3, min_vox=6, max_n=900):
    """取内部最亮的一小撮体素，连通标记后返回质心（体素坐标）与体积。"""
    vals = v[np.broadcast_to(msk, v.shape)]
    thr = np.percentile(vals, q)
    b = (v > thr) & np.broadcast_to(msk, v.shape)
    b = ndimage.binary_opening(b, np.ones((2, 2, 2)))
    lab, n = ndimage.label(b)
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0)
    sz = np.bincount(lab.ravel())[1:]
    keep = np.nonzero(sz >= min_vox)[0] + 1
    if keep.size == 0:
        return np.zeros((0, 3)), np.zeros(0)
    if keep.size > max_n:
        keep = keep[np.argsort(sz[keep - 1])[::-1][:max_n]]
    cen = np.array(ndimage.center_of_mass(b, lab, list(keep)))
    return cen, sz[keep - 1]


def rasterize(pts, shape, sigma=1.2):
    g = np.zeros(shape, np.float32)
    p = np.round(pts).astype(int)
    ok = np.all((p >= 0) & (p < np.array(shape)), axis=1)
    p = p[ok]
    if len(p):
        np.add.at(g, (p[:, 0], p[:, 1], p[:, 2]), 1.0)
    return ndimage.gaussian_filter(g, sigma)


def phase_corr(a, b):
    sh = tuple(int(2 ** np.ceil(np.log2(max(x, y) * 1.3)))
               for x, y in zip(a.shape, b.shape))
    ax = tuple(range(3))
    A = np.fft.rfftn(a, sh, axes=ax); B = np.fft.rfftn(b, sh, axes=ax)
    R = A * np.conj(B); R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.irfftn(R, sh, axes=ax)
    k = int(np.argmax(c)); pk = float(c.flat[k])
    s = np.unravel_index(k, c.shape)
    s = tuple(int(x) - (d if x > d // 2 else 0) for x, d in zip(s, c.shape))
    return pk, s, c


def rot_z(pts, ang_deg, centre):
    t = np.deg2rad(ang_deg)
    c, s = np.cos(t), np.sin(t)
    q = pts.copy()
    dy = pts[:, 1] - centre[0]; dx = pts[:, 2] - centre[1]
    q[:, 1] = centre[0] + c * dy - s * dx
    q[:, 2] = centre[1] + s * dy + c * dx
    return q


def run(gid, verbose=True):
    t0 = time.time()
    sample, v_fine = JOBS[gid]
    C = np.load(HERE / f"vol_{gid}_coarse.npy")[4:-4]
    F = np.load(HERE / f"vol_{gid}_fine14.npy")[3:-3]
    print(f"\n{'='*74}\n{sample} ({gid})   粗 {C.shape}   细 {F.shape}")

    mC, cenC = interior(C)
    mF, cenF = interior(F)
    pC, szC = bright_grains(C, mC)
    pF, szF = bright_grains(F, mF)
    print(f"  亮矿物颗粒：粗扫 {len(pC)} 个（中位 {np.median(szC) if len(szC) else 0:.0f} 体素）"
          f"   细扫 {len(pF)} 个（中位 {np.median(szF) if len(szF) else 0:.0f} 体素）")
    if len(pC) < 30 or len(pF) < 30:
        print("  >>> 颗粒太少，该方法不适用")
        return dict(gid=gid, sample=sample, verdict="不适用",
                    nC=len(pC), nF=len(pF), ncc=0, sig=0, angle=0,
                    flip=False, z0=0, zlen=0, inlier=0, secs=int(time.time()-t0))

    shape = (C.shape[0], C.shape[1], C.shape[2])
    gC = rasterize(pC, shape)

    best = None
    for flip in (False, True):
        pf = pF.copy()
        if flip:
            pf[:, 0] = (F.shape[0] - 1) - pf[:, 0]
        for ang in np.arange(0, 360, 3.0):
            q = rot_z(pf, ang, cenF)
            q[:, 1] += cenC[0] - cenF[0]
            q[:, 2] += cenC[1] - cenF[1]
            g = rasterize(q, shape)
            pk, s, c = phase_corr(gC, g)
            flat = np.sort(c.ravel())[-2000:]
            sig = pk / np.median(flat) if np.median(flat) > 0 else 0
            if best is None or pk > best[0]:
                best = (pk, sig, ang, flip, s)
        if verbose:
            print(f"  [{'翻转' if flip else '正向'}] 扫完，当前最佳 "
                  f"{best[2]:.0f}° 峰 {best[0]:.4f} 显著性 {best[1]:.2f}")

    pk, sig, ang, flip, sh = best
    print(f"\n  最佳：{'翻转' if flip else '正向'}  {ang:.0f}°  峰 {pk:.4f}  "
          f"显著性 {sig:.2f}  位移 {sh}")

    # 内点率：把细扫点按最佳变换搬过去，看多少落在粗扫某个点的 2 体素内
    pf = pF.copy()
    if flip:
        pf[:, 0] = (F.shape[0] - 1) - pf[:, 0]
    q = rot_z(pf, ang, cenF)
    q[:, 1] += cenC[0] - cenF[0]
    q[:, 2] += cenC[1] - cenF[1]
    q[:, 0] += sh[0]; q[:, 1] += sh[1]; q[:, 2] += sh[2]
    from scipy.spatial import cKDTree
    tree = cKDTree(pC)
    d, _ = tree.query(q)
    inl = float(np.mean(d < 2.5))
    print(f"  内点率（<2.5 体素 = 35 μm）= {inl*100:.1f}%   "
          f"最近邻距离中位 {np.median(d):.2f} 体素")

    verdict = ("通过" if (inl > 0.25 and sig > 2.0)
               else ("边缘" if inl > 0.15 else "失败"))
    print(f"  耗时 {time.time()-t0:.0f}s   >>> 判定：{verdict}")
    return dict(gid=gid, sample=sample, flip=flip, angle=float(ang),
                peak=round(pk, 5), sig=round(float(sig), 2),
                inlier=round(inl, 4), med_dist=round(float(np.median(d)), 2),
                nC=len(pC), nF=len(pF), verdict=verdict,
                secs=int(time.time() - t0))


if __name__ == "__main__":
    gids = sys.argv[1:] or ["G01"]
    rows = [run(g) for g in gids]
    out = HERE / "registration_v4.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n" + "=" * 74)
    for r in rows:
        print(f"{r['sample']:<8}{r['gid']:<5} 内点率 {r.get('inlier',0)*100:5.1f}%  "
              f"显著性 {r.get('sig',0):.2f}   {r['verdict']}")
