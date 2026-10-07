"""消倾斜：把柱轴摆正，再存回缓存供配准用。

实测 G01 粗扫倾角 4.97°、细扫 1.26°，差 3.71°；柱长范围内两端错位 45 体素，
而点云内点阈值是 2.5 体素 —— 这是此前所有配准失败的直接原因。

用**拟合的轴线**逐层平移（不是逐层原始质心），只去掉系统性倾斜，
不把结构本身扭掉。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tilt import centroids, fit_axis  # noqa: E402


def destilt(v, return_fit=False):
    cy, cx = centroids(v)
    f = fit_axis(cy, cx)
    if f is None:
        raise RuntimeError("轴线拟合失败")
    py, px, tilt, _ = f
    z = np.arange(v.shape[0])
    ty = np.polyval(py, z); tx = np.polyval(px, z)
    cy0, cx0 = v.shape[1] / 2.0, v.shape[2] / 2.0
    out = np.empty_like(v)
    for k in range(v.shape[0]):
        out[k] = ndimage.shift(v[k], (cy0 - ty[k], cx0 - tx[k]),
                               order=1, mode="constant", cval=float(v[k].min()))
    if return_fit:
        # (py, px) 是轴心随 z 的一次拟合，(cy0, cx0) 是平移到的目标中心。
        # 导出全分辨率配对时要用它把这一步剪切还原回原始坐标。
        return out, tilt, dict(py=[float(c) for c in np.atleast_1d(py)],
                               px=[float(c) for c in np.atleast_1d(px)],
                               cy0=float(cy0), cx0=float(cx0), nz=int(v.shape[0]))
    return out, tilt


def main(gid="G01"):
    for tag in ("coarse", "fine14"):
        src = HERE / f"vol_{gid}_{tag}.npy"
        dst = HERE / f"vol_{gid}_{tag}_dt.npy"
        v = np.load(src)
        w, tilt = destilt(v)
        np.save(dst, w.astype(np.float32))
        cy2, cx2 = centroids(w)
        f2 = fit_axis(cy2, cx2)
        print(f"{gid} {tag}: 倾角 {tilt:.2f}° -> "
              f"{f2[2] if f2 else float('nan'):.2f}°   写入 {dst.name}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["G01"]))
