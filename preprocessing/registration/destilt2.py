"""消倾斜（改正版）：用**旋转**把柱轴摆正，不是用逐层平移。

原来 destilt.py 是逐层平移 —— 那是剪切。试样倾斜的物理正解是刚体旋转，
剪切和旋转在倾角一阶上就差一个同量级的剪切分量。两次扫描倾角不同、方向也
不同，于是给两个体各留下一个不同的假剪切，配准后表现为 3-11% 的各向异性
仿射残差（实测 剪切 = 1.36 x sqrt(粗倾角^2+细倾角^2)，r = 0.937）。

改成旋转之后，两个体之间就只剩真正的刚体+等比缩放关系。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tilt import centroids, fit_axis


def rodrigues(v, u):
    """把单位向量 v 转到 u 的旋转矩阵。"""
    v = v / np.linalg.norm(v); u = u / np.linalg.norm(u)
    k = np.cross(v, u); s = np.linalg.norm(k); c = float(v @ u)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    k = k / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def destilt(v, return_fit=False, order=1):
    cy, cx = centroids(v)
    f = fit_axis(cy, cx)
    if f is None:
        raise RuntimeError("轴线拟合失败")
    py, px, tilt, _ = f
    a = float(np.atleast_1d(py)[0]); b = float(np.atleast_1d(px)[0])
    R = rodrigues(np.array([1.0, a, b]), np.array([1.0, 0.0, 0.0]))
    c = (np.array(v.shape) - 1) / 2.0
    # 试样轴在体中心高度处的位置 -> 要搬到体中心
    zc = (v.shape[0] - 1) / 2.0
    p0 = np.array([zc, float(np.polyval(py, zc)), float(np.polyval(px, zc))])
    M = R.T                                    # 输出 -> 输入
    off = p0 - M @ c
    out = ndimage.affine_transform(v.astype(np.float32), M, offset=off,
                                   order=order, mode="constant",
                                   cval=float(np.percentile(v[::9], 1)))
    if return_fit:
        return out, tilt, dict(R=[[float(x) for x in r] for r in R],
                               p0=[float(x) for x in p0],
                               c=[float(x) for x in c], tilt=float(tilt),
                               nz=int(v.shape[0]))
    return out, tilt


if __name__ == "__main__":
    from tilt import centroids as _c, fit_axis as _f
    gid = sys.argv[1] if len(sys.argv) > 1 else "G01"
    for tag in ("C", "F"):
        p = HERE / f"cache_{gid}_{tag}.npy"
        if not p.exists():
            continue
        v = np.load(p)
        cy, cx = _c(v); ff = _f(cy, cx)
        print(f"{gid} {tag}: 现有缓存的残余倾角 {ff[2] if ff else float('nan'):.3f}°")
