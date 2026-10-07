"""把「缩放→翻转→旋转→裁框→精配」这一串合成成一次仿射，只插值一遍。

为什么必须这么做
  这一串原来是四次 order=1 的插值。线性插值本身就是个低通滤波器：
  半 体素 偏移下，波长 4.6 体素处幅度只剩 0.78，到奈奎斯特直接归零。
  而我们要量的恰恰是「高频还剩多少」—— 用线性插值量，量到的一大半是
  我自己抹掉的，不是仪器抹掉的。合成成一次 order=3 样条插值就干净了。
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "registration"
CACHE = HERE.parent / "cache"


def H(M=None, t=None):
    A = np.eye(4)
    if M is not None:
        A[:3, :3] = M
    if t is not None:
        A[:3, 3] = t
    return A


def rot_yx(deg):
    """与 ndimage.rotate(V, deg, axes=(1,2)) 等价的 输出->输入 矩阵（符号已实测核对）。"""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], float)


def rot3(az, ay, ax):
    def r(i, j, t):
        M = np.eye(3); c, s = np.cos(t), np.sin(t)
        M[i, i] = c; M[j, j] = c; M[i, j] = -s; M[j, i] = s
        return M
    return r(1, 2, az) @ r(0, 2, ay) @ r(0, 1, ax)


def chain(shape0, out_n, m, flip, ang, ddz, dy, dx, daz, day, dax, ds):
    """返回 (M, off, 输出形状)，使 out[p] = F[M @ p + off]。"""
    nz0, ny0, nx0 = shape0
    shpA = tuple(int(round(n * m)) for n in shape0)          # zoom 后
    # A: zoom（grid_mode=False 下 out o 对应 in o*(n_in-1)/(n_out-1)）
    sA = np.diag([(n0 - 1) / max(na - 1, 1) for n0, na in zip(shape0, shpA)])
    A = H(sA)
    # B: 翻转 z 与 y
    if flip:
        B = H(np.diag([-1.0, -1.0, 1.0]), [shpA[0] - 1, shpA[1] - 1, 0.0])
    else:
        B = np.eye(4)
    # C: 面内旋转（reshape=False，形状不变，绕体心）
    cC = (np.array(shpA) - 1) / 2.0
    R = rot_yx(ang)
    C = H(R, cC - R @ cC)
    # D: fit_in 到 out_n（只动 y,x，保持中心）
    n1 = shpA[1]
    o = (n1 - out_n) // 2 if n1 >= out_n else -((out_n - n1) // 2)
    D = H(t=[0.0, o, o])
    shpD = (shpA[0], out_n, out_n)
    # E: 精配的小相似变换（绕输出体心）
    cE = (np.array(shpD) - 1) / 2.0
    ME = rot3(np.deg2rad(daz), np.deg2rad(day), np.deg2rad(dax)).T / (1.0 + ds)
    E = H(ME, cE - ME @ (cE + np.array([ddz, dy, dx])))
    T = A @ B @ C @ D @ E
    return T[:3, :3], T[:3, 3], shpD


def place(F, out_n, order=3, **kw):
    M, off, shp = chain(F.shape, out_n, **kw)
    return ndimage.affine_transform(F.astype(np.float32), M, offset=off,
                                    output_shape=shp, order=order,
                                    mode="constant", cval=0.0)


def params(gid):
    base = {r["gid"]: r for r in csv.DictReader(
        (RES / "reg_final.csv").open(encoding="utf-8-sig"))}[gid]
    ref = {r["gid"]: r for r in csv.DictReader(
        (RES / "reg_refined.csv").open(encoding="utf-8-sig"))}[gid]
    ang0 = float(base["angle"])
    return dict(m=float(base["m"]), flip=base["flip"] == "True", ang=ang0,
                ddz=float(ref["ddz"]), dy=float(ref["dy"]), dx=float(ref["dx"]),
                daz=float(ref["angle"]) - ang0, day=float(ref["tilt_y"]),
                dax=float(ref["tilt_x"]), ds=float(ref["m"]) / float(base["m"]) - 1.0
                ), int(base["dz"]), base, ref


if __name__ == "__main__":
    # 自检：合成矩阵必须复现逐步链条的结果
    rng = np.random.default_rng(0)
    V = ndimage.gaussian_filter(rng.standard_normal((40, 52, 52)), 1.5).astype(np.float32)
    for flip in (False, True):
        for ang in (0.0, 23.7, -155.2):
            m = 0.94
            W = ndimage.zoom(V, (m, m, m), order=1).astype(np.float32)
            if flip:
                W = W[::-1, ::-1, :]
            W = ndimage.rotate(W, ang, axes=(1, 2), reshape=False, order=1,
                               mode="constant")
            n0 = 44
            n1 = W.shape[1]
            if n1 >= n0:
                o = (n1 - n0) // 2; Wc = W[:, o:o+n0, o:o+n0]
            else:
                o = (n0 - n1) // 2
                Wc = np.zeros((W.shape[0], n0, n0), np.float32)
                Wc[:, o:o+n1, o:o+n1] = W
            Z = place(V, n0, order=1, m=m, flip=flip, ang=ang, ddz=0, dy=0, dx=0,
                      daz=0, day=0, dax=0, ds=0)
            k = min(Wc.shape[0], Z.shape[0])
            a = Wc[2:k-2, 6:-6, 6:-6]; b = Z[2:k-2, 6:-6, 6:-6]
            cc = float((a*b).sum() / (np.sqrt((a*a).sum()*(b*b).sum()) + 1e-12))
            print(f"  flip={flip!s:<5} ang={ang:7.1f}  与逐步链条相关 {cc:.5f}"
                  f"   形状 {Wc.shape} vs {Z.shape}")
