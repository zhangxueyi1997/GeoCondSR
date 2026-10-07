"""极坐标 + 二维 FFT 搜索：一次算出**所有角度 × 所有 z 偏移**的归一化互相关。

为什么换成极坐标
  面内旋转在极坐标里就是 theta 方向的平移。于是「转 120 个角、每个角做一次
  三维旋转插值」变成「一次极坐标重采样 + 一次二维 FFT」，快两三个数量级。
  角度密度不再受算力限制，比例尺也就能扫得足够细。

同时修掉旧流水线的一个硬伤
  旧代码把「翻转」写成 F[::-1]，那是**镜像**不是翻转，刚体转不出来，
  只可能给出假峰 —— G02/G03 的解正是落在这个分支上。
  真正的端对端翻转是绕面内轴转 180°：z 反向**并且** theta 反向。
"""
from __future__ import annotations
import numpy as np


def polar_grid(ny, nx, cy, cx, rmax, nr, nth):
    r = (np.arange(nr) + 0.5) / nr * rmax
    th = np.arange(nth) / nth * 2 * np.pi
    y = cy + r[:, None] * np.sin(th)[None, :]
    x = cx + r[:, None] * np.cos(th)[None, :]
    y0 = np.floor(y).astype(np.int64); x0 = np.floor(x).astype(np.int64)
    fy = (y - y0).astype(np.float32); fx = (x - x0).astype(np.float32)
    y0 = np.clip(y0, 0, ny - 2); x0 = np.clip(x0, 0, nx - 2)
    idx = (y0 * nx + x0).ravel()
    w = [( (1-fy)*(1-fx) ).ravel(), ( (1-fy)*fx ).ravel(),
         ( fy*(1-fx) ).ravel(), ( fy*fx ).ravel()]
    off = [0, 1, nx, nx + 1]
    return idx, off, w, r


def to_polar(V, cy, cx, rmax, nr, nth):
    """(z,y,x) -> (r, theta, z)，双线性。"""
    nz, ny, nx = V.shape
    idx, off, w, r = polar_grid(ny, nx, cy, cx, rmax, nr, nth)
    flat = V.reshape(nz, ny * nx)
    out = np.zeros((nz, nr * nth), np.float32)
    for o, ww in zip(off, w):
        out += flat[:, idx + o] * ww
    return np.ascontiguousarray(out.reshape(nz, nr, nth).transpose(1, 2, 0)), r


def radial_rescale(P, r_src, r_dst):
    """沿 r 重采样：把模板放大 m 倍就是取 r_src = r_dst/m 处的值。"""
    nr_d = len(r_dst)
    out = np.zeros((nr_d,) + P.shape[1:], np.float32)
    j = np.searchsorted(r_src, r_dst) - 1
    ok = (j >= 0) & (j + 1 < len(r_src))
    jj = j[ok]
    t = ((r_dst[ok] - r_src[jj]) / (r_src[jj+1] - r_src[jj])).astype(np.float32)
    out[ok] = P[jj] * (1 - t)[:, None, None] + P[jj+1] * t[:, None, None]
    return out


def z_rescale(P, m):
    """沿 z 放大 m 倍（线性插值）。"""
    if abs(m - 1.0) < 1e-5:
        return P
    nz = P.shape[2]
    nz2 = int(round(nz * m))
    s = np.arange(nz2) / m
    j = np.clip(s.astype(np.int64), 0, nz - 2)
    t = (s - j).astype(np.float32)
    return P[:, :, j] * (1 - t) + P[:, :, j+1] * t


def corr_all(PC, PB, rw):
    """返回 (n_theta, n_dz) 的归一化互相关。rw = 每个 r 的面积权重 sqrt(r)。"""
    nr, nth, nzC = PC.shape
    nzB = PB.shape[2]
    L = int(2 ** np.ceil(np.log2(nzC + nzB)))
    acc = np.zeros((nth, L // 2 + 1), np.complex128)
    eB = 0.0
    eC = np.zeros(nzC)
    for i in range(nr):
        a = PC[i] * rw[i]; b = PB[i] * rw[i]
        acc += np.fft.rfft2(a, s=(nth, L)) * np.conj(np.fft.rfft2(b, s=(nth, L)))
        eB += float((b.astype(np.float64) ** 2).sum())
        eC += (a.astype(np.float64) ** 2).sum(axis=0)
    num = np.fft.irfft2(acc, s=(nth, L))[:, :nzC - nzB + 1]
    cs = np.concatenate([[0.0], np.cumsum(eC)])
    win = cs[nzB:] - cs[:-nzB]
    den = np.sqrt(np.maximum(win, 1e-12) * max(eB, 1e-12))
    return num / den[None, :]


# ---------------------------------------------------------------- 加速版
from scipy import fft as _sf


class Correlator:
    """预存粗扫一侧的 FFT；每换一个比例尺只需重算模板一侧。

    与 corr_all 等价，但 (1) 用 scipy 多线程 FFT，(2) 复数单精度，
    (3) 长度取 next_fast_len 而非 2 的幂，(4) 三维一次变换而非逐 r 循环。
    """

    def __init__(self, PC, rw, nzB_max):
        self.nr, self.nth, self.nzC = PC.shape
        self.rw = rw
        self.L = _sf.next_fast_len(self.nzC + nzB_max)
        A = (PC * rw[:, None, None]).astype(np.float32)
        self.CF = _sf.rfft2(A, s=(self.nth, self.L), axes=(1, 2),
                            workers=-1).astype(np.complex64)
        self.eC = (A.astype(np.float64) ** 2).sum(axis=(0, 1))   # 每层能量
        cs = np.concatenate([[0.0], np.cumsum(self.eC)])
        self._cs = cs

    def __call__(self, PB):
        nzB = PB.shape[2]
        B = (PB * self.rw[:, None, None]).astype(np.float32)
        BF = _sf.rfft2(B, s=(self.nth, self.L), axes=(1, 2),
                       workers=-1).astype(np.complex64)
        acc = np.einsum("rtk,rtk->tk", self.CF, BF.conj()).astype(np.complex128)
        num = _sf.irfft2(acc, s=(self.nth, self.L), workers=-1)[:, :self.nzC - nzB + 1]
        win = self._cs[nzB:] - self._cs[:-nzB]
        eB = float((B.astype(np.float64) ** 2).sum())
        den = np.sqrt(np.maximum(win, 1e-12) * max(eB, 1e-12))
        return num / den[None, :]
