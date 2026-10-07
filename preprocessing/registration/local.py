"""逐子块求错位：刚体配准之后，还剩多少「位置相关」的残差？

整体平移不影响逐 k 的相干谱，但**位置相关**的错位会：不同子块的相位不同，
平均下来高频就被抵消掉，看上去就像「这个尺度的信息没了」。

两次扫描的锥角差很大（细扫试样 3 mm / 物距 10 mm，半锥角 8.5°；
粗扫 3 mm / 60 mm，1.4°），重建的锥束伪影和几何畸变完全不是一回事，
残留位置相关畸变是完全可能的。这里直接量出来。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register3 import interior
from template_match import prep
import compose

BLK, STEP = 64, 32


def subvox_shift(a, b, up=8):
    """相位相关 + 抛物线插值，给出 b 相对 a 的亚体素平移。"""
    w = np.hanning(a.shape[0])
    W3 = w[:, None, None] * w[None, :, None] * w[None, None, :]
    A = np.fft.fftn((a - a.mean()) * W3)
    B = np.fft.fftn((b - b.mean()) * W3)
    R = A * B.conj()
    R /= np.abs(R) + 1e-12
    c = np.fft.ifftn(R).real
    k = np.unravel_index(int(np.argmax(c)), c.shape)
    out = []
    for ax in range(3):
        n = c.shape[ax]
        i = k[ax]
        im, ip = (i - 1) % n, (i + 1) % n
        y0 = c[k[:ax] + (im,) + k[ax+1:]]
        y1 = c[k]
        y2 = c[k[:ax] + (ip,) + k[ax+1:]]
        d = 0.0
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            d = 0.5 * (y0 - y2) / den
        v = i + d
        if v > n / 2:
            v -= n
        out.append(v)
    return np.array(out), float(c.max())


def run(gid):
    kw, dz, base, ref = compose.params(gid)
    C0 = np.load(HERE / f"cache_{gid}_C.npy")
    F0 = np.load(HERE / f"cache_{gid}_F.npy")
    mC, _ = interior(C0)
    Cg = C0[4:-4]
    W = compose.place(F0[3:-3], Cg.shape[1], order=3, **kw)
    n = min(W.shape[0], Cg.shape[0] - dz)
    C = prep(Cg[dz:dz + n], mC)
    B = prep(W[:n], mC)

    locs, sh, pk = [], [], []
    for z in range(0, n - BLK + 1, STEP):
        for y in range(0, C.shape[1] - BLK + 1, STEP):
            for x in range(0, C.shape[2] - BLK + 1, STEP):
                if not mC[y:y + BLK, x:x + BLK].all():
                    continue
                d, p = subvox_shift(C[z:z+BLK, y:y+BLK, x:x+BLK],
                                    B[z:z+BLK, y:y+BLK, x:x+BLK])
                if np.abs(d).max() > 6:
                    continue
                locs.append((z + BLK/2, y + BLK/2, x + BLK/2)); sh.append(d); pk.append(p)
    L = np.array(locs); S = np.array(sh); P = np.array(pk)
    print(f"{gid}  子块 {len(S)} 个   相位峰值中位 {np.median(P):.3f}")
    print(f"  错位 (体素)  z {S[:,0].mean():+.2f}±{S[:,0].std():.2f}   "
          f"y {S[:,1].mean():+.2f}±{S[:,1].std():.2f}   "
          f"x {S[:,2].mean():+.2f}±{S[:,2].std():.2f}")
    print(f"  去掉整体平移后的散布（这部分刚体配准修不掉）："
          f"{np.linalg.norm(S - S.mean(0), axis=1).mean():.2f} 体素 平均, "
          f"{np.percentile(np.linalg.norm(S - S.mean(0), axis=1), 90):.2f} 体素 90%")
    # 是不是随位置系统变化？对 (z,y,x) 做一次线性回归看解释了多少
    A = np.c_[np.ones(len(L)), L - L.mean(0)]
    expl = []
    for j in range(3):
        coef, *_ = np.linalg.lstsq(A, S[:, j], rcond=None)
        res = S[:, j] - A @ coef
        v0 = S[:, j].var()
        expl.append(1 - res.var() / max(v0, 1e-12))
    print(f"  其中能被「位置的线性函数」解释的比例： z {expl[0]:.2f}  "
          f"y {expl[1]:.2f}  x {expl[2]:.2f}   （高=还有系统性畸变没修）")
    np.savez(HERE / f"local_{gid}.npz", loc=L, shift=S, peak=P)
    return L, S


if __name__ == "__main__":
    for g in sys.argv[1:] or ["G02"]:
        try:
            run(g)
        except Exception:
            import traceback; traceback.print_exc()
