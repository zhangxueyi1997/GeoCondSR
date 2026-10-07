"""亚体素精配：在粗搜的解上再优化 (dz, dy, dx, 角度, 比例) 五个量。

为什么非做不可
  粗搜的角度步长 0.7°、平移只到整体素。粗扫差 1 个体素 = 细扫差 7 个体素，
  而超分要学的正是这 7 个体素尺度上的细节 —— 配不到亚体素，配对数据就是噪声。
  叠色图上「每个亮粒红一个青一个」就是这个残差的样子。

做法：把已经摆好的细扫体当作起点，只搜一个小的相似变换（旋转+各向同性缩放+平移），
用 Nelder-Mead 直接最大化逐层 NCC 的均值。一次 affine_transform 一遍插值，够快。
"""
from __future__ import annotations
import sys, csv, time
from pathlib import Path
import numpy as np
from scipy import ndimage, optimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register3 import interior
from template_match import prep
from psearch3 import fit_in

V_COARSE = 14.0


def rot3(az, ay, ax):
    """(z,y,x) 指标系下的三个基本旋转之积。az 是面内旋转，ay/ax 是两个倾角。"""
    def r(i, j, t):
        M = np.eye(3); c, s = np.cos(t), np.sin(t)
        M[i, i] = c; M[j, j] = c; M[i, j] = -s; M[j, i] = s
        return M
    return r(1, 2, az) @ r(0, 2, ay) @ r(0, 1, ax)


def similarity(W, ddz, dy, dx, daz, day, dax, ds):
    """小相似变换：绕体心的三轴旋转 + 各向同性缩放 + 平移。

    比只修面内旋转多了 day/dax 两个倾角 —— 两次扫描样品的摆放倾角不同，
    消倾斜只把轴拉直了，轴向之外的残余倾斜还得在这里收掉。"""
    c = (np.array(W.shape) - 1) / 2.0
    R = rot3(np.deg2rad(daz), np.deg2rad(day), np.deg2rad(dax))
    M = R.T / (1.0 + ds)                       # 输出 -> 输入
    off = c - M @ (c + np.array([ddz, dy, dx]))
    return ndimage.affine_transform(W, M, offset=off, order=1, mode="constant")


def score(A, B):
    n = min(A.shape[0], B.shape[0])
    num = (A[:n] * B[:n]).sum(axis=(1, 2))
    den = np.sqrt((A[:n] ** 2).sum(axis=(1, 2)) * (B[:n] ** 2).sum(axis=(1, 2))) + 1e-12
    return float(np.mean(num / den)), num / den


def run(gid, csv_in="reg_final.csv"):
    t0 = time.time()
    r = [x for x in csv.DictReader((HERE / csv_in).open(encoding="utf-8-sig"))
         if x["gid"] == gid][0]
    m = float(r["m"]); flip = r["flip"] == "True"
    ang = float(r["angle"]); dz = int(r["dz"])
    C = np.load(HERE / f"cache_{gid}_C.npy"); F = np.load(HERE / f"cache_{gid}_F.npy")
    mC, _ = interior(C); mF, _ = interior(F)
    Cp = prep(C[4:-4], mC); Fp = prep(F[3:-3], mF)

    W0 = ndimage.zoom(Fp, (m, m, m), order=1).astype(np.float32)
    if flip:
        W0 = W0[::-1, ::-1, :]
    W0 = fit_in(ndimage.rotate(W0, ang, axes=(1, 2), reshape=False, order=1,
                               mode="constant"), Cp.shape[1])
    n = min(W0.shape[0], Cp.shape[0] - dz)
    A = Cp[dz:dz + n]
    base, _ = score(A, W0[:n])

    NP = 7

    def neg(p):
        W = similarity(W0, p[0], p[1], p[2], p[3], p[4], p[5], p[6] * 0.01)
        return -score(A, W[:n])[0]

    # 起点全零时 Nelder-Mead 的默认初始单纯形只有 2.5e-4，等于没动；显式给步长
    step = np.array([1.0, 1.0, 1.0, 0.4, 0.4, 0.4, 0.3])
    simp = np.vstack([np.zeros(NP)] + [np.eye(NP)[i] * step[i] for i in range(NP)])
    res = optimize.minimize(neg, np.zeros(NP), method="Nelder-Mead",
                            options=dict(xatol=0.008, fatol=1e-5, maxfev=900,
                                         initial_simplex=simp))
    p = res.x
    W = similarity(W0, *p[:6], p[6] * 0.01)
    fin, per = score(A, W[:n])
    m_fin = m * (1 + p[6] * 0.01)
    print(f"{gid}/{r['sample']}  逐层 NCC {base:.3f} -> {fin:.3f}   "
          f"dz{p[0]:+.2f} dy{p[1]:+.2f} dx{p[2]:+.2f}  面内{p[3]:+.3f}° "
          f"倾角{p[4]:+.3f}/{p[5]:+.3f}°  比例{p[6]*0.01*100:+.3f}%   "
          f"({time.time()-t0:.0f}s, {res.nfev} 次)", flush=True)
    ratio_fin = float(r["ratio"]) * (1 + p[6] * 0.01)
    return dict(gid=gid, sample=r["sample"], v_panel=r["v_panel"],
                v_true=round(V_COARSE / ratio_fin, 4),
                dev_pct=round((V_COARSE / ratio_fin) / float(r["v_panel"]) * 100 - 100, 1),
                ratio=round(ratio_fin, 4), m=round(m_fin, 5), flip=flip,
                angle=round(ang + p[3], 3), dz=dz, ddz=round(float(p[0]), 3),
                dy=round(float(p[1]), 3), dx=round(float(p[2]), 3),
                tilt_y=round(float(p[4]), 3), tilt_x=round(float(p[5]), 3),
                z_mm=round(dz * V_COARSE / 1000, 2),
                ncc_before=round(base, 4), ncc=round(fin, 4),
                per_min=round(float(per.min()), 3),
                per_pos=f"{int((per > 0.05).sum())}/{n}",
                verdict=r["verdict"], secs=int(time.time() - t0))


if __name__ == "__main__":
    out = HERE / "reg_refined.csv"
    rows = []
    for g in sys.argv[1:]:
        try:
            rows.append(run(g))
        except Exception:
            import traceback; traceback.print_exc()
        if rows:
            old = (list(csv.DictReader(out.open(encoding="utf-8-sig")))
                   if out.exists() else [])
            keep = [x for x in old if x["gid"] not in {y["gid"] for y in rows}]
            allr = sorted(keep + [{k: str(v) for k, v in x.items()} for x in rows],
                          key=lambda x: x["gid"])
            with out.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(allr[0].keys()))
                w.writeheader(); w.writerows(allr)
