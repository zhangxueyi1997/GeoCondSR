"""用逐子块错位直接最小二乘拟合一个仿射改正，补上刚体模型缺的自由度。

逐子块测出来的残差 98-99% 是位置的线性函数 —— 说明缺的是各向异性缩放/剪切，
不是非刚性形变。那就不必上形变场，一个 3x4 仿射就够，且是闭式解、没有迭代风险。
拟合完再测一遍，残差应当掉到接近相位相关本身的精度（约 0.1 体素）。
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "registration"
CACHE = HERE.parent / "cache"
sys.path.insert(0, str(HERE))
from register3 import interior
from template_match import prep
import compose
from local import subvox_shift, BLK, STEP

V_COARSE = 14.0


def block_shifts(C, B, msk):
    locs, sh, pk = [], [], []
    n = min(C.shape[0], B.shape[0])
    for z in range(0, n - BLK + 1, STEP):
        for y in range(0, C.shape[1] - BLK + 1, STEP):
            for x in range(0, C.shape[2] - BLK + 1, STEP):
                if not msk[y:y + BLK, x:x + BLK].all():
                    continue
                d, p = subvox_shift(C[z:z+BLK, y:y+BLK, x:x+BLK],
                                    B[z:z+BLK, y:y+BLK, x:x+BLK])
                if np.abs(d).max() > 6:
                    continue
                locs.append((z + BLK/2 - 0.5, y + BLK/2 - 0.5, x + BLK/2 - 0.5))
                sh.append(d); pk.append(p)
    return np.array(locs), np.array(sh), np.array(pk)


def fit_affine(L, S, sign):
    """求 (Mc, oc)：校正后的取样点 p' = Mc @ p + oc。"""
    A = np.c_[np.ones(len(L)), L]
    coef = np.linalg.lstsq(A, sign * S, rcond=None)[0]        # (4,3)
    Mc = np.eye(3) + coef[1:].T
    oc = coef[0]
    return Mc, oc


def apply_corr(kw, out_n, Fshape, corr):
    M, off, shp = compose.chain(Fshape, out_n, **kw)
    Mc, oc = corr
    return M @ Mc, M @ oc + off, shp


def place_corr(F, out_n, order, corr, **kw):
    M, off, shp = apply_corr(kw, out_n, F.shape, corr)
    return ndimage.affine_transform(F.astype(np.float32), M, offset=off,
                                    output_shape=shp, order=order,
                                    mode="constant", cval=0.0)


def perslice(A, B):
    n = min(A.shape[0], B.shape[0])
    num = (A[:n] * B[:n]).sum(axis=(1, 2))
    den = np.sqrt((A[:n]**2).sum(axis=(1, 2)) * (B[:n]**2).sum(axis=(1, 2))) + 1e-12
    return float(np.median(num / den))


def run(gid, iters=3):
    kw, dz, base, ref = compose.params(gid)
    C0 = np.load(CACHE / f"cache_{gid}_C.npy")
    F0 = np.load(CACHE / f"cache_{gid}_F.npy")
    mC, _ = interior(C0)
    Cg = C0[4:-4]; Fg = F0[3:-3]
    corr = (np.eye(3), np.zeros(3))
    sign = None
    print(f"\n{gid}/{base['sample']}")
    for it in range(iters):
        W = place_corr(Fg, Cg.shape[1], 3, corr, **kw)
        n = min(W.shape[0], Cg.shape[0] - dz)
        Cp = prep(Cg[dz:dz + n], mC); Bp = prep(W[:n], mC)
        L, S, P = block_shifts(Cp, Bp, mC)
        rms = float(np.sqrt((S**2).sum(1).mean()))
        ncc = perslice(Cp, Bp)
        print(f"  第 {it} 轮：子块 {len(S)}  错位均方根 {rms:.3f} 体素  "
              f"逐层 NCC {ncc:.3f}", flush=True)
        if rms < 0.12 or it == iters - 1:
            break
        if sign is None:                        # 头一轮定相位相关的符号
            best = None
            for sg in (+1, -1):
                c2 = fit_affine(L, S, sg)
                Mc = c2[0] @ corr[0]
                oc = c2[0] @ corr[1] + c2[1]
                W2 = place_corr(Fg, Cg.shape[1], 1, (Mc, oc), **kw)
                n2 = min(W2.shape[0], Cg.shape[0] - dz)
                v = perslice(prep(Cg[dz:dz+n2], mC), prep(W2[:n2], mC))
                if best is None or v > best[0]:
                    best = (v, sg)
            sign = best[1]
            print(f"    相位相关符号定为 {sign:+d}")
        c2 = fit_affine(L, S, sign)
        corr = (c2[0] @ corr[0], c2[0] @ corr[1] + c2[1])
    Mc, oc = corr
    # 把仿射改正拆成可读的量
    U, sv, Vt = np.linalg.svd(Mc)
    print(f"  仿射改正的三个主缩放： {sv[0]:.5f} {sv[1]:.5f} {sv[2]:.5f}"
          f"   （各向同性的话三个应一样）")
    return dict(gid=gid, sample=base["sample"], nblk=len(S),
                rms_vox=round(rms, 3), per_ncc=round(ncc, 4),
                sv=[round(float(x), 6) for x in sv],
                Mc=[[round(float(v), 8) for v in row] for row in Mc],
                oc=[round(float(v), 6) for v in oc], dz=dz)


if __name__ == "__main__":
    out = RES / "affine_corr.json"
    db = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for g in sys.argv[1:]:
        try:
            db[g] = run(g)
            out.write_text(json.dumps(db, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        except Exception:
            import traceback; traceback.print_exc()
