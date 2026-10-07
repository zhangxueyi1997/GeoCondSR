"""配准 v8：把细扫当**模板**，在粗扫里找它的位置（带重叠归一化）。

用户指出的要点：细扫只是柱子的一段，粗扫是整根。这是「小的在大的里面定位」，
不该用相位相关 —— 短体要补零到长体长度，大部分相关量是「数据×零」，
真峰被稀释，且不同 z 偏移下有效重叠量不同却没有归一化。

正确做法是归一化互相关模板匹配：
    NCC(θ, dz) = <C[dz:dz+n], R(F,θ)> / sqrt( ||C[dz:dz+n]||² · ||R(F,θ)||² )
分子沿 z 用 FFT 算（一次批量 rfft 搞定所有 (y,x) 列），
分母用沿 z 的累积和，两者都 O(N log N)。

面内两者都已裁到 ±2.3 mm 且细扫覆盖完整横截面（视场 5.42 mm > 试样 4 mm），
所以面内只剩绕 z 旋转，无需搜平移。
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register3 import interior  # noqa: E402

JOBS = {"G01": ("ZCQ-1", 2.700016), "G04": ("ZSC-1", 2.499891),
        "G03": ("ZYN-3", 2.499891)}
VOX = 14.0
SUF = "_dt"


def prep(v, msk, hp_sigma=4.0):
    """高通去碗状趋势 -> 掩膜内零均值 -> 掩膜外置零。"""
    out = np.empty_like(v, dtype=np.float32)
    for z in range(v.shape[0]):
        s = v[z].astype(np.float32)
        out[z] = (s - ndimage.gaussian_filter(s, hp_sigma)) * msk
    m = np.broadcast_to(msk, out.shape)
    out -= out[m].mean()
    out *= m
    return out


def ncc_vs_dz(C, Fr):
    """返回每个 z 偏移的归一化互相关。C 长、Fr 短。"""
    nC, nF = C.shape[0], Fr.shape[0]
    L = int(2 ** np.ceil(np.log2(nC + nF)))
    # 分子：沿 z 的互相关，(y,x) 上求和
    Cf = np.fft.rfft(C, L, axis=0)
    Ff = np.fft.rfft(Fr[::-1], L, axis=0)          # 反序 = 相关
    num = np.fft.irfft(Cf * Ff, L, axis=0).sum(axis=(1, 2))
    num = num[nF - 1: nF - 1 + (nC - nF + 1)]
    # 分母：窗内 ||C||，用累积和
    e = (C.astype(np.float64) ** 2).sum(axis=(1, 2))
    cs = np.concatenate([[0.0], np.cumsum(e)])
    win = cs[nF:] - cs[:-nF]                       # 长度 nC-nF+1
    den = np.sqrt(np.maximum(win, 1e-12) * float((Fr.astype(np.float64) ** 2).sum()))
    return num / den


def run(gid, verbose=True):
    t0 = time.time()
    sample, v_fine = JOBS[gid]
    C0 = np.load(HERE / f"v2_{gid}_coarse.npy")[4:-4]
    F0 = np.load(HERE / f"v2_{gid}_fine.npy")[3:-3]
    # 两边面内像素数可能差 1（真实倍率非整数导致的取整），统一裁到同尺寸
    n = min(C0.shape[1], C0.shape[2], F0.shape[1], F0.shape[2])
    C0 = C0[:, :n, :n]; F0 = F0[:, :n, :n]
    mC, _ = interior(C0); mF, _ = interior(F0)
    msk = mC & mF
    C = prep(C0, msk); F = prep(F0, msk)
    print(f"\n{'='*74}\n{sample} ({gid})  粗 {C.shape} 细 {F.shape}  "
          f"可搜 z 偏移 0–{C.shape[0]-F.shape[0]}")

    best = None
    for lvl, (down, step, rng) in enumerate([(4, 3.0, None), (2, 1.0, 6.0),
                                             (1, 0.4, 2.0)]):
        Cd = C[:, ::down, ::down] if down > 1 else C
        Fd = F[:, ::down, ::down] if down > 1 else F
        angles = (np.arange(0, 360, step) if rng is None
                  else np.arange(best[1] - rng, best[1] + rng + 1e-9, step))
        rows = []
        for ang in angles:
            for flip in ((False, True) if lvl == 0 else (best[3],)):
                G = Fd[::-1] if flip else Fd
                Gr = ndimage.rotate(G, ang, axes=(1, 2), reshape=False,
                                    order=1, mode="constant", prefilter=False)
                r = ncc_vs_dz(Cd, Gr)
                k = int(np.argmax(r))
                rows.append((float(r[k]), float(ang), k, flip, r))
        rows.sort(key=lambda t: -t[0])
        best = rows[0]
        # 显著性：最佳值 vs 所有 (角, dz) 的分布
        pool = np.concatenate([t[4] for t in rows])
        sig = (best[0] - pool.mean()) / (pool.std() + 1e-12)
        if verbose:
            print(f"  L{lvl+1} 1/{down}  步长 {step}°  {len(angles)} 角  ->  "
                  f"{best[1]:.2f}°{'（翻转）' if best[3] else ''}  "
                  f"dz={best[2]}  NCC {best[0]:.4f}  {sig:.1f}σ", flush=True)

    ncc, ang, dz, flip, prof = best
    pool = prof
    sig = (ncc - pool.mean()) / (pool.std() + 1e-12)
    print(f"\n  最佳：{'翻转 ' if flip else ''}{ang:.2f}°  dz={dz} 层 = "
          f"{dz*VOX/1000:.2f} mm  NCC {ncc:.4f}  {sig:.1f}σ")
    print(f"  细扫段落在粗扫 {dz*VOX/1000:.2f}–{(dz+F.shape[0])*VOX/1000:.2f} mm")
    verdict = "通过" if (ncc > 0.30 and sig > 6) else \
              ("边缘" if ncc > 0.15 and sig > 4 else "失败")
    print(f"  耗时 {time.time()-t0:.0f}s   >>> {verdict}")

    np.save(HERE / f"tm_{gid}_prof.npy", prof)
    Gr = ndimage.rotate(F[::-1] if flip else F, ang, axes=(1, 2),
                        reshape=False, order=1, mode="constant")
    np.save(HERE / f"tm_{gid}_coarse.npy", C0[dz:dz+F.shape[0]])
    np.save(HERE / f"tm_{gid}_fine.npy",
            ndimage.rotate(F0[::-1] if flip else F0, ang, axes=(1, 2),
                           reshape=False, order=1, mode="constant"))
    return dict(gid=gid, sample=sample, flip=flip, angle=round(ang, 2), dz=dz,
                z_mm=round(dz * VOX / 1000, 2), ncc=round(ncc, 4),
                sigma=round(float(sig), 1), verdict=verdict,
                secs=int(time.time() - t0))


if __name__ == "__main__":
    gids = sys.argv[1:] or ["G01"]
    rows = [run(g) for g in gids]
    out = HERE / "template_match.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    print("\n" + "=" * 74)
    for r in rows:
        print(f"{r['sample']:<8}{r['gid']:<5} {r['angle']:6.2f}°  "
              f"dz {r['z_mm']:.2f} mm  NCC {r['ncc']:.4f}  {r['sigma']:.1f}σ  "
              f"{r['verdict']}")
