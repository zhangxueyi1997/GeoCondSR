"""搜索器 v3：极坐标 FFT + 比例尺迭代（用分段斜率直接解，不靠网格）+ 宽域兜底。

判真假只看两条硬指标，不看 NCC 绝对值：
  · 分段共线  —— 细扫切 8 段各自定位，落点必须落在一条直线上（假峰凑不出来）
  · 逐层 NCC  —— 回笛卡尔坐标，层层为正
"""
from __future__ import annotations
import sys, time, csv, json, itertools
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register3 import interior
from template_match import prep
from polar import to_polar, radial_rescale, z_rescale, Correlator
from pipeline import JOBS, rock_diameter_px, PROJ, COARSE, V_COARSE

NTH = 512


def build(gid, rfrac):
    C = np.load(HERE / f"cache_{gid}_C.npy")
    F = np.load(HERE / f"cache_{gid}_F.npy")
    mC, (cyC, cxC) = interior(C)
    mF, (cyF, cxF) = interior(F)
    R = np.sqrt(min(mC.sum(), mF.sum()) / np.pi) / 0.90
    rmax = rfrac * R
    nr = max(24, int(rmax))
    Cp = prep(C[4:-4], mC)
    Fp = prep(F[3:-3], mF)
    r_grid = (np.arange(nr) + 0.5) / nr * rmax
    PC, _ = to_polar(Cp, cyC, cxC, rmax, nr, NTH)
    PF, r_src = to_polar(Fp, cyF, cxF, 0.90 * R, 2 * nr, NTH)
    return dict(PC=PC, PF=PF, r_src=r_src, r_grid=r_grid,
                rw=np.sqrt(r_grid).astype(np.float32), R=R, Cp=Cp, Fp=Fp)


def tmpl(S, m, flip):
    P = z_rescale(radial_rescale(S["PF"], S["r_src"], S["r_grid"] / m), m)
    if flip:                       # 端对端翻转 = z 反向 + theta 反向
        P = P[:, ::-1, ::-1]
    return np.ascontiguousarray(P)


def best_of(cor, S, ms, flips):
    out = None
    for flip in flips:
        for m in ms:
            r = cor(tmpl(S, m, flip))
            k = int(np.argmax(r))
            a, d = np.unravel_index(k, r.shape)
            v = float(r.flat[k])
            if out is None or v > out[0]:
                out = (v, float(m), bool(flip), a * 360.0 / NTH, int(d),
                       float((v - r.mean()) / (r.std() + 1e-12)))
    return out


def anchors(cor, S, m, flip, K=8):
    B = tmpl(S, m, flip)
    e = np.linspace(0, B.shape[2], K + 1).astype(int)
    xs, ys, ns = [], [], []
    for k in range(K):
        blk = np.ascontiguousarray(B[:, :, e[k]:e[k + 1]])
        if blk.shape[2] < 10:
            continue
        r = cor(blk)
        j = int(np.argmax(r))
        _, d = np.unravel_index(j, r.shape)
        xs.append((e[k] + e[k + 1]) / 2)
        ys.append(d + blk.shape[2] / 2)
        ns.append(float(r.flat[j]))
    xs = np.array(xs); ys = np.array(ys)
    # Theil-Sen：对离群段稳健。最小二乘会被一个跑飞的段整段带偏（G04 即如此）
    sl = [ (ys[j]-ys[i])/(xs[j]-xs[i])
           for i, j in itertools.combinations(range(len(xs)), 2) if xs[j] != xs[i] ]
    b = float(np.median(sl)); a = float(np.median(ys - b*xs))
    res = ys - (a + b*xs)
    return b, int((np.abs(res) < 5).sum()), len(xs), ys, np.array(ns)


def fit_in(W, n0):
    n1 = W.shape[1]
    if n1 >= n0:
        o = (n1 - n0) // 2
        return np.ascontiguousarray(W[:, o:o + n0, o:o + n0])
    o = (n0 - n1) // 2
    t = np.zeros((W.shape[0], n0, n0), np.float32)
    t[:, o:o + n1, o:o + n1] = W
    return t


def shift3(A, B):
    """相位相关求整数平移，用于面内（及 z 上的亚层）微调。"""
    fa = np.fft.rfftn(A, axes=(0, 1, 2)); fb = np.fft.rfftn(B, axes=(0, 1, 2))
    x = fa * np.conj(fb); x /= np.abs(x) + 1e-9
    c = np.fft.irfftn(x, A.shape, axes=(0, 1, 2))
    k = np.unravel_index(int(np.argmax(c)), c.shape)
    return tuple(int(v) - (sz if v > sz // 2 else 0) for v, sz in zip(k, A.shape))


def perslice(A, B):
    n = min(A.shape[0], B.shape[0])
    return np.array([float((A[i] * B[i]).sum() /
                     (np.sqrt((A[i] ** 2).sum() * (B[i] ** 2).sum()) + 1e-12))
                     for i in range(n)])


def attempt(gid, rfrac, ms, log):
    S = build(gid, rfrac)
    cor = Correlator(S["PC"], S["rw"], int(S["PF"].shape[2] * max(ms)) + 4)
    b = best_of(cor, S, ms, (False, True))
    b = best_of(cor, S, np.arange(b[1] - 0.012, b[1] + 0.0121, 0.002), (b[2],))
    ncc, m, flip, ang, dz, sig = b
    for _ in range(3):                        # 用分段斜率直接解比例尺
        sl, good, K, ys, ns = anchors(cor, S, m, flip)
        if good < 6 or abs(sl - 1) < 0.0015:
            break
        m2 = float(np.clip(m * sl, ms.min(), ms.max()))
        c = best_of(cor, S, np.arange(m2 - 0.004, m2 + 0.0041, 0.001), (flip,))
        log.append(f"    斜率 {sl:.4f}  m {m:.4f} -> {c[1]:.4f}   "
                   f"NCC {ncc:.4f} -> {c[0]:.4f}")
        if c[0] <= ncc + 1e-4:
            break
        ncc, m, flip, ang, dz, sig = c
    sl, good, K, ys, ns = anchors(cor, S, m, flip)
    log.append(f"  m={m:.4f} {'翻转' if flip else '正向'} {ang:.1f}° dz={dz} "
               f"NCC {ncc:.4f} {sig:.0f}sigma  斜率 {sl:.4f} 共线 {good}/{K}")
    log.append("  分段：" + " ".join(f"{y:.0f}({n:.2f})" for y, n in zip(ys, ns)))

    best = None
    for sgn in (1, -1):
        W = ndimage.zoom(S["Fp"], (m, m, m), order=1).astype(np.float32)
        if flip:
            W = W[::-1, ::-1, :]
        W = fit_in(ndimage.rotate(W, sgn * ang, axes=(1, 2), reshape=False,
                                  order=1, mode="constant"), S["Cp"].shape[1])
        n = min(W.shape[0], S["Cp"].shape[0] - dz)
        per = perslice(S["Cp"][dz:dz + n], W[:n])
        if best is None or np.median(per) > np.median(best[1]):
            best = (sgn, per, W)
    sgn, per, W = best
    # 面内微调：两边中心各自估出来，差几个体素就足以压低相关
    n = min(W.shape[0], S["Cp"].shape[0] - dz)
    ddz, dy, dx = shift3(S["Cp"][dz:dz + n], W[:n])
    if abs(dy) + abs(dx) + abs(ddz) > 0:
        d2 = int(np.clip(dz - ddz, 0, S["Cp"].shape[0] - n))
        Wm = ndimage.shift(W, (0, dy, dx), order=1)
        per2 = perslice(S["Cp"][d2:d2 + n], Wm[:n])
        if np.median(per2) > np.median(per):
            log.append(f"  面内微调 dz{-ddz:+d} dy{dy:+d} dx{dx:+d}  "
                       f"逐层中位 {np.median(per):.3f} -> {np.median(per2):.3f}")
            per, W, dz = per2, Wm, d2
    med = float(np.median(per))
    pos = int((per > 0.05).sum())
    ok = med > 0.15 and pos > 0.85 * len(per) and good >= 6 and abs(sl - 1) < 0.03
    log.append(f"  逐层 NCC 中位 {med:.3f}  区间 {per.min():.3f}-{per.max():.3f}  "
               f">0.05 的 {pos}/{len(per)}   {'通过' if ok else '失败'}")
    return dict(ok=ok, m=m, flip=flip, angle=sgn * ang, dz=dz, ncc=ncc,
                sigma=sig, slope=sl, colin=f"{good}/{K}", per_med=med,
                per_pos=f"{pos}/{len(per)}", score=med), per


def run(gid):
    t0 = time.time()
    log = []
    sample, v_panel, cdir = JOBS[gid]
    mf = HERE / f"meta_{gid}.json"
    if mf.exists():
        ratio0 = json.loads(mf.read_text(encoding="utf-8"))["ratio"]
    else:                                       # 早期缓存没存 meta，按原法重算
        dC, _ = rock_diameter_px(sorted((COARSE / cdir).glob("*.tif")))
        dF, _ = rock_diameter_px(sorted((PROJ / "dataset" / gid / "small_ct" /
                                         "raw16").glob("*.tif")))
        ratio0 = dF / dC
    print(f"\n{'=' * 72}\n{gid} / {sample}   直径比倍率 {ratio0:.3f}", flush=True)

    r, per = attempt(gid, 0.75, np.arange(0.86, 1.161, 0.01), log)
    if not r["ok"]:
        log.append("  一次不过，改用 60% 半径 + 宽比例域重试")
        r2, per2 = attempt(gid, 0.60, np.arange(0.68, 1.301, 0.01), log)
        if r2["score"] > r["score"]:
            r, per = r2, per2
    print("\n".join(log), flush=True)

    ratio_fin = ratio0 * r["m"]
    v_fin = V_COARSE / ratio_fin
    print(f"  -> 真实倍率 {ratio_fin:.3f}   真实体素 {v_fin:.4f} um "
          f"(面板 {v_panel:.4f}，差 {(v_fin / v_panel - 1) * 100:+.1f}%)   "
          f"dz={r['dz']} = {r['dz'] * V_COARSE / 1000:.2f} mm   "
          f">>> {'通过' if r['ok'] else '失败'}  ({time.time() - t0:.0f}s)", flush=True)
    np.save(HERE / f"per_{gid}.npy", per)
    return dict(gid=gid, sample=sample, v_panel=v_panel, v_true=round(v_fin, 4),
                dev_pct=round((v_fin / v_panel - 1) * 100, 1),
                ratio=round(ratio_fin, 3), m=round(r["m"], 4), flip=r["flip"],
                angle=round(r["angle"], 1), dz=r["dz"],
                z_mm=round(r["dz"] * V_COARSE / 1000, 2), ncc=round(r["ncc"], 4),
                sigma=round(r["sigma"], 1), slope=round(r["slope"], 4),
                colin=r["colin"], per_med=round(r["per_med"], 3),
                per_pos=r["per_pos"], verdict="通过" if r["ok"] else "失败",
                secs=int(time.time() - t0))


if __name__ == "__main__":
    out = HERE / "reg_final.csv"
    rows = []
    for g in sys.argv[1:]:
        try:
            rows.append(run(g))
        except Exception:
            import traceback
            traceback.print_exc()
        if rows:
            old = (list(csv.DictReader(out.open(encoding="utf-8-sig")))
                   if out.exists() else [])
            keep = [r for r in old if r["gid"] not in {x["gid"] for x in rows}]
            allr = sorted(keep + [{k: str(v) for k, v in r.items()} for r in rows],
                          key=lambda r: r["gid"])
            with out.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(allr[0].keys()))
                w.writeheader()
                w.writerows(allr)
    print("\n" + "=" * 72)
    for r in rows:
        print(f"{r['gid']:<5}{r['sample']:<9} v={r['v_true']:.4f}um "
              f"({r['dev_pct']:+.1f}%)  {r['angle']:+7.1f}°"
              f"{'翻' if r['flip'] else '  '} dz {r['z_mm']:5.2f}mm  "
              f"NCC {r['ncc']:.3f} 斜率 {r['slope']:.4f} 共线 {r['colin']} "
              f"逐层 {r['per_med']:.3f} {r['per_pos']}  {r['verdict']}")
