"""生产流水线：逐样品定标定尺 -> 裁框 -> 消倾斜 -> 模板匹配。

G01 已验证：面板体素错 21%，真实倍率 6.550（面板给 5.185），
按真实倍率配准成功（334.00°，dz=399，逐层 NCC 0.547-0.728），
按面板倍率七次全败。

每个样品重复同一套：
  1. 从原始切片量岩石直径（最外侧半高穿越，自然排除护套 —— 护套灰度低于半高）
  2. 由「同一块岩石两次扫描必须一样大」定出真实倍率
  3. 按真实倍率块平均、同物理裁框、消倾斜
  4. 模板匹配（细扫是短模板，在长粗扫里定位；带重叠归一化）
"""
from __future__ import annotations
import os

import csv
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loadvol import load_cropped, load_downsampled, physical_bbox, specimen_centre
from destilt import destilt
from register3 import interior
from template_match import ncc_vs_dz, prep

PROJ = Path(os.environ.get('GEOCOND_RAW_PROJECT', 'raw_project'))   # contains dataset/<group>/small_ct/raw16 (fine scans)
COARSE = Path(os.environ.get('GEOCOND_RAW_COARSE', 'raw_coarse'))   # coarse (14 um) miniplug scans
V_COARSE = 14.0

JOBS = {  # gid: (样品, 面板体素, 粗扫目录)
    "G01": ("ZCQ-1",   2.700016, "ZCQ-1-14"),
    "G04": ("ZSC-1",   2.499891, "ZSC-1-14"),
    "G03": ("ZYN-3",   2.499891, "ZYN-3-14"),
    "G05": ("ZSC-11",  2.500082, "ZSC-11-14"),
    "G06": ("ZSC-13",  2.500082, "ZSC-13-14"),
    "G07": ("ZSC-15",  2.500082, "ZSC-15-14"),
    "G08": ("ZSC-3",   2.499891, "ZSC-3-14"),
    "G09": ("ZSC-5",   2.499891, "ZSC-5-14"),
    "G10": ("ZSC-7",   2.499891, "ZSC-7-14"),
    "G11": ("ZSC-9",   2.500082, "ZSC-9-14"),
    "G12": ("ZSD-1",   2.699796, "ZSD-1-14"),
    "G13": ("ZSHX-1",  2.699796, "ZSHX-1-14"),
    "G14": ("ZSX-1",   2.700016, "ZSX-1-14"),
    "G15": ("ZSX-3",   2.700016, "ZSX-3-14"),
    "G16": ("ZYN-1",   2.499964, "ZYN-1-14"),
    "G17": ("ZYN-5",   2.499891, "ZYN-5-14"),
    "G18": ("ZCQ-2-1", 2.700016, "ZCQ-2-1-14"),
    "G19": ("ZCQ-3-1", 2.700016, "ZCQ-3-1-14"),
    "G20": ("ZFJ-1",   2.700016, "ZFJ-1-14"),
    "G02": ("ZGZ-1",   2.499964, "ZGZ-1-14"),
}


def rock_diameter_px(files, nz=9):
    """岩石直径（像素）。最外侧半高穿越 —— 护套灰度低于半高，自然被排除。"""
    zs = np.clip(np.linspace(len(files)*0.25, len(files)*0.75, nz).astype(int),
                 0, len(files)-1)
    ds = []
    for i in zs:
        a = tifffile.imread(str(files[i])).astype(np.float64)
        cor = 60
        bg = float(np.median(np.r_[a[:cor,:cor].ravel(), a[:cor,-cor:].ravel(),
                                   a[-cor:,:cor].ravel(), a[-cor:,-cor:].ravel()]))
        core = a[a.shape[0]//4:-a.shape[0]//4, a.shape[1]//4:-a.shape[1]//4]
        body = float(np.median(core))
        if body - bg < 0.02*(float(np.percentile(a,99.5))-bg):
            body = float(np.percentile(a, 99.5))
        m = ndimage.binary_fill_holes(ndimage.binary_closing(
            a > bg + 0.35*(body-bg), np.ones((9,9))))
        lab, n = ndimage.label(m)
        if n == 0: continue
        sz = ndimage.sum(m, lab, range(1, n+1))
        cy, cx = ndimage.center_of_mass(m, lab, [int(np.argmax(sz))+1])[0]
        yy, xx = np.indices(a.shape)
        r = np.hypot(yy-cy, xx-cx).ravel()
        rmax = float(min(cy, cx, a.shape[0]-cy, a.shape[1]-cx))
        nb = 600
        k = np.clip((r/rmax*nb).astype(int), 0, nb-1)
        c = np.bincount(k, minlength=nb); s = np.bincount(k, weights=a.ravel(), minlength=nb)
        p = np.where(c > 15, s/np.maximum(c,1), np.nan)
        rr = (np.arange(nb)+.5)/nb*rmax
        plateau = float(np.nanmedian(p[:int(.12*nb)]))
        half = .5*(plateau+bg)
        above = np.where(np.isfinite(p) & (p >= half))[0]
        above = above[above < int(.96*nb)]
        if above.size == 0: continue
        j = int(above[-1])
        if j+1 < nb and np.isfinite(p[j+1]) and p[j] != p[j+1]:
            t = (p[j]-half)/(p[j]-p[j+1]); rad = rr[j] + t*(rr[j+1]-rr[j])
        else: rad = rr[j]
        ds.append(2*rad)
    return float(np.median(ds)), float(np.std(ds))


def run(gid):
    t0 = time.time()
    sample, v_panel, cdir = JOBS[gid]
    cf = sorted((COARSE / cdir).glob("*.tif"))
    ff = sorted((PROJ / "dataset" / gid / "small_ct" / "raw16").glob("*.tif"))
    print(f"\n{'='*76}\n{gid} / {sample}   粗 {len(cf)} 张 / 细 {len(ff)} 张", flush=True)

    dC, sC = rock_diameter_px(cf)
    dF, sF = rock_diameter_px(ff)
    D_mm = dC * V_COARSE / 1000
    v_true = D_mm * 1000 / dF
    ratio = V_COARSE / v_true
    print(f"  岩石直径 粗 {dC:.1f}±{sC:.1f} px = {D_mm:.3f} mm   细 {dF:.1f}±{sF:.1f} px")
    print(f"  真实体素 {v_true:.4f} μm （面板 {v_panel:.4f}，差 "
          f"{(v_true/v_panel-1)*100:+.1f}%）  真实倍率 {ratio:.3f}", flush=True)

    half = min(1.9, (2006*v_true/1000)/2*0.93)
    cc = HERE / f"cache_{gid}_C.npy"; fc = HERE / f"cache_{gid}_F.npy"
    if cc.exists() and fc.exists():
        C = np.load(cc); F = np.load(fc)
        print(f"  用缓存  粗 {C.shape} 细 {F.shape}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        bC = physical_bbox(cf, V_COARSE, half)
        bF = physical_bbox(ff, v_true, half)
        C, _ = load_cropped(cf, bbox=bC); C, tc = destilt(C)
        F, _ = load_downsampled(ff, ratio, bbox=bF); F, tf = destilt(F)
        n = min(C.shape[1], C.shape[2], F.shape[1], F.shape[2])
        C = C[:, :n, :n]; F = F[:, :n, :n]
        np.save(cc, C.astype(np.float32)); np.save(fc, F.astype(np.float32))
        print(f"  裁框 ±{half:.2f} mm  粗 {C.shape} 细 {F.shape}  "
              f"倾角 {tc:.2f}°/{tf:.2f}°  ({time.time()-t0:.0f}s)", flush=True)

    mC, _ = interior(C); mF, _ = interior(F)
    msk = mC & mF
    Cp = prep(C[4:-4], msk); Fp = prep(F[3:-3], msk)

    # 比例尺由**配准自己**解出，不用直径法。
    # 直径法在 G04 上散布达 8%（G01 只有 0.8%），8% 比例误差 = 边缘错位 11 体素，
    # 足以摧毁配准 —— 实测 G04 因此失败（NCC 0.0138）。
    # 不必为每个比例重新块平均（6 分钟）：对已平均好的小体做 zoom 即可。
    def scaled(V, m):
        if abs(m - 1.0) < 1e-4:
            return V
        W = ndimage.zoom(V, (m, m, m), order=1)
        # 裁/补到原面内尺寸，保持圆心不动
        out = np.zeros_like(V)
        n0 = V.shape[1]; n1 = W.shape[1]
        if n1 >= n0:
            o = (n1 - n0) // 2
            k = min(V.shape[0], W.shape[0])
            out[:k] = W[:k, o:o+n0, o:o+n0]
        else:
            o = (n0 - n1) // 2
            k = min(V.shape[0], W.shape[0])
            out[:k, o:o+n1, o:o+n1] = W[:k]
        return out

    best = None
    for lvl, (down, step, arng, ms) in enumerate([
            (4, 3.0, None, np.arange(0.76, 1.261, 0.03)),
            (2, 1.0, 6.0,  None),
            (1, 0.4, 2.0,  None)]):
        Cd = Cp[:, ::down, ::down] if down > 1 else Cp
        rows = []
        mlist = ms if ms is not None else [best[5]]
        for m in mlist:
            Fm = scaled(Fp, m)
            Fd = Fm[:, ::down, ::down] if down > 1 else Fm
            angs = (np.arange(0, 360, step) if arng is None
                    else np.arange(best[1]-arng, best[1]+arng+1e-9, step))
            for ang in angs:
                for flip in ((False, True) if lvl == 0 else (best[3],)):
                    G = Fd[::-1] if flip else Fd
                    Gr = ndimage.rotate(G, ang, axes=(1, 2), reshape=False,
                                        order=1, mode="constant", prefilter=False)
                    r = ncc_vs_dz(Cd, Gr); k = int(np.argmax(r))
                    rows.append((float(r[k]), float(ang), k, flip, r, float(m)))
        rows.sort(key=lambda t: -t[0]); best = rows[0]
        if lvl == 0:
            print(f"  L1 比例扫描 -> m={best[5]:.3f} (倍率 {ratio*best[5]:.3f})"
                  f"  {best[1]:.0f}°  NCC {best[0]:.4f}", flush=True)
    ncc, ang, dz, flip, prof, mbest = best
    ratio_fin = ratio * mbest
    v_fin = V_COARSE / ratio_fin
    sig = (ncc - prof.mean())/(prof.std()+1e-12)
    verdict = "通过" if (ncc>0.20 and sig>5) else ("边缘" if ncc>0.12 else "失败")
    print(f"  -> {'翻转 ' if flip else ''}{ang:.2f}°  dz={dz} = {dz*V_COARSE/1000:.2f} mm"
          f"  NCC {ncc:.4f}  {sig:.1f}σ   >>> {verdict}  ({time.time()-t0:.0f}s)",
          flush=True)
    return dict(gid=gid, sample=sample, v_panel=v_panel,
                v_true=round(v_fin,4),
                dev_pct=round((v_fin/v_panel-1)*100,1),
                ratio=round(ratio_fin,3), m=round(mbest,3),
                D_mm=round(D_mm,3), flip=flip, angle=round(ang,2), dz=dz,
                z_mm=round(dz*V_COARSE/1000,2), ncc=round(ncc,4),
                sigma=round(float(sig),1), verdict=verdict, secs=int(time.time()-t0))


if __name__ == "__main__":
    gids = sys.argv[1:] or ["G04"]
    out = HERE / "pipeline_results.csv"
    rows = []
    if out.exists():
        rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    done = {r["gid"] for r in rows}
    for g in gids:
        if g in done:
            print(f"{g} 已完成，跳过"); continue
        try:
            rows.append(run(g))
        except Exception as e:
            print(f"  !! {g} 失败: {e}")
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            w.writerows(rows)
    print("\n" + "="*76)
    for r in rows:
        print(f"{r['gid']:<5}{str(r['sample']):<10} 真实体素 {r['v_true']} μm "
              f"({r['dev_pct']}%)  {r['angle']}°  dz {r['z_mm']} mm  "
              f"NCC {r['ncc']}  {r['verdict']}")
