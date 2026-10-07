"""只造缓存体，不做搜索。搜索改由 psearch.py 负责（快两个数量级）。"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loadvol import load_cropped, load_downsampled, physical_bbox
from destilt2 import destilt   # 旋转版；旧的逐层平移版是剪切，会留下假的各向异性
from pipeline import JOBS, rock_diameter_px, PROJ, COARSE, V_COARSE

for gid in sys.argv[1:]:
    cc = HERE / f"cache_{gid}_C.npy"; fc = HERE / f"cache_{gid}_F.npy"
    if cc.exists() and fc.exists():
        print(f"{gid} 已有缓存，跳过", flush=True); continue
    t0 = time.time()
    try:
        sample, v_panel, cdir = JOBS[gid]
        cf = sorted((COARSE / cdir).glob("*.tif"))
        ff = sorted((PROJ / "dataset" / gid / "small_ct" / "raw16").glob("*.tif"))
        dC, sC = rock_diameter_px(cf); dF, sF = rock_diameter_px(ff)
        D_mm = dC * V_COARSE / 1000
        v_diam = D_mm * 1000 / dF                     # 直径比给的先验（脆弱）
        # 直径法在圆柱不圆、边缘有胶带或光晕时会整段失灵（实测 G08 细扫散布
        # ±2181 px、G10 ±148 px），先验一错缓存就按错误分辨率块平均掉了，
        # 之后再怎么搜也补不回来。改用与样品无关的固定先验：
        # 已配准样品的真实体素一律是面板值的 0.75-0.84 倍，取 0.80 居中，
        # 搜索范围 m∈[0.86,1.16] 足以覆盖。比例尺最终由配准自己定。
        v_true = 0.80 * v_panel
        ratio = V_COARSE / v_true
        # 裁框只比岩石大一圈就够。粗扫每张 8 MB 却只用中间一小块，
        # 框越小读盘越少 —— 这是整条流水线的瓶颈。
        rad_mm = D_mm / 2
        half = min(1.9, (2006 * v_true / 1000) / 2 * 0.93, max(1.15, rad_mm * 1.10))
        bC = physical_bbox(cf, V_COARSE, half); bF = physical_bbox(ff, v_true, half)
        C, _ = load_cropped(cf, bbox=bC); shp_C0 = C.shape; C, tc, ftC = destilt(C, True, order=3)
        F, _ = load_downsampled(ff, ratio, bbox=bF); shp_F0 = F.shape
        F, tf, ftF = destilt(F, True, order=3)
        n = min(C.shape[1], C.shape[2], F.shape[1], F.shape[2])
        C = C[:, :n, :n]; F = F[:, :n, :n]
        np.save(cc, C.astype(np.float32)); np.save(fc, F.astype(np.float32))
        # 导出全分辨率配对时需要把缓存坐标还原回原始切片坐标
        (HERE / f"meta_{gid}.json").write_text(json.dumps(dict(
            gid=gid, sample=sample, v_panel=v_panel, cdir=cdir,
            n_coarse=len(cf), n_fine=len(ff), d_coarse_px=dC, d_fine_px=dF,
            D_mm=D_mm, v_from_diam=v_diam, v_prior=v_true, ratio=ratio, f_int=max(1, int(round(ratio))),
            half_mm=half, bbox_C=[int(x) for x in bC], bbox_F=[int(x) for x in bF],
            shape_C0=list(shp_C0), shape_F0=list(shp_F0), n_square=int(n),
            destilt_C=ftC, destilt_F=ftF, tilt_C=float(tc), tilt_F=float(tf),
        ), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{gid}/{sample}  直径 粗 {dC:.1f}±{sC:.1f} 细 {dF:.1f}±{sF:.1f}  "
              f"直径法 v={v_diam:.4f} 先验 v={v_true:.4f}μm "
              f"(面板 {v_panel:.4f})  倍率 {ratio:.3f}  "
              f"粗{C.shape} 细{F.shape}  倾角 {tc:.2f}/{tf:.2f}  ({time.time()-t0:.0f}s)",
              flush=True)
    except Exception as e:
        print(f"{gid} 出错: {e}", flush=True)
