"""测两个体各自的柱轴倾斜，并用「逐层质心归中」把它消掉。

此前所有配准只搜绕 z 的旋转，默认两次装夹都竖直。但两次上机相隔四个半月、
重新装夹，倾角必然不同。柱长 10 mm、倾角差 2°，两端就横向错开 0.35 mm = 25 体素，
足以摧毁任何刚体配准 —— 这很可能就是 G01 内点率只有 1% 的原因。

修法：逐层把试样质心平移到体中心。对近竖直圆柱，这同时消掉倾斜与横向漂移，
且不需要解三维旋转。代价是它不是严格刚体变换，但对本问题是良好近似。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for _f in (r"C:\Windows\Fonts\msyh.ttc",):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
VOX = 14.0


def centroids(v):
    """逐层试样质心（体素坐标）。未检出的层填 NaN。"""
    cor = 50
    cy = np.full(v.shape[0], np.nan)
    cx = np.full(v.shape[0], np.nan)
    for z in range(v.shape[0]):
        a = v[z]
        bg = float(np.median(np.r_[a[:cor, :cor].ravel(), a[:cor, -cor:].ravel(),
                                   a[-cor:, :cor].ravel(), a[-cor:, -cor:].ravel()]))
        body = float(np.median(a[a.shape[0]//3:-a.shape[0]//3,
                                 a.shape[1]//3:-a.shape[1]//3]))
        if body - bg < 30:
            continue
        m = a > bg + 0.5 * (body - bg)
        m = ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((5, 5))))
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        sz = ndimage.sum(m, lab, range(1, n + 1))
        if sz.max() < 0.02 * m.size:
            continue
        c = ndimage.center_of_mass(m, lab, [int(np.argmax(sz)) + 1])[0]
        cy[z], cx[z] = c
    return cy, cx


def fit_axis(cy, cx):
    z = np.arange(len(cy))
    ok = np.isfinite(cy) & np.isfinite(cx)
    # 只用中段，避开端部
    lo, hi = int(len(cy) * 0.15), int(len(cy) * 0.85)
    ok &= (z >= lo) & (z < hi)
    if ok.sum() < 20:
        return None
    py = np.polyfit(z[ok], cy[ok], 1)
    px = np.polyfit(z[ok], cx[ok], 1)
    tilt_deg = np.degrees(np.arctan(np.hypot(py[0], px[0])))
    return py, px, tilt_deg, ok


def report(gid):
    C = np.load(HERE / f"vol_{gid}_coarse.npy")[4:-4]
    F = np.load(HERE / f"vol_{gid}_fine14.npy")[3:-3]
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for k, (v, name) in enumerate([(C, "粗扫 14 μm"), (F, "细扫→14 μm")]):
        cy, cx = centroids(v)
        f = fit_axis(cy, cx)
        if f is None:
            print(f"{name}: 质心拟合失败"); continue
        py, px, tilt, ok = f
        z = np.arange(len(cy))
        span = (z[ok].max() - z[ok].min())
        drift = np.hypot(py[0], px[0]) * span
        print(f"{name}:  倾角 {tilt:.2f}°   中段 {span} 层 = {span*VOX/1000:.2f} mm 内"
              f"质心漂移 {drift:.1f} 体素 = {drift*VOX:.0f} μm")
        out[name] = (tilt, drift, py, px)
        ax[k].plot(z, cy - np.nanmean(cy), lw=1.1, label="y 质心")
        ax[k].plot(z, cx - np.nanmean(cx), lw=1.1, label="x 质心")
        ax[k].plot(z[ok], np.polyval(py, z[ok]) - np.nanmean(cy), "--", lw=1.4,
                   color="#d94f2b", label=f"拟合轴 {tilt:.2f}°")
        ax[k].plot(z[ok], np.polyval(px, z[ok]) - np.nanmean(cx), "--", lw=1.4,
                   color="#8a6a1c")
        ax[k].set_xlabel("层号"); ax[k].set_ylabel("质心偏移 (体素)")
        ax[k].legend(fontsize=8); ax[k].grid(alpha=.3)
        ax[k].set_title(f"{name}   倾角 {tilt:.2f}°", fontsize=10)
    names = list(out)
    if len(names) == 2:
        t1, t2 = out[names[0]][0], out[names[1]][0]
        d1, d2 = out[names[0]][1], out[names[1]][1]
        print(f"\n两者倾角差 {abs(t1-t2):.2f}°")
        print(f"若不校正，柱长范围内两端最大错位可达 {max(d1, d2):.0f} 体素 "
              f"= {max(d1,d2)*VOX:.0f} μm")
        print(f">>> {'必须校正' if max(d1,d2) > 5 else '影响不大'}"
              f"（判据：错位 > 5 体素即足以破坏点云匹配，因内点阈值是 2.5 体素）")
    fig.suptitle(f"{gid} 柱轴倾斜", fontsize=12)
    plt.tight_layout()
    p = HERE / f"tilt_{gid}.png"
    fig.savefig(p, dpi=112)
    print("图:", p)


if __name__ == "__main__":
    report(*(sys.argv[1:] or ["G01"]))
