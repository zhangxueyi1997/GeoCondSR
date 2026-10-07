"""把实测的 MTF 曲线做成训练可用的三维卷积核 H。

为什么需要
  README §5 的损失里有一项**退化一致性**：把网络输出降回 14 μm 应还原输入。
  第 5 步实测出了退化算子，但交付的是**径向平均的 MTF 曲线**，
  训练里没法直接用 —— 需要一个能作用在体上的算子。

  注意这里的 H 指的是「真实扫描相对**理想块平均**多出来的那部分模糊」。
  完整的退化链是：2 μm 体 → 块平均到 14 μm → 卷 H → 加噪。
  块平均那一步是确定的，不用估；H 是仪器的份额。

怎么做
  1. 取各样品 MTF 的中位曲线（逐样品的也一并存，跨仪器时要重估）
  2. 实测只到奈奎斯特 |k|=0.5，而三维 k 立方的角上 |k|=0.866，
     超出部分按**在测量带上段拟合的 log MTF ~ k² 斜率**外推，并夹到 [0,1]
  3. 在 N³ 的 k 网格上按 |k| 插值出各向同性的 H(k)，逆变换得实空间核
  4. 截到 SUPP³ 的支撑，重新归一使和为 1（保证平均衰减不变 = 直流增益为 1）

自检（不是自证）
  把核作用到 I（真值块平均到 14 μm 网格），再与 R（真实粗扫）重算互谱。
  若 H 估对了，**残余 MTF 应当接近 1** —— 即卷完之后 I 已经和 R 一样糊。
  直接比「卷完的谱 vs 实测 MTF」是循环论证，那样测不出对错。

输出 results/degradation/H_kernel.npz
    kernel   (SUPP, SUPP, SUPP) float32，和为 1
    mtf_med / k      中位 MTF 曲线与对应频率（周/粗体素）
    mtf_each / gids  逐样品曲线
    nps_med          噪声功率谱（径向），供需要时按谱注噪
"""
from __future__ import annotations
import os
import sys, json, csv, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = Path(os.environ.get('GEOCOND_WORK', str(HERE.parent / 'work')))   # intermediate preprocessing products
RES = PROJ / "results"
REG = RES / "registration"
OUT = RES / "degradation"
sys.path.insert(0, str(HERE))
import spectrum

V_COARSE = 14.0
NGRID = 48                  # 造核用的 k 网格边长
SUPP = 9                    # 核的支撑（粗体素），奇数


def median_mtf():
    """各样品 MTF 的中位曲线。"""
    fs = sorted(REG.glob("mtf_G*.npz"))
    if not fs:
        raise SystemExit("没有 mtf_G??.npz，先跑 code/mtf.py")
    gids, curves, nps = [], [], []
    k = None
    for f in fs:
        d = np.load(f)
        gids.append(f.stem.split("_")[1])
        curves.append(d["mtf"])
        nps.append(d["Snn"] / max(d["Snn"].sum(), 1e-30))
        k = d["k"]
    C = np.array(curves)
    return k, np.median(C, 0), C, np.array(gids), np.median(np.array(nps), 0)


def extend(k, m):
    """把 MTF 外推到 |k| = 0.866（三维 k 立方的角）。"""
    sel = (k > k.max() * 0.5) & (m > 1e-3)
    if sel.sum() >= 3:
        # log m = a + b k²，用测量带上段拟合，取斜率外推
        A = np.c_[np.ones(sel.sum()), k[sel] ** 2]
        a, b = np.linalg.lstsq(A, np.log(m[sel]), rcond=None)[0]
    else:
        a, b = 0.0, -10.0
    kk = np.linspace(0, 0.9, 400)
    out = np.interp(kk, k, m, left=m[0], right=np.nan)
    far = ~np.isfinite(out)
    out[far] = np.exp(a + b * kk[far] ** 2)
    return kk, np.clip(out, 0.0, 1.0)


def build(kk, mm, n=NGRID, supp=SUPP):
    f = np.fft.fftfreq(n)
    K = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2
                + f[None, None, :] ** 2)
    H = np.interp(K.ravel(), kk, mm).reshape(K.shape)
    ker = np.real(np.fft.ifftn(H))
    ker = np.fft.fftshift(ker)
    c = n // 2
    h = supp // 2
    K3 = ker[c - h:c + h + 1, c - h:c + h + 1, c - h:c + h + 1].copy()
    energy = float(K3.sum() / ker.sum())          # 截断丢了多少
    K3 /= K3.sum()                                # 直流增益归 1
    return K3.astype(np.float32), energy


def verify(gid, K3):
    """把核作用到 I，再与 R 重算互谱：残余 MTF 应接近 1。"""
    from scipy import ndimage
    R, I, msk, ref = spectrum.placed(gid, order=3)
    J = ndimage.convolve(I.astype(np.float32), K3, mode="nearest")
    out = {}
    for tag, B in (("卷之前", I), ("卷之后", J)):
        Srr, Sbb, Srb, nb = spectrum.spectra(R, B, msk)
        H2 = np.maximum(np.abs(Srb) ** 2 - Srr * Sbb / nb, 0.0) \
            / np.maximum(Sbb ** 2, 1e-30)
        Hm = spectrum.radial(np.sqrt(H2))
        lo = float(np.median(Hm[1:5]))
        out[tag] = Hm / max(lo, 1e-30)
    return out, ref


if __name__ == "__main__":
    k, med, C, gids, nps = median_mtf()
    kk, mm = extend(k, med)
    K3, energy = build(kk, mm)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "H_kernel.npz", kernel=K3, k=k, mtf_med=med, mtf_each=C,
             gids=gids, nps_med=nps, k_ext=kk, mtf_ext=mm,
             supp=SUPP, v_coarse=V_COARSE)
    print("=" * 92)
    print(f"退化算子 H：{len(gids)} 个样品的中位 MTF → {SUPP}³ 三维核")
    print("=" * 92)
    idx = [i for i in (1, 2, 4, 8, 16, 24, 31) if i < len(k)]
    print("波长 μm : " + " ".join(f"{V_COARSE/k[i]:7.0f}" for i in idx))
    print("中位 MTF: " + " ".join(f"{med[i]:7.3f}" for i in idx))
    print(f"\n截断到 {SUPP}³ 保留了 {energy*100:.1f}% 的核能量")
    print(f"核中心权重 {K3[SUPP//2, SUPP//2, SUPP//2]:.4f}，"
          f"最负权重 {K3.min():+.5f}（负瓣来自 MTF 的非高斯滚降，正常）")
    print(f"核的和 {K3.sum():.6f}（应为 1：直流增益不变，平均衰减不被改动）")

    print("\n" + "=" * 92)
    print("自检：把核作用到 I，再与真实 R 重算互谱 —— 残余 MTF 应向 1 靠拢")
    print(f"{'gid':<5}{'样品':<10}{'':<6}" + "".join(f"{V_COARSE/k[i]:>9.0f}μm"
                                                    for i in idx))
    ok = 0
    for g in gids[:4]:
        try:
            o, ref = verify(g, K3)
            for tag in ("卷之前", "卷之后"):
                print(f"{g if tag=='卷之前' else '':<5}"
                      f"{ref['sample'] if tag=='卷之前' else '':<10}{tag:<6}"
                      + "".join(f"{o[tag][i]:>10.3f}" for i in idx))
            a = np.mean([abs(o["卷之前"][i] - 1) for i in idx])
            b = np.mean([abs(o["卷之后"][i] - 1) for i in idx])
            print(f"{'':<21}偏离 1 的平均幅度：{a:.3f} → {b:.3f}"
                  + ("   ✅ 变好" if b < a else "   ❌ 没变好"))
            ok += int(b < a)
        except Exception as e:
            print(f"{g} 自检失败 {type(e).__name__}: {e}")
    print(f"\n{ok}/4 个样品自检通过。")
    print(f"写入 {OUT / 'H_kernel.npz'}")
    print("\n用法：完整退化链 = 2 μm 体 → 块平均到 14 μm → 卷 kernel → 按 nps 注噪。")
    print("     块平均那一步是确定的不用估；kernel 只是仪器相对理想块平均多出的模糊。")
