"""可识别谱：配准好的真实对子，逐频率量「低分辨率里还剩多少真信息」。

记
  C(k)  真实的 14 μm 扫描
  B(k)  同一块岩石的高分辨率扫描块平均到 14 μm 网格（近似无噪的「真值」）

相干谱
      gamma^2(k) = |S_CB|^2 / (S_CC * S_BB)
直接就是「频率 k 处两次扫描共享的信息比例」：
  = 1  该尺度的结构在低分辨率里完整保留，超分只是去噪，必然可学；
  = 0  该尺度在低分辨率里与真值毫无关系，任何模型输出的都是编的。
gamma^2 掉到 0.5 的波长，就是这块岩石的**可识别深度**。

两个必须踩准的点
  1. 单块体算出来的 gamma^2 恒等于 1（自我相关），必须先在很多子块上分别求谱、
     把三个谱各自平均，再相除。
  2. gamma^2 要**逐 k 矢量**算完再做径向平均，不能先把 S_CB 沿球壳平均。
     残余整体错位 delta 给 S_CB 带上相位 exp(2 pi i k·delta)，同一球壳上不同方向
     的相位不同；先沿壳平均会让它们互相抵消，于是「配准差一点」被误读成
     「这个尺度的信息没了」。逐 k 算则整体平移只改相位不改模，自动免疫。
  3. N 个子块估出的 gamma^2 有 +(1-g2)/N 的正偏，按 (N*g2-1)/(N-1) 去偏。

gamma^2 对任何确定性线性滤波（插值、块平均、MTF）都不变，只反映信噪结构：
gamma^2 = 0.5 就是粗扫在该频率上信噪比等于 1 —— 经典的噪声极限分辨率判据。
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register3 import interior
import compose

V_COARSE = 14.0
BLK = 64
STEP = 32


def placed(gid, order=3):
    """按精配结果把细扫体摆到粗扫坐标里（合成成一次插值），返回 (C, B, 掩膜)。"""
    kw, dz, base, ref = compose.params(gid)
    C = np.load(HERE / f"cache_{gid}_C.npy")
    F = np.load(HERE / f"cache_{gid}_F.npy")
    mC, _ = interior(C)
    Cg = C[4:-4]; Fg = F[3:-3]
    db = json.loads((HERE / "affine_corr.json").read_text(encoding="utf-8"))
    if gid not in db:
        raise SystemExit(f"{gid} 还没做仿射改正（先跑 affinefit.py）")
    e = db[gid]
    import affinefit
    W = affinefit.place_corr(Fg, Cg.shape[1], order,
                             (np.array(e["Mc"]), np.array(e["oc"])), **kw)
    n = min(W.shape[0], Cg.shape[0] - dz)
    return Cg[dz:dz + n], W[:n], mC, ref


def blocks(shape, msk, n):
    """在试样内部铺满不重叠一半的子块。"""
    nz, ny, nx = n, shape[1], shape[2]
    out = []
    for z in range(0, nz - BLK + 1, STEP):
        for y in range(0, ny - BLK + 1, STEP):
            for x in range(0, nx - BLK + 1, STEP):
                if msk[y:y + BLK, x:x + BLK].all():
                    out.append((z, y, x))
    return out


def spectra(C, B, msk):
    win = np.hanning(BLK)
    W3 = win[:, None, None] * win[None, :, None] * win[None, None, :]
    locs = blocks(C.shape, msk, C.shape[0])
    if len(locs) < 8:
        raise SystemExit(f"可用子块只有 {len(locs)} 个，太少")
    Scc = Sbb = Scb = 0.0
    for z, y, x in locs:
        a = C[z:z+BLK, y:y+BLK, x:x+BLK].astype(np.float64)
        b = B[z:z+BLK, y:y+BLK, x:x+BLK].astype(np.float64)
        a = (a - a.mean()) * W3
        b = (b - b.mean()) * W3
        A = np.fft.fftn(a); Bf = np.fft.fftn(b)
        Scc = Scc + (A * A.conj()).real
        Sbb = Sbb + (Bf * Bf.conj()).real
        Scb = Scb + A * Bf.conj()
    return Scc / len(locs), Sbb / len(locs), Scb / len(locs), len(locs)


def radial(S, nbin=None):
    k = np.fft.fftfreq(BLK)
    KK = np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + k[None, None, :]**2)
    nbin = nbin or BLK // 2
    idx = np.clip((KK / 0.5 * nbin).astype(int), 0, nbin - 1)
    cnt = np.bincount(idx.ravel(), minlength=nbin)
    if np.iscomplexobj(S):
        re = np.bincount(idx.ravel(), weights=S.real.ravel(), minlength=nbin)
        im = np.bincount(idx.ravel(), weights=S.imag.ravel(), minlength=nbin)
        return (re + 1j * im) / np.maximum(cnt, 1)
    s = np.bincount(idx.ravel(), weights=S.ravel(), minlength=nbin)
    return s / np.maximum(cnt, 1)


def run(gid, order=3):
    C, B, msk, r = placed(gid, order)
    Scc, Sbb, Scb, nb = spectra(C, B, msk)
    g2v = np.abs(Scb)**2 / np.maximum(Scc * Sbb, 1e-30)      # 逐 k 矢量
    g2v = np.clip((nb * g2v - 1.0) / (nb - 1.0), 0.0, 1.0)    # 去偏
    g2 = radial(g2v)
    cc = radial(Scc); bb = radial(Sbb); cb = radial(Scb)
    nbin = len(g2)
    kk = (np.arange(nbin) + 0.5) / nbin * 0.5 / V_COARSE * 1000     # 1/mm
    lam = 1000.0 / np.maximum(kk, 1e-9)                             # 波长 μm
    # 可识别深度：gamma^2 跌破 0.5 的波长（线性插值）
    depth = np.nan
    for i in range(1, nbin):
        if g2[i] < 0.5 <= g2[i - 1]:
            t = (g2[i - 1] - 0.5) / (g2[i - 1] - g2[i])
            depth = lam[i - 1] + t * (lam[i] - lam[i - 1])
            break
    print(f"{gid}/{r['sample']}  子块 {nb} 个  体素 {r['v_true']} μm")
    print("   波长 μm : " + " ".join(f"{lam[i]:7.0f}" for i in (1, 2, 3, 4, 6, 8, 12, 16, 24, 31)))
    print("   gamma^2 : " + " ".join(f"{g2[i]:7.3f}" for i in (1, 2, 3, 4, 6, 8, 12, 16, 24, 31)))
    print(f"   >>> 可识别深度 (gamma^2=0.5) = {depth:.0f} μm"
          f"  = {depth/V_COARSE:.2f} 个粗体素 = {depth/float(r['v_true']):.1f} 个细体素")
    np.savez(HERE / f"spec_{gid}.npz", k=kk, lam=lam, g2=g2, Scc=cc, Sbb=bb,
             Scb=cb, nblocks=nb)
    return dict(gid=gid, sample=r["sample"], v_true=r["v_true"], nblocks=nb,
                depth_um=round(float(depth), 1),
                depth_coarse_vox=round(float(depth) / V_COARSE, 3),
                g2_at_2vox=round(float(np.interp(2 * V_COARSE, lam[::-1], g2[::-1])), 3),
                g2_at_4vox=round(float(np.interp(4 * V_COARSE, lam[::-1], g2[::-1])), 3))


if __name__ == "__main__":
    rows = []
    for g in sys.argv[1:]:
        try:
            rows.append(run(g))
        except Exception:
            import traceback; traceback.print_exc()
    if rows:
        out = HERE / "identifiability.csv"
        old = (list(csv.DictReader(out.open(encoding="utf-8-sig")))
               if out.exists() else [])
        keep = [x for x in old if x["gid"] not in {y["gid"] for y in rows}]
        allr = sorted(keep + [{k: str(v) for k, v in x.items()} for x in rows],
                      key=lambda x: x["gid"])
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(allr[0].keys()))
            w.writeheader(); w.writerows(allr)
        print("\n写入", out.name)
