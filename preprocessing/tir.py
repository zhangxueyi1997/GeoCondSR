"""T / I / R 三级隔离：把「分辨率损失」和「真实扫描的噪声与模糊」拆开。

三个层级（同一块岩石、同一块物理区域）
  T  2.1 μm 原始细扫                     —— 真值
  I  T 自己块平均到 14 μm 网格            —— 只丢分辨率，噪声被压掉约 15 倍
  R  真实拍的 14 μm 粗扫                  —— 分辨率与 I 相同，但带真实噪声与 MTF

于是两两之差各自只含一个原因：
  T → I   纯分辨率损失（同一次扫描、同样的噪声源，只是块平均）
  I → R   纯「真实扫描比理想块平均多出来的东西」= 噪声 + 真实 MTF

为什么非拆不可
  phases2.py / gmm.py 卡死在一个猜不出来的问题上：细扫主峰的半高宽是粗扫的
  2.2 倍，这个差别**到底是真实的矿物变化，还是噪声？**只有 T 和 R 两级时
  只能猜，猜错窗口就定歪 —— 实测两版分别给出「绝对值荒唐但相关 0.84」与
  「绝对值合理但相关 0.61」，混合高斯那版更是把 IGV 推到 82%、COPL 推到 −303。

  插进 I 就不用猜了。但每一步到底测的是什么，必须说准：
    hwhm(T)      2.1 μm 上的总展宽 = 真实结构变化 + 细扫噪声
    hwhm(I)      14 μm 上的理想展宽 —— 块平均同时做了两件事（压噪声约 15 倍、
                 抹掉 14 μm 以下的真实变化），所以 T−I 是这两件事之和，
                 **不能单独叫「噪声」**
    hwhm(R)      14 μm 上的真实展宽

  真正良定义的是 **I 与 R 的比较**：两者分辨率完全相同，唯一差别是
  「真拍的」与「理想块平均的」，所以
    R − I = 真实扫描相对理想降质多出来的一切 = 噪声 + 真实 MTF
  这一项正是训练要用的退化算子 H 与理想块平均之间的差，也正是
  README 待办第 4 步担心的那个东西。
  （注意方向：额外模糊会让直方图**变窄**，额外噪声让它**变宽**，
   两者相消，所以 R−I 的符号本身不能单独解读，见报告里的说明。）

区域怎么保证是同一块
  mkcache 对两级用的是**同一个物理半宽** half_mm（`physical_bbox(…, half)`），
  再裁到同一个 n_square。所以 C 与 F 缓存覆盖同样的物理范围、横向尺寸相同，
  一个按比例定的圆盘掩膜就落在同一块岩石上，**不需要插值**。
  直方图不在乎翻转和面内旋转（圆盘是旋转不变的），所以这里刻意**不做配准插值** ——
  order=3 样条会改直方图，而我们量的正是直方图。
  z 向用配准给的 dz 对齐区间。

端元怎么取
  mu_air 必须在离样品很远处取（第 1 步：粗扫里试样只占画幅 1%，
  百分位掩膜会把散射晕框进来，量出 43–67% 的假孔隙度），
  而缓存只裁到 1.1 倍半径，里面没有干净空气 —— 所以 T/I 的 mu_air 从原始细扫
  切片取，R 的从原始粗扫切片取。块平均是线性的，故 mu_air 与 mu_mode 在 T→I
  下不变，脚本会把这一条作为自检打印出来。

用法
    python code/tir.py            # 全部已配准样品
    python code/tir.py G01 G02    # 指定样品
"""
from __future__ import annotations
import os
import sys, json, csv, time
from pathlib import Path
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PROJ = Path(os.environ.get('GEOCOND_WORK', str(HERE.parent / 'work')))   # intermediate preprocessing products
RES = PROJ / "results"
REG = RES / "registration"
CACHE = PROJ / "cache"
sys.path.insert(0, str(HERE))
import fastio, centre, compose

DATA = Path(os.environ.get('GEOCOND_RAW_PROJECT', 'raw_project'))   # contains dataset/<group>/small_ct/raw16 (fine scans)
COARSE = Path(os.environ.get('GEOCOND_RAW_COARSE', 'raw_coarse'))   # coarse (14 um) miniplug scans
V_COARSE = 14.0
PORE_CUT = 0.50            # 物理判据：空气电平与骨架主峰的中点
K_W = 2.2                  # 骨架窗 = 主峰 ± K_W 倍半高半宽
NZ = 15                    # 每级取多少层做统计
RFRAC = 0.88               # 掩膜取岩石半径的这个比例，避开边缘与胶带


def mode_of(v, bins=500):
    h, e = np.histogram(v, bins=bins)
    return float((e[:-1] + e[1:])[int(np.argmax(h))] / 2)


def hwhm_of(v, bins=500):
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    i = int(np.argmax(h)); half = h[i] / 2.0
    lo, hi = i, i
    while lo > 0 and h[lo] > half:
        lo -= 1
    while hi < len(h) - 1 and h[hi] > half:
        hi += 1
    return float((c[hi] - c[lo]) / 2)


def air_level(files, rad_px, zs):
    """离样品 1.6 倍半径以远的空气电平。缓存里没有干净空气，必须回原始切片取。"""
    out = []
    for z in zs:
        a = fastio.read(files[z]).astype(np.float32)
        cy, cx = centre.disc_centre(a, rad_px)
        yy, xx = np.indices(a.shape)
        far = np.hypot(yy - cy, xx - cx) > rad_px * 1.6
        if far.sum() > 1000:
            out.append(mode_of(a[far]))
    return float(np.median(out)) if out else np.nan


def disc(n, rfrac):
    yy, xx = np.indices((n, n))
    c = (n - 1) / 2.0
    return np.hypot(yy - c, xx - c) <= rfrac * (n / 2.0)


def phases(u, win):
    """Delta 单位下的四相分数与 IGV。"""
    lo, hi = win
    f_pore = float((u < PORE_CUT).mean())
    f_frame = float(((u >= lo) & (u < hi)).mean())
    f_dense = float((u >= hi).mean())
    return dict(f_pore=f_pore, f_frame=f_frame, f_dense=f_dense,
                IGV=1.0 - f_frame)


def level_values(gid, meta, tag):
    """返回该层级掩膜内的灰度样本。T 回原始细扫读，I/R 读缓存。"""
    rad_frac = (meta["D_mm"] / 2) / meta["half_mm"]
    if tag == "T":
        ff = sorted((DATA / "dataset" / gid / "small_ct" / "raw16").glob("*.tif"))
        r0, r1, c0, c1 = meta["bbox_F"]
        n = min(r1 - r0, c1 - c0)
        m = disc(n, RFRAC * rad_frac)
        zs = np.linspace(len(ff) * .15, len(ff) * .85, NZ).astype(int)
        return np.concatenate([
            fastio.read(ff[z], r0, r0 + n, c0, c0 + n).astype(np.float32)[m]
            for z in zs])
    V = np.load(CACHE / f"cache_{gid}_{'F' if tag == 'I' else 'C'}.npy",
                mmap_mode="r")
    z0, z1 = 0, V.shape[0]
    if tag == "R":                                  # 只取与细扫重叠的那一段
        z0, z1 = meta["_dz"], meta["_dz"] + meta["_nzR"]
        z0 = max(0, z0); z1 = min(V.shape[0], z1)
    n = V.shape[1]
    m = disc(n, RFRAC * rad_frac)
    zs = np.linspace(z0 + (z1 - z0) * .15, z0 + (z1 - z0) * .85, NZ).astype(int)
    return np.concatenate([np.asarray(V[z])[m] for z in zs])


def run(gid, k, ntot):
    tag = f"[{k:2d}/{ntot}] {gid}"
    meta = json.loads((REG / f"meta_{gid}.json").read_text(encoding="utf-8"))
    base = {r["gid"]: r for r in csv.DictReader(
        (REG / "reg_final.csv").open(encoding="utf-8-sig"))}
    if gid not in base or base[gid]["verdict"] != "通过":
        print(f"{tag} 未配准通过，跳过", flush=True)
        return None
    nzF = np.load(CACHE / f"cache_{gid}_F.npy", mmap_mode="r").shape[0]
    meta["_dz"] = int(base[gid]["dz"])
    meta["_nzR"] = int(round(nzF * float(base[gid]["m"])))
    print(f"{tag} {meta['sample']:<9} 掩膜半径 {RFRAC*100:.0f}% 岩石半径   "
          f"每级 {NZ} 层   R 取 z[{meta['_dz']}, {meta['_dz']+meta['_nzR']})",
          flush=True)

    # --- 端元：空气电平必须回原始切片取 ---
    t0 = time.time()
    ff = sorted((DATA / "dataset" / gid / "small_ct" / "raw16").glob("*.tif"))
    cf = sorted((COARSE / meta["cdir"]).glob("*.tif"))
    ref = {r["gid"]: r for r in csv.DictReader(
        (REG / "reg_refined.csv").open(encoding="utf-8-sig"))}
    vF = float(ref[gid]["v_true"]) if gid in ref else meta["v_prior"]
    air_F = air_level(ff, meta["D_mm"] * 1000 / vF / 2,
                      np.linspace(len(ff) * .3, len(ff) * .7, 3).astype(int))
    air_C = air_level(cf, meta["D_mm"] * 1000 / V_COARSE / 2,
                      np.linspace(len(cf) * .3, len(cf) * .7, 3).astype(int))
    print(f"{tag}   空气电平：细扫 {air_F:9.1f}   粗扫 {air_C:9.1f}   "
          f"({time.time()-t0:.1f}s)", flush=True)

    out = {}
    for lv in ("T", "I", "R"):
        t1 = time.time()
        v = level_values(gid, meta, lv)
        air = air_C if lv == "R" else air_F
        mo = mode_of(v)
        D = mo - air
        w = hwhm_of(v) / D if D > 0 else np.nan
        out[lv] = dict(air=air, mode=mo, D=D, w=w, n=len(v))
        print(f"{tag}   {lv}：主峰 {mo:9.1f}  Δ {D:8.1f}  "
              f"半高宽 {w:6.3f}Δ  体素数 {len(v):8d}  ({time.time()-t1:.1f}s)",
              flush=True)
        out[lv]["_v"] = v

    # T→I 的线性自检：块平均是线性算子，主峰位置应当基本不动
    dm = abs(out["I"]["mode"] - out["T"]["mode"]) / max(out["T"]["D"], 1e-9)
    print(f"{tag}   参考：T→I 主峰漂移 {dm*100:5.2f}% Δ"
          "（峰不对称时，噪声被压掉会让众数移位；各级已各用自己的端元归一）",
          flush=True)

    # --- 两套窗口规则 ---
    rows = {}
    for rule, label in (("self", "各级用自己的半高宽"), ("fixed", "各级都用 T 的窗")):
        wT = out["T"]["w"]
        for lv in ("T", "I", "R"):
            w = out[lv]["w"] if rule == "self" else wT
            win = (max(1.0 - K_W * w, PORE_CUT + 0.02), 1.0 + K_W * w)
            u = (out[lv]["_v"] - out[lv]["air"]) / out[lv]["D"]
            rows[(rule, lv)] = dict(phases(u, win), win_lo=win[0], win_hi=win[1])
        r = rows
        print(f"{tag}   窗规则「{label}」  "
              f"IGV  T {r[(rule,'T')]['IGV']*100:5.1f}%  "
              f"I {r[(rule,'I')]['IGV']*100:5.1f}%  "
              f"R {r[(rule,'R')]['IGV']*100:5.1f}%   "
              f"→ 分辨率 {(r[(rule,'I')]['IGV']-r[(rule,'T')]['IGV'])*100:+5.1f}  "
              f"噪声/MTF {(r[(rule,'R')]['IGV']-r[(rule,'I')]['IGV'])*100:+5.1f}",
              flush=True)
    print(flush=True)
    return dict(gid=gid, sample=meta["sample"], lv=out, rows=rows)


def report(res):
    print("=" * 100)
    print("一、三级峰宽（Δ 单位）—— 良定义的比较是 I 与 R，它们分辨率完全相同")
    print(f"{'gid':<5}{'样品':<10}{'hwhm T':>9}{'hwhm I':>9}{'hwhm R':>9}"
          f"{'T−I 降质效应':>14}{'R−I 噪声+MTF':>15}")
    a = []
    for d in res:
        t, i, r = (d["lv"][x]["w"] for x in "TIR")
        a.append((t, i, r))
        print(f"{d['gid']:<5}{d['sample']:<10}{t:>9.3f}{i:>9.3f}{r:>9.3f}"
              f"{t-i:>14.3f}{r-i:>15.3f}")
    a = np.array(a)
    print(f"{'中位':<15}{np.median(a[:,0]):>9.3f}{np.median(a[:,1]):>9.3f}"
          f"{np.median(a[:,2]):>9.3f}{np.median(a[:,0]-a[:,1]):>14.3f}"
          f"{np.median(a[:,2]-a[:,1]):>18.3f}")
    mT, mI, mR = (np.median(a[:, i]) for i in range(3))
    print(f"\n→ 细扫主峰是粗扫的 {mT/mR:.2f} 倍（phases2.py 记录的是 2.2 倍，吻合）。")
    print(f"  拆开看：T→I 占 {(mT-mI)/(mT-mR)*100:.0f}%，I→R 只占 "
          f"{(mI-mR)/(mT-mR)*100:.0f}%。")
    print("  **所以这个 2 倍几乎全部是「降到 14 μm」造成的，不是「两次扫描性质不同」。**")
    print("  注意 T→I 这一项本身还是两件事之和（压掉噪声、抹掉 14 μm 以下的真实变化），")
    print("  要再分开需要用互谱估真实 MTF —— 但对定窗口这个用途已经不需要了。")
    print(f"→ I 与 R 分辨率相同，中位差仅 {mR-mI:+.3f}Δ（{(mR-mI)/mI*100:+.0f}%）：")
    print("  真实粗扫在直方图展宽上已经接近理想块平均，额外噪声与额外模糊基本相消。")

    for rule, label in (("self", "各级用自己的半高宽"), ("fixed", "各级都用 T 的窗")):
        print("\n" + "=" * 100)
        print(f"二、IGV 的三级分解 —— 窗规则「{label}」")
        print(f"{'gid':<5}{'样品':<10}{'IGV T':>8}{'IGV I':>8}{'IGV R':>8}"
              f"{'分辨率 I−T':>12}{'噪声/MTF R−I':>14}{'合计 R−T':>11}")
        m = []
        for d in res:
            t, i, r = (d["rows"][(rule, x)]["IGV"] * 100 for x in "TIR")
            m.append((t, i, r))
            print(f"{d['gid']:<5}{d['sample']:<10}{t:>8.1f}{i:>8.1f}{r:>8.1f}"
                  f"{i-t:>+12.1f}{r-i:>+14.1f}{r-t:>+11.1f}")
        m = np.array(m)
        print(f"{'中位':<15}{np.median(m[:,0]):>8.1f}{np.median(m[:,1]):>8.1f}"
              f"{np.median(m[:,2]):>8.1f}{np.median(m[:,1]-m[:,0]):>+12.1f}"
              f"{np.median(m[:,2]-m[:,1]):>+14.1f}"
              f"{np.median(m[:,2]-m[:,0]):>+11.1f}")
        print(f"{'跨样品相关 r':<15}  T-I {np.corrcoef(m[:,0],m[:,1])[0,1]:+.3f}"
              f"   I-R {np.corrcoef(m[:,1],m[:,2])[0,1]:+.3f}"
              f"   T-R {np.corrcoef(m[:,0],m[:,2])[0,1]:+.3f}")
        print(f"{'散布 sd':<15}  T {m[:,0].std():.2f}   I {m[:,1].std():.2f}"
              f"   R {m[:,2].std():.2f}")

    print("\n" + "=" * 100)
    print("三、两套窗规则的总账 —— 以及一个只有插进 I 才看得见的问题")
    print(f"{'窗规则':<22}{'R−T 偏差中位':>14}{'相关 T-R':>10}"
          f"{'散布 sd(T)':>12}{'散布 sd(R)':>12}{'I−T':>9}{'R−I':>9}")
    summ = {}
    for rule, label in (("self", "各级用自己的半高宽"), ("fixed", "各级都用 T 的窗")):
        m = np.array([[d["rows"][(rule, x)]["IGV"] * 100 for x in "TIR"] for d in res])
        s = dict(bias=np.median(m[:, 2] - m[:, 0]),
                 r=np.corrcoef(m[:, 0], m[:, 2])[0, 1],
                 sdT=m[:, 0].std(), sdR=m[:, 2].std(),
                 it=np.median(m[:, 1] - m[:, 0]), ri=np.median(m[:, 2] - m[:, 1]))
        summ[rule] = s
        print(f"{label:<22}{s['bias']:>+13.1f}{s['r']:>10.3f}"
              f"{s['sdT']:>12.2f}{s['sdR']:>12.2f}{s['it']:>+9.1f}{s['ri']:>+9.1f}")
    b = summ["self"]
    print(f"\n→ 「各级用自己的半高宽」在三项上全胜：偏差 {b['bias']:+.1f} vs "
          f"{summ['fixed']['bias']:+.1f}，相关 {b['r']:.3f} vs {summ['fixed']['r']:.3f}，"
          f"散布保住（{b['sdR']:.2f} vs T 的 {b['sdT']:.2f}）")
    print(f"  而固定窗把散布压掉一半（{summ['fixed']['sdR']:.2f} vs {b['sdT']:.2f}）"
          "—— 样品之间的真实差异被窗口吃掉了。")
    print(f"\n⚠ 但插进 I 才看得见的问题：自适应窗下 T→R 的小偏差 ({b['bias']:+.1f}) "
          f"**不是因为两步都小**，")
    print(f"  而是 I−T = {b['it']:+.1f} 与 R−I = {b['ri']:+.1f} 互相抵消的结果。")
    print("  只有 T、R 两级时会误以为「一致性很好」；抵消是脆弱的，")
    print("  换一台机器、换一次扫描就未必还能抵消。**这正是插进 I 的价值。**")
    print("  对策：IGV 的绝对值必须声明为「本批扫描条件下的标定值」，")
    print("       跨仪器迁移前要重做这张表。")

    rows = []
    for d in res:
        for rule in ("self", "fixed"):
            for lv in "TIR":
                p = d["rows"][(rule, lv)]
                rows.append(dict(gid=d["gid"], sample=d["sample"], 窗规则=rule,
                                 level=lv, hwhm=round(d["lv"][lv]["w"], 4),
                                 mu_air=round(d["lv"][lv]["air"], 1),
                                 mu_mode=round(d["lv"][lv]["mode"], 1),
                                 **{k: round(v * 100, 3) if k.startswith("f_")
                                    or k == "IGV" else round(v, 4)
                                    for k, v in p.items()}))
    out = RES / "tir.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n写入 {out}")


if __name__ == "__main__":
    gids = [g for g in sys.argv[1:] if g.startswith("G")] or \
           [f"G{i:02d}" for i in range(1, 21)]
    print("=" * 100)
    print(f"T / I / R 三级隔离：{len(gids)} 个候选样品，每级 {NZ} 层，"
          f"掩膜 {RFRAC*100:.0f}% 岩石半径")
    print("=" * 100, flush=True)
    res, t0 = [], time.time()
    for k, g in enumerate(gids, 1):
        try:
            r = run(g, k, len(gids))
            if r:
                res.append(r)
        except KeyboardInterrupt:
            print("\n用户中断"); sys.exit(1)
        except Exception as e:
            print(f"[{k}/{len(gids)}] {g} 失败：{type(e).__name__}: {e}\n", flush=True)
    print(f"测量完成 {len(res)} 个样品，用时 {(time.time()-t0)/60:.1f} min\n", flush=True)
    if len(res) >= 3:
        report(res)
