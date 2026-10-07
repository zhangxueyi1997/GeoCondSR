"""训练成因瓶颈超分网络。默认跑 GPU 1。

三条按需求做的事
  1. **高频输出模型内容**：每 LOG_EVERY 步往 JSONL 写一行（所有损失分量、
     ĉ 逐变量的预测 vs 真值、各模块梯度范数、显存、吞吐、时间戳），
     每 DUMP_EVERY 步把真实的体数据存成 npz（lr / hr / 预测 / 三段分解 /
     c_true / c_pred）—— 以后画图直接读，不用重跑。
  2. **断点续跑**：每 CKPT_EVERY 步原子写一次检查点（模型 + 优化器 + 步数 +
     全部随机数状态）。`--resume auto` 自动找最新的接着跑。
     原子写 = 先写 .tmp 再 rename，**保存到一半掉电也不会毁掉旧检查点**。
  3. **死机可查**：每步更新心跳文件（时间戳 + 步数 + 显存），
     异常连完整 traceback 一起写进日志。事后一看就知道死在哪一步、什么状态。

用法
    python train.py --root data --fold SC --out runs/spade_SC
    python train.py ... --resume auto            # 接着上次跑
    python train.py ... --free                   # 消融：换 64 维自由特征
    python train.py ... --gpu 0
"""
from __future__ import annotations
import argparse, json, os, random, signal, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import numpy as _np
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PairDataset, folds, load_H, VAR_W, PHI_IDX7
from models import PORE_CUT
from models import (GenesisSR, bands, degrade, lowpass, genesis_phi, genesis_phi_arr, ARR_SCALE2,
                    chi_soft, thick_soft, chi_local,
                    CUT1_UM, CUT2_UM, N_C, GIVEN_MODES, ARR_MODES, ARR_SCALE,
                    BOTH_MODES,
                    Discriminator, hinge_d, hinge_g)

C_NAMES = ["IGV", "f_pore", "mu_inter", "f_dense"]
# |Φ(真值块) − c| 的实测口径差（第 12 步 §四闸一，全项目 560 块）。死区用它当 δ。
GC_FLOOR = [0.0295, 0.0179, 0.0288, 0.0210]
STOP = {"flag": False}


def _sig(*_):
    STOP["flag"] = True
    print("\n收到中断信号：存完检查点再退出（不要再按一次）", flush=True)


# ------------------------------------------------------------------ 损失

@torch.amp.autocast("cuda", enabled=False)
def radial_power(x, nbin=16):
    """径向平均的对数功率。<15 μm 带只匹配它，不做逐点损失。

    ⚠ 与 models.lowpass 同理，必须 fp32：cuFFT 半精度只支持 2 的幂尺寸。"""
    x = x.float()
    n = x.shape[-1]
    X = torch.fft.fftn(x, dim=(-3, -2, -1))
    P = (X.real ** 2 + X.imag ** 2).mean(1)
    k = torch.fft.fftfreq(n, device=x.device)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2
                    + k[None, None, :] ** 2)
    idx = (kk / 0.5 * nbin).long().clamp(0, nbin - 1).reshape(-1)
    P = P.reshape(P.shape[0], -1)
    out = torch.zeros(P.shape[0], nbin, device=x.device, dtype=P.dtype)
    cnt = torch.zeros(nbin, device=x.device, dtype=P.dtype)
    out.index_add_(1, idx, P)
    cnt.index_add_(0, idx, torch.ones_like(P[0]))
    return out / cnt.clamp_min(1)


def spec_loss(pred, true, nbin=16):
    """谱形损失。地板由**真值自己的功率**定，不能用固定的 1e-12 —— 生成器是
    零初始化的，log(0+1e-12) = −27.6 会让这一项无界、梯度爆掉后被裁剪，
    结果就是学不动（实测这一项卡在 17.6 不降，把总损失整个压住）。
    以真值均功率的 1e-4 为地板后，最坏情况被限在 log(1e4) ≈ 9.2。"""
    Pp, Pt = radial_power(pred, nbin), radial_power(true, nbin)
    floor = (Pt.mean(1, keepdim=True) * 1e-4).clamp_min(1e-20).detach()
    return F.l1_loss(torch.log(Pp + floor), torch.log(Pt + floor))


# ---- 第 51 步：方向敏感的谱约束（治斜条纹）----
# 诊断（第 50 步）：旧 spec_loss 只比径向平均功率 —— 不看方向、不看能量是否挤在少数波矢上。
# 一道平面波（斜条纹）与各向同性的真实纹理在它眼里得分相同；训练又没开增广，
# 网络于是用固定方向的条纹凑够每个尺度的能量（同一模型 9 张切片方向一致，周期 ≈ 粗扫分辨极限）。
# 两处补：(1) 功率按「半径 × 13 个方向」分箱匹配 —— 13 个方向（3 面 + 6 棱 + 4 顶点）
#   在八面体群下互相置换，与增广相容；(2) 逐壳谱平坦度匹配：mean(log P) − log(mean P)，
#   各向同性随机纹理 ≈ −0.58，条纹（能量挤在少数分量）远低于此 —— 专罚「凑数式」集中。
_U13 = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
        [0, 1, 1], [0, 1, -1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]
_DIRB = {}


def _dir_bins(n, device, nbin=16):
    key = (n, str(device), nbin)
    if key not in _DIRB:
        k = torch.fft.fftfreq(n, device=device)
        V = torch.stack(torch.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3)
        R = V.norm(dim=1)
        U = torch.tensor(_U13, dtype=torch.float32, device=device)
        U = U / U.norm(dim=1, keepdim=True)
        d = ((V @ U.T).abs() / R[:, None].clamp_min(1e-12)).argmax(1)
        r = (R / 0.5 * nbin).long().clamp(0, nbin - 1)
        idx = r * len(_U13) + d
        one = torch.ones_like(R)
        cnt = torch.zeros(nbin * len(_U13), device=device).index_add_(0, idx, one)
        rcnt = torch.zeros(nbin, device=device).index_add_(0, r, one)
        _DIRB[key] = (idx, cnt, r, rcnt)
    return _DIRB[key]


@torch.amp.autocast("cuda", enabled=False)
def spec_dir_loss(pred, true, nbin=16):
    """方向分箱对数功率 L1 + 逐壳谱平坦度 L1。只在真值该带有能量的壳上计（支撑掩码）。"""
    n = pred.shape[-1]
    idx, cnt, r, rcnt = _dir_bins(n, pred.device, nbin)
    Pp = torch.fft.fftn(pred.float(), dim=(-3, -2, -1)).abs().pow(2).mean(1).reshape(pred.shape[0], -1)
    Pt = torch.fft.fftn(true.float(), dim=(-3, -2, -1)).abs().pow(2).mean(1).reshape(true.shape[0], -1)
    B = Pp.shape[0]
    # (1) 半径 × 方向
    Dp = torch.zeros(B, cnt.numel(), device=Pp.device).index_add_(1, idx, Pp) / cnt.clamp_min(1)
    Dt = torch.zeros(B, cnt.numel(), device=Pp.device).index_add_(1, idx, Pt) / cnt.clamp_min(1)
    fl = (Dt.mean(1, keepdim=True) * 1e-4).clamp_min(1e-20).detach()
    Mt = (Dt.detach() > Dt.detach().amax(1, keepdim=True) * 1e-3) & (cnt > 0)
    l_dir = ((torch.log(Dp + fl) - torch.log(Dt + fl)).abs() * Mt).sum() / Mt.sum().clamp_min(1)
    # (2) 逐壳平坦度。地板 = **各自**该壳均功率的 1e-3（detach）——
    #   初版用真值的壳均功率当地板，实测平坦度项在输出 ×0.5/×1/×2 时为 2.88/3.45/3.89：
    #   奖励「缩小幅度」，而地质正是经 LevelHead 调细节幅度起作用的 → 会悄悄挤掉地质效应。
    #   地板随自身缩放后，平坦度对整体幅度严格不变，幅度只由方向项/径向项按真值管。
    St = torch.zeros(B, nbin, device=Pp.device).index_add_(1, r, Pt) / rcnt.clamp_min(1)
    Sp = torch.zeros(B, nbin, device=Pp.device).index_add_(1, r, Pp) / rcnt.clamp_min(1)

    def lg(P, S):
        f2 = (S.detach() * 1e-3).clamp_min(1e-30)
        return (torch.zeros(B, nbin, device=P.device).index_add_(1, r, torch.log(P + f2[:, r]))
                / rcnt.clamp_min(1)) - torch.log(S + f2)
    Ms = (St.detach() > St.detach().amax(1, keepdim=True) * 1e-3) & (rcnt > 0)
    l_flat = ((lg(Pp, Sp) - lg(Pt, St)).abs() * Ms).sum() / Ms.sum().clamp_min(1)
    return l_dir + l_flat, l_dir.detach(), l_flat.detach()


def _fake_c(lr, mu, sd):
    """4 个任意 LR 统计量（均值/标准差/25 分位/75 分位），标准化后顶替地质 c。
    第 28 步的证伪对照：结构与真实 c 完全对称，只是没有地质含义。"""
    v = lr.flatten(1).float()
    q = torch.quantile(v, torch.tensor([0.25, 0.75], device=v.device), dim=1)
    f = torch.stack([v.mean(1), v.std(1), q[0], q[1]], 1)
    m = torch.tensor(mu, device=f.device, dtype=f.dtype)
    s_ = torch.tensor(sd, device=f.device, dtype=f.dtype).clamp_min(1e-6)
    return ((f - m) / s_ * 0.05 + 0.1).to(lr.dtype)     # 缩到与真实 c 同量级


def proj_hard(hr, lr, Hk, n=3):
    """硬 H 投影（可微）。与 evaluate.py 中推理用的是同一个算子，
    只是迭代轮数可调 —— 训练时每一步都做 8 轮太贵。"""
    k = 16
    off = (lr.shape[-1] - k) // 2
    sl = slice(off, off + k)
    lrc = lr[..., sl, sl, sl]
    for _ in range(n):
        hr = hr + F.interpolate(lrc - degrade(hr, Hk), size=hr.shape[-3:],
                                mode="trilinear", align_corners=False)
    return hr


def cfield_target(hr, win, n=4):
    """真值地质场 (B, 3, n, n, n)：IGV / f_pore / f_dense，每格 224/n μm。

    与 cfield2.Acc.vals 同一定义（已核验：按体积平均回块级，与 cfield2 的 c_blk
    相关 0.989~0.993、平均绝对差 0.005~0.008）。win=(lo,hi) 是逐样品骨架窗，
    batch 里本来就有，所以不需要改 dataset。

    剔除 mu_inter：跨岩性留一检验 R² = -0.31（n=2/4/8 全为负），做成场只注入噪声。
    为什么值得做成场：实测块内子块间 σ / 块间 σ = IGV 1.37、f_pore 1.06、
    f_dense 1.27 —— 用 4 个块级常数，扔掉的地质变化比留下的还多。"""
    B = hr.shape[0]
    D, H, W = hr.shape[-3:]
    d, h, w = D // n, H // n, W // n
    x = hr[:, 0, :d * n, :h * n, :w * n].reshape(B, n, d, n, h, n, w)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, n ** 3, -1).float()
    lo = win[:, 0].view(B, 1, 1).float()
    hi = win[:, 1].view(B, 1, 1).float()
    igv = 1.0 - ((x >= lo) & (x < hi)).float().mean(-1)
    pore = (x < PORE_CUT).float().mean(-1)
    dense = (x >= hi).float().mean(-1)
    return torch.stack([igv, pore, dense], 1).reshape(B, 3, n, n, n)


def compute_losses(out, hr, lr, c_true, w, win, Hk, cfg, out2=None, geo_t=None):
    # 第 42 步：结构类损失（top/loc/por）改算在**投影后**的输出上，与评测同口径。
    # 逐体素 L1 与谱匹配仍用原始输出：它们不受这个错位影响，一次只动一个变量。
    _hp = (proj_hard(out["hr"], lr, Hk, cfg.proj_train)
           if getattr(cfg, "proj_train", 0) > 0 else out["hr"])
    """分三段 + 退化一致性。返回 (总损失, 各分量 dict)。

    out2：同一 (LR, c) 再抽一次 z 的输出（阶段二有模式寻求时提供）。
    有它时成因一致性作用在**两次 z 的 Φ 均值**上而不是单样本上 ——
    c 是 0.5 mm 格的均值，真值块绕它有 0.018~0.030 的自然散布；
    逐样本把 Φ 拉向 c 等于消灭这个散布，这正是 gi 比 given 更收缩
    （第 13 步护栏，预测 1 被证伪）的机制。约束条件均值、散布交给 D。"""
    t_a, t_mid, t_fine = bands(hr)
    arr = cfg.mode in ARR_MODES          # 排布列是否启用（c_mse 与 Φ 都要用）
    L = {}
    # >31 μm：确定性，有唯一答案
    L["a_l1"] = F.l1_loss(out["a"], t_a)
    # 31→15 μm：可推断带。先用 L1 + 谱匹配；对抗项留钩子（见 README_GPU §三）
    L["mid_l1"] = F.l1_loss(out["mid"], t_mid)
    # 第 51 步：spec_keep_radial=1 时保留旧径向项。实测（check51，6 折×40 留出块）它「支持正确地质」：
    #   真 c 的径向谱损失低于配错 c 的块占 79%，而新方向/平坦度项在条纹阶段被结构误差主导、
    #   对 c 无分辨（~49%）。保留它，过渡期地质的幅度信号不丢；新两项只负责罚条纹。
    if getattr(cfg, "spec_dir", 0):
        L["mid_spec"], L["mid_dir"], L["mid_flat"] = spec_dir_loss(out["mid"], t_mid)
        if getattr(cfg, "spec_keep_radial", 0):
            L["mid_rad"] = spec_loss(out["mid"], t_mid); L["mid_spec"] = L["mid_spec"] + L["mid_rad"]
    else:
        L["mid_spec"] = spec_loss(out["mid"], t_mid)
    # <15 μm：不可约带。**只匹配统计量，不做逐点损失**
    if getattr(cfg, "spec_dir", 0):
        L["fine_spec"], L["fine_dir"], L["fine_flat"] = spec_dir_loss(out["fine"], t_fine)
        if getattr(cfg, "spec_keep_radial", 0):
            L["fine_rad"] = spec_loss(out["fine"], t_fine); L["fine_spec"] = L["fine_spec"] + L["fine_rad"]
    else:
        L["fine_spec"] = spec_loss(out["fine"], t_fine)
    L["fine_std"] = F.l1_loss(out["fine"].std(dim=(-3, -2, -1)),
                              t_fine.std(dim=(-3, -2, -1)))
    # 瓶颈监督（自由特征消融时没有 c 可监督）。4 维或 7 维（第 15 步）都要监督 ——
    # 初版写死 == 4，7 维时这项为零，编码器排布 3 维的输出行范数精确为 0.000（从没训过）。
    if out["c"].shape[1] == c_true.shape[1]:
        ct_ = c_true
        if arr and c_true.shape[1] > N_C:
            # 编码器预测**缩放后**的排布量（与 SPADE 输入同一套常数），
            # 否则原始量级会让 c_mse 炸掉（第 15 步教训）
            s7 = torch.tensor([1.] * N_C + [ARR_SCALE2[c] for c in cfg._arr_cols],
                              device=hr.device, dtype=c_true.dtype)
            ct_ = c_true / s7
        L["c_mse"] = (w * (out["c"] - ct_) ** 2).mean()
    else:
        L["c_mse"] = torch.zeros((), device=hr.device)
    # 成因一致性：**把条件真正焊在输出上**。
    # 前面的 c_mse 只管「E 猜得准不准」，没有任何一项检查**生成出来的图
    # 到底有没有那个成因**。于是即便 c 给对了，G 也可以完全不理它。
    # Φ 是 cfield.cvals 的可微版（见 models.genesis_phi），作用在 out["hr"] 上，
    # 要求它测出来的四个数等于实测的 c —— 这才叫「从地质因素反演」。
    # 只有 gi 变体开这一项：given vs gi 正好是这条损失的消融。
    aux = {}
    # 第 18 步：排布列可变后，Φ 只作用在**成分 4 项**上（新列的可微泛函另行实现）。
    # given7 不带 Φ，不受影响；gi7 暂等价于「4 项 Φ + 7 维条件」。
    PHI = genesis_phi
    tgt = c_true[:, :N_C]
    wgt = w[:, :N_C]
    if cfg.mode in ("gi", "gi7") or out2 is not None:
        ph = PHI(out["hr"], win[:, 0], win[:, 1])
        aux["ph1"] = ph
        if out2 is not None:
            ph2 = PHI(out2["hr"], win[:, 0], win[:, 1])
            aux["ph2"] = ph2
            ph = 0.5 * (ph + ph2)              # 约束 z 平均，不约束单样本
    # ---- 第 17 步：Φ 一致性的两个修法（逐块硬压导致跟踪崩溃，3 折里 2 折）----
    # 症状（直接观测）：gi 的连通性散布比在崩溃折上同步塌陷
    #   YN 0.776→0.580、CQ 0.916→0.732，而未崩的 SC 0.776→0.781 持平。
    # 机制：c 测在 0.5 mm 格上，块只覆盖中心 224 μm，**真值块自己对 c 就偏离
    #   0.016~0.030**（逐折实测，见第 17 步）。逐块硬压 Φ=c 等于强迫模型抹掉
    #   这个本该存在的偏离，把块间差异压平 —— 而块间差异正是逐块跟踪衡量的量。
    #   这是同一失效模式第三次出现（第 15 步逐块压 χ → 跟踪 0.086）。
    # A 死区：只罚超出口径差 δ 的部分，保留逐块地质因果。
    # B 批级：只约束这一批的均值，与 w_top 同一原理（批级有效、逐块打架）。
    if cfg.mode in ("gi", "gi7"):
        d = (ph - tgt).abs()          # Φ 现在一律 4 项，与 4 维变体严格可比
        if cfg.gc_batch:
            # B：只对齐批均值，逐块自由
            d = (ph.mean(0) - tgt.mean(0)).abs()
            L["gc"] = (wgt.mean(0) * d).mean()
        else:
            if cfg.gc_deadzone > 0:
                # A：死区。δ 按变量给（成分 4 项用实测口径差 × 系数；新增项已被 sc 归一）
                dz = torch.tensor(GC_FLOOR, device=hr.device, dtype=d.dtype) * cfg.gc_deadzone
                if d.shape[1] > len(GC_FLOOR):
                    dz = torch.cat([dz, dz.new_full((d.shape[1] - len(GC_FLOOR),), dz.mean())])
                d = torch.relu(d - dz)
            L["gc"] = (wgt * d).mean()
    else:
        L["gc"] = torch.zeros((), device=hr.device)
    # 拓扑批级矩匹配（第 16 步）。诊断：生成体的孔隙团数只有真值的 14%、χ 只有 1/8,
    # 逐块连通性跟踪已达 r=0.92 且散布接近信息极限，**误差的 79% 是系统性偏差**。
    # 系统性缺陷要用系统性手段修：只要求「这一批的平均拓扑 = 真值这一批的平均」,
    # 逐块自由度留给 LR 数据。这与第 15 步失败的做法正相反 —— 那里给每块压一个
    # χ 目标，与 LR 打架，把跟踪从 0.916 打到 0.086。
    if cfg.w_top > 0:
        cp, ct_ = chi_soft(_hp), chi_soft(hr)
        tp, tt_ = thick_soft(_hp), thick_soft(hr)
        L["top"] = ((cp.mean() - ct_.mean()).abs() / ARR_SCALE[0]
                    + (tp.mean() - tt_.mean()).abs() / ARR_SCALE[2])
        aux["chi_p"], aux["chi_t"] = float(cp.mean().detach()), float(ct_.mean().detach())
    else:
        L["top"] = torch.zeros((), device=hr.device)
    # ---- 第 27 步：局部拓扑**分布**匹配 ----
    # 批级 χ 一个批次只给 1 个标量，抵不过百万体素的 L1；子块版给 64 个数/块。
    # 排序后比对 ⇒ 与位置无关：只约束「这块该有多少拓扑、怎么分布」，
    # 不指定「哪里该有洞」，因此不与 LR 数据打架（第 15 步的失败模式）。
    if cfg.w_loc > 0:
        cl_p = chi_local(_hp, cfg.loc_n)
        cl_t = chi_local(hr, cfg.loc_n)
        L["loc"] = ((cl_p.sort(1).values - cl_t.sort(1).values).abs().mean()
                    / ARR_SCALE[0])
        aux["loc_p"] = float(cl_p.mean().detach())
        aux["loc_t"] = float(cl_t.mean().detach())
    else:
        L["loc"] = torch.zeros((), device=hr.device)
    # ---- 第 38 步：块孔隙度损失 ----
    # 现有损失项全是零均值带通量，没有一项在管灰度水平，
    # 而孔隙度恰恰是水平量 —— 水平通道即便接上了，没有损失也推不动它。
    if cfg.w_por > 0:
        pp = torch.sigmoid((PORE_CUT - _hp.float()) / 0.02).mean(dim=(1, 2, 3, 4))
        pt_ = (hr.float() < PORE_CUT).float().mean(dim=(1, 2, 3, 4))
        L["por"] = (pp - pt_).abs().mean()
    else:
        L["por"] = torch.zeros((), device=hr.device)
    # ---- 第 39 步：显式地质中间表示的监督 ----
    # 监督信号来自**细扫**，推理时不存在 => privileged information 蒸馏，不是泄题。
    # filled 模式当年失败的直接死因是 c_mse 只占总损失 0.1%（编码器等于自由），
    # 所以 w_geo 必须把这一项提到总损失的 10~20%。监督密度 192 个数 vs 原来 4 个。
    # w_geo=0 即「同容量自由场」对照臂：结构完全相同，只是中间场不受地质监督。
    if cfg.w_geo > 0 and out.get("geo") is not None and geo_t is not None:
        L["geo"] = (out["geo"].float() - geo_t.float()).abs().mean()
    else:
        L["geo"] = torch.zeros((), device=hr.device)
    # ---- 第 24 步：批级 chi **方差**匹配 ----
    # w_top 管均值、w_cov 管与 c 的斜率，**没人管块间差异本身**。
    # 实测网络 chi 块间 std 仅真值 0.27~0.41 倍，chi 的 R^2 0.311，
    # 低于只用 8 个 LR 标量的线性回归 0.660 —— 已有信息被扔掉了一半。
    if cfg.w_var > 0:
        xp2, xt2 = chi_soft(out["hr"]), chi_soft(hr)
        if xp2.numel() > 1:
            L["var"] = (xp2.std() - xt2.std()).abs() / ARR_SCALE[0]
            aux["chistd_p"] = float(xp2.std().detach())
            aux["chistd_t"] = float(xt2.std().detach())
        else:
            L["var"] = torch.zeros((), device=hr.device)
    else:
        L["var"] = torch.zeros((), device=hr.device)
    # ---- 第 21 步：批级**协方差**匹配（w_top 的盲区补丁）----
    # w_top 只压批均值，而批均值用一个**与 c 无关的常数**修正就能满足 ——
    # 第 20 步阈值门控实测证实了这一点：门开/门关跨阈值体素 1.13%（门是活的），
    # 换 c 只有 0.50%（c 不掌舵）。门照梯度学，梯度里没有 c，学出来就是常数。
    #
    # 前提（CQ 折 150 块实测）：c→chi 的偏回归系数（已扣掉粗扫均值/标准差可得的部分）
    # 真值合计 125700、模型仅 20581 —— **复现 16%，且四维方向全对、幅度小 5 倍**。
    # 这是 c 相对 LR 真正有独立话语权的地方（最大团占比的偏 beta 被粗扫削掉 85%，chi 只削掉 14%）。
    #
    # 形式：去掉各自均值后，chi 的**块间误差**不许与 c 相关。
    #   L_cov = | mean_b( u_center * [(chi_p - mean chi_p) - (chi_t - mean chi_t)] ) |
    # 均值那部分归 w_top 管，这一项专管斜率；常数修正的协方差恒为 0，交不了差。
    # 仍是**分布级**约束（不给任何一块指定目标），与第 15/17 步逐块硬压的失败路线相反。
    if cfg.w_cov > 0 and c_true is not None and c_true.shape[0] > 1:
        xp, xt = chi_soft(out["hr"]), chi_soft(hr)
        cs = torch.tensor(cfg._c_std, device=hr.device, dtype=xp.dtype)
        u = c_true[:, :len(cfg._c_std)].to(xp.dtype) / cs
        u = u - u.mean(0, keepdim=True)
        e = (xp - xp.mean()) - (xt - xt.mean())
        L["cov"] = (u * e.unsqueeze(1)).mean(0).abs().sum() / ARR_SCALE[0]
        aux["cov_p"] = float((u * (xp - xp.mean()).unsqueeze(1)).mean(0).abs().sum().detach())
        aux["cov_t"] = float((u * (xt - xt.mean()).unsqueeze(1)).mean(0).abs().sum().detach())
    else:
        L["cov"] = torch.zeros((), device=hr.device)
    # ---- 第 19 步：直接约束不可约带的**谱斜率** ----
    # 实测（4 折，各 40 块）：生成体细带的标准差基本对了（相关 0.61~0.92），
    # 但**谱斜率四折全部偏高一倍**（3.3~4.3 vs 真值 1.9~2.5）——
    # 斜率偏高 = 高频衰减太快 = 细结构被系统性抹掉，正是「团数仅真值 14%、χ 仅 1/8」
    # 在频域的对应。**第一次在频域看到同一个病。**
    #
    # 为什么已有的 fine_spec 没管住：它是 16 个频段对数功率的 L1，细带只占最高几段，
    # 而那几段的功率比低频低几个数量级，在求平均时权重被淹没 ——
    # **名义上覆盖细带，实际上对细带几乎没有梯度。**
    # 修法：直接压斜率本身（尺度无关的标量，不会被功率量级淹没）。
    # 这个量恰好是地质可预测的（样品间占比 81.8%、c 能解释 R²=0.497），
    # 所以它也是「地质因素」真正能发挥作用的地方。
    if cfg.w_fband > 0:
        def slope(x):
            # 高频段 / 中频段的对数功率之比，代理谱斜率；可微且便宜
            P = radial_power(x.float(), 16)
            return (P[:, 4:8].mean(1).clamp_min(1e-20).log()
                    - P[:, 10:14].mean(1).clamp_min(1e-20).log())
        L["fband"] = (slope(out["fine"]) - slope(t_fine)).abs().mean()
    else:
        L["fband"] = torch.zeros((), device=hr.device)
    # 退化一致性：必要不充分，幻觉正好活在退化算子的零空间里
    o = (lr.shape[-1] - hr.shape[-1] // 7) // 2
    s = slice(o, o + hr.shape[-1] // 7)
    L["degrade"] = F.l1_loss(degrade(out["hr"], Hk), lr[..., s, s, s])
    # 逐体素 L1 与谱匹配分开加权：L1 是「条件均值」这股力的本体，阶段二要压它；
    # 谱匹配是分布目标，不压多样性，要保留。pilot5 把两者一起砍了，谱误差反而变差。
    w_l1 = cfg.w_mid if getattr(cfg, "w_mid_l1", None) is None else cfg.w_mid_l1
    tot = (cfg.w_a * L["a_l1"] + w_l1 * L["mid_l1"] + cfg.w_mid * L["mid_spec"]
           + cfg.w_fine * (L["fine_spec"] + L["fine_std"])
           + cfg.w_c * L["c_mse"] + cfg.w_deg * L["degrade"]
           + cfg.w_gc * L["gc"] + cfg.w_top * L["top"]
           + cfg.w_fband * L["fband"] + cfg.w_cov * L["cov"]
           + cfg.w_var * L["var"] + cfg.w_loc * L["loc"]
           + cfg.w_por * L["por"] + cfg.w_geo * L["geo"])
    return tot, {k: float(v.detach()) for k, v in L.items()}, aux


# ------------------------------------------------------------------ 检查点

def save_ckpt(path, model, opt, step, cfg, best):
    """原子写：先 .tmp 再 rename，保存中掉电不会毁掉旧的。"""
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                    best=best, cfg=vars(cfg),
                    rng_torch=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all(),
                    rng_np=np.random.get_state(), rng_py=random.getstate()), tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, dev):
    d = torch.load(path, map_location=dev, weights_only=False)
    model.load_state_dict(d["model"]); opt.load_state_dict(d["opt"])
    torch.set_rng_state(d["rng_torch"].cpu())
    try:
        torch.cuda.set_rng_state_all([s.cpu() for s in d["rng_cuda"]])
    except Exception:
        pass
    np.random.set_state(d["rng_np"]); random.setstate(d["rng_py"])
    return d["step"], d.get("best", float("inf"))


# ------------------------------------------------------------------ 主

def _mk_winit(seed):
    # 每个 worker 单独播种：dataset.py 的翻转增强用的是 python 的 random
    def _f(w):
        import random as _r
        _r.seed(seed + w)
        np.random.seed(seed + w)
        torch.manual_seed(seed + w)
    return _f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("GEOCOND_DATA", "data"))
    p.add_argument("--out", default="runs/run1")
    p.add_argument("--fold", default="SC", help="留一岩性折；dev 用开发划分")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--free", action="store_true", help="等价于 --mode free（保留兼容）")
    p.add_argument("--mode", default="",
                   help="bottleneck / free / none / sealed / sealed_free")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", default="")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--dump_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--keep_ckpt", type=int, default=3)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--w_a", type=float, default=1.0)
    p.add_argument("--w_mid", type=float, default=1.0)
    p.add_argument("--w_fine", type=float, default=0.5)
    p.add_argument("--w_c", type=float, default=5.0)
    # 成因一致性的权重。|Φ(真值块) − c| 实测 0.018~0.030（体积不同带来的下限），
    # 其余损失量级 O(0.1)，取 2.0 让它可见但不压住重建项。
    p.add_argument("--w_gc", type=float, default=2.0)
    p.add_argument("--w_deg", type=float, default=1.0)
    # 第 16 步：拓扑批级矩匹配的权重。0 = 关（默认，旧行为不变）
    p.add_argument("--w_top", type=float, default=0.0)
    # 第 18 步：排布列可选（「一次只加一个」）。逗号分隔，如 --arr_cols pore_env
    # 可选：pore_env rim_idx skel_len het_igv COPL CEPL chi
    p.add_argument("--arr_cols", default="pore_env")
    # 第 19 步：不可约带统计一致性的权重。0 = 关
    p.add_argument("--w_fband", type=float, default=0.0)
    # 第 20 步：阈值门控残差头（零初始化，默认关）。诊断见 models.ThroatGate
    p.add_argument("--throat", type=int, default=0)
    p.add_argument("--w_var", type=float, default=0.0,
                   help="第 24 步：批级 chi 方差匹配（逼出块间差异）")
    p.add_argument("--z_global", type=int, default=0,
                   help="第 36 步：噪声退化成每块一个全局向量，与 c 同量级竞争")
    p.add_argument("--proj_train", type=int, default=0,
                   help="第 42 步：训练时先做 N 轮硬 H 投影再算结构类损失（0=关，评测用 8 轮）")
    p.add_argument("--seal", type=int, default=0,
                   help="第 40 步：切断生成体对粗扫的直接通路，逼它只能经由地质条件")
    p.add_argument("--geo", type=int, default=0,
                   help="第 39 步：显式地质中间表示的场分辨率 n（0=关，建议 4 即 56 μm）")
    p.add_argument("--geo_true", type=int, default=0,
                   help="第 39 步：直接喂真值场（**上界测试，泄题，只测天花板**）")
    p.add_argument("--w_geo", type=float, default=0.0,
                   help="第 39 步：地质场监督权重；0 = 同容量自由场对照臂")
    p.add_argument("--level", type=int, default=0,
                   help="第 38 步：接上 c 驱动的水平通道（零初始化）")
    p.add_argument("--w_por", type=float, default=0.0,
                   help="第 38 步：块孔隙度损失，唯一管灰度水平的一项")
    p.add_argument("--seed", type=int, default=-1,
                   help="终局实验：固定随机种子。两臂用同一 seed 构成配对比较")
    p.add_argument("--zero_z", type=int, default=0,
                   help="第 34 步：z 恒为 0，堵死噪声这条逃生通道")
    p.add_argument("--c_src", default="",
                   help="第 31 步：c 的支撑 blk/cell/ann（空=沿用原 0.5mm 格 c）")
    p.add_argument("--fake_c", type=int, default=0,
                   help="第 28 步：把 4 个地质数换成 4 个任意 LR 统计量（证伪对照）")
    p.add_argument("--w_loc", type=float, default=0.0,
                   help="第 27 步：局部拓扑分布匹配（子块 χ 排序后 L1）")
    p.add_argument("--loc_n", type=int, default=4,
                   help="每边切几份，默认 4 ⇒ 64 个子块")
    p.add_argument("--wta", type=int, default=0,
                   help="第 26 步：Winner-Takes-All，抽 K 个候选只罚最好的（0 关）")
    p.add_argument("--only_throat", type=int, default=0,
                   help="第 23 步：冻死主网络，只训阈值门（配合把各 L1 权重置 0）")
    p.add_argument("--w_cov", type=float, default=0.0,
                   help="第 21 步：批级 c-chi 协方差匹配（w_top 只管均值，常数即可满足）")
    # 第 17 步：Φ 一致性的修法。0/0 = 旧行为（逐块硬压）
    p.add_argument("--gc_deadzone", type=float, default=0.0, help="A：死区宽度 = 系数 × 实测口径差")
    p.add_argument("--gc_batch", type=int, default=0, help="B：Φ 只约束批均值")
    # ---- 第 14 步：对抗微调阶段（ESRGAN 两阶段配方的第二阶段）----
    p.add_argument("--init_from", default="",
                   help="阶段一检查点路径：只载入模型权重，优化器重建，step 归零")
    p.add_argument("--adv", type=int, default=0, help="1 = 开投影式条件判别器")
    p.add_argument("--w_adv", type=float, default=0.05)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--aug", type=int, default=0, help="1 = 48 倍八面体增广")
    # 第一次 pilot 两套学习率下 D 都在 ~150 步内完全分开真假（hinge 0.002），
    # 且 a_l1 被拽高 —— 不是调参问题：D 看整张图，>31 μm 那部分 G 本就画得好，
    # 判别力全来自「31 μm 以下糊不糊」，太容易赢；对抗梯度还流进了 A 路。
    # 修法：D 只看生成的两个频带（见循环），外加两项标准稳定器：
    p.add_argument("--d_noise", type=float, default=0.05,
                   help="D 输入的实例噪声初始 σ（Δ 单位），线性退火到 0")
    p.add_argument("--r1", type=float, default=1.0, help="惰性 R1 惩罚系数 γ，0 关")
    p.add_argument("--r1_every", type=int, default=4)
    # pilot3 诊断出的另两条（第 14 步 §6.4）：
    #   · 可恢复带相关 0.825→0.736 —— 对抗梯度沿 G 的输入（实例归一化后的 a_out）
    #     倒灌回 A 路。A 路有唯一答案、已训 15000 步，阶段二直接冻结。
    #   · 换 8 个 z 成因变量只动 1~4% —— G 不听 z，塌成确定性函数，D 再强也抬不起
    #     离散度。逐层注噪（models.NoiseInject）+ 模式寻求项把 z 的作用推起来。
    # pilot5 的 1000 步读数（第 14 步 §6.5）：z 敏感度 750 步时到 0.015，D 一饱和就塌回 0.005；
    # 且 w_mid 同时乘 mid_l1 与 mid_spec，降它把谱项也砍了。→ 三个参数：
    p.add_argument("--w_mid_l1", type=float, default=None,
                   help="可推断带逐体素 L1 的权重；不给则等于 w_mid。阶段二只降它，谱项保留")
    p.add_argument("--d_every", type=int, default=1, help="每 N 个 G 步更新一次 D，不让 D 赢太快")
    p.add_argument("--d_noise_floor", type=float, default=0.0,
                   help="实例噪声退火的下限 σ_min，不让 D 最后看到干净输入")
    p.add_argument("--freeze_a", type=int, default=0, help="1 = 阶段二冻结 A 路")
    p.add_argument("--spec_dir", type=int, default=0,
                   help="第 51 步：1 = mid/fine 谱项换成「半径×13 方向」分箱 + 逐壳平坦度（治斜条纹）")
    p.add_argument("--afeat", type=int, default=0,
                   help="第 52 步：1 = A 路深层特征（实例归一化）接进 G 的 28³/56³ 层，零初始化")
    p.add_argument("--spec_keep_radial", type=int, default=0,
                   help="第 51 步：1 = 在方向+平坦度之外保留旧径向谱项（它支持正确地质，过渡期不丢地质信号）")
    p.add_argument("--w_ms", type=float, default=0.0,
                   help="模式寻求项权重（MSGAN）：惩罚「换 z 不换输出」，0 关")
    cfg = p.parse_args()
    # 终局实验：固定全部随机源。两臂同 seed ⇒ 初始权重、数据顺序、噪声序列完全相同，
    # 唯一差别是喂进去的 c，从而把「运行之间」那一份主导方差差分掉。
    if cfg.seed >= 0:
        import random as _random
        _random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        # cuDNN 的 3D 卷积反向默认走非确定性原子加，必须关掉，
        # 否则同 seed 两次仍会分叉，配对设计失去意义。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    cfg._arr_cols = []
    if not cfg.mode:
        cfg.mode = "free" if cfg.free else "bottleneck"
    cfg.free = cfg.mode in ("free", "sealed_free")      # 评估脚本按它挑编码器

    root = Path(cfg.root).expanduser()
    out = Path(cfg.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / "dumps").mkdir(exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    dev = torch.device(f"cuda:{cfg.gpu}")
    torch.cuda.set_device(dev)
    # 三维卷积的默认算法很慢，让 cuDNN 自己挑；TF32 对本任务精度足够
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

    fs = {f["fold"]: f for f in folds(root)}
    if cfg.fold == "ALL":
        # 主模型：用全部 14 个可用样品训练，不留岩性。
        # 留一岩性的 7 个折是**评估协议**（证明机制），这一个是**交付的模型**
        # ——论文里展示输出、画例图、以及将来真拿去用的，都是它。
        import json as _j
        sp = _j.loads((root / "meta" / "splits.json").read_text(encoding="utf-8"))
        tr_g = [r["gid"] for r in sp["samples"] if r["usable"]]
        te_g = []
    elif cfg.fold in fs:
        tr_g, te_g = fs[cfg.fold]["train"], fs[cfg.fold]["test"]
    else:
        from dataset import dev_split
        tr_g, te_g = dev_split(root)
    arr_cols = [x for x in cfg.arr_cols.split(",") if x] if cfg.mode in ARR_MODES else None
    ds = PairDataset(root, tr_g, unit_split="train",    # 留出 1/10 供"见过岩性"评估
                     aug=bool(cfg.aug), arr=arr_cols or False,
                     c_src=cfg.c_src or None)
    _g = None
    if cfg.seed >= 0:
        _g = torch.Generator(); _g.manual_seed(cfg.seed)     # 取样顺序也锁死
    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers,
                    pin_memory=True, drop_last=True, persistent_workers=cfg.workers > 0,
                    generator=_g,
                    worker_init_fn=_mk_winit(cfg.seed) if cfg.seed >= 0 else None)
    Hk = torch.from_numpy(load_H(root)[0]).to(dev)

    # 第 21 步：协方差项要把 c 标准化。统计**只取训练折**（留一岩性协议下无泄漏），
    # 启动时算一次并写进 meta，评估端可复核。
    cfg._c_std = [1.0] * N_C
    if cfg.w_cov > 0:
        _n = min(512, len(ds))
        _ii = _np.random.default_rng(0).choice(len(ds), _n, replace=False)
        _cc = _np.array([ds[int(i)]["c"][:N_C] for i in _ii], dtype=_np.float64)
        cfg._c_std = [float(max(x, 1e-6)) for x in _cc.std(0)]
        print("协方差项：训练折 c 标准差 = "
              + ", ".join("%s %.4f" % (n, v) for n, v in zip(C_NAMES, cfg._c_std)), flush=True)

    # 第 28 步：假 c 的标准化统计，只用训练折（无泄漏），启动时算一次
    cfg._fk_mu, cfg._fk_sd = [0.0] * 4, [1.0] * 4
    if cfg.fake_c:
        _ii = _np.random.default_rng(0).choice(len(ds), min(512, len(ds)), replace=False)
        _v = _np.stack([ds[int(i)]["lr"].ravel() for i in _ii])
        _f = _np.stack([_v.mean(1), _v.std(1),
                        _np.percentile(_v, 25, axis=1),
                        _np.percentile(_v, 75, axis=1)], 1)
        cfg._fk_mu = [float(x) for x in _f.mean(0)]
        cfg._fk_sd = [float(x) for x in _f.std(0)]
        print("假 c 统计（训练折）：mu %s  sd %s"
              % (_np.round(cfg._fk_mu, 4).tolist(), _np.round(cfg._fk_sd, 4).tolist()),
              flush=True)

    cfg._arr_cols = arr_cols or []
    model = GenesisSR(mode=cfg.mode, arr_cols=arr_cols,
                      throat=bool(cfg.throat),
                      zero_z=bool(cfg.zero_z),
                      z_global=bool(cfg.z_global),
                      level=bool(cfg.level),
                      geo=int(cfg.geo),
                      geo_true=bool(cfg.geo_true),
                      seal=bool(cfg.seal),
                      afeat=bool(getattr(cfg, "afeat", 0))).to(dev)
    # 第 23 步：只训门。主网络冻死 ⇒ 像素保真度由构造保证，不必靠 L1 去守，
    # 于是可以把 L1 权重全置 0，让批级统计项单独驱动门 —— L1 再也压不住 c。
    if cfg.only_throat:
        if not cfg.throat:
            raise SystemExit("--only_throat 需要同时 --throat 1")
        for n_, q_ in model.named_parameters():
            q_.requires_grad = "throat" in n_
        _tn = [n_ for n_, q_ in model.named_parameters() if q_.requires_grad]
        print("只训阈值门：可训张量 %d 个，参数 %d 个（主网络已冻结）"
              % (len(_tn), sum(q_.numel() for q_ in model.parameters() if q_.requires_grad)),
              flush=True)
    npar = sum(x.numel() for x in model.parameters())
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp))

    # 阶段二：从阶段一检查点起步 —— 只拿模型权重，优化器重建，step 归零。
    # ESRGAN 的标准做法：L1 预训练好的 G 再上对抗，比从零训稳得多也快得多。
    if cfg.init_from:
        d0 = torch.load(Path(cfg.init_from).expanduser(), map_location=dev,
                        weights_only=False)
        # 第 15 步热启动：4 维条件的检查点载入 7 维模型 —— SPADE 第一层 1x1x1 卷积的
        # 输入通道 4→7、编码器输出 4→7，新增通道零填充。起步时新 3 维对输出无影响，
        # 与 4 维模型严格等价，再由训练把排布通道学起来。
        sd0 = d0["model"]; msd = model.state_dict(); padded = 0
        for k, v in list(sd0.items()):
            if k in msd and msd[k].shape != v.shape:
                tgt_ = torch.zeros_like(msd[k])
                if v.dim() == 5 and msd[k].shape[1] > v.shape[1]:        # 1x1x1 conv 输入通道
                    tgt_[:, :v.shape[1]] = v
                elif v.dim() == 2 and msd[k].shape[0] > v.shape[0]:      # Linear 输出
                    tgt_[:v.shape[0]] = v
                elif v.dim() == 1 and msd[k].shape[0] > v.shape[0]:      # bias
                    tgt_[:v.shape[0]] = v
                else:
                    raise RuntimeError(f"无法热启动：{k} {tuple(v.shape)} -> {tuple(msd[k].shape)}")
                sd0[k] = tgt_; padded += 1
        if padded:
            print(f"热启动零填充 {padded} 个张量（条件维度扩展，新增通道权重为 0，起步与原模型等价）", flush=True)
        # strict=False：阶段二新增的逐层注噪参数（NoiseInject.scale，初始 0）
        # 在阶段一检查点里没有，缺了就用初始值，行为与阶段一完全一致。
        miss, unexp = model.load_state_dict(sd0, strict=False)
        print(f"阶段一权重来自 {cfg.init_from}（原 step {d0.get('step')}）  "
              f"新增参数 {len(miss)} 个（应全为 ni_*.scale）  多余 {len(unexp)} 个",
              flush=True)
        bad = [k for k in miss if not ((".scale" in k and "ni_" in k) or "throat" in k
                                       or k.startswith("g.fa4.") or k.startswith("g.fa2."))]
        if bad or unexp:
            raise RuntimeError(f"检查点与模型不匹配：缺 {bad}  多 {list(unexp)}")
    if cfg.freeze_a:
        for p_ in model.a.parameters():
            p_.requires_grad_(False)
        print("A 路已冻结（阶段二不再更新可恢复带）", flush=True)
    D, optD = None, None
    if cfg.adv:
        D = Discriminator().to(dev)
        optD = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.99))
        print(f"投影式条件判别器 {sum(x.numel() for x in D.parameters())/1e6:.2f} M  "
              f"w_adv {cfg.w_adv}  lr_d {cfg.lr_d}  增广 {'开' if cfg.aug else '关'}",
              flush=True)

    step, best = 0, float("inf")
    if cfg.resume:
        cks = sorted((out / "ckpt").glob("step*.pt"))
        pth = Path(cfg.resume) if cfg.resume != "auto" else (cks[-1] if cks else None)
        if pth and pth.exists():
            step, best = load_ckpt(pth, model, opt, dev)
            print(f"从 {pth.name} 续跑，step={step}", flush=True)
            dp = pth.with_name(pth.name.replace("step", "D", 1))
            if D is not None and dp.exists():
                dd = torch.load(dp, map_location=dev, weights_only=False)
                D.load_state_dict(dd["D"]); optD.load_state_dict(dd["optD"])
                print(f"判别器从 {dp.name} 续跑", flush=True)
        else:
            print("没有找到检查点，从头开始", flush=True)

    logf = (out / "log.jsonl").open("a", encoding="utf-8")
    meta = dict(fold=cfg.fold, train=tr_g, test=te_g, n_units=len(ds),
                eval_seen="PairDataset(train_gids, unit_split='heldout')",
                eval_unseen="PairDataset(test_gids)",
                params=npar, free=cfg.free, mode=cfg.mode, cfg=vars(cfg),
                gpu=torch.cuda.get_device_name(dev))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print("=" * 84)
    print(f"折 {cfg.fold}  训练 {len(tr_g)} 样 / {len(ds)} 单元   测试 {te_g}")
    MODE_DESC = {"bottleneck": "成因瓶颈（4 个数，G 看原样 a_out）",
                 "free": "自由特征（64 维，G 看原样 a_out）",
                 "none": "无条件对照（G 只看 a_out + 噪声）",
                 "sealed": "封闭瓶颈（4 个数，G 看不到 a_out）",
                 "filled": "**填充式瓶颈**（4 个数，G 看归一化 a_out）",
                 "filled_free": "填充式自由特征（64 维，G 看归一化 a_out）"}
    MODE_DESC["sealed_free"] = "封闭自由特征（64 维，G 看不到 a_out）"
    MODE_DESC["given"] = "**实测成因**（4 个数来自 2.1 μm 细扫，不是从 LR 猜的）"
    MODE_DESC["given_perm"] = "实测成因但**配错**（证伪对照）"
    MODE_DESC["gi"] = "**成因反演**（实测 c + 成因一致性损失 Φ(x̂)=c）"
    MODE_DESC["given7"] = "**7 维实测成因**（4 成分 + χ/团密度/厚度 排布量，第 15 步）"
    MODE_DESC["gi7"] = "**7 维成因反演**（7 维实测 c + 6 项可微 Φ，含软 χ 与 2V/S）"
    MODE_DESC["given7_perm"] = "7 维实测但**排布 3 维配错**（成分正确，证伪排布是否被用上）"
    # ⚠ 用 .get 而不是 []：新增变体时忘了补描述就会一起步 KeyError 崩掉，
    #   实测 filled 全组 8 个运行就是这么全军覆没的。描述不该有杀伤力。
    print(f"{MODE_DESC.get(cfg.mode, cfg.mode)}  "
          f"参数 {npar/1e6:.2f} M   设备 cuda:{cfg.gpu} "
          f"{torch.cuda.get_device_name(dev)}")
    print(f"日志 {out/'log.jsonl'}   体数据 {out/'dumps'}   检查点 {out/'ckpt'}")
    print("=" * 84, flush=True)

    t0, tlast, it = time.time(), time.time(), iter(dl)
    try:
        while step < cfg.steps and not STOP["flag"]:
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            lr_ = b["lr"].to(dev, non_blocking=True)
            hr_ = b["hr"].to(dev, non_blocking=True)
            ct = b["c"].to(dev, non_blocking=True)
            if cfg.fake_c:
                # 第 28 步证伪对照：4 个任意 LR 统计量顶替 4 个地质数。
                # 同样是 4 个数、同样可从粗扫图算出，唯一差别是没有地质含义。
                ct = _fake_c(lr_, cfg._fk_mu, cfg._fk_sd)
            w = b["w"].to(dev, non_blocking=True)
            wn = b["win"].to(dev, non_blocking=True)
            gt_field = cfield_target(hr_, wn, cfg.geo) if cfg.geo else None

            with torch.amp.autocast("cuda", enabled=bool(cfg.amp)):
                # 第 26 步 WTA：抽 K 个候选，只对最接近真值的那个回传梯度。
                # 先 no_grad 挑赢家（显存只占一次前向），再只对赢家重跑带梯度的一次。
                if cfg.wta > 1:
                    with torch.no_grad():
                        Bw = lr_.shape[0]
                        zs = [torch.randn(Bw, model.g.nz, *hr_.shape[-3:],
                                          device=lr_.device, dtype=torch.float32)
                              for _ in range(cfg.wta)]
                        errs = []
                        for zk in zs:
                            ok = model(lr_, z=zk, c_true=ct, geo_t=gt_field)
                            errs.append((ok["hr"] - hr_).abs().flatten(1).mean(1))
                        win = torch.stack(errs, 0).argmin(0)     # 逐样本各自的赢家
                        zbest = torch.stack([zs[int(win[i])][i] for i in range(Bw)], 0)
                    o = model(lr_, z=zbest, c_true=ct, geo_t=gt_field)
                else:
                    o = model(lr_, c_true=ct, geo_t=gt_field)
                # 阶段二有模式寻求时，同一 (LR, c) 再抽一次 z：
                # 成因一致性作用在两次的 Φ 均值上（见 compute_losses），
                # 模式寻求作用在两次的 Φ 之差上（见下）。
                o2 = (model(lr_, c_true=ct, geo_t=gt_field)
                      if cfg.w_ms > 0 else None)
                tot, parts, aux = compute_losses(o, hr_, lr_, ct, w, wn, Hk, cfg,
                                                 out2=o2, geo_t=gt_field)
            if D is not None:
                # D 只看**生成的两个频带**（31 μm 以下）—— 三段分治的直接推论：
                # >31 μm 由 A 路确定性复原、有唯一答案，不该受对抗压力；
                # 第一次 pilot 让 D 看整张图，D 靠「>31 μm 对不对」轻松赢，
                # 对抗梯度又流进 A 路把可恢复带拽坏（a_l1 0.125→0.16）。
                # 现在 fake = G 的 mid+fine，real = 真值滤掉 >31 μm 后的残余；
                # 对抗梯度只到 G，A 路碰不到。D 一律看**实测** c（所有变体相同）。
                with torch.no_grad():
                    real_b = (hr_.float() - lowpass(hr_, CUT1_UM)).detach()
                fake_b = (o["mid"] + o["fine"]).float()
                # 噪声线性退火但不到零（d_noise_floor）：pilot5 里 σ 退到 0.016 时 D 饱和、
                # z 敏感度从 0.015 塌回 0.005。D 永远不该看到完全干净的输入。
                sig = max(cfg.d_noise_floor,
                          cfg.d_noise * max(0.0, 1.0 - step / max(cfg.steps, 1)))
                nz = lambda x: x + sig * torch.randn_like(x) if sig > 0 else x
                if step % cfg.d_every == 0:
                    # D 每 d_every 个 G 步才更新一次：不让 D 赢太快（G 每步都拿对抗梯度）
                    ld = hinge_d(D(nz(real_b), ct), D(nz(fake_b.detach()), ct))
                    if cfg.r1 > 0 and step % (cfg.r1_every * cfg.d_every) == 0:
                        # 惰性 R1：只对 real 求 D 的输入梯度，每 r1_every 次 D 更新做一次，
                        # 系数按间隔放大（Karras 2020）
                        rb = nz(real_b).requires_grad_(True)
                        dr = D(rb, ct).sum()
                        g1, = torch.autograd.grad(dr, rb, create_graph=True)
                        r1 = g1.pow(2).sum(dim=(1, 2, 3, 4)).mean()
                        ld = ld + 0.5 * cfg.r1 * cfg.r1_every * r1
                        parts["r1"] = float(r1.detach())
                    optD.zero_grad(set_to_none=True)
                    ld.backward()
                    optD.step()
                else:
                    with torch.no_grad():
                        ld = hinge_d(D(nz(real_b), ct), D(nz(fake_b), ct))   # 只记录
                # G 的对抗项：D 的参数会被这次 backward 挂上梯度，
                # 但下一轮 optD.zero_grad 会清掉，不会被 optD.step 用到。
                lg = hinge_g(D(fake_b, ct))
                tot = tot + cfg.w_adv * lg.to(tot.dtype)
                parts["d_loss"] = float(ld.detach())
                parts["g_adv"] = float(lg.detach())
                parts["d_sig"] = sig
            if cfg.w_ms > 0:
                # 模式寻求（MSGAN, Mao 2019），但作用在 **Φ** 上而不是体素上：
                # 冒烟实测逐体素差起步就有 0.16 —— 高频抖动轻松满足体素级差异，
                # 而那正是诊断出的「无意义的 z 敏感」；要惩罚的是「换 z 不换成因」。
                # z 在 forward 内部抽，分母 |z1−z2| 在期望上是常数，省掉。
                dphi = (aux["ph1"] - aux["ph2"]).abs().mean()
                ms = 1.0 / (dphi + 1e-3)
                tot = tot + cfg.w_ms * ms.to(tot.dtype)
                parts["ms"] = float(ms.detach())
                parts["z_dphi"] = float(dphi.detach())      # 目标：抬到真值块间散布量级 ~0.02
            opt.zero_grad(set_to_none=True)
            scaler.scale(tot).backward()
            scaler.unscale_(opt)
            # none 变体没有编码器（model.e is None），必须跳过，
            # 否则一起步就 AttributeError —— 实测 SC_none 在第 0 步崩掉。
            gn = {n: float(torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0))
                  for n, m in (("A", model.a), ("E", model.e), ("G", model.g))
                  if m is not None}
            scaler.step(opt); scaler.update()
            step += 1

            # 心跳：死机后一看就知道停在哪一步、什么状态
            (out / "heartbeat.json").write_text(json.dumps(dict(
                step=step, t=time.strftime("%Y-%m-%d %H:%M:%S"),
                loss=float(tot.detach()),
                mem_gb=torch.cuda.memory_allocated(dev) / 1e9)), encoding="utf-8")

            if step % cfg.log_every == 0:
                dt = time.time() - tlast; tlast = time.time()
                cp = o["c"].detach().float()
                rec = dict(step=step, t=time.strftime("%H:%M:%S"),
                           elapsed_min=round((time.time() - t0) / 60, 2),
                           total=round(float(tot.detach()), 5),
                           **{k: round(v, 5) for k, v in parts.items()},
                           grad=  {k: round(v, 3) for k, v in gn.items()},
                           lr=opt.param_groups[0]["lr"],
                           mem_gb=round(torch.cuda.max_memory_allocated(dev) / 1e9, 2),
                           peak_gb=round(torch.cuda.max_memory_allocated(dev) / 1e9, 2),
                           it_s=round(cfg.log_every / max(dt, 1e-9), 2))
                if cfg.mode in ("bottleneck", "sealed"):
                    # 瓶颈学到没有 —— 逐变量看预测 vs 真值，这是机制是否成立的直接证据
                    rec["c_pred"] = [round(float(x), 4) for x in cp.mean(0)]
                    rec["c_true"] = [round(float(x), 4) for x in ct.mean(0)]
                    rec["c_mae"] = [round(float(x), 4)
                                    for x in (cp - ct).abs().mean(0)]
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n"); logf.flush()
                s = "  ".join(f"{k} {v:.4f}" for k, v in parts.items())
                extra = ("  ĉMAE " + "/".join(f"{x:.3f}" for x in rec["c_mae"])
                         if "c_mae" in rec else "")
                print(f"[{step:6d}] 总 {rec['total']:.4f}  {s}{extra}  "
                      f"{rec['it_s']:.1f} it/s  {rec['mem_gb']:.1f}G", flush=True)

            if step % cfg.dump_every == 0:
                # 存真实体数据，以后画图直接读，不用重跑
                with torch.no_grad():
                    ta, tm, tf = bands(hr_[:1].float())
                np.savez_compressed(
                    out / "dumps" / f"step{step:07d}.npz",
                    lr=lr_[:1].float().cpu().numpy(), hr=hr_[:1].float().cpu().numpy(),
                    pred=o["hr"][:1].detach().float().cpu().numpy(),
                    pred_a=o["a"][:1].detach().float().cpu().numpy(),
                    pred_mid=o["mid"][:1].detach().float().cpu().numpy(),
                    pred_fine=o["fine"][:1].detach().float().cpu().numpy(),
                    true_a=ta.cpu().numpy(), true_mid=tm.cpu().numpy(),
                    true_fine=tf.cpu().numpy(),
                    c_true=ct[:1].cpu().numpy(),
                    c_pred=o["c"][:1].detach().float().cpu().numpy(),
                    gid=np.array(b["gid"][0]), step=step)
                print(f"  ↳ 存体数据 dumps/step{step:07d}.npz", flush=True)

            if step % cfg.ckpt_every == 0:
                save_ckpt(out / "ckpt" / f"step{step:07d}.pt", model, opt, step,
                          cfg, best)
                if D is not None:
                    tmpD = out / "ckpt" / f"D{step:07d}.pt.tmp"
                    torch.save(dict(D=D.state_dict(), optD=optD.state_dict()), tmpD)
                    os.replace(tmpD, out / "ckpt" / f"D{step:07d}.pt")
                cks = sorted((out / "ckpt").glob("step*.pt"))
                for old in cks[:-cfg.keep_ckpt]:
                    old.unlink()
                    dold = old.with_name(old.name.replace("step", "D", 1))
                    if dold.exists():
                        dold.unlink()
                print(f"  ↳ 检查点 ckpt/step{step:07d}.pt", flush=True)

    except torch.cuda.OutOfMemoryError:
        print("\n❌ 显存不足。降 --bs，或给 PairDataset 传 hr_crop=56 只取中心块。",
              flush=True)
        logf.write(json.dumps(dict(step=step, error="OOM")) + "\n")
        raise
    except Exception:
        tb = traceback.format_exc()
        print("\n❌ 崩了：\n" + tb, flush=True)
        logf.write(json.dumps(dict(step=step, error=tb), ensure_ascii=False) + "\n")
        raise
    finally:
        save_ckpt(out / "ckpt" / f"step{step:07d}.pt", model, opt, step, cfg, best)
        logf.close()
        print(f"\n已存最终检查点 step{step:07d}.pt（共 {step} 步，"
              f"{(time.time()-t0)/60:.1f} min）", flush=True)


if __name__ == "__main__":
    main()
