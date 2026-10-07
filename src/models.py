"""成因瓶颈超分网络：A 路 + E 编码器 + 硬瓶颈 + G 生成器，以及消融用的自由特征变体。

三段分治（边界都是实测的，不是设定）
    > 31 μm    可恢复带   A 路负责，确定性，L1
    31→15 μm   可推断带   G 负责，受 ĉ 约束的条件生成
    < 15 μm    不可约带   G 负责，但只匹配统计量

尺寸
    LR 36³ @ 14 μm（0.5 mm 立方，正好一个 ĉ 格）
    HR 112³ @ 2.0 μm —— **只覆盖 LR 中心的 16 个粗体素**（224 μm），不是整块。
    所以 A 路要「看 36³ 的上下文、只产出中心 16³ 对应的 112³」。

硬瓶颈
    E 输出**只有 4 个数**，E 与 G 之间没有任何跳连。
    关于这块岩石的一切只能从这 4 个数过去 —— 这是全文机制所在。
    消融时把它换成 64 维无监督自由特征（FreeBottleneck），其余全同。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

V_COARSE, V_FINE, UP = 14.0, 2.0, 7
CUT1_UM, CUT2_UM = 31.0, 15.0        # 两条实测天花板
N_C = 4                              # ĉ：IGV, f_pore, mu_inter, f_dense

PORE_CUT = 0.50                      # 与 cfield.py 同一个物理判据
TAU_PHI = 0.02                       # 成因泛函的阈值软化宽度，见 genesis_phi
GIVEN_MODES = ("given", "given_perm", "gi")   # 条件用**实测** c 的那一族
ZERO_NOISE = False      # 第 34 步：True 时逐层注噪也关掉（与 zero_z 配套）
Z_GLOBAL = False        # 第 36 步：True 时两条噪声通道都退化成每块一个全局数
BOTH_MODES = ("both", "both_perm")   # 第 29 步：cond = [ĉ, 实测c]，8 维

# ---- 第 15 步：排布类成因变量（Minkowski / 拓扑一族），c 从 4 维扩到 7 维 ----
# 准入实测（350 块）：整块信度 χ 0.888 / 团密度 0.918 / 厚度 0.889（各向异性是 −0.02），
# 粗扫 + 现有 4 个 c 读不出：χ 84%、团密度 68%、厚度 30% 为真新信息；
# 三者合起来把块连通性的解释力从 R²=0.60 抬到 0.69。可加性 ⇒ REV 小，这是关键。
N_C_ARR = 7                          # IGV, f_pore, mu_inter, f_dense, chi, ncomp, thick
N_PHI_ARR = 6                        # Φ 可微项：前 4 + chi_soft + thick(2V/S)；ncomp 不可微，只做条件
ARR_MODES = ("given7", "gi7", "given7_perm")
CHI_SCALE = 1e6                      # χ 按每 1e6 体素计
# 第一次 7 维 pilot 失败的原因：排布量原始量级 χ ~2800（std 2041）、团密度 ~2600（std 1707）、
# 厚度 ~4 μm（std 2.1），而成分量 ~0.1（std 0.05–0.11）—— 差四个数量级。直接送进 SPADE，
# 零填充的新权重一拿到梯度就把调制层淹掉：4 个成分旋钮全塌（IGV +0.089 → +0.006）、
# 离散度 0.78 → 0.60、排布配错与配对无差别。**进 SPADE 前按固定常数缩到成分量的尺度**
# （每维 std ≈ 0.06）；同一套常数也用于 gc 损失。固定常数而非 batch 统计：全折一致、可复现。
ARR_SCALE = (34000.0, 28000.0, 35.0)   # 第 15 步的 chi, ncomp, thick（已证伪，保留对照）
# 第 18 步的候选列，按实测跨块 std 缩到成分量的尺度（目标 std ≈ 0.06）。
# 缩放必须按实测定 —— 第 15 步没缩放，2000 倍的输入把整个调制层淹掉，4 个成分旋钮全塌。
ARR_SCALE2 = {"pore_env": 3.53, "rim_idx": 1.69, "skel_len": 73.05,
              "het_igv": 1.05, "COPL": 2.08, "CEPL": 1.97, "chi": 30462.75}


@torch.amp.autocast("cuda", enabled=False)
def chi_soft(x, tau=TAU_PHI):
    """软 Euler 示性数密度（每 1e6 体素）。立方复形 χ = n0 − n1 + n2 − n3
    （体素 − 邻接对 + 满 2×2 方 − 满 2×2×2 立方），对应 6-连通孔隙前景，
    与 evaluate.connectivity 的标号一致。软版把指示函数换成 sigmoid，乘积可微。
    实测 τ=0.02 与硬版相关 0.9987（有恒定偏置，故目标一律用软泛函本身定义）。"""
    x = x.float()
    s = torch.sigmoid((PORE_CUT - x) / tau)                    # (B,1,D,H,W) 孔隙软指示
    d = (-3, -2, -1)
    n0 = s.sum(d)
    n1 = ((s[..., 1:, :, :] * s[..., :-1, :, :]).sum(d) +
          (s[..., :, 1:, :] * s[..., :, :-1, :]).sum(d) +
          (s[..., :, :, 1:] * s[..., :, :, :-1]).sum(d))
    n2 = ((s[..., 1:, 1:, :] * s[..., :-1, 1:, :] * s[..., 1:, :-1, :] * s[..., :-1, :-1, :]).sum(d) +
          (s[..., 1:, :, 1:] * s[..., :-1, :, 1:] * s[..., 1:, :, :-1] * s[..., :-1, :, :-1]).sum(d) +
          (s[..., :, 1:, 1:] * s[..., :, :-1, 1:] * s[..., :, 1:, :-1] * s[..., :, :-1, :-1]).sum(d))
    n3 = (s[..., 1:, 1:, 1:] * s[..., :-1, 1:, 1:] * s[..., 1:, :-1, 1:] * s[..., 1:, 1:, :-1] *
          s[..., :-1, :-1, 1:] * s[..., :-1, 1:, :-1] * s[..., 1:, :-1, :-1] * s[..., :-1, :-1, :-1]).sum(d)
    nvox = float(x.shape[-1] * x.shape[-2] * x.shape[-3])
    return ((n0 - n1 + n2 - n3) / nvox * CHI_SCALE)[:, 0]


@torch.amp.autocast("cuda", enabled=False)
def chi_local(x, n=4, tau=TAU_PHI):
    """把体切成 n³ 个子块，各算软 χ，返回 (B, n³)。

    第 27 步：批级 χ 一个批次只给 1 个标量，信号太弱；子块版给 n³ 个。
    子块直接 reshape 进 batch 维，复用 chi_soft，无额外实现风险。"""
    B, C, D, H, W = x.shape
    d, h, w = D // n, H // n, W // n
    y = x[:, :, :d * n, :h * n, :w * n]
    y = y.reshape(B, C, n, d, n, h, n, w)
    y = y.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B * n ** 3, C, d, h, w)
    return chi_soft(y, tau).reshape(B, n ** 3)


@torch.amp.autocast("cuda", enabled=False)
def thick_soft(x, tau=0.05, vox_um=V_FINE):
    """厚度代理 2V/S（μm）：孔隙体积 / 孔隙表面积。与 EDT 局部厚度相关 0.902。"""
    x = x.float()
    s = torch.sigmoid((PORE_CUT - x) / tau)
    gz = s[..., 1:, :, :] - s[..., :-1, :, :]
    gy = s[..., :, 1:, :] - s[..., :, :-1, :]
    gx = s[..., :, :, 1:] - s[..., :, :, :-1]
    # 三个方向裁到公共尺寸再合成梯度模
    n = min(gz.shape[-3], gy.shape[-3], gx.shape[-3]); m = min(gz.shape[-2], gy.shape[-2], gx.shape[-2])
    k = min(gz.shape[-1], gy.shape[-1], gx.shape[-1])
    g = torch.sqrt(gz[..., :n, :m, :k] ** 2 + gy[..., :n, :m, :k] ** 2 + gx[..., :n, :m, :k] ** 2 + 1e-12)
    d = (-3, -2, -1)
    sv = g.mean(d) / vox_um
    return (2.0 * s.mean(d) / sv.clamp_min(1e-9))[:, 0]


def genesis_phi_arr(x, lo, hi):
    """7 维条件对应的 6 项可微泛函：前 4 项同 genesis_phi，再接 chi_soft 与 thick_soft。
    返回 (B, 6)。团密度（c 的第 6 维）不可微，不在此列。"""
    return torch.cat([genesis_phi(x, lo, hi), chi_soft(x)[:, None], thick_soft(x)[:, None]], 1)


# --------------------------------------------------- 成因测量泛函（可微）

@torch.amp.autocast("cuda", enabled=False)
def genesis_phi(x, lo, hi, tau=TAU_PHI):
    """把 cfield.cvals 写成可微形式，于是能作用在**生成出来的图**上。

    这是「准入规则」终于兑现的地方：当初之所以只收下能写成线性泛函或低阶矩的
    变量（体积分数、区域均值），踢掉 D50 / 分选 / 配位数那些实例几何量，
    就是因为前者对图像可微 —— 可以反过来当约束用，后者不行。

    与 cfield.cvals 逐字对应，只把硬阈值换成 sigmoid：
        IGV      = 1 − P(lo ≤ u < hi)        骨架窗之外
        f_pore   = P(u < 0.5Δ)               可分辨孔隙
        mu_inter = E[u | u < lo]             粒间区平均衰减
        f_dense  = P(u ≥ hi)                 高密度矿物

    τ = 0.02 是实测定的：在 560 个 patch 上与硬阈值的偏差
    IGV 0.0012 / f_pore 0.0007 / f_dense 0.0003 / mu_inter 0.0090，
    可以忽略；τ = 0.05 就把 mu_inter 偏掉 0.058，不能用。

    ⚠ 必须 fp32：阈值宽度 0.02 意味着 x/τ 可达 ±50，且要在 112³ = 1.4M 个
    体素上求和，半精度既容易溢出又丢有效位。

    x  (B,1,D,H,W) Δ 单位     lo/hi  (B,) 每样品的骨架窗
    返回 (B, 4)
    """
    x = x.float()
    lo = lo.float().reshape(-1, 1, 1, 1, 1)
    hi = hi.float().reshape(-1, 1, 1, 1, 1)
    d = (-3, -2, -1)
    p_lo = torch.sigmoid((x - lo) / tau)          # u ≥ lo
    p_hi = torch.sigmoid((x - hi) / tau)          # u ≥ hi
    p_pore = torch.sigmoid((PORE_CUT - x) / tau)  # u < 0.5
    p_inter = 1.0 - p_lo                          # u < lo
    mu_inter = (x * p_inter).sum(d) / p_inter.sum(d).clamp_min(1e-3)
    return torch.stack([1.0 - (p_lo * (1.0 - p_hi)).mean(d),
                        p_pore.mean(d), mu_inter, p_hi.mean(d)], -1)[:, 0]


# ------------------------------------------------------------------ 频带

@torch.amp.autocast("cuda", enabled=False)
def lowpass(x, lam_um, vox_um=V_FINE):
    """理想低通：只保留波长 > lam_um 的内容。与项目里测天花板用的是同一个定义。

    ⚠ 必须跑 float32：cuFFT 在半精度下只支持 2 的幂尺寸，而我们的 112 不是，
    AMP 开着会直接报错。频域运算一律退回 fp32。"""
    x = x.float()
    n = x.shape[-1]
    k = torch.fft.fftfreq(n, device=x.device)
    kk = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2
                    + k[None, None, :] ** 2)
    lam = vox_um / kk.clamp_min(1e-9)
    m = (lam > lam_um).to(x.dtype)
    return torch.fft.ifftn(torch.fft.fftn(x, dim=(-3, -2, -1)) * m,
                           dim=(-3, -2, -1)).real


def bands(hr):
    """把真值切成三段，供分段损失用。返回 (>31, 31-15, <15)。"""
    a = lowpass(hr, CUT1_UM)
    b = lowpass(hr, CUT2_UM) - a
    return a, b, hr - a - b


# ------------------------------------------------------------------ 积木

class Res3d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.GroupNorm(8, c),
                               nn.SiLU(), nn.Conv3d(c, c, 3, padding=1))
        self.n = nn.GroupNorm(8, c)

    def forward(self, x):
        return F.silu(self.n(x + self.b(x)))


class SPADE(nn.Module):
    """空间自适应归一化：先归一化特征，再用 ĉ 算出的缩放与偏移去调制。

    ĉ 训练时是 (B, N_C) 的向量（一个 patch 一个格），推理时可以是
    (B, N_C, z, y, x) 的场 —— 两种都接，这样训练与整体推理用同一套代码。"""

    def __init__(self, c, n_cond, hidden=64):
        super().__init__()
        self.n_cond = n_cond
        if n_cond == 0:                       # none 变体：完全不做条件化
            self.n = nn.GroupNorm(8, c)       # 退化成带仿射的普通归一化
            return
        self.n = nn.GroupNorm(8, c, affine=False)
        # 第一层用 1x1x1：条件在一个 patch 内是常数（推理时也只是 0.5 mm 网格的
        # 平滑场），对它做 3x3x3 卷积不增加任何信息，却让 64 维的消融变体比
        # 4 维的瓶颈变体贵 16 倍 —— 那会让两个变体的算力不对等，对比就不干净了。
        self.mlp = nn.Sequential(nn.Conv3d(n_cond, hidden, 1), nn.SiLU())
        # g/b 也用 1x1x1：整条调制路径对条件逐点，于是「条件在 patch 内是常数」时
        # 可以只在 1³ 上算完再广播，与逐点算**严格等价**。若 g/b 用 3x3x3，
        # 常数场经零填充卷积在内部等于「27 个权重之和 x 值」，而在 1³ 上算只用到
        # 中心权重 —— 两者不等价，快路径就不能用。条件本身是 0.5 mm 的平滑场，
        # 3x3x3 也买不到什么。
        self.g = nn.Conv3d(hidden, c, 1)
        self.b = nn.Conv3d(hidden, c, 1)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)

    def forward(self, x, c):
        if self.n_cond == 0:
            return self.n(x)
        if c.dim() == 2:
            # 训练时 ĉ 在一个 patch 内是常数：只在 1³ 上算完再广播。
            # 因为 mlp/g/b 全是 1x1x1，这与在 112³ 上逐点算**严格等价**，
            # 但省掉把条件展开成 112³ 张量（64 维时那是 360 MB）及其上的全部卷积。
            # 不做这一步时，消融变体（64 维条件）比瓶颈变体（4 维）慢 1.6 倍，
            # 纯属浪费，而且让两个变体的算力不对等。
            h = self.mlp(c[..., None, None, None])
            return self.n(x) * (1 + self.g(h)) + self.b(h)      # 后三维自动广播
        if c.shape[-3:] != x.shape[-3:]:                        # 推理：ĉ 是场
            c = F.interpolate(c, size=x.shape[-3:], mode="nearest")
        h = self.mlp(c)
        return self.n(x) * (1 + self.g(h)) + self.b(h)


class NoiseInject(nn.Module):
    """逐层注噪（StyleGAN 式）：x + s ⊙ ε，ε 逐体素 N(0,1) 单张噪声图，s 逐通道可学。

    为什么需要（第 14 步 pilot3 诊断）：同一块 LR + 同一个 c，换 8 个 z，
    成因变量的波动只有真值块间波动的 1~4% —— 生成器塌成了 (LR, c) 的确定性函数，
    只把输入端的 z 当高频抖动撒在体素上，动不了结构。D 只能在 G *能*画出不同结构时
    才有东西可推，所以离散度卡在 0.74 与 D 强弱无关。
    s 初始为 0 ⇒ 阶段一检查点载入后行为**完全不变**；模式寻求损失再把 s 推起来。"""

    def __init__(self, ch):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, ch, 1, 1, 1))

    def forward(self, x):
        # 第 34 步：逐层注噪是第二条「逃生通道」—— 它自己抽 ε，不走输入端的 z。
        # 只堵 z 时实测残余噪声（0.159）仍压过 c（0.070），故必须一并关掉。
        if ZERO_NOISE:
            return x
        if Z_GLOBAL:
            # 第 36 步：整块共用一个数 —— 噪声只能整体推一下这一层的激活，
            # 不能再逐体素涂纹理。空间结构必须另找来源（LR 形态 / c）。
            e = torch.randn(x.shape[0], 1, 1, 1, 1, device=x.device, dtype=x.dtype)
            return x + self.scale.to(x.dtype) * e
        return x + self.scale.to(x.dtype) * torch.randn_like(x[:, :1])


class GeoField(nn.Module):
    # 第 39 步：粗扫 -> 空间地质场 c_hat (nc, n, n, n)，每格 224/n μm。
    # 通道 = IGV / f_pore / f_dense（mu_inter 跨岩性 R² 为负，剔除）。
    # 卷积在**整块 36³（504 μm）**上算，感受野吃得到目标块外围的上下文；
    # 输出只裁到中心块对应的范围，再池化到 n³ —— 与真值场的定义域对齐。
    def __init__(self, n=4, ch=32, nc=3, blk_frac=16.0 / 36.0):
        super().__init__()
        self.n, self.nc, self.blk_frac = n, nc, blk_frac
        self.body = nn.Sequential(
            nn.Conv3d(1, ch, 3, padding=1), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(ch, ch * 2, 3, padding=1), nn.SiLU(),
            nn.Conv3d(ch * 2, ch * 2, 3, padding=1), nn.SiLU())
        self.head = nn.Conv3d(ch * 2, nc, 1)

    def forward(self, lr):
        h = self.body(lr)
        sz = h.shape[-1]
        k = max(self.n, int(round(sz * self.blk_frac)))
        o = (sz - k) // 2
        h = h[..., o:o + k, o:o + k, o:o + k]
        return self.head(F.adaptive_avg_pool3d(h, self.n))


class LevelHead(nn.Module):
    # 第 38 步：c 驱动**细带振幅** —— 唯一能在不破坏 H 自洽的前提下改变孔隙度的自由度。
    # 硬投影钉住的是 degrade(hr)=lr，即模糊后的图；细带是高频，模糊后基本消失，
    # 故缩放它不违反实测粗扫图，却改变灰度在阈值附近的散布 => 改变孔隙度。
    # 实测（投影后）：x0.5 -> -0.005，x1.5 -> +0.017，x2.0 -> +0.048，
    # 而真值块孔隙度散布是 0.060 => 量程覆盖 81%，且随增益单调。
    # 只看 c、不看 LR => 它造成的任何变化必然是 c 驱动的。
    # 末层零初始化 => 起步输出与原模型逐位相同。
    def __init__(self, n_cond, hidden=32):
        super().__init__()
        self.n_cond = n_cond
        self.f = nn.Sequential(nn.Linear(n_cond, hidden), nn.SiLU(),
                               nn.Linear(hidden, 1))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, fine, c):
        if c.dim() > 2:
            c = c.mean(dim=tuple(range(2, c.dim())))
        g = self.f(c.to(self.f[0].weight.dtype)).to(fine.dtype)
        return fine * (1.0 + g.view(-1, 1, 1, 1, 1))


class ThroatGate(nn.Module):
    """阈值门控残差：让条件的修正量**只作用在临界体素**上（第 20 步）。

    诊断依据（冻结模型实测，CQ 折 24 块）：
      · SPADE 调制幅度 47.5%，通路本身不弱；
      · 但换掉 c 只让**0.65% 的体素**改变「是不是孔隙」的判定；
      · 而连通性完全由阈值穿越决定 —— **条件的力气撒在了不影响连通性的地方**。
    这一次性解释了此前四次失败（pore_env、χ、gi、谱斜率）：变量再好，
    作用不到阈值穿越上，终点就不动。

    头寸（反事实实测）：把 |x−0.5Δ|<0.10 的体素（约 4.89%）全推成孔隙，
    最大团占比 0.436 → 0.490，而真值均值 0.497 —— **系统性偏差几乎消除**。
    更窄的带（0.01~0.02，约 1% 体素）摆动仅 0.015~0.028，无效，
    正对应当前那 0.65%。所以 tau 取 0.10，不能更小。
    上限也要写明：逐块覆盖率仅 29%（真值块间 std 0.234 > 可摆动 0.133），
    **能修系统性偏差，修不了块间散布**。

    地质含义：喉道开不开取决于胶结，而喉道正是强度贴着阈值的那一小撮体素。

    out 零初始化 ⇒ 起步恒等，旧检查点热启动后行为完全不变。"""

    def __init__(self, ch, n_cond, tau=0.10):
        super().__init__()
        self.tau = tau
        self.sp = SPADE(ch, n_cond)
        self.out = nn.Conv3d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, feat, c):
        w = torch.exp(-((x - PORE_CUT) / self.tau) ** 2)   # 临界带的门
        return x + w * self.out(F.silu(self.sp(feat, c)))


class SpadeRes(nn.Module):
    def __init__(self, c, n_cond):
        super().__init__()
        self.s1, self.s2 = SPADE(c, n_cond), SPADE(c, n_cond)
        self.c1 = nn.Conv3d(c, c, 3, padding=1)
        self.c2 = nn.Conv3d(c, c, 3, padding=1)

    def forward(self, x, c):
        h = self.c1(F.silu(self.s1(x, c)))
        return x + self.c2(F.silu(self.s2(h, c)))


# ------------------------------------------------------------------ A 路

class APath(nn.Module):
    """可恢复带复原：看 36³ 上下文，产出中心 16³ 对应的 112³，限带在 >31 μm。

    为什么要看全 36³ 再裁中心：边界处的结构需要上下文才能复原，
    直接喂 16³ 会在块的边缘出现接缝 —— 推理时按格平铺会很明显。

    为什么通道数逐级减半：这一路只负责 λ>31 μm，也就是 ≥15.5 个高分辨体素的
    内容 —— **在 112³ 上堆厚通道是纯浪费**。粗的信息用粗的网格算。"""

    def __init__(self, ch=48, nres=4, ctx=36, core=16):
        super().__init__()
        self.ctx, self.core = ctx, core
        self.stem = nn.Sequential(nn.Conv3d(1, ch, 3, padding=1), nn.SiLU())
        self.body = nn.Sequential(*[Res3d(ch) for _ in range(nres)])
        self.up1 = nn.Sequential(nn.Conv3d(ch, ch // 2, 3, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.Conv3d(ch // 2, ch // 4, 3, padding=1), nn.SiLU())
        self.out = nn.Conv3d(ch // 4, 1, 3, padding=1)

    def forward(self, lr, ret_feat=False):
        h = self.body(self.stem(lr))
        o = (self.ctx - self.core) // 2
        sl = slice(o, o + self.core)
        h = h[..., sl, sl, sl]
        feat = h                     # 第 52 步：中心 16³ 的残差块深层特征，供 G 定位细节
        n = self.core * UP
        h = self.up1(F.interpolate(h, size=(n // 4,) * 3, mode="trilinear",
                                   align_corners=False))
        h = self.up2(F.interpolate(h, size=(n // 2,) * 3, mode="trilinear",
                                   align_corners=False))
        h = F.interpolate(h, size=(n,) * 3, mode="trilinear", align_corners=False)
        base = F.interpolate(lr[..., sl, sl, sl], size=(n,) * 3,
                             mode="trilinear", align_corners=False)
        return (base + self.out(h), feat) if ret_feat else base + self.out(h)


# ------------------------------------------------------------------ E 编码器

class Encoder(nn.Module):
    """LR 36³ -> ĉ（N_C 个数）。全卷积 + 全局池化，推理时可扫成场。"""

    def __init__(self, ch=32, n_out=N_C):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1, ch, 3, stride=1, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv3d(ch, ch * 2, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 2), nn.SiLU(),
            Res3d(ch * 2),
            nn.Conv3d(ch * 2, ch * 4, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 4), nn.SiLU(),
            Res3d(ch * 4))
        self.head = nn.Sequential(nn.Linear(ch * 4, 128), nn.SiLU(),
                                  nn.Linear(128, n_out))

    def forward(self, lr):
        h = self.f(lr)
        return self.head(h.mean(dim=(-3, -2, -1)))


class FreeBottleneck(nn.Module):
    """消融用：同样的主干，但输出 64 维**无监督**自由特征。

    这是全文最关键的对照 —— 预期它在见过的岩性上更好、在没见过的岩性上明显崩。"""

    def __init__(self, ch=32, dim=64):
        super().__init__()
        self.e = Encoder(ch, n_out=dim)
        self.dim = dim

    def forward(self, lr):
        return self.e(lr)


# ------------------------------------------------------------------ G 生成器

class Generator(nn.Module):
    """输入只有：A 路输出 + ĉ + 噪声 z。**没有任何来自 E 的跳连。**

    渐进式：重的条件化计算放在低分辨率，逐级上采样，最后才到 112³。
    这不只是为了快 —— **物理上本来就该这样**：
      可推断带 31→15 μm 的波长 ≥ 7.5 个高分辨体素，在 28³ 网格（8 μm/体素）
      上做条件化完全够；只有 <15 μm 的不可约带才需要最细的网格，
      而那一带按实测有 92% 由样品级统计决定（第 10 步），用一个轻量的
      噪声驱动头就够，不需要厚通道。

    实测：全程 112³ 的老版本 0.34 it/s，改成渐进式后快一个量级。"""

    def __init__(self, n_cond=N_C, ch=64, nres=4, nz=8, core=16, up=UP,
                 zero_z=False, z_global=False, level=False,
                 see_a=True, norm_a=False, throat=False, afeat=0):
        super().__init__()
        # see_a=False 是 **sealed** 变体：G 的主体看不到 A 路输出，
        # A 路只在最后相加。这才让瓶颈真正生效 ——
        # ĉ 与 a_out 都是同一张低分辨图的函数，而低分辨图里只剩 >31 μm 的内容，
        # 所以 **a_out 能传的信息是 ĉ 的超集**：把它喂进 G，那几个数永远冗余，
        # 瓶颈在信息论上恒等于没有。实测 16 个运行的消融全部无差异，就是这个原因。
        # norm_a=True 是 **filled** 变体：G 看得见 A 路，但先做实例归一化。
        #
        # 为什么这样才对：A 路输出里同时含两样东西 ——
        #   形状（颗粒在哪、怎么排）：**必须给 G**，否则细结构会被放到颗粒内部去，
        #                            物理上荒谬；sealed 变体就是这么残废的。
        #   灰度统计（整体多亮、反差多大）：**不能给 G**，因为 ĉ 的四个变量
        #                            (IGV / f_pore / mu_inter / f_dense)
        #                            本质上全是灰度分布的统计量 —— 给了就等于把
        #                            瓶颈要传的东西从旁路又送了一遍。
        # 实例归一化正好只洗掉后者、保留前者：于是 ĉ 成为**必需品**，
        # 瓶颈既真实生效又不残废。A 路给素描，ĉ 给着色。
        self.see_a, self.norm_a = see_a, norm_a
        self.nz, self.n_full = nz, core * up
        # 第 34 步：堵死噪声这条逃生通道 —— z 恒为 0 后，
        # 给定 LR 之后输出的一切块间变化只能来自 c。
        self.zero_z = bool(zero_z)
        # 第 36 步：z 退化成每块 nz 个数。原来 z 是 8x112^3=1124 万个数（还直接
        # 进细带头），而 c 只有 4 个 —— 优化上地质根本不是噪声的对手。
        self.z_global = bool(z_global)
        n4 = self.n_full // 4                      # 28³，条件化主战场
        self.n4, self.n2 = n4, self.n_full // 2
        self.stem = nn.Conv3d((1 if see_a else 0) + nz, ch, 3, padding=1)
        self.body = nn.ModuleList([SpadeRes(ch, n_cond) for _ in range(nres)])
        self.mid4 = nn.ModuleList([SpadeRes(ch // 2, n_cond) for _ in range(2)])
        self.red1 = nn.Conv3d(ch, ch // 2, 3, padding=1)
        self.mid2 = SpadeRes(ch // 4, n_cond)
        self.red2 = nn.Conv3d(ch // 2, ch // 4, 3, padding=1)
        self.out_mid = nn.Conv3d(ch // 4, 1, 3, padding=1)
        # <15 μm 不可约带：轻量噪声头，只在最细网格上跑一层
        self.fine_head = nn.Sequential(nn.Conv3d(nz + ch // 4, ch // 4, 3, padding=1),
                                       nn.SiLU())
        self.fine_sp = SPADE(ch // 4, n_cond)
        self.out_fine = nn.Conv3d(ch // 4, 1, 3, padding=1)
        for m in (self.out_mid, self.out_fine):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        # 逐层注噪（初始尺度 0，对旧检查点透明；见 NoiseInject）
        self.ni_body = nn.ModuleList([NoiseInject(ch) for _ in range(nres)])
        self.ni_mid4 = nn.ModuleList([NoiseInject(ch // 2) for _ in range(2)])
        self.ni_mid2 = NoiseInject(ch // 4)
        self.ni_fine = NoiseInject(ch // 4)
        # 第 20 步：阈值门控残差（零初始化，默认关）
        self.throat = ThroatGate(ch // 4, n_cond) if throat else None
        # 第 38 步：水平通道（零初始化，默认关）
        self.level = LevelHead(n_cond) if level else None
        # 第 52 步：接入 A 路深层特征（学 EDSR：重建要用粗扫的深层特征，而不是一张平滑图）。
        # 终评看到的毛病是「斑块、多余小孔、位置不对」：G 只看 a_out + 噪声，不知道孔该落在哪。
        # 特征先做实例归一化再进 —— 与 norm_a 同一道理：只传形状/位置，洗掉灰度统计，
        # 灰度统计是地质 c 的职责，不洗则 c 被旁路冗余。1x1x1 零初始化 ⇒ 热启动时逐位等价。
        self.fa4 = self.fa2 = None
        if afeat:
            self.fa4 = nn.Conv3d(afeat, ch, 1)
            self.fa2 = nn.Conv3d(afeat, ch // 2, 1)
            for m in (self.fa4, self.fa2):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, a_out, c, z=None, a_feat=None):
        B = a_out.shape[0]
        fa = None
        if self.fa4 is not None and a_feat is not None:
            fa = F.instance_norm(a_feat.float()).to(a_out.dtype)
        if self.zero_z:
            z = torch.zeros(B, self.nz, *a_out.shape[-3:], device=a_out.device,
                            dtype=a_out.dtype)
        elif self.z_global and z is None:
            # 每块一个全局向量再广播 => 随机性只定「方向」，不定「逐体素内容」
            z = torch.randn(B, self.nz, 1, 1, 1, device=a_out.device,
                            dtype=a_out.dtype).expand(-1, -1, *a_out.shape[-3:])
        elif z is None:
            z = torch.randn(B, self.nz, *a_out.shape[-3:], device=a_out.device,
                            dtype=a_out.dtype)
        # 降到 28³ 做条件化
        if self.see_a:
            a_in = a_out
            if self.norm_a:      # 洗掉灰度统计，只留形状
                m = a_out.mean(dim=(-3, -2, -1), keepdim=True)
                sd = a_out.std(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-5)
                a_in = (a_out - m) / sd
            x = torch.cat([a_in, z], 1)
        else:
            x = z
        h = self.stem(F.interpolate(x, size=(self.n4,) * 3, mode="trilinear",
                                    align_corners=False))
        if fa is not None:
            h = h + self.fa4(F.interpolate(fa, size=(self.n4,) * 3, mode="trilinear",
                                           align_corners=False))
        for b, ni in zip(self.body, self.ni_body):
            h = ni(b(h, c))
        h = self.red1(F.interpolate(h, size=(self.n2,) * 3, mode="trilinear",
                                    align_corners=False))
        if fa is not None:
            h = h + self.fa2(F.interpolate(fa, size=(self.n2,) * 3, mode="trilinear",
                                           align_corners=False))
        for b, ni in zip(self.mid4, self.ni_mid4):
            h = ni(b(h, c))
        h = self.red2(F.interpolate(h, size=(self.n_full,) * 3, mode="trilinear",
                                    align_corners=False))
        h = self.ni_mid2(self.mid2(h, c))
        mid = self.out_mid(h)
        f = self.fine_sp(self.ni_fine(self.fine_head(torch.cat([z, h], 1))), c)
        fine = self.out_fine(F.silu(f))
        if self.level is not None:
            fine = self.level(fine, c)          # 细带振幅由 c 调制（零初始化）
        hr = a_out + mid + fine
        if self.throat is not None:
            hr = self.throat(hr, h, c)
        return mid, fine, hr


# ------------------------------------------------------------------ 判别器

class Discriminator(nn.Module):
    """投影式条件判别器 D(x, c)（第 14 步定案，让地质管排布的机制）。

    3D PatchGAN 主干 + 谱归一化，112³ 四次下采样到 7³ 逐块判别。
    条件 c 经 MLP 嵌入后与主干特征做内积（Miyato & Koyama 2018）：
        D(x, c) = f(x) + <φ(c), h(x)>
    投影项逼 D 真的用 c 去判「这是不是**成因为 c 的**岩石」，
    而不是退化成无条件的「像不像岩石」。

    为什么排布只能这样管：排布类描述子在 224 μm 块上测不准（第 13 步 §7.2，
    子块相关 r=−0.02，REV 问题），不能写成显式约束。判别器从数据里学
    p(排布 | 成因)，把「胶结重 → 孔不连通」这类关系在数据里绑定。

    训练时 D 一律接收**实测** c（所有变体相同），只有 G 的条件来源不同 ——
    这样消融差异只能归因于 G 知不知道 c。推理时 D 丢弃。
    全程 fp32：谱归一化的幂迭代在半精度下不稳。"""

    def __init__(self, ch=32, n_cond=N_C):
        super().__init__()
        SN = nn.utils.spectral_norm
        self.n_cond = n_cond
        self.body = nn.Sequential(
            SN(nn.Conv3d(1, ch, 4, 2, 1)), nn.LeakyReLU(0.2, True),          # 56³
            SN(nn.Conv3d(ch, ch * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True),     # 28³
            SN(nn.Conv3d(ch * 2, ch * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True), # 14³
            SN(nn.Conv3d(ch * 4, ch * 8, 4, 2, 1)), nn.LeakyReLU(0.2, True)) # 7³
        self.out = SN(nn.Conv3d(ch * 8, 1, 3, 1, 1))
        if n_cond > 0:
            self.proj = nn.Sequential(nn.Linear(n_cond, 64), nn.SiLU(),
                                      SN(nn.Linear(64, ch * 8)))

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x, c=None):
        h = self.body(x.float())
        o = self.out(h)
        if c is not None and self.n_cond > 0:
            e = self.proj(torch.nan_to_num(c.float()))[:, :, None, None, None]
            o = o + (h * e).sum(1, keepdim=True)
        return o


def hinge_d(d_real, d_fake):
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def hinge_g(d_fake):
    return -d_fake.mean()


# ------------------------------------------------------------------ 整体

class GenesisSR(nn.Module):
    """十个变体，靠 mode 切换。前七个是「条件从 LR 猜」，后三个是「条件是实测」。

      bottleneck   E->4 个物理数，G 看 a_out + ĉ + z        （原主模型）
      free         E->64 维自由特征，其余同上                （原消融）
      none         G 只看 a_out + z，**完全没有条件**        （缺失的对照）
      sealed       E->4 个物理数，**G 完全看不到 a_out**（过度矫正，见下）
      sealed_free  E->64 维自由特征，同上
      filled       E->4 个物理数，**G 看归一化后的 a_out**（保形状、去灰度统计）
      filled_free  E->64 维自由特征，同上   ← **filled vs filled_free 才是最终的消融**

    前三个共享同一个漏洞：a_out 是低分辨图的函数，而低分辨图里只剩 >31 μm
    的内容，所以 a_out 能传的信息是 ĉ 的超集 —— 条件是冗余的，
    瓶颈测不出任何东西（实测 7 折 8 个指标全部无差异，p >= 0.45）。
    sealed 把旁路完全堵死，但**矫枉过正**：G 连颗粒在哪都不知道，
    只能生成位置乱放的纹理（孔隙会落到颗粒内部，物理上荒谬）。
    filled 是正解：A 路输出先做实例归一化再喂 G ——
    **形状留下（定位要用），灰度统计洗掉（那正是 ĉ 要传的）**，
    于是 ĉ 成为必需品而 G 仍知道该把细结构放在哪。

    ------------------------------------------------------------------
    但以上七个变体共享一个更深的问题：**条件始终是 ĉ = E(LR)。**
    ĉ 是 LR 的确定性函数，所以无论瓶颈堵得多严，条件都带不进 LR 以外的
    任何信息 —— 换句话说，「地质约束」在这七个变体里只是重参数化，
    不可能变好，消融也不可能有信号。这与检验够不够狠无关。

    真正的地质信息在**实测的 c** 里：c 测在 2.1 μm 细扫上，而 E 从 LR 只能
    猜到 R² = 0.72 / 0.79 / 0.70 / 0.63，**残差就是 21~37% 的新信息**。
    下面三个变体把它放进来，构成一条信息阶梯（架构完全相同，只有那 4 个数
    的来源不同，所以差异只能归因于信息本身）：

      given        c = **实测值**            信息阶梯第 2 级
      given_perm   c = 实测值但**配错**       证伪对照：若与 given 同样好，
                                            说明 G 压根没用这 4 个数
      gi           given + **成因一致性损失**  完整方法：用可微泛函 Φ 要求
                                            Φ(生成图) = c，把条件真正焊在
                                            输出上，而不只是「可用」

    与之配对的对照是 filled（c = E(LR)，第 1 级）和 none（无条件，第 0 级）。
    **filled → given 的落差 = 地质测量值本身的价值；
      given → gi 的落差 = 成因一致性约束的价值；
      given vs given_perm = 因果性证据。**
    """

    def __init__(self, mode="bottleneck", ch_a=48, ch_e=32, ch_g=64, nz=8,
                 arr_cols=None, throat=False, zero_z=False, z_global=False,
                 level=False, geo=0, geo_true=False, seal=False, afeat=False):
        super().__init__()
        global ZERO_NOISE
        ZERO_NOISE = bool(zero_z)           # 与 zero_z 配套：两条噪声通道一起堵
        global Z_GLOBAL
        Z_GLOBAL = bool(z_global)           # 下面构造的 NoiseInject 会读到它
        assert mode in ("bottleneck", "free", "none", "sealed", "sealed_free",
                        "filled", "filled_free") + GIVEN_MODES + ARR_MODES + BOTH_MODES
        self.mode = mode
        self.a = APath(ch=ch_a)
        if mode in ARR_MODES:
            # 条件 = 4 个成分 + arr_cols 里选中的排布列。
            # 第 18 步起列数可变（「一次只加一个」），arr_cols 给名字，缩放按 ARR_SCALE2。
            self.arr_cols = list(arr_cols) if arr_cols else ["chi", "ncomp", "thick"]
            n = N_C + len(self.arr_cols)
            self.e = Encoder(ch_e, n_out=n); ncond = n
        elif mode in BOTH_MODES:
            # 第 29 步：编码器照常预测 4 维 ĉ，条件是 [ĉ, 实测c] 共 8 维。
            # 网络保留 filled 的全部自由度，额外拿到实测 c，用不用由它自己决定。
            self.e = Encoder(ch_e); ncond = 2 * N_C
        elif mode in ("bottleneck", "sealed", "filled") or mode in GIVEN_MODES:
            self.e = Encoder(ch_e); ncond = N_C
        elif mode in ("free", "sealed_free", "filled_free"):
            self.e = FreeBottleneck(ch_e); ncond = self.e.dim
        else:
            self.e = None; ncond = 0
        # 第 39 步：显式地质中间表示。条件由 4 个常数换成 3 通道的空间场。
        self.geo_n = int(geo)
        self.geo_true = bool(geo_true)
        self.geo = GeoField(n=self.geo_n) if self.geo_n and not self.geo_true else None
        if self.geo_n:
            ncond = 3
        self.g = Generator(
            n_cond=ncond, ch=ch_g, nz=nz, zero_z=zero_z, z_global=z_global,
            level=level,
            # 第 40 步：seal 把 see_a 从 mode 里解耦 —— 这样「完全切断」
            # 才能和**实测条件**（given/空间场）组合。原来只有 E 推的 ĉ 能走这条路。
            see_a=(mode not in ("sealed", "sealed_free")) and not seal,
            norm_a=(mode in ("filled", "filled_free") or mode in GIVEN_MODES
                    or mode in ARR_MODES or mode in BOTH_MODES), throat=throat,
            afeat=(ch_a if afeat else 0))
        self.afeat = bool(afeat)

    def _scale_arr(self, cond):
        """排布列按实测 std 缩到成分量的尺度再进 SPADE；成分 4 维原样。
        旋钮试验（c_force）走同一条路，所以扫描仍用原始单位。"""
        cols = getattr(self, "arr_cols", None)
        if not cols or cond.shape[1] != N_C + len(cols):
            return cond
        s = torch.tensor([ARR_SCALE2[c] for c in cols],
                         device=cond.device, dtype=cond.dtype)
        return torch.cat([cond[:, :N_C], cond[:, N_C:] / s], 1)

    def forward(self, lr, z=None, c_true=None, c_force=None, geo_t=None):
        a_feat = None
        if getattr(self, "afeat", False):
            a_out, a_feat = self.a(lr, ret_feat=True)
        else:
            a_out = self.a(lr)
        c_hat = self.e(lr) if self.e is not None else \
            lr.new_zeros(lr.shape[0], 0)
        if c_force is not None:
            # 可控性检验：外部直接拧旋钮，绕过 E 与实测值。
            # 对 64 维自由特征无定义 —— 那 64 维没有任何一维叫「IGV」，
            # 你没法要求它「把 IGV 调到 0.35」。这不是它输了一点点，
            # 是这件事对它**不成立**。
            if c_force.shape[1] != self.g.body[0].s1.n_cond:
                raise RuntimeError(
                    f"{self.mode} 的条件是 {self.g.body[0].s1.n_cond} 维，"
                    f"不能用 {c_force.shape[1]} 维的物理量去拧")
            cond = self._scale_arr(torch.nan_to_num(c_force).to(a_out.dtype))
        elif self.mode in ARR_MODES:
            # 第 15 步：7 维实测条件（4 成分 + χ、团密度、厚度）。
            # given7_perm 只把**后 3 维（排布）配错**，成分保持正确 ——
            # 这样证伪的是「排布信息有没有被用上」，而不是整个条件。
            if c_true is None:
                raise RuntimeError(f"{self.mode} 变体的条件是实测 7 维 c，forward 必须收到 c_true")
            cond = torch.nan_to_num(c_true).to(a_out.dtype)
            if self.mode == "given7_perm":
                # 只把排布列配错，成分保持正确 —— 证伪的是「排布信息有没有被用上」
                cond = torch.cat([cond[:, :N_C], torch.roll(cond[:, N_C:], 1, 0)], 1)
            cond = self._scale_arr(cond)
        elif self.mode in BOTH_MODES:
            # 第 29 步公平检验：网络自己推的 ĉ 与实测 c 并排给它，8 维。
            # both_perm 把**实测那一半**配错，检验实测值的内容是否被用上。
            if c_true is None:
                raise RuntimeError(f"{self.mode} 变体需要实测 c")
            cm = torch.nan_to_num(c_true[:, :N_C]).to(a_out.dtype)
            if self.mode == "both_perm":
                cm = torch.roll(cm, 1, 0)
            cond = torch.cat([c_hat, cm], 1)
        elif self.mode in GIVEN_MODES:
            # 条件是**实测值**，不是从同一张图里猜出来的。
            # 这正是前七个变体测不出任何东西的原因：ĉ = E(LR) 与 a_out 同为
            # LR 的函数，条件在信息论上恒为冗余，换成什么都一样。
            # c 实测在 2.1 μm 细扫上，而 E 只能猜到 R² 0.72/0.79/0.70/0.63 ——
            # 残差 c − E(LR) 就是低分辨图里**根本没有**的那 21~37% 地质信息。
            if c_true is None:
                raise RuntimeError(
                    f"{self.mode} 变体的条件是实测 c，forward 必须收到 c_true")
            cond = torch.nan_to_num(c_true).to(a_out.dtype)
            if self.mode == "given_perm":
                cond = torch.roll(cond, 1, 0)      # 配错：证伪对照，见类注释
        else:
            cond = c_hat
        geo_pred = None
        if self.geo_n:
            # geo_true：直接喂真值场（**上界测试，泄题，只用来测天花板**）
            # 否则：由 GeoField 从粗扫预测，训练时受细扫实测场监督
            if self.geo_true:
                if geo_t is None:
                    raise RuntimeError("geo_true 需要通过 geo_t 传入真值地质场")
                cond = torch.nan_to_num(geo_t).to(a_out.dtype)
            else:
                geo_pred = self.geo(lr)
                cond = geo_pred.to(a_out.dtype)
        mid, fine, hr = self.g(a_out, cond, z, a_feat=a_feat)
        return dict(a=a_out, c=c_hat, cond=cond, mid=mid, fine=fine, hr=hr,
                    geo=geo_pred)


def degrade(hr, kernel, up=UP):
    """HR(2 μm) -> LR(14 μm)：块平均 + 卷实测 H 核。退化一致性损失用。

    kernel 是 results/degradation/H_kernel.npz 里的 9³ 实测核，
    **不要用高斯近似** —— 实测滚降比高斯快得多（拟合最大误差 0.83）。"""
    b, c = hr.shape[0], hr.shape[1]
    blk = F.avg_pool3d(hr, up)
    k = kernel.to(hr.dtype).to(hr.device)[None, None]
    pad = k.shape[-1] // 2
    return F.conv3d(F.pad(blk, (pad,) * 6, mode="replicate"), k)
