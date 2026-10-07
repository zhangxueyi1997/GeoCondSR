# -*- coding: utf-8 -*-
"""第 54 步架构：确定性均值通路 M + 地质调制的残差纹理生成器 G。
分工按「能不能从粗扫推出来」，不再按频率：
  M（均值通路）：粗扫 36³ → 深残差主干 → 裁中心 16³ → 放大到 112³ → 全分辨率尾部卷积 → μ（条件均值，全频段）。
     第 53 步实测：<31 μm 细节有相当部分可从粗扫推出（逐点相关 0.48/0.20、位置对准 0.68/0.59），旧 A 路按设计只管 >31 μm，把它丢了。
  G（残差纹理）：输入 μ（实例归一化）+ M 的深层特征（实例归一化，只传位置/形状）+ 噪声；
     地质 c 经 SPADE 逐层调制 + 幅度头 —— 位置由图像定，形态由地质定。输出零均值残差 r，x = μ + r。
     推不出来的残差占细节方差 88%（第 53 步），地质的主作用点在这里。
G 直接复用 models.Generator（afeat 接口），权重可从 _rot2 的 G 热启动。"""
import torch, torch.nn as nn, torch.nn.functional as F
from models import Res3d, Generator, UP, N_C


class MeanPath(nn.Module):
    def __init__(self, ch=64, nres=12, hc=32, ctx=36, core=16):
        super().__init__()
        self.ctx, self.core, self.ch = ctx, core, ch
        self.stem = nn.Conv3d(1, ch, 3, padding=1)
        self.body = nn.Sequential(*[Res3d(ch) for _ in range(nres)])
        self.fuse = nn.Conv3d(ch, ch, 3, padding=1)
        self.up = nn.Conv3d(ch, hc, 3, padding=1)
        self.tail = nn.Sequential(nn.Conv3d(hc, hc, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(hc, hc, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(hc, 1, 3, padding=1))

    def forward(self, lr, ret_feat=False):
        h = self.stem(lr); h = h + self.fuse(self.body(h))          # 全局残差
        o = (self.ctx - self.core) // 2; sl = slice(o, o + self.core)
        feat = h[..., sl, sl, sl]
        n = self.core * UP
        base = F.interpolate(lr[..., sl, sl, sl], size=(n,) * 3, mode='trilinear', align_corners=False)
        u = F.interpolate(self.up(feat), size=(n,) * 3, mode='trilinear', align_corners=False)
        mu = base + self.tail(F.silu(u))                            # 全分辨率尾部：可表达可推断的细节
        return (mu, feat) if ret_feat else mu


class CleanRB(nn.Module):
    """干净残差块（学 EDSR 的经验：超分回归不做归一化，直通路不经任何变换，修正量缩 0.1）。
    第 54 步实测：Res3d（相加后再 GroupNorm+SiLU）叠 12 块的均值通路，细节位置对准只有 EDSR 的一半（0.37–0.41 vs 0.66–0.72），
    L1 0.19–0.21 vs 0.16–0.18 —— 每块都把直通路归一化一次，灰度信息逐块丢失。"""

    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv3d(c, c, 3, padding=1); self.b = nn.Conv3d(c, c, 3, padding=1)

    def forward(self, x):
        return x + 0.1 * self.b(F.relu(self.a(x)))


class MeanPath2(MeanPath):
    """第 54 步修正版均值通路：主干换成干净残差块（16 块，与 EDSR 深度相当），其余与 MeanPath 相同。"""

    def __init__(self, ch=64, nres=16, hc=32, ctx=36, core=16):
        super().__init__(ch=ch, nres=1, hc=hc, ctx=ctx, core=core)
        self.body = nn.Sequential(*[CleanRB(ch) for _ in range(nres)])


def load_mean(path, dev):
    """按检查点里的参数名自动识别版本（Res3d 版含 body.0.b.0.weight，干净版含 body.0.a.weight）。"""
    sd = torch.load(path, map_location=dev, weights_only=False)['model']
    M = (MeanPath2() if 'body.0.a.weight' in sd else MeanPath()).to(dev)
    M.load_state_dict(sd); return M


class CPred(nn.Module):
    """第 55 步：从粗扫预测地质变量 ĉ（工程可用：推理只需粗扫 36³）。
    监督标签是细扫实测的 c_blk —— 细扫只在训练时作「特权信息」出现，推理时不需要。
    输出为原始量纲（内部按训练折的均值/标准差标准化学习）。"""

    def __init__(self, mu=None, sd=None):
        super().__init__()
        from models import Encoder
        self.e = Encoder(32, n_out=N_C)
        self.register_buffer('mu', torch.zeros(N_C) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.ones(N_C) if sd is None else torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, lr):
        return self.e(lr).float() * self.sd + self.mu


def load_cpred(path, dev):
    m = CPred().to(dev); m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)['model']); return m.eval()


class GeoResSR(nn.Module):
    """推理：x = μ + r(μ, feat, c, z)。评测时再做 8 轮硬 H 投影（与既往同口径）。"""

    def __init__(self, ch_m=64, nres_m=12):
        super().__init__()
        self.m = MeanPath(ch=ch_m, nres=nres_m)
        self.g = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=ch_m)

    def forward(self, lr, c, z=None):
        with torch.no_grad():
            mu, feat = self.m(lr, ret_feat=True)
        mid, fine, x = self.g(mu, c, z, a_feat=feat)
        return dict(mu=mu, r=mid + fine, hr=x)
