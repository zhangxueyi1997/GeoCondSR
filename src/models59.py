# -*- coding: utf-8 -*-
"""第 59 步：近年强基线的三维实现（与 EDSR-3D / SRGAN-3D 同数据接口：输入 36³ 粗扫上下文，输出中心 16³ 对应的 112³）。
SwinIR3D —— SwinIR（Liang et al., 2021）的三维版：3D 窗口自注意力（窗口 6³、相对位置偏置、隔层移位窗口）+ 残差 Swin 组，
            在 36³ 低分辨网格上提特征，取中心 16³，三线性上采样到 112³ 后两层卷积，叠加在三线性插值上（与 EDSR-3D 同一重建头）。
DiffSR3D —— SR3 / SRDiff 式条件扩散（Saharia et al., 2022; Li et al., 2022）：以三线性插值与粗扫编码特征为条件，
            扩散生成「细扫 − 三线性」残差；3D U-Net（112³→56³→28³→14³，14³ 层自注意力），ε 预测、余弦噪声表，DDIM 采样。"""
import math, torch, torch.nn as nn, torch.nn.functional as F

UPN, K = 112, 16


def center_up(lr, k=K, n=UPN):
    o = (lr.shape[-1] - k) // 2; s = slice(o, o + k)
    return F.interpolate(lr[..., s, s, s], size=(n,) * 3, mode='trilinear', align_corners=False)


# ------------------------------------------------------------------ SwinIR-3D
def win_part(x, w):          # (B,D,H,W,C) -> (B, nW, w³, C)
    B, D, H, W, C = x.shape
    x = x.view(B, D // w, w, H // w, w, W // w, w, C).permute(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(B, -1, w ** 3, C)


def win_merge(x, w, D, H, W):   # (B, nW, w³, C) -> (B,D,H,W,C)
    B, C = x.shape[0], x.shape[-1]
    x = x.view(B, D // w, H // w, W // w, w, w, w, C).permute(0, 1, 4, 2, 5, 3, 6, 7)
    return x.reshape(B, D, H, W, C)


class WinAttn3D(nn.Module):
    def __init__(s, dim, w, heads):
        super().__init__()
        s.w, s.h = w, heads; s.qkv = nn.Linear(dim, 3 * dim); s.proj = nn.Linear(dim, dim)
        s.bias = nn.Parameter(torch.zeros(heads, (2 * w - 1) ** 3)); nn.init.trunc_normal_(s.bias, std=0.02)
        c = torch.stack(torch.meshgrid(*[torch.arange(w)] * 3, indexing='ij')).flatten(1)
        r = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (w - 1)
        s.register_buffer('idx', r[..., 0] * (2 * w - 1) ** 2 + r[..., 1] * (2 * w - 1) + r[..., 2], persistent=False)

    def forward(s, x, mask=None):   # x (B, nW, N, C); mask (nW, N, N) or None
        B, nW, N, C = x.shape
        q, k, v = s.qkv(x).view(B, nW, N, 3, s.h, C // s.h).permute(3, 0, 1, 4, 2, 5)   # each (B,nW,h,N,d)
        bias = s.bias[:, s.idx][None, None]                                              # 1,1,h,N,N
        if mask is not None: bias = bias + mask[None, :, None]                           # 1,nW,h,N,N
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return s.proj(o.transpose(2, 3).reshape(B, nW, N, C))


class SwinBlock3D(nn.Module):
    def __init__(s, dim, w, heads, shift, mlp=2.0):
        super().__init__()
        s.w, s.shift = w, shift; s.n1, s.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim); s.attn = WinAttn3D(dim, w, heads)
        s.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(s, x, mask):
        B, D, H, W, C = x.shape; h = s.n1(x)
        if s.shift: h = torch.roll(h, (-s.shift,) * 3, (1, 2, 3))
        o = win_merge(s.attn(win_part(h, s.w), mask if s.shift else None), s.w, D, H, W)
        if s.shift: o = torch.roll(o, (s.shift,) * 3, (1, 2, 3))
        x = x + o
        return x + s.mlp(s.n2(x))


class RSTB3D(nn.Module):
    def __init__(s, dim, depth, w, heads):
        super().__init__()
        s.blocks = nn.ModuleList([SwinBlock3D(dim, w, heads, 0 if i % 2 == 0 else w // 2) for i in range(depth)])
        s.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(s, x, mask):          # x (B,D,H,W,C)
        h = x
        for b in s.blocks: h = b(h, mask)
        return x + s.conv(h.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


class SwinIR3D(nn.Module):
    def __init__(s, dim=96, depths=(4, 4, 4, 4), heads=6, w=6, hc=32, size=36):
        super().__init__()
        s.w = w; s.head = nn.Conv3d(1, dim, 3, padding=1)
        s.layers = nn.ModuleList([RSTB3D(dim, d, w, heads) for d in depths]); s.norm = nn.LayerNorm(dim)
        s.after = nn.Conv3d(dim, dim, 3, padding=1); s.up = nn.Conv3d(dim, hc, 3, padding=1)
        s.tail = nn.Sequential(nn.Conv3d(hc, hc, 3, padding=1), nn.ReLU(), nn.Conv3d(hc, 1, 3, padding=1))
        # 移位窗口的注意力掩码（Swin 标准做法，三维）
        sh = w // 2; img = torch.zeros(1, size, size, size, 1); cnt = 0
        sl = (slice(0, -w), slice(-w, -sh), slice(-sh, None))
        for a in sl:
            for b in sl:
                for c in sl: img[:, a, b, c, :] = cnt; cnt += 1
        mw = win_part(img, w)[0, :, :, 0]                                   # nW, N
        m = mw[:, None, :] - mw[:, :, None]
        s.register_buffer('mask', m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0), persistent=False)

    def forward(s, lr):
        x = s.head(lr); h = x.permute(0, 2, 3, 4, 1)
        for l in s.layers: h = l(h, s.mask)
        x = x + s.after(s.norm(h).permute(0, 4, 1, 2, 3))
        o = (x.shape[-1] - K) // 2; c = slice(o, o + K)
        u = F.interpolate(s.up(x[..., c, c, c]), size=(UPN,) * 3, mode='trilinear', align_corners=False)
        return center_up(lr) + s.tail(F.relu(u))


# ------------------------------------------------------------------ 3D 条件扩散
def temb(t, dim):
    half = dim // 2; f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]; return torch.cat([a.sin(), a.cos()], 1)


class RB3(nn.Module):
    def __init__(s, ci, co, tdim):
        super().__init__()
        s.n1, s.c1 = nn.GroupNorm(8, ci), nn.Conv3d(ci, co, 3, padding=1); s.t = nn.Linear(tdim, co)
        s.n2, s.c2 = nn.GroupNorm(8, co), nn.Conv3d(co, co, 3, padding=1)
        s.sk = nn.Conv3d(ci, co, 1) if ci != co else nn.Identity()

    def forward(s, x, te):
        h = s.c1(F.silu(s.n1(x))) + s.t(te)[:, :, None, None, None]
        return s.sk(x) + s.c2(F.silu(s.n2(h)))


class Attn3(nn.Module):
    def __init__(s, c, heads=4):
        super().__init__(); s.n = nn.GroupNorm(8, c); s.qkv = nn.Conv3d(c, 3 * c, 1); s.o = nn.Conv3d(c, c, 1); s.h = heads

    def forward(s, x):
        B, C, D, H, W = x.shape
        q, k, v = s.qkv(s.n(x)).view(B, 3, s.h, C // s.h, D * H * W).permute(1, 0, 2, 4, 3)
        o = F.scaled_dot_product_attention(q, k, v).permute(0, 1, 3, 2).reshape(B, C, D, H, W)
        return x + s.o(o)


class LREnc(nn.Module):
    """粗扫 36³ 上下文 → 中心 16³ 特征 → 112³（SRDiff 的 LR 编码器思路）。"""
    def __init__(s, c=32, nb=4, co=16):
        super().__init__()
        s.h = nn.Conv3d(1, c, 3, padding=1)
        s.b = nn.Sequential(*[nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.SiLU(), nn.Conv3d(c, c, 3, padding=1)) for _ in range(nb)])
        s.o = nn.Conv3d(c, co, 3, padding=1)

    def forward(s, lr):
        x = s.h(lr)
        for blk in s.b: x = x + blk(x)
        o = (x.shape[-1] - K) // 2; c = slice(o, o + K)
        return F.interpolate(s.o(x[..., c, c, c]), size=(UPN,) * 3, mode='trilinear', align_corners=False)


class UNet3D(nn.Module):
    def __init__(s, cin, ch=(32, 64, 128, 128), tdim=128):
        super().__init__()
        s.tdim = tdim; s.tm = nn.Sequential(nn.Linear(tdim, tdim * 2), nn.SiLU(), nn.Linear(tdim * 2, tdim))
        s.inp = nn.Conv3d(cin, ch[0], 3, padding=1)
        s.down = nn.ModuleList(); s.ds = nn.ModuleList(); c0 = ch[0]
        for i, c in enumerate(ch):
            s.down.append(nn.ModuleList([RB3(c0, c, tdim), RB3(c, c, tdim)])); c0 = c
            s.ds.append(nn.Conv3d(c, c, 3, stride=2, padding=1) if i < len(ch) - 1 else nn.Identity())
        s.mid1, s.att, s.mid2 = RB3(c0, c0, tdim), Attn3(c0), RB3(c0, c0, tdim)
        s.up = nn.ModuleList()
        for i, c in reversed(list(enumerate(ch))):
            s.up.append(nn.ModuleList([RB3(c0 + c, c, tdim), RB3(c, c, tdim)])); c0 = c
        s.out = nn.Sequential(nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv3d(c0, 1, 3, padding=1))

    def forward(s, x, t):
        te = s.tm(temb(t, s.tdim)); h = s.inp(x); skips = []
        for (r1, r2), d in zip(s.down, s.ds):
            h = r2(r1(h, te), te); skips.append(h); h = d(h)
        h = s.mid2(s.att(s.mid1(h, te)), te)
        for (r1, r2), sk in zip(s.up, reversed(skips)):
            if h.shape[-1] != sk.shape[-1]: h = F.interpolate(h, size=sk.shape[-3:], mode='nearest')
            h = r2(r1(torch.cat([h, sk], 1), te), te)
        return s.out(h)


class DiffSR3D(nn.Module):
    """ε 预测。x0 = (细扫 − 三线性) / SCALE。"""
    SCALE = 0.5

    def __init__(s, T=1000):
        super().__init__()
        s.enc = LREnc(); s.net = UNet3D(cin=1 + 1 + 16); s.T = T
        t = torch.arange(T + 1, dtype=torch.float64) / T
        ab = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2; ab = (ab / ab[0]).clamp(1e-5, 1.0)
        s.register_buffer('abar', ab[1:].float())                             # ᾱ_t, t = 0..T−1

    def cond(s, lr):
        base = center_up(lr); return base, torch.cat([base, s.enc(lr)], 1)

    def loss(s, lr, hr):
        base, cnd = s.cond(lr); x0 = (hr - base) / s.SCALE
        t = torch.randint(0, s.T, (hr.shape[0],), device=hr.device); a = s.abar[t].view(-1, 1, 1, 1, 1)
        e = torch.randn_like(x0); xt = a.sqrt() * x0 + (1 - a).sqrt() * e
        return F.mse_loss(s.net(torch.cat([xt, cnd], 1), t).float(), e.float())

    @torch.no_grad()
    def sample(s, lr, steps=50, seed=7):
        base, cnd = s.cond(lr); g = torch.Generator(device=lr.device); g.manual_seed(seed)
        x = torch.randn(base.shape, generator=g, device=lr.device, dtype=base.dtype)
        ts = torch.linspace(s.T - 1, 0, steps, device=lr.device).long()
        for i, t in enumerate(ts):
            a = s.abar[t]; a_prev = s.abar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=lr.device)
            e = s.net(torch.cat([x, cnd], 1), t.repeat(x.shape[0])).float()
            x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-8, 8)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e                   # DDIM, η = 0
        return base + s.SCALE * x0
