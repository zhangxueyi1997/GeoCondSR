# -*- coding: utf-8 -*-
"""第 59 步：GPU 格子玻尔兹曼绝对渗透率 —— 稀疏版（只存流体节点）。物理与 lbm59.py 逐项相同：
D3Q19，TRT（Λ = 3/16；τ+ 默认 0.6，Λ 固定时渗透率与粘度无关，已验证），半程反弹；沿流动方向与镜像拼接成周期几何（无入口/出口效应）；垂直流向的四个侧面各设 1 层固体墙；
体积力 F 驱动；只保留同时连通两端的孔隙团（6 连通），不连通记 0；k_格子 = ν⟨u⟩/F，⟨·⟩ 取样品全部体素（含固体）。
稀疏实现：流体节点压成一维，迁移用预先算好的上游索引 gather，上游为固体时取本点反方向碰后分布（半程反弹）。多个样品拼成一批。
自检：python lbm59s.py test（平行板、方管与解析解；并与稠密版 lbm59.py 对比同一几何）。"""
import sys, math, numpy as np, torch
from scipy import ndimage as ndi

C = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]])
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
TAU_P = 1.0; LAM = 3.0 / 16.0; TAU_M = 0.5 + LAM / (TAU_P - 0.5); NU = (TAU_P - 0.5) / 3.0


def spanning(pore):
    """保留同时触及第 0 轴首末层的孔隙团。"""
    lab, n = ndi.label(pore, structure=ndi.generate_binary_structure(3, 1))
    if n == 0: return np.zeros_like(pore)
    keep = np.intersect1d(np.unique(lab[0][lab[0] > 0]), np.unique(lab[-1][lab[-1] > 0]))
    return np.isin(lab, keep) if len(keep) else np.zeros_like(pore)


def _geometry(s):
    """s: (Z,Y,X) 贯通孔隙。返回拼接镜像并加侧墙后的流体掩膜（2Z,Y,X）。"""
    Z = s.shape[0]; fl = np.zeros((2 * Z,) + s.shape[1:], bool); fl[:Z] = s; fl[Z:] = s[::-1]
    fl[:, 0, :] = False; fl[:, -1, :] = False; fl[:, :, 0] = False; fl[:, :, -1] = False
    return fl


@torch.no_grad()
def permeability(pores, dev='cuda:0', F=1e-5, max_it=8000, tol=5e-3, check=500, max_nodes=6_000_000, verbose=False, tau=0.6, aitken=False, kfloor=1e-4):
    """pores: list of (Z,Y,X) bool（第 0 轴为流动方向）。返回格子单位渗透率列表；permeability.iters 为各批最大步数。"""
    out = [0.0] * len(pores); geo = []
    for i, p in enumerate(pores):
        s = spanning(p)
        if s.any(): geo.append((i, _geometry(s)))
    permeability.iters = 0
    batch, nn_ = [], 0
    for i, g in geo:                                                 # 按流体节点数分批
        n = int(g.sum())
        if batch and nn_ + n > max_nodes:
            _run(batch, out, dev, F, max_it, tol, check, verbose, tau, aitken, kfloor); batch, nn_ = [], 0
        batch.append((i, g)); nn_ += n
    if batch: _run(batch, out, dev, F, max_it, tol, check, verbose, tau, aitken, kfloor)
    return out


def _run(batch, out, dev, F, max_it, tol, check, verbose=False, tau=0.6, aitken=False, kfloor=1e-4):
    """tau：τ+（TRT 在 Λ = 3/16 下渗透率与粘度无关，减小 τ+ 可加快压力场弛豫）；aitken：对指数收敛尾部做 Aitken Δ² 外推（试验项，默认关）。
    收敛：每 check 步 ⟨u⟩ 的相对变化 ≤ tol；格子渗透率低于 kfloor（约 0.4 mD）的块视为近不渗透，不参与判据。"""
    TAU_P = tau; TAU_M = 0.5 + LAM / (TAU_P - 0.5); NU = (TAU_P - 0.5) / 3.0
    idx_all, src_all, seg, vol = [], [], [], []
    off = 0
    for b, (i, g) in enumerate(batch):
        shp = np.array(g.shape); flat = np.flatnonzero(g); n = len(flat)
        cmap = -np.ones(g.size, np.int64); cmap[flat] = np.arange(n) + off
        zyx = np.stack(np.unravel_index(flat, g.shape), 1)
        src = np.empty((19, n), np.int64)
        for q in range(19):
            up = (zyx - C[q]) % shp                                    # 上游节点（周期）
            src[q] = cmap[np.ravel_multi_index(up.T, g.shape)]           # −1 = 固体（反弹）
        src_all.append(src); seg.append(np.full(n, b)); vol.append(g.size); off += n
    src = torch.from_numpy(np.concatenate(src_all, 1)).to(dev); seg = torch.from_numpy(np.concatenate(seg)).to(dev)
    Nn = src.shape[1]; B = len(batch); vol = torch.tensor(vol, dtype=torch.float64, device=dev)
    solid = src < 0; srcc = src.clamp_min(0)
    w = torch.tensor(W, dtype=torch.float32, device=dev)[:, None]
    cz, cy, cx = [torch.tensor(C[:, k], dtype=torch.float32, device=dev)[:, None] for k in range(3)]
    force = 3.0 * w * cz * F
    opp = torch.tensor(OPP, device=dev)
    f = w.expand(19, Nn).clone()
    prev = None; it = 0; hist = []; ext_prev = None
    for it in range(1, max_it + 1):
        rho = f.sum(0); uz = (f * cz).sum(0) / rho; uy = (f * cy).sum(0) / rho; ux = (f * cx).sum(0) / rho
        cu = cz * uz + cy * uy + cx * ux; usq = uz * uz + uy * uy + ux * ux
        neq = f - w * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
        neqo = neq[opp]
        fc = f - 0.5 * (neq + neqo) / TAU_P - 0.5 * (neq - neqo) / TAU_M + force
        f = torch.where(solid, fc[opp], torch.gather(fc, 1, srcc))    # 迁移 + 半程反弹
        if it % check == 0:
            rho = f.sum(0); uz = ((f * cz).sum(0) + 0.5 * F) / rho       # 速度含半个力的修正
            m = torch.zeros(B, dtype=torch.float64, device=dev).index_add_(0, seg, uz.double()) / vol
            hist.append(m)
            if aitken and len(hist) >= 3:
                d1, d2 = hist[-1] - hist[-2], hist[-2] - hist[-3]; den = d1 - d2
                ext = torch.where(den.abs() > 1e-30, hist[-1] - d1 * d1 / torch.where(den.abs() > 1e-30, den, torch.ones_like(den)), hist[-1])
                ext = torch.where((d1 * d2 > 0) & (d1.abs() < d2.abs()), ext, hist[-1])          # 只在单调、收缩时外推
                if verbose: print('  第 %d 步  k格子 %s  外推 %s' % (it, np.round((NU * m / F).cpu().numpy(), 5), np.round((NU * ext / F).cpu().numpy(), 5)), flush=True)
                if ext_prev is not None and torch.all((ext - ext_prev).abs() <= tol * ext.abs().clamp_min(1e-30)): m = ext; break
                ext_prev = ext
            else:
                if verbose: print('  第 %d 步  k格子 %s' % (it, np.round((NU * m / F).cpu().numpy(), 5)), flush=True)
                live = NU * m / F > kfloor
                if prev is not None and torch.all(((m - prev).abs() <= tol * m.abs().clamp_min(1e-30)) | ~live): break
            prev = m
    permeability.iters = max(permeability.iters, it)
    k = (NU * m / F).cpu().numpy()
    for b, (i, _) in enumerate(batch): out[i] = float(k[b])


def directional(pore, dev='cuda:0', **kw):
    """三个方向（z、y、x）的渗透率：把流动轴换到第 0 轴。"""
    return [permeability([np.ascontiguousarray(np.moveaxis(pore, a, 0))], dev, **kw)[0] for a in range(3)]


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'test':
    import time
    dev = 'cuda:%s' % (sys.argv[2] if len(sys.argv) > 2 else '0'); Z, N = 40, 34
    def rect(a, b):
        s_ = sum(math.tanh(n * math.pi * b / (2 * a)) / n ** 5 for n in range(1, 200, 2))
        return a * a / 12 * (1 - 192 * a / (math.pi ** 5 * b) * s_)
    for h in (6, 10, 16):
        p = np.zeros((Z, N, N), bool); y0 = (N - h) // 2; p[:, y0:y0 + h, 1:N - 1] = True
        t0 = time.time(); k = permeability([p], dev)[0]; ana = rect(h, N - 2) * h * (N - 2) / (N * N)
        print('矩形缝 %d×%d  LBM %.4f  解析 %.4f  比 %.3f  （%d 步，%.1f s）' % (h, N - 2, k, ana, k / ana, permeability.iters, time.time() - t0), flush=True)
    for a in (8, 12, 20):
        p = np.zeros((Z, N, N), bool); o = (N - a) // 2; p[:, o:o + a, o:o + a] = True
        k = permeability([p], dev)[0]; ana = 0.0351443 * a ** 2 * a ** 2 / (N * N)
        print('方管 a=%d  LBM %.4f  解析 %.4f  比 %.3f' % (a, k, ana, k / ana), flush=True)
    p = np.zeros((Z, N, N), bool); p[5:30, 10:20, 10:20] = True; print('不连通 →', permeability([p], dev)[0])
    # 与稠密版同一随机多孔几何对比
    import lbm59
    g = np.random.default_rng(0); r = ndi.gaussian_filter(g.standard_normal((48, 40, 40)), 2.0); p = r > np.percentile(r, 70)
    t0 = time.time(); ks = permeability([p, p[:, ::-1]], dev); ts = time.time() - t0
    t0 = time.time(); kd = lbm59.permeability([p, p[:, ::-1]], dev); td = time.time() - t0
    print('随机多孔（孔隙率 0.30）稀疏 %s  稠密 %s  （%.1f s 对 %.1f s）' % (np.round(ks, 5), np.round(kd, 5), ts, td), flush=True)
