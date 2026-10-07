# -*- coding: utf-8 -*-
"""第 59 步：GPU 格子玻尔兹曼绝对渗透率（D3Q19，两松弛时间 TRT，Λ = 3/16，半程反弹）。
几何：孔隙 = 1。沿 z 方向体积力驱动；样品沿 z 与其镜像拼接成周期几何；x、y 四个侧面各设 1 层固体墙。
只保留同时连通两端的孔隙团（6 连通）；不连通则渗透率记 0，不做模拟。
k_格子 = ν⟨u_z⟩ / F（⟨·⟩ 取样品区全部体素，含固体）；物理渗透率 = k_格子 × dx²。
自检（python lbm59.py test）：平行板与方管，与解析解比较。"""
import sys, math, numpy as np, torch
from scipy import ndimage as ndi

C = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]])
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
TAU_P = 1.0; LAM = 3.0 / 16.0; TAU_M = 0.5 + LAM / (TAU_P - 0.5); NU = (TAU_P - 0.5) / 3.0


def spanning(pore):
    """保留同时触及 z 两端（第 0 轴首末层）的孔隙团。"""
    lab, n = ndi.label(pore, structure=ndi.generate_binary_structure(3, 1))
    if n == 0: return np.zeros_like(pore)
    keep = np.intersect1d(np.unique(lab[0][lab[0] > 0]), np.unique(lab[-1][lab[-1] > 0]))
    return np.isin(lab, keep) if len(keep) else np.zeros_like(pore)


@torch.no_grad()
def permeability(pores, dev='cuda:0', F=1e-5, max_it=20000, tol=1e-4, buf=4, check=250, batch=6):
    if len(pores) > batch:                     # 分批，控制显存
        out, its = [], 0
        for i in range(0, len(pores), batch):
            out += _perm(pores[i:i + batch], dev, F, max_it, tol, check); its = max(its, _perm.iters)
        permeability.iters = its; return out
    out = _perm(pores, dev, F, max_it, tol, check); permeability.iters = _perm.iters; return out


@torch.no_grad()
def _perm(pores, dev, F, max_it, tol, check):
    """pores: list of (Z,Y,X) bool arrays（z 为流动方向，尺寸相同）。返回格子单位渗透率列表（按全样品体积平均）。"""
    out = [0.0] * len(pores); idx = []; geo = []
    for i, p in enumerate(pores):
        s = spanning(p)
        if s.any(): idx.append(i); geo.append(s)
    _perm.iters = 0
    if not idx: return out
    Z, Y, X = geo[0].shape; B = len(geo)
    fl = np.zeros((B, 2 * Z, Y, X), bool)                                         # 沿 z 镜像拼接成周期几何（无入口/出口效应）
    for b, s in enumerate(geo): fl[b, :Z] = s; fl[b, Z:] = s[::-1]
    fl[:, :, 0, :] = False; fl[:, :, -1, :] = False; fl[:, :, :, 0] = False; fl[:, :, :, -1] = False   # 侧墙
    fluid = torch.from_numpy(fl).to(dev); solid = ~fluid
    w = torch.tensor(W, dtype=torch.float32, device=dev).view(19, 1, 1, 1, 1)
    f = w.expand(19, B, *fl.shape[1:]).clone() * fluid[None]
    # 方向约定：C[:,0]→z（dim 2），C[:,1]→y（dim 3），C[:,2]→x（dim 4）；体积力沿 z，用 C[:,0]
    force = (3.0 * torch.tensor(W, dtype=torch.float32, device=dev) * torch.tensor(C[:, 0], dtype=torch.float32, device=dev) * F).view(19, 1, 1, 1, 1)
    cz, cy, cx = [torch.tensor(C[:, k], dtype=torch.float32, device=dev).view(19, 1, 1, 1, 1) for k in range(3)]
    zmask = torch.ones((1, 1, 2 * Z, 1, 1), device=dev)
    shifts = [tuple(int(v) for v in C[i]) for i in range(19)]
    prev = None; sl = slice(0, 2 * Z)
    for it in range(1, max_it + 1):
        rho = f.sum(0).clamp_min(1e-12); uz = (f * cz).sum(0) / rho; uy = (f * cy).sum(0) / rho; ux = (f * cx).sum(0) / rho
        cu = cz * uz + cy * uy + cx * ux; usq = uz * uz + uy * uy + ux * ux
        neq = f - w * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq); del cu, usq      # 非平衡部分
        neqo = neq[OPP]
        fc = f - 0.5 * (neq + neqo) / TAU_P - 0.5 * (neq - neqo) / TAU_M + force; del neq, neqo
        fc = fc * fluid[None]
        # 流动 + 半程反弹：从 x−c 来的若是固体，则取本点反方向的碰后分布
        fn = torch.empty_like(fc)
        for i in range(19):
            dz, dy, dx = shifts[i]
            src = torch.roll(fc[i], shifts=(dz, dy, dx), dims=(1, 2, 3))
            bs = torch.roll(solid, shifts=(dz, dy, dx), dims=(1, 2, 3))
            fn[i] = torch.where(bs, fc[OPP[i]], src)
        f = fn * fluid[None]; del fn, fc
        if it % check == 0:
            rho = f.sum(0).clamp_min(1e-12); uz = torch.where(fluid, ((f * cz).sum(0) + 0.5 * F * zmask[0]) / rho, torch.zeros_like(rho))   # 速度含半个力的修正
            m = uz[:, sl].mean(dim=(1, 2, 3))
            if prev is not None and torch.all((m - prev).abs() <= tol * m.abs().clamp_min(1e-30)): break
            prev = m
    _perm.iters = it
    k = (NU * m / F).cpu().numpy()
    for j, i in enumerate(idx): out[i] = float(k[j])
    return out


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'test':
    dev = 'cuda:0'; Z, N = 40, 34
    def rect(a, b):   # 矩形截面 a×b（a≤b）Poiseuille 平均速度系数：u = rect·G/μ
        s_ = sum(math.tanh(n * math.pi * b / (2 * a)) / n ** 5 for n in range(1, 200, 2))
        return a * a / 12 * (1 - 192 * a / (math.pi ** 5 * b) * s_)
    # 矩形缝：y 方向宽 h，x 方向宽 N−2（两侧各 1 层墙）
    for h in (6, 10, 16):
        p = np.zeros((Z, N, N), bool); y0 = (N - h) // 2; p[:, y0:y0 + h, 1:N - 1] = True
        import time; t0 = time.time(); k = permeability([p], dev)[0]; ana = rect(h, N - 2) * h * (N - 2) / (N * N)
        print('矩形缝 %d×%d  LBM %.4f  解析 %.4f  比 %.3f  （%d 步，%.1f s）' % (h, N - 2, k, ana, k / ana, permeability.iters, time.time() - t0), flush=True)
    # 方管：a×a 正方截面；解析 k = 0.035144·a² × (a²/N²)（方管的 Poiseuille 系数）
    for a in (8, 12, 20):
        p = np.zeros((Z, N, N), bool); o = (N - a) // 2; p[:, o:o + a, o:o + a] = True
        k = permeability([p], dev)[0]; ana = 0.0351443 * a ** 2 * a ** 2 / (N * N)
        print('方管 a=%d  LBM %.4f  解析 %.4f  比 %.3f' % (a, k, ana, k / ana), flush=True)
    # 不连通：孔隙团不触两端 → 0
    p = np.zeros((Z, N, N), bool); p[5:30, 10:20, 10:20] = True; print('不连通 →', permeability([p], dev)[0])
