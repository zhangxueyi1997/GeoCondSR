"""论文三 训练数据加载器 —— 在 A100 上用，读传输包 p3bundle。

两套数据，喂两个不同的部件
  PairDataset   读 pairs/：(lr 36³, hr 112³, c 4) —— 训练 A 路与 G 生成器
                2800 个单元。hr 只覆盖 lr 中心的 224 μm，不是整块。
  CellDataset   读 cache/ + cfield/：(lr 36³, c 4) —— 只训练 E 编码器
                7139 个格子，是 PairDataset 的 2.5 倍，**因为 E 不需要高分辨目标**。
                训 E 时优先用这个，样本量大得多。

约定（接收端最容易搞错的地方，全写在这里）
  · lr / hr 都**已经归到 Δ 单位**（0 = 空气电平，1 = 骨架主峰）。不要再做归一化。
  · hr 体素 2.00 μm，正好是 lr（14 μm）的 7 细分。
  · c 的四个变量依次是 IGV、f_pore、mu_inter、f_dense。
  · 评估**一律用留一岩性折，且逐折报** —— 7 折里有 4 折测试集只有 1 个样品，
    报平均会把这件事盖住。

ĉ 的变量权重（由实测方差分解定，不是拍的）
  变量        样品内占比   含义                          建议权重
  IGV           41%      空间变化真实                    1.0
  f_pore        28%      空间变化真实                    1.0
  mu_inter       8%      基本是**样品级**的量             0.3（降空间权重）
  f_dense       70%      **斑块状**，逐格监督最有价值      1.0
                         但在 G11/G15 上几乎恒为零 → 这两样上按样品掩掉

用法
    from dataset import PairDataset, CellDataset, folds, VAR_W
    for fold in folds(os.environ.get("GEOCOND_DATA", "data")):            # 7 个留一岩性折
        tr = PairDataset(os.environ.get("GEOCOND_DATA", "data"), fold["train"])
        te = PairDataset(os.environ.get("GEOCOND_DATA", "data"), fold["test"])
"""
from __future__ import annotations
import json, warnings, random, itertools
from pathlib import Path
import numpy as np

# 八面体群：6 种轴置换 x 8 种翻转 = 48 个元素。
# 合法性（第 14 步）：H 核实测各向同性（x<->y、z<->面内互换差为 0），
# c 是直方图量、旋转不变，lr/hr 同变换 —— 所以 48 种全部合法。
# 2400 块 x 48 = 11.5 万等效样本；原先每块看 25 遍纯属记忆。
_PERMS = list(itertools.permutations(range(3)))


def octahedral(x, k):
    """第 k 个（0..47）八面体群元素作用在 (C, D, H, W) 立方数组上。"""
    perm, flips = _PERMS[k // 8], k % 8
    y = np.transpose(x, (0,) + tuple(p + 1 for p in perm))
    for ax in range(3):
        if flips >> ax & 1:
            y = np.flip(y, axis=ax + 1)
    return np.ascontiguousarray(y)

NAMES = ["IGV", "f_pore", "mu_inter", "f_dense"]
VAR_W = np.array([1.0, 1.0, 0.3, 1.0], np.float32)      # 见文件头
DEAD_FDENSE = {"G11", "G15"}                            # f_dense 几乎恒为零
UP = 7
CELL = 36


# ----------------------------------------------------------------- 划分

def folds(root):
    """返回 7 个留一岩性折：[{fold, train:[gid], test:[gid], note}]。"""
    sp = json.loads((Path(root).expanduser() / "meta" / "splits.json")
                    .read_text(encoding="utf-8"))
    return sp["leave_one_facies"]


def dev_split(root):
    sp = json.loads((Path(root).expanduser() / "meta" / "splits.json")
                    .read_text(encoding="utf-8"))
    return sp["dev"]["train"], sp["dev"]["val"]


def load_win(root):
    """每个样品的骨架窗 (lo, hi)，Δ 单位 —— 成因泛函 Φ 要用。

    窗是逐样品算的（主峰 ± 2.2 × 自己的半高宽，第 4 步的结论：固定窗会把
    样品间散布压掉一半），所以必须跟着 batch 走，不能写死一个常数。
    存在 cfield/{gid}.npz 的 norm[4:6]。"""
    w = {}
    for f in sorted((Path(root).expanduser() / "cfield").glob("G*.npz")):
        n = np.load(f)["norm"]
        w[f.stem] = (float(n[4]), float(n[5]))
    if not w:
        raise RuntimeError("cfield/ 里没有骨架窗 —— 包拷全了吗？")
    return w


def load_arr(root, sub="arr2", cols=None):
    """排布类成因变量旁文件。

    sub="arr"  第 15 步的 chi/ncomp/thick（已证伪，保留供对照）
    sub="arr2" 第 18 步准入通过的 7 列：
               pore_env rim_idx skel_len het_igv COPL CEPL chi
    cols：要用哪几列的名字；None = 全用。**「一次只加一个」靠它实现。**
    返回 ({(gid, 文件名): 向量}, 列名列表)。"""
    d0 = None; out = {}
    for f in sorted((Path(root).expanduser() / sub).glob("G*.npz")):
        d = np.load(f)
        names = [str(x) for x in d["names"]]
        idx = list(range(len(names))) if cols is None else [names.index(c) for c in cols]
        d0 = [names[i] for i in idx]
        for fn, v in zip(d["files"], d["vals"]):
            out[(f.stem, str(fn))] = v.astype(np.float32)[idx]
    if not out:
        raise RuntimeError(f"{sub}/ 里没有排布变量 —— 先跑 export_arr2.py")
    return out, d0


# 7 维条件的权重：前 4 同 VAR_W；chi 1.0、ncomp 0.3（只做条件、不可微、噪声较大）、thick 1.0
VAR_W7 = np.array([1.0, 1.0, 0.3, 1.0, 1.0, 0.3, 1.0], np.float32)
NAMES7 = NAMES + ["chi", "ncomp", "thick"]
PHI_IDX7 = [0, 1, 2, 3, 4, 6]          # Φ 可微的 6 项在 7 维 c 里的下标（跳过 ncomp）


def var_mask(gid):
    """哪些 c 分量在这个样品上有信号。f_dense 在 G11/G15 上恒为零，掩掉。"""
    m = np.ones(4, np.float32)
    if gid in DEAD_FDENSE:
        m[3] = 0.0
    return m


# ----------------------------------------------------------------- 配对

class PairDataset:
    """pairs/：训练 A 路与 G。__getitem__ 返回 dict(lr, hr, c, w, gid)。"""

    def __init__(self, root, gids=None, hr_crop=None, unit_split=None, hold=10,
                 aug=False, arr=False, c_src=None):
        """unit_split: None 全要 / "train" 训练用 / "heldout" 留出评估用。
        aug: 训练时随机取 48 个八面体变换之一，lr/hr 同变换；评估时必须关。
        arr: 第 15 步，c 扩成 7 维（+chi, ncomp, thick，从 arr/ 旁文件按文件名对齐）。
             这三个量旋转不变（可加/拓扑量），增广下不用跟着变。

        为什么需要留出 patch：消融那张交叉图要「**见过的岩性**」一栏，
        而训练 15000 步 x bs4 = 6 万次采样、每个 patch 被看 33 遍，
        直接拿训练过的 patch 评估就是作弊。所以每个样品固定留出 1/hold 的
        patch 不参与训练，专门用来量「见过的岩性、没见过的位置」。
        划分按排序后的下标定，确定性的，训练与评估两边算出来一致。"""
        self.root = Path(root).expanduser()
        d = self.root / "pairs"
        gids = gids or sorted(p.name for p in d.iterdir() if p.is_dir())
        self.items = []
        for g in gids:
            fs = sorted((d / g).glob("*.npz"))
            for i, f in enumerate(fs):
                held = (i % hold) == (hold - 1)
                if unit_split == "train" and held:
                    continue
                if unit_split == "heldout" and not held:
                    continue
                self.items.append((g, f))
        if not self.items:
            raise RuntimeError(f"{d} 下没有找到任何 npz —— 包拷全了吗？")
        self.hr_crop = hr_crop          # 显存紧张时可只取 hr 中心的一小块
        self.win = load_win(self.root)  # 逐样品骨架窗，成因一致性损失要用
        self.aug = bool(aug)
        # 第 31 步：c 可按不同支撑供给（cfield2 的 blk / cell / ann）。
        # 实测成因信息在 224 μm 块尺度上远强于 0.5 mm 格尺度（+0.32 vs +0.05）。
        self.c_src, self.cf2 = c_src, {}
        if c_src:
            import glob as _g
            for _f in sorted(_g.glob(str(self.root / "cfield2" / "*.npz"))):
                _d = np.load(_f)
                self.cf2[Path(_f).stem] = (_d["c_" + c_src], _d["valid"])
            if not self.cf2:
                raise RuntimeError("c_src 已指定但 cfield2/ 下没有找到任何场")
        self.arr, self.arr_cols = (None, [])
        if arr:
            cols = arr if isinstance(arr, (list, tuple)) else None
            self.arr, self.arr_cols = load_arr(self.root, "arr2", cols)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        gid, f = self.items[i]
        z = np.load(f)
        lr, hr, c = z["lr"], z["hr"], z["c"]
        if self.hr_crop and self.hr_crop < hr.shape[0]:
            o = (hr.shape[0] - self.hr_crop) // 2
            s = slice(o, o + self.hr_crop)
            hr = hr[s, s, s]
        lr, hr = lr[None].astype(np.float32), hr[None].astype(np.float32)
        if self.aug:
            # 用 Python 的 random：DataLoader 每个 worker 会给它单独播种，
            # numpy 的不会（所有 worker 抽同一串数，增广就白做了）。
            k = random.randrange(48)
            lr, hr = octahedral(lr, k), octahedral(hr, k)
        if self.c_src and gid in self.cf2:
            # 按 cell 下标取该支撑的 c；该格无效时退回原 c（保持样本数不变）
            fld, val = self.cf2[gid]
            iz, iy, ix = (int(z["cell"][0]), int(z["cell"][1]), int(z["cell"][2]))
            if (0 <= iz < val.shape[0] and 0 <= iy < val.shape[1]
                    and 0 <= ix < val.shape[2] and val[iz, iy, ix]):
                v = fld[iz, iy, ix]
                if np.isfinite(v).all():
                    c = v.astype(np.float32)
        c = np.nan_to_num(c.astype(np.float32)); w = VAR_W * var_mask(gid)
        if self.arr is not None:
            ex = self.arr[(gid, f.name)]                 # 选中的排布列
            c = np.concatenate([c, ex]).astype(np.float32)
            w = np.concatenate([w, np.ones(len(ex), np.float32)]).astype(np.float32)
        return dict(lr=lr,                               # (1,36,36,36)
                    hr=hr,                               # (1,112,112,112)
                    c=c,                                 # 4 维或 7 维
                    w=w,
                    win=np.asarray(self.win[gid], np.float32),   # (lo, hi)
                    gid=gid)


# ----------------------------------------------------------------- ĉ 格

class CellDataset:
    """cache/ + cfield/：只训 E。样本量是 PairDataset 的 2.5 倍。

    LR 直接从粗扫体按 cfield 的网格切，不需要任何高分辨数据。"""

    def __init__(self, root, gids=None, mmap=True):
        self.root = Path(root).expanduser()
        cf = self.root / "cfield"
        gids = gids or sorted(p.stem for p in cf.glob("G*.npz"))
        self.vol, self.items = {}, []
        for g in gids:
            f = cf / f"{g}.npz"
            if not f.exists():
                warnings.warn(f"{g} 没有 cfield，跳过"); continue
            d = np.load(f)
            cache = self.root / "cache" / str(d["lr_cache"])
            if not cache.exists():
                warnings.warn(f"{g} 没有 {cache.name}，跳过"); continue
            self.vol[g] = (np.load(cache, mmap_mode="r" if mmap else None),
                           int(d["lr_z0"]), d["norm"][0], d["norm"][1])
            cell, stride = int(d["grid"][0]), int(d["grid"][1])
            C, V = d["c"], d["valid"]
            for iz, iy, ix in np.argwhere(V):
                self.items.append((g, iz * stride, iy * stride, ix * stride,
                                   C[iz, iy, ix].astype(np.float32), cell))
        if not self.items:
            raise RuntimeError("没有任何有效格 —— cache/ 与 cfield/ 都拷了吗？")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        g, z0, y0, x0, c, cell = self.items[i]
        V, zoff, air, D = self.vol[g]
        z0 = z0 + zoff
        lr = np.asarray(V[z0:z0 + cell, y0:y0 + cell, x0:x0 + cell], np.float32)
        if lr.shape != (cell,) * 3:                     # 贴边的格子补掉
            out = np.full((cell,) * 3, air, np.float32)
            out[:lr.shape[0], :lr.shape[1], :lr.shape[2]] = lr
            lr = out
        return dict(lr=((lr - air) / D)[None].astype(np.float32),
                    c=c, w=(VAR_W * var_mask(g)), gid=g)


# ----------------------------------------------------------------- 退化算子

def load_H(root):
    """返回 (9³ 卷积核, 噪声功率谱)。完整退化链见 README。"""
    d = np.load(Path(root).expanduser() / "degradation" / "H_kernel.npz")
    return d["kernel"].astype(np.float32), d["nps_med"].astype(np.float32)


def degrade(hr, kernel, up=UP):
    """把 HR（2 μm）降回 LR（14 μm）：块平均 → 卷 H。退化一致性损失用。

    numpy 版，供自检；训练里请用 torch 的 avg_pool3d + conv3d 重写，才能反传。"""
    from scipy import ndimage
    n = hr.shape[-1] // up * up
    x = hr[..., :n, :n, :n]
    s = x.shape[:-3] + (n // up, up, n // up, up, n // up, up)
    blk = x.reshape(s).mean(axis=(-5, -3, -1))          # 块平均
    return ndimage.convolve(blk, kernel, mode="nearest")


# ----------------------------------------------------------------- 自检

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEOCOND_DATA", "data")
    print(f"包路径 {root}")
    fs = folds(root)
    print(f"留一岩性折 {len(fs)} 个：")
    for f in fs:
        print(f"  {f['fold']:<5} 测试 {f['n_test']} 样 / 训练 {f['n_train']} 样"
              f"   {f.get('note','')}")
    p = PairDataset(root)
    c = CellDataset(root)
    print(f"\nPairDataset {len(p)} 个单元   CellDataset {len(c)} 个格")
    a, b = p[0], c[0]
    print(f"pair: lr {a['lr'].shape} {a['lr'].min():.3f}~{a['lr'].max():.3f}  "
          f"hr {a['hr'].shape} {a['hr'].min():.3f}~{a['hr'].max():.3f}  c {a['c']}")
    print(f"cell: lr {b['lr'].shape} {b['lr'].min():.3f}~{b['lr'].max():.3f}  "
          f"c {b['c']}  权重 {b['w']}")
    k, nps = load_H(root)
    print(f"\nH 核 {k.shape} 和={k.sum():.6f}（应为 1）  噪声谱 {nps.shape}")
    lo = degrade(a["hr"][0], k)
    print(f"退化自检：hr {a['hr'].shape[1:]} → {lo.shape}（应为 16³，"
          f"因为 hr 只覆盖 lr 中心 16 个粗体素）")
    o = (36 - 16) // 2
    ctr = a["lr"][0, o:o + 16, o:o + 16, o:o + 16]
    print(f"  降回来的均值 {lo.mean():.4f} vs lr 中心均值 {ctr.mean():.4f}  "
          f"差 {abs(lo.mean()-ctr.mean()):.4f}（应 < 0.05）")
