"""坐标映射：把配准后粗扫坐标里的一个点，映回原始 2.1 μm 切片的坐标。

这是打通「真实粗扫特征 → 2.1 μm 真值」最后一环所缺的东西。
正向链条（细扫 TIFF → 配准后的粗扫坐标）一共五步，逐步写出来：

  A 裁框      physical_bbox：fine(z,y,x) -> (z, y-r0F, x-c0F)
  B 块平均    block_mean_2d(f) + 每 f 层求均值：
              输出 index m 覆盖输入 [m*f, (m+1)*f)，故 输入 = m*f + (f-1)/2
  C 残差缩放  ndimage.zoom(resid), resid = f/ratio, grid_mode=False：
              输出 o -> 输入 o*(n_in-1)/(n_out-1)
  D 消倾斜    destilt: out[p] = in[Md p + offd]，Md = R^T，offd = p0 - Md c
  E 摆位      affinefit.place_corr: out[p] = Fg[M p + off]，Fg = cache_F[3:-3]

所以反过来，给定配准后坐标 p：
  q = M p + off + (3,0,0)          -> cache_F 坐标
  o = Md q + offd                  -> 消倾斜之前
  m = o * (n_in-1)/(n_out-1)       -> 块平均之后
  cropped = m*f + (f-1)/2          -> 裁框之后
  fine = cropped + (0, r0F, c0F)   -> 原始细扫切片坐标

形状自洽性可以先验算（本模块 `shapes()` 会核对）：
  G01  (1952-234)//6 = 286 -> round(286*0.9258) = 265 = shape_F0[1]  ✅
       ceil(1601/6) = 267  -> round(267*0.9258) = 247 = shape_F0[0]  ✅

⚠ 本模块只映射**点**，不做重采样。因为我们要量的是相分数（直方图），
  而直方图对面内旋转不敏感（取的是以该点为心的方框），
  重采样反而会改直方图 —— 与 tir.py 同一个理由。
"""
from __future__ import annotations
import os
import sys, json, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = Path(os.environ.get('GEOCOND_WORK', str(HERE.parent / 'work')))   # intermediate preprocessing products
REG = PROJ / "results" / "registration"
CACHE = PROJ / "cache"
sys.path.insert(0, str(HERE))
import compose, affinefit

DATA = Path(os.environ.get('GEOCOND_RAW_PROJECT', 'raw_project'))   # contains dataset/<group>/small_ct/raw16 (fine scans)


def shapes(meta, nfiles):
    """(块平均后形状 n_in, 缩放后形状 n_out)，并核对与 shape_F0 一致。"""
    f = int(meta["f_int"])
    r0, r1, c0, c1 = meta["bbox_F"]
    n_in = np.array([int(np.ceil(nfiles / f)), (r1 - r0) // f, (c1 - c0) // f], float)
    n_out = np.array(meta["shape_F0"], float)
    pred = np.round(n_in * (f / meta["ratio"]))
    ok = bool((pred == n_out).all())
    return n_in, n_out, ok, pred


class Mapper:
    """placed(配准后粗扫) 坐标 -> 原始细扫切片坐标。"""

    def __init__(self, gid):
        self.gid = gid
        self.meta = json.loads((REG / f"meta_{gid}.json").read_text(encoding="utf-8"))
        self.files = sorted((DATA / "dataset" / gid / "small_ct"
                             / "raw16").glob("*.tif"))
        kw, dz, base, ref = compose.params(gid)
        self.dz, self.ref = dz, ref
        nzF, n = (np.load(CACHE / f"cache_{gid}_F.npy", mmap_mode="r").shape[0],
                  np.load(CACHE / f"cache_{gid}_C.npy", mmap_mode="r").shape[1])
        e = json.loads((REG / "affine_corr.json").read_text(encoding="utf-8"))[gid]
        self.M, self.off, self.shp = affinefit.apply_corr(
            kw, n, (nzF - 6, n, n), (np.array(e["Mc"]), np.array(e["oc"])))
        d = self.meta["destilt_F"]
        self.Md = np.array(d["R"]).T
        self.offd = np.array(d["p0"]) - self.Md @ np.array(d["c"])
        self.n_in, self.n_out, self.ok, self.pred = shapes(self.meta, len(self.files))
        self.f = int(self.meta["f_int"])
        self.r0, self.c0 = self.meta["bbox_F"][0], self.meta["bbox_F"][2]

    def __call__(self, pts):
        """pts (N,3) 配准后坐标 -> (N,3) 细扫切片坐标 (z,y,x)，浮点。"""
        p = np.atleast_2d(np.asarray(pts, float))
        q = p @ self.M.T + self.off + np.array([3.0, 0.0, 0.0])
        o = q @ self.Md.T + self.offd
        m = o * (self.n_in - 1) / np.maximum(self.n_out - 1, 1)
        cropped = m * self.f + (self.f - 1) / 2.0
        return cropped + np.array([0.0, self.r0, self.c0])

    def scale(self):
        """一个配准后粗体素对应多少细体素（三轴平均）。"""
        a = self(np.zeros(3))[0]
        d = []
        for i in range(3):
            e = np.zeros(3); e[i] = 1.0
            d.append(np.linalg.norm(self(e)[0] - a))
        return float(np.mean(d))


if __name__ == "__main__":
    db = json.loads((REG / "affine_corr.json").read_text(encoding="utf-8"))
    print(f"{'gid':<5}{'形状自洽':>10}{'预测 n_out':>22}{'实际 shape_F0':>22}"
          f"{'粗/细体素比':>12}")
    for g in sorted(db):
        try:
            mp = Mapper(g)
            print(f"{g:<5}{'✅' if mp.ok else '❌ 不一致':>10}"
                  f"{str(mp.pred.astype(int).tolist()):>22}"
                  f"{str(mp.n_out.astype(int).tolist()):>22}"
                  f"{mp.scale():>12.3f}")
        except Exception as e:
            print(f"{g:<5} 失败 {type(e).__name__}: {e}")
