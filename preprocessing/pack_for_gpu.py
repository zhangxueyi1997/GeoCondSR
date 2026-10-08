"""把训练需要的东西理成一个自描述的传输包（拷贝到移动硬盘或上传都用它）。

为什么要单独打包而不是随便拷
  1. 原始数据 240 GB，训练只需要其中 18 GB。搞错会白拷一天。
  2. 接收端（A100）没有本项目的任何上下文，包里必须带清单、校验和、
     以及"这些文件各是什么、怎么用"的说明 —— 否则过两周自己都认不出来。
  3. 校验和是必须的：18 GB 走 USB 或网络，静默损坏一个 npz 会让训练
     莫名其妙地崩，而且很难查。

包含什么（与不包含什么，都在生成的 MANIFEST.md 里写清楚）
  pairs/        配对训练数据，G 生成器的目标
  cache/        粗扫缓存，E 编码器的 LR 输入体
  cfield/       ĉ 场，E 的监督目标
  degradation/  退化算子 H 的三维核
  meta/         留一岩性划分、逐样品几何元信息

用法
    python preprocessing/pack_for_gpu.py --dest /path/to/data          # 拷到目标目录
    python preprocessing/pack_for_gpu.py --dest /path/to/data --dry    # 只算体积与清单，不拷
    python preprocessing/pack_for_gpu.py --dest ... --verify           # 拷完后校验
"""
from __future__ import annotations
import os
import sys, json, shutil, hashlib, argparse, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = Path(os.environ.get('GEOCOND_WORK', str(HERE.parent / 'work')))   # intermediate preprocessing products
RES = PROJ / "results"
PAIRS = PROJ / "pairs_npz"   # output of export_pairs.py

# (包内目录, 源, 通配, 说明)
PARTS = [
    ("pairs", PAIRS, "*/*.npz",
     "配对训练数据。每个 npz：lr(36³ 粗体素, Δ单位) / hr(112³ @2.0μm, Δ单位) "
     "/ c(4 个成因变量)。G 生成器的训练目标。"),
    ("cache", PROJ / "cache", "cache_G*_C.npy",
     "真实 14 μm 粗扫缓存（已消倾斜）。E 编码器的 LR 输入体；"
     "7139 个 ĉ 格的 LR 全从这里按 cfield 的 grid 切。"),
    ("cfield", RES / "cfield", "G*.npz",
     "ĉ 场：0.5 mm 网格（36 粗体素，步长 18）上的 4 个成因变量真值 + 有效掩膜 "
     "+ 归一化常数。E 编码器的监督目标。"),
    ("degradation", RES / "degradation", "*.npz",
     "退化算子 H 的 9³ 三维核 + 噪声功率谱。退化一致性损失用。"
     "完整退化链 = 2μm 体 → 块平均到 14μm → 卷 kernel → 按 nps 注噪。"),
    ("meta", RES, "splits.json",
     "留一岩性划分（最终评估用）+ 开发划分（调超参用）。必须写死不许漂移。"),
    ("meta", RES / "registration", "meta_G*.json",
     "逐样品几何元信息（裁框、倍率、消倾斜矩阵）。"),
    ("meta", RES / "registration", "reg_refined.csv",
     "配准精配结果，含实测细扫体素 v_true。"),
]

NOT_INCLUDED = [
    ("原始细扫 TIFF", "约 240 GB", "配对数据已把需要的部分抽出来，训练不用"),
    ("cache_G*_F.npy", "994 MB", "细扫缓存，只用于配准与测量，训练不用"),
    ("results 下的各种 csv / 中间缓存", "约 1 GB", "分析产物，训练不用"),
]


def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect():
    out = []
    for sub, src, pat, note in PARTS:
        if not src.exists():
            print(f"⚠ 源不存在，跳过：{src}")
            continue
        fs = sorted(src.glob(pat))
        if not fs:
            print(f"⚠ 没有匹配 {src}\\{pat}")
            continue
        out.append(dict(sub=sub, src=src, files=fs, note=note,
                        nbytes=sum(f.stat().st_size for f in fs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--nohash", action="store_true", help="跳过校验和（快，但不推荐）")
    a = ap.parse_args()
    dest = Path(a.dest)

    groups = collect()
    total = sum(g["nbytes"] for g in groups)
    print("=" * 92)
    print(f"传输包 → {dest}")
    print("=" * 92)
    print(f"{'包内目录':<14}{'文件数':>8}{'体积':>12}   来源")
    for g in groups:
        print(f"{g['sub']:<14}{len(g['files']):>8}{g['nbytes']/1e9:>10.2f} GB"
              f"   {g['src']}")
    print(f"{'合计':<14}{sum(len(g['files']) for g in groups):>8}"
          f"{total/1e9:>10.2f} GB")
    print()
    print("不包含（避免白拷）：")
    for n, s, why in NOT_INCLUDED:
        print(f"  · {n:<28}{s:>10}   {why}")
    if a.dry:
        print("\n--dry：只算不拷。")
        return

    dest.mkdir(parents=True, exist_ok=True)
    man, t0, done = [], time.time(), 0
    for g in groups:
        od = dest / g["sub"]
        od.mkdir(parents=True, exist_ok=True)
        for f in g["files"]:
            # pairs 是 {gid}/{idx}.npz，要保住子目录
            rel = f.relative_to(g["src"])
            tgt = od / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            if not (a.verify and tgt.exists()):
                shutil.copy2(f, tgt)
            h = "" if a.nohash else sha(tgt)
            if a.verify and not a.nohash and h != sha(f):
                print(f"❌ 校验不符：{rel}")
            man.append(dict(path=f"{g['sub']}/{rel.as_posix()}",
                            bytes=f.stat().st_size, sha256=h))
            done += f.stat().st_size
            if len(man) % 200 == 0:
                el = time.time() - t0
                print(f"  {len(man):5d} 个文件  {done/1e9:6.2f}/{total/1e9:.2f} GB  "
                      f"{done/1e6/max(el,1e-9):6.1f} MB/s  "
                      f"剩约 {(total-done)/max(done/max(el,1e-9),1)/60:5.1f} min",
                      flush=True)

    (dest / "manifest.json").write_text(
        json.dumps(dict(total_bytes=total, n_files=len(man),
                        created=time.strftime("%Y-%m-%d %H:%M"), files=man),
                   ensure_ascii=False), encoding="utf-8")

    lines = ["# 论文三 训练数据包", "",
             f"生成于 {time.strftime('%Y-%m-%d %H:%M')}，"
             f"共 {len(man)} 个文件、{total/1e9:.2f} GB。", "",
             "## 这些是什么", ""]
    seen = set()
    for g in groups:
        if g["sub"] in seen:
            continue
        seen.add(g["sub"])
        lines += [f"### `{g['sub']}/`", "", g["note"], ""]
    lines += ["## 关键约定", "",
              "- `lr` / `hr` 都已归到 **Δ 单位**（0 = 空气电平，1 = 骨架主峰），"
              "不要再做别的归一。",
              "- `hr` 的体素是 **2.00 μm**，正好是 `lr`（14 μm）的 7 细分。",
              "- ĉ 的 4 个变量依次是 **IGV、f_pore、mu_inter、f_dense**。",
              "  ⚠ `f_dense` 在 G11/G15 上几乎恒为零；`mu_inter` 基本是样品级的量"
              "（样品内只占 8% 方差）——**损失应按变量加权，不要等权**。",
              "- 评估**一律用 `meta/splits.json` 的留一岩性折，且必须逐折报**："
              "7 折里有 4 折的测试集只有 1 个样品，报平均会把这件事盖住。", "",
              "## 校验", "",
              "```bash", "python verify_bundle.py   # 或用 manifest.json 里的 sha256",
              "```", ""]
    (dest / "README.md").write_text("\n".join(lines), encoding="utf-8")

    el = time.time() - t0
    print(f"\n完成：{len(man)} 个文件、{total/1e9:.2f} GB，用时 {el/60:.1f} min")
    print(f"写入 {dest/'manifest.json'} 与 {dest/'README.md'}")


if __name__ == "__main__":
    main()
