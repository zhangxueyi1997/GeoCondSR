"""调度：缓存一造好就顺次跑完 粗搜 -> 亚体素精配 -> 仿射微调 -> 可识别谱。

mkcache 的写入顺序是 两个 .npy -> meta json，所以 meta 存在就等于缓存已完整。
每一步各自写自己的结果文件，中断后重跑会自动跳过已完成的。
"""
from __future__ import annotations
import csv, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
ALL = [f"G{i:02d}" for i in range(1, 21)]


def ready(g):
    return ((HERE / f"cache_{g}_C.npy").exists()
            and (HERE / f"cache_{g}_F.npy").exists()
            and (HERE / f"meta_{g}.json").exists())


def csv_rows(name):
    f = HERE / name
    if not f.exists():
        return {}
    return {r["gid"]: r for r in csv.DictReader(f.open(encoding="utf-8-sig"))}


def json_keys(name):
    f = HERE / name
    if not f.exists():
        return set()
    try:
        return set(json.loads(f.read_text(encoding="utf-8")))
    except Exception:
        return set()


def run(stage, g):
    print(f"[{time.strftime('%H:%M')}] {stage} {g}", flush=True)
    p = subprocess.run([PY, "-X", "utf8", "-u", f"{stage}.py", g], cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for l in [l for l in (p.stdout or "").splitlines() if l.strip()][-3:]:
        print("    " + l, flush=True)
    if p.returncode != 0:
        print("    !! " + (p.stderr or "")[-400:], flush=True)


def pending(g):
    """返回该样品下一步要跑的阶段名，全部跑完返回 None。"""
    srch = csv_rows("reg_final.csv")
    if g not in srch:
        return "psearch3"
    if srch[g]["verdict"] != "通过":
        return None                      # 粗搜就没过，后面不用跑
    if g not in csv_rows("reg_refined.csv"):
        return "refine"
    if g not in json_keys("affine_corr.json"):
        return "affinefit"
    if g not in csv_rows("identifiability.csv"):
        return "spectrum"
    return None


if __name__ == "__main__":
    t0 = time.time()
    while True:
        moved = False
        for g in ALL:
            if not ready(g):
                continue
            while (st := pending(g)) is not None:
                run(st, g)
                moved = True
        if all(ready(g) and pending(g) is None for g in ALL):
            break
        if not moved:
            if time.time() - t0 > 5 * 3600:
                print("超时退出", flush=True)
                break
            time.sleep(45)

    print("\n" + "=" * 100, flush=True)
    srch = csv_rows("reg_final.csv"); refn = csv_rows("reg_refined.csv")
    aff = json.loads((HERE / "affine_corr.json").read_text(encoding="utf-8")) \
        if (HERE / "affine_corr.json").exists() else {}
    idf = csv_rows("identifiability.csv")
    ok = 0
    print(f"{'样品':<6}{'岩性':<9}{'体素μm':>8}{'vs面板':>8}{'角度':>9}"
          f"{'深度mm':>8}{'共线':>6}{'仿射残差':>9}{'逐层NCC':>9}{'各向异性':>9}"
          f"{'可识别深度':>11}")
    for g in ALL:
        s = srch.get(g)
        if not s:
            print(f"{g:<6}—— 无结果"); continue
        ok += s["verdict"] == "通过"
        r = refn.get(g, {}); a = aff.get(g, {}); d = idf.get(g, {})
        sv = a.get("sv")
        aniso = f"{(sv[0]/sv[2]-1)*100:.1f}%" if sv else "-"
        print(f"{g:<6}{s['sample']:<9}{r.get('v_true', s['v_true']):>8}"
              f"{r.get('dev_pct', s['dev_pct']):>7}%{s['angle']:>8}°"
              f"{'翻' if s['flip'] == 'True' else ' '}{s['z_mm']:>7}"
              f"{s['colin']:>6}{a.get('rms_vox', '-'):>9}"
              f"{a.get('per_ncc', r.get('ncc', '-')):>9}{aniso:>9}"
              f"{d.get('depth_um', '-'):>9} μm")
    print(f"\n共 {ok}/{len(ALL)} 个配准成功", flush=True)
