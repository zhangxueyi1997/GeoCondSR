"""阴性对照：把 A 样品配好的细扫，去和**另一块岩石**的粗扫算相干谱。

两者毫无关系，gamma^2 必须全程接近 0。若不为零，说明这套流程会凭空造出
相干性（子块加窗、去偏、径向平均任何一环出错都可能），所有结论都作废。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compose, affinefit, spectrum as sp
from register3 import interior

SHOW = (2, 3, 4, 6, 8, 12, 16, 20, 24)


def g2_of(C, B, msk):
    Scc, Sbb, Scb, nb = sp.spectra(C, B, msk)
    g = np.abs(Scb) ** 2 / np.maximum(Scc * Sbb, 1e-30)
    g = np.clip((nb * g - 1) / (nb - 1), 0, 1)
    return sp.radial(g), nb


def main(a, b):
    db = json.loads((HERE / "affine_corr.json").read_text(encoding="utf-8"))
    kw, dz, base, _ = compose.params(a)
    CA = np.load(HERE / f"cache_{a}_C.npy")
    FA = np.load(HERE / f"cache_{a}_F.npy")
    CB = np.load(HERE / f"cache_{b}_C.npy")
    msk, _ = interior(CA)
    Cg = CA[4:-4]
    e = db[a]
    W = affinefit.place_corr(FA[3:-3], Cg.shape[1], 3,
                             (np.array(e["Mc"]), np.array(e["oc"])), **kw)
    n = min(W.shape[0], Cg.shape[0] - dz)

    g_ok, nb0 = g2_of(Cg[dz:dz + n], W[:n], msk)
    lam = 1000.0 / ((np.arange(len(g_ok)) + .5) / len(g_ok) * 0.5 / 14.0 * 1000)
    print("  波长 μm            " + " ".join(f"{lam[i]:6.0f}" for i in SHOW))
    print(f"  {a} 配自己的粗扫   " + " ".join(f"{g_ok[i]:6.3f}" for i in SHOW)
          + f"   ({nb0} 块)")

    Cg2 = CB[4:-4]
    k = min(Cg2.shape[1], Cg.shape[1])
    nn = min(n, Cg2.shape[0])
    g_bad, nb1 = g2_of(Cg2[:nn, :k, :k], W[:nn, :k, :k], msk[:k, :k])
    print(f"  {a} 配 {b} 的粗扫   " + " ".join(f"{g_bad[i]:6.3f}" for i in SHOW)
          + f"   ({nb1} 块)")
    print(f"  -> 阴性对照最大值 {g_bad[2:].max():.3f}  （应接近 0）")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
