import sys, json, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from lbm59s import spanning
N = 112; un = lambda row: np.unpackbits(row)[:N ** 3].reshape(N, N, N).astype(bool)
def chord_z(m):
    a = m.reshape(N, -1).T.astype(np.int8); d = np.diff(np.pad(a, ((0, 0), (1, 1))), axis=1)
    L = np.argwhere(d == -1)[:, 1] - np.argwhere(d == 1)[:, 1]; return L.mean() if len(L) else 0
for f in ('CQ', 'SC', 'YN'):
    D = np.load(str(OUT_ROOT) + '/eval59/out/eval59a_%s_masks.npz' % f); C = json.load(open(str(OUT_ROOT) + '/eval59/out/lbm59_%s.json' % f))
    sel = C['细扫']['sel']; kt = np.array(C['细扫']['kz']); top = np.argsort(kt)[-5:][::-1]
    print(f)
    for t in top:
        i = sel[t]; row = []
        for v in ('细扫', 'EDSR-3D', '本文·仅粗扫'):
            m = un(D[v][i]); s = spanning(m); k = kt[t] if v == '细扫' else C[v]['kz'][t]
            row.append('%s k %6.0f φ %.3f φ贯通 %.3f 弦长z %.1f' % (v[:4], k, m.mean(), s.mean(), chord_z(m)))
        print('  ' + ' | '.join(row))
