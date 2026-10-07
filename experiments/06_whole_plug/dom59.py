# 大圆柱粗扫 vs 小柱粗扫（训练域）：灰度分位与径向功率谱（36³ 窗口，Hann 窗）
import numpy as np, glob, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
P = str(DATA_ROOT) + '/plugs/'
T = json.load(open(P + 'plug_table.json', encoding='utf-8'))
A = {v['group']: k for k, v in T.items() if v['ab'] == 'A'}
n = 36; w1 = np.hanning(n); W3 = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
f = np.fft.fftfreq(n); fr = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2)
edges = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.87])
def psd(W):
    out = []
    for v in W:
        v = v.astype(np.float32); p = np.abs(np.fft.fftn((v - v.mean()) * W3)) ** 2 / W3.sum()
        out.append([p[(fr >= edges[i]) & (fr < edges[i + 1])].mean() for i in range(len(edges) - 1)])
    return np.mean(out, 0)
def q(W): return np.percentile(W.astype(np.float32), [2, 10, 50, 90, 98])
res = {}
print('频带中心（周/粗体素）', np.round((edges[:-1] + edges[1:]) / 2, 3))
for g in ['G01', 'G02', 'G03', 'G05', 'G06', 'G07', 'G09', 'G11', 'G12', 'G13', 'G15', 'G16', 'G17', 'G19']:
    fs = sorted(glob.glob(str(DATA_ROOT) + '/pairs_npz/%s/*.npz' % g)); idx = np.random.default_rng(0).choice(len(fs), min(150, len(fs)), replace=False)
    Ws = np.stack([np.load(fs[i])['lr'] for i in idx]); Wl = np.load(P + A[g] + '.npz')['win'][::2]
    ps, pl = psd(Ws), psd(Wl); res[g] = dict(ps=ps.tolist(), pl=pl.tolist(), qs=q(Ws).tolist(), ql=q(Wl).tolist())
    print(g, A[g], '分位 小 %s | 大 %s' % (np.round(q(Ws), 3), np.round(q(Wl), 3)))
    print('    功率比 大/小', np.round(pl / ps, 2))
json.dump(res, open(str(DATA_ROOT) + '/plugs/dom59.json', 'w'))
