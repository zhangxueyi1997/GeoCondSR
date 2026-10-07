# -*- coding: utf-8 -*-
"""整柱“粗扫 + 稀疏细扫”端到端检验的汇总（口径见 data/plugs/prereg_wcsparse_20261002.md）。
数据：data/plugs/out/wcsparse59_<折>.json（sparse / sparse_all / coarse，有地质版 _m59n + _g59n）；
对照：data/plugs/out/plugsr59_<折>_n59n_cup.json 的 cal 条件（无地质整柱流程、EDSR-3D、SRGAN-3D、粗扫阈值、三线性），只取同一批 26 根柱。"""
import json, csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; P = FIGDATA / 'plugs'
F = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
FINE = {r['组号']: float(r['细扫可分辨孔隙度']) / 100 for r in csv.DictReader(open(H / 'tab01_samples.csv', encoding='utf-8-sig')) if r['细扫可分辨孔隙度'] not in ('', 'nan')}
W = {}; B = {}
for f in F:
    fp = P / 'out' / ('wcsparse59_%s.json' % f)
    if fp.exists(): W.update(json.load(open(fp, encoding='utf-8'))['plugs'])
    r = json.load(open(P / 'out' / ('plugsr59_%s_n59n_cup.json' % f), encoding='utf-8'))
    B.update({p: v for p, v in r['plugs'].items() if v['code'] == f})
PIDS = sorted(W, key=lambda p: (W[p]['group'], W[p]['ab']))
print('柱数', len(PIDS))
HE = np.array([W[p]['phi_he'] / 100 for p in PIDS]); FI = np.array([FINE[W[p]['group']] for p in PIDS]); AB = np.array([W[p]['ab'] for p in PIDS])
GRP = np.array([W[p]['group'] for p in PIDS])
X = {'本文·粗扫+稀疏细扫（主）': [np.mean(W[p]['phi']['sparse']) for p in PIDS],
     '本文·粗扫+全部细扫（参照）': [np.mean(W[p]['phi']['sparse_all']) for p in PIDS],
     '本文有地质·只用粗扫': [np.mean(W[p]['phi']['coarse']) for p in PIDS],
     '本文无地质（已有整柱流程）': [np.mean(B[p]['cal']['本文·仅粗扫']) for p in PIDS],
     'EDSR-3D': [np.mean(B[p]['cal']['EDSR-3D']) for p in PIDS], 'SRGAN-3D': [np.mean(B[p]['cal']['SRGAN-3D']) for p in PIDS],
     '粗扫阈值': [np.mean(B[p]['cal']['粗扫阈值']) for p in PIDS], '三线性': [np.mean(B[p]['cal']['三线性']) for p in PIDS]}
X = {k: np.array(v) for k, v in X.items()}
TIGHT = np.isin(GRP, ['G02', 'G09', 'G11', 'G12'])
print('\n%-26s %7s %7s %7s | %7s %7s | %8s %8s' % ('方法', 'r氦', 'ρ氦', '比值', '致密超氦', '致密比', '偏差*', 'MAE*'))
for k, x in X.items():
    e = x - FI
    print('%-26s %7.3f %7.3f %7.3f | %7d %7.2f | %+8.4f %8.4f' % (k, pearsonr(x, HE)[0], spearmanr(x, HE)[0], np.median(x / HE),
          int((x[TIGHT] > HE[TIGHT]).sum()), np.median(x[TIGHT] / HE[TIGHT]), e.mean(), np.abs(e).mean()))
print('（* 相对小柱细扫，与稀疏细扫有信息重叠，只作参考）；小柱细扫与氦：r %.3f ρ %.3f 比值 %.3f' % (pearsonr(FI, HE)[0], spearmanr(FI, HE)[0], np.median(FI / HE)))
for ab in ('A', 'B'):
    m = AB == ab
    print('\n%s 柱（%s，%d 根）：' % (ab, '平行样校准' if ab == 'A' else '本柱校准', m.sum()))
    for k in ('本文·粗扫+稀疏细扫（主）', '本文无地质（已有整柱流程）', 'EDSR-3D', 'SRGAN-3D'):
        x = X[k][m]; print('  %-26s r氦 %.3f ρ氦 %.3f 比值 %.3f' % (k, pearsonr(x, HE[m])[0], spearmanr(x, HE[m])[0], np.median(x / HE[m])))
print('\n逐柱（氦 / 细扫 / 稀疏 / 无地质）：')
for i, p in enumerate(PIDS):
    print('  %-6s %s%s 氦 %5.2f 细扫 %5.2f 稀疏 %5.2f 全部 %5.2f 有地质粗扫 %5.2f 无地质 %5.2f' % (p, GRP[i], AB[i], 100 * HE[i], 100 * FI[i], 100 * X['本文·粗扫+稀疏细扫（主）'][i],
          100 * X['本文·粗扫+全部细扫（参照）'][i], 100 * X['本文有地质·只用粗扫'][i], 100 * X['本文无地质（已有整柱流程）'][i]))
