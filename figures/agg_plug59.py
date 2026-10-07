# -*- coding: utf-8 -*-
"""第 59 步：整柱粗扫超分的柱级汇总。数据：data/plugs/out/plugsr59_<折><模型>.json（plugsr59.py）。
每根柱 300 个窗口（中心 224 μm），柱级 2 μm 可分辨孔隙率 = 窗口孔隙率平均。
对照：
  细扫真值：同组小柱细扫的 2 μm 可分辨孔隙率（表 1，tab01_samples.csv；只有配准成功的 14 组）；小柱取自 B 柱，A、B 为平行样；
  氦孔隙度：各柱实验室值（physical.py，plug_table.json）；
  小柱粗扫对照：同一模型在测试小柱粗扫配对块上的输出与细扫真值（域差异检查）。
山西（SX）、福建（FJ）不属任何测试折：取 6 个折模型的平均，单列，不进入主统计。
评价口径（2026-09-28 19:55 在看到完整标定结果之前登记）：
  ① 柱级超分孔隙率与氦孔隙度的 Pearson / Spearman 相关（测试折全部柱）；
  ② 与同组小柱细扫 2 μm 孔隙率的偏差、平均绝对误差与相关（有细扫的测试组 × A/B）；
  ③ 超分孔隙率 / 氦孔隙度的中位数，与小柱细扫的该比值（中位 0.50，2.4 节）对照；
  所有方法用同一套输入；标定前（原始）与标定后都报告。
用法：python agg_plug59.py [模型后缀，默认 _n54b]"""
import json, sys, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; D = FIGDATA / 'plugs'
GS = sys.argv[1] if len(sys.argv) > 1 else '_n54b'; SUF = sys.argv[2] if len(sys.argv) > 2 else ''   # SUF：'_cup' 读标定版输出
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
R = {f: json.load(open(D / 'out' / ('plugsr59_%s%s%s.json' % (f, GS, SUF)), encoding='utf-8')) for f in FOLDS if (D / 'out' / ('plugsr59_%s%s%s.json' % (f, GS, SUF))).exists()}
T = json.load(open(D / 'plug_table.json', encoding='utf-8'))
FINE = {r['组号']: float(r['细扫可分辨孔隙度']) / 100 for r in csv.DictReader(open(H / 'tab01_samples.csv', encoding='utf-8-sig')) if r['细扫可分辨孔隙度'] not in ('', 'nan')}
MM = ['粗扫阈值', '三线性', 'EDSR-3D', 'SRGAN-3D', '本文·仅粗扫']
CONDS = ['raw', 'harm', 'cal', 'cal_harm']; CN = {'raw': '原始', 'harm': '谱减', 'cal': '标定', 'cal_harm': '标定+谱减'}
print('模型', GS, '；已有折', list(R))
P = {}
for f, r in R.items():
    for pid, v in r['plugs'].items():
        own = v['code'] == f
        for cond in [c for c in CONDS if c in v]:
            for m in MM:
                P.setdefault((pid, cond, m), []).append((own, float(np.mean(v[cond][m]))))
rows = []
for pid, t in sorted(T.items(), key=lambda x: (x[1]['group'], x[1]['ab'])):
    if (pid, 'raw', MM[0]) not in P: continue
    own = [x for x in P[(pid, 'raw', MM[0])] if x[0]]
    rec = dict(pid=pid, group=t['group'], ab=t['ab'], code=t['code'], he=t['phi_he'] / 100, fine=FINE.get(t['group'], np.nan), main=bool(own))
    for cond in [c for c in CONDS if (pid, c, MM[0]) in P]:
        for m in MM:
            vals = P[(pid, cond, m)]; sel = [x[1] for x in vals if x[0]] or [x[1] for x in vals]
            rec[cond + '|' + m] = float(np.mean(sel))
    rows.append(rec)
print('\n| 柱 | 组 | 产地 | 氦孔隙度 | 细扫真值 | ' + ' | '.join('%s（原/校正）' % m for m in MM) + ' |')
for r in rows:
    print('| %s | %s%s | %s | %.3f | %s | ' % (r['pid'], r['group'], r['ab'], r['code'] + ('' if r['main'] else '*'), r['he'], '%.4f' % r['fine'] if np.isfinite(r['fine']) else '—')
          + ' | '.join(' / '.join('%.4f' % r[c + '|' + m] for c in CONDS if c + '|' + m in r) for m in MM) + ' |')
M = [r for r in rows if r['main'] and np.isfinite(r['fine'])]
print('\n主统计：有细扫真值且属测试折的柱 %d 根（%d 组 × A/B）' % (len(M), len({r['group'] for r in M})))
CC = [c for c in CONDS if all(c + '|' + MM[0] in r for r in M)]
for cond in CC:
    print('  输入：%s' % CN[cond])
    for m in MM:
        p = np.array([r[cond + '|' + m] for r in M]); t_ = np.array([r['fine'] for r in M])
        print('    %-10s 偏差 %+.4f  MAE %.4f  |ln比| 中位 %.2f  Pearson %.2f  Spearman %.2f' % (m, np.mean(p - t_), np.mean(np.abs(p - t_)), np.median(np.abs(np.log(np.maximum(p, 1e-4) / t_))), np.corrcoef(p, t_)[0, 1], spearmanr(p, t_)[0]))
A = [r for r in rows if r['main']]
he = np.array([r['he'] for r in A])
print('\n与氦孔隙度（属测试折的全部柱 %d 根）的相关：' % len(A))
for cond in CC:
    print('  %s：' % CN[cond] + '，'.join('%s Pearson %.2f / Spearman %.2f' % (m, np.corrcoef([r[cond + '|' + m] for r in A], he)[0, 1], spearmanr([r[cond + '|' + m] for r in A], he)[0]) for m in MM))
print('\n超分孔隙率 / 氦孔隙度 的中位数（测试折全部柱；小柱细扫的该比值中位 0.50）：')
for cond in CC: print('  %s：' % CN[cond] + '，'.join('%s %.2f' % (m, np.median([r[cond + '|' + m] / r['he'] for r in A])) for m in MM))
t_ = np.array([r['fine'] for r in M]); print('  参照：细扫真值与氦孔隙度（%d 根） Pearson %.2f / Spearman %.2f' % (len(M), np.corrcoef(t_, [r['he'] for r in M])[0, 1], spearmanr(t_, [r['he'] for r in M])[0]))
print('\n小柱粗扫对照（同一模型、测试小柱配对块，域内）：')
for f, r in R.items():
    for g, v in r['small'].items():
        print('  %s %s 细扫真值 %.4f | ' % (f, g, np.mean(v['细扫真值'])) + ' '.join('%s %.4f' % (m, np.mean(v[m])) for m in MM))
json.dump(rows, open(D / ('agg_plug59%s%s.json' % (GS, SUF)), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
