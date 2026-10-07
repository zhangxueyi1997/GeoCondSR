# -*- coding: utf-8 -*-
"""第 59 步渗流汇总（LBM，z 向渗透率 + 三向贯通判别）。数据：data/eval/eval59/lbm59_<折>.json（lbm_eval59.py 输出）。
口径：
  贯通一致率：200 块 × 3 方向，方法与细扫同为贯通/同为不贯通的比例；
  漏判率：细扫贯通而方法不贯通的比例；误判率：细扫不贯通而方法贯通的比例；
  渗透率（细扫 z 向贯通的块，每折至多 60 块）：
    块平均 k 之比 = Σk方法 / Σk细扫（不贯通记 0；块并联时的等效渗透率之比）；
    共同子集 |Δlog10 k| 中位数：所有被比较方法都贯通的块（同一批块，才可直接比较）；
    log10 k 相关：该方法贯通的块上与细扫的皮尔逊相关。"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; D = FIGDATA / 'eval/eval59'
FOLDS = [f for f in ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX'] if (D / ('lbm59_%s.json' % f)).exists()]
L = {f: json.load(open(D / ('lbm59_%s.json' % f), encoding='utf-8')) for f in FOLDS}
MS = [m for m in ['三线性', 'EDSR-3D', 'SRGAN-3D', 'SwinIR-3D', '扩散', '本文·仅粗扫', '本文·粗扫+稀疏细扫'] if all(m in L[f] for f in FOLDS)]
CMP = [m for m in MS if m != '三线性']
print('折：', FOLDS, '；方法：', MS)
rows = {}
for f in FOLDS:
    T = np.array(L[f]['细扫']['span']); kt = np.array(L[f]['细扫']['kz'])
    com = (kt > 0).copy()
    for m in CMP: com &= np.array(L[f][m]['kz']) > 0
    for m in MS:
        P = np.array(L[f][m]['span']); kp = np.array(L[f][m]['kz']); ok = (kp > 0) & (kt > 0)
        r = dict(agree=float((P == T).mean()), miss=float((T & ~P).sum() / max(T.sum(), 1)), false=float((~T & P).sum() / max((~T).sum(), 1)),
                 n_t=int(len(kt)), n_ok=int(ok.sum()),
                 kratio=float(kp.sum() / kt.sum()) if len(kt) and kt.sum() > 0 else np.nan,
                 dlog_com=float(np.median(np.abs(np.log10(kp[com]) - np.log10(kt[com])))) if com.sum() >= 3 and m in CMP else np.nan,
                 r_log=float(np.corrcoef(np.log10(kp[ok]), np.log10(kt[ok]))[0, 1]) if ok.sum() >= 5 else np.nan, n_com=int(com.sum()))
        rows[(f, m)] = r
print('\n各折（细扫 z 向贯通块数 / 共同子集块数）：', {f: (rows[(f, MS[0])]['n_t'], rows[(f, MS[0])]['n_com']) for f in FOLDS})
for key, nm, fmt in (('agree', '贯通一致率', '%.3f'), ('miss', '漏判率', '%.3f'), ('false', '误判率', '%.3f'), ('kratio', '块平均k之比', '%.2f'),
                     ('dlog_com', '共同子集|Δlog10k|中位', '%.3f'), ('r_log', 'log10k相关', '%.3f')):
    print('\n%s' % nm)
    for m in MS:
        v = [rows[(f, m)][key] for f in FOLDS]
        print('  %-14s' % m, ' '.join(('%-6s ' + fmt) % (f, x) if np.isfinite(x) else '%-6s   —  ' % f for f, x in zip(FOLDS, v)), ' | 均值', fmt % np.nanmean(v) if np.isfinite(v).any() else '—')
# 计分：本文·仅粗扫 对 各基线，贯通一致率逐折占优数
print('\n贯通一致率逐折比较（本文·仅粗扫 高于对方的折数 / 可比折数；GZ 近无贯通不计）')
FF = [f for f in FOLDS if rows[(f, MS[0])]['n_t'] >= 5]
for m in MS:
    if m.startswith('本文'): continue
    w = sum(rows[(f, '本文·仅粗扫')]['agree'] > rows[(f, m)]['agree'] for f in FF)
    wk = sum(abs(np.log10(rows[(f, '本文·仅粗扫')]['kratio'])) < abs(np.log10(rows[(f, m)]['kratio'])) for f in FF if rows[(f, m)]['kratio'] > 0)
    print('  对 %s：贯通一致率 %d/%d；块平均 k 更接近细扫 %d/%d' % (m, w, len(FF), wk, len(FF)))
json.dump({'%s|%s' % k: v for k, v in rows.items()}, open(D / 'agg_lbm59.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------- 表 6（markdown）：贯通一致率与误判率取 6 折均值；漏判率与渗透率各项只取细扫贯通块 ≥5 的折 ----------
FP = [f for f in FOLDS if rows[(f, MS[0])]['n_t'] >= 5]
NM = {'三线性': '三线性', 'EDSR-3D': 'EDSR-3D', 'SRGAN-3D': 'SRGAN-3D', 'SwinIR-3D': 'SwinIR-3D', '扩散': '扩散模型', '本文·仅粗扫': '本文（仅粗扫）', '本文·粗扫+稀疏细扫': '本文（加稀疏细扫）'}
print('\n**表 6** 渗流模拟结果（贯通一致率、误判率为 %d 折均值；其余为细扫 z 向贯通块不少于 5 块的 %d 折均值：%s）\n' % (len(FOLDS), len(FP), '、'.join(FP)))
print('| 指标 | 方向 | ' + ' | '.join(NM[m] for m in MS) + ' |'); print('|---|---|' + '---|' * len(MS))
for key, nm, s, ff, fmt in (('agree', '贯通判别一致率', 1, FOLDS, '%.3f'), ('miss', '漏判率', -1, FP, '%.3f'), ('false', '误判率', -1, FOLDS, '%.3f'),
                            ('dlog_com', r'共同块 $|\Delta\log_{10}k|$ 中位数', -1, FP, '%.3f'), ('r_log', r'$\log_{10}k$ 相关系数', 1, FP, '%.2f'), ('kratio', '块平均 $k$ 与细扫之比', 0, FP, '%.2f')):
    ff = [f for f in ff if all(np.isfinite(rows[(f, m)][key]) for m in CMP)]                    # 只取所有被比较方法都有值的折
    v = [np.mean([rows[(f, m)][key] for f in ff]) if np.isfinite([rows[(f, m)][key] for f in ff]).all() else np.nan for m in MS]   # 有缺折的方法记 —
    if key in ('dlog_com', 'r_log'): print('<!-- %s 所用折：%s -->' % (key, '、'.join(ff)))
    if s: best = int(np.nanargmax(np.array(v) * s))
    else: best = int(np.nanargmin(np.abs(np.log(np.array(v)))))
    print('| %s | %s | ' % (nm, {1: '↑', -1: '↓', 0: '→1'}[s]) + ' | '.join('—' if not np.isfinite(x) else (('**' + fmt + '**') if i == best else fmt) % x for i, x in enumerate(v)) + ' |')
print('\n逐折占优（本文·仅粗扫 对 各基线；共同块 |Δlog10k| 与 log10k 相关只计有值的折）：')
for m in MS:
    if m.startswith('本文') or m == '三线性': continue
    for key, s, ff in (('agree', 1, [f for f in FOLDS if f != 'GZ']), ('miss', -1, FP), ('dlog_com', -1, FP), ('kratio', 0, FP)):
        a = [rows[(f, '本文·仅粗扫')][key] for f in ff]; b = [rows[(f, m)][key] for f in ff]
        if s: w = sum((x - y) * s > 0 for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y))
        else: w = sum(abs(np.log(x)) < abs(np.log(y)) for x, y in zip(a, b) if x > 0 and y > 0)
        print('  对 %s %s：%d/%d' % (m, key, w, sum(np.isfinite(x) and np.isfinite(y) for x, y in zip(a, b))))
