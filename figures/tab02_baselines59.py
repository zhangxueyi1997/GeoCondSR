# -*- coding: utf-8 -*-
"""表 2（第 59 步终版）：只用粗扫时 7 种方法的指标，6 折均值（每折 200 块，同一批块与噪声种子）。
数据：data/eval/eval59/eval59_<折>.json（三线性、EDSR-3D、SRGAN-3D、SwinIR-3D、扩散、本文·仅粗扫）；均值通路取 eval55c 的“仅均值通路”（同一批块）。
推理用时：单块、批大小 1、含 8 次硬投影（扩散为 50 步 DDIM），eval59.py 实测。计分规则同 3.4 节（≥5/6 折占优得 1 分）。"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; E = FIGDATA / 'eval'
F = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']
R = {f: json.load(open(E / 'eval59' / ('eval59_%s.json' % f), encoding='utf-8')) for f in F}
A = {f: json.load(open(E / 'eval55c' / ('eval55_%s.json' % f), encoding='utf-8')) for f in F}
for f in F:   # 同一批块核对：本文在两次评价中的数值一致
    assert abs(R[f]['本文·仅粗扫']['孔隙度误差'] - A[f]['新·无地质']['孔隙度误差']) < 1e-9
COL = [('三线性', '三线性'), ('EDSR-3D', 'EDSR-3D'), ('SRGAN-3D', 'SRGAN-3D'), ('SwinIR-3D', 'SwinIR-3D'), ('扩散', '扩散模型'), ('均值通路', '均值通路'), ('本文·仅粗扫', '本文')]
get = lambda m, f: A[f]['仅均值通路'] if m == '均值通路' else R[f][m]
ROWS = [('PSNR', 'PSNR (dB)', 1, '%.2f'), ('SSIM', 'SSIM', 1, '%.3f'), ('Dice', 'Dice', 1, '%.3f'), ('孔隙度误差', '孔隙率误差', -1, '%.4f'),
        ('比表面相对误差', '比表面相对误差ᵃ', -1, '%.3f'), ('欧拉数误差', '欧拉数误差', -1, '%.3f'), ('弦长相对误差', '弦长相对误差', -1, '%.3f'),
        ('S2距离', '两点函数距离', -1, '%.4f'), ('连通跟踪', '连通跟踪', 1, '%.3f'), ('连通度MAE', '连通误差', -1, '%.3f'), ('碎裂度误差', '碎裂度误差', -1, '%.3f'),
        ('条纹指数', '条纹指数ᵇ', -1, '%.3f'), ('细节位置对准', '细节位置对准', 1, '%.3f')]
MET = [r for r in ROWS[:11]]
print('**表{tab:baselines}** 只用粗扫时各方法在 6 个测试产地上的指标（6 折均值，每折 200 块；粗体为各行最优）\n')
print('| 指标 | 方向 | ' + ' | '.join(n for _, n in COL) + ' |'); print('|---|---|' + '---|' * len(COL))
for k, name, s, fmt in ROWS:
    v = [np.mean([get(m, f)[k] for f in F]) for m, _ in COL]
    best = max(v) if s > 0 else min(v)
    cells = [('**' + fmt % x + '**') if (fmt % x) == (fmt % best) else fmt % x for x in v]
    print('| %s | %s | ' % (name, '↑' if s > 0 else '↓') + ' | '.join(cells) + ' |')
t = [np.mean([R[f][m]['每块用时秒'] for f in F]) if m != '均值通路' else np.nan for m, _ in COL]
print('| 推理用时 (s/块)ᶜ | ↓ | ' + ' | '.join('%.3f' % x if np.isfinite(x) else '—' for x in t) + ' |')
sc = []
for m, _ in COL:
    if m == '本文·仅粗扫': sc.append('—'); continue
    wa = wb = 0
    for k, _, s, _ in MET:
        w = sum((R[f]['本文·仅粗扫'][k] - get(m, f)[k]) * s > 0 for f in F); wa += w >= 5; wb += (6 - w) >= 5
    sc.append('%d:%d' % (wa, wb))
print('| 计分（本文 : 该方法） | | ' + ' | '.join(sc) + ' |')
noSD = [f for f in F if f != 'SD']
ss = {m: np.mean([get(m, f)['比表面相对误差'] for f in noSD]) for m, _ in COL}
print('\nᵃ 山东折的比表面误差为离群值（本文 %.2f），去掉该折后的 5 折均值：本文 %.2f，EDSR-3D %.2f，SRGAN-3D %.2f，SwinIR-3D %.2f，扩散模型 %.2f。ᵇ 细扫为 0.025。ᶜ 单个目标块、批大小 1、一块 GPU，含 8 次硬投影；扩散模型为 50 步 DDIM 采样。' %
      (R['SD']['本文·仅粗扫']['比表面相对误差'], ss['本文·仅粗扫'], ss['EDSR-3D'], ss['SRGAN-3D'], ss['SwinIR-3D'], ss['扩散']))
print('\n本文相对四个深度基线中最好者的变化：')
for k, name, s, fmt in ROWS[3:11]:
    vo = np.mean([R[f]['本文·仅粗扫'][k] for f in F]); vb = [np.mean([R[f][m][k] for f in F]) for m in ['EDSR-3D', 'SRGAN-3D', 'SwinIR-3D', '扩散']]
    b = min(vb) if s < 0 else max(vb); print('  %-8s 本文 %.4f 最好基线 %.4f  变化 %+.1f%%' % (name, vo, b, 100 * (vo - b) / b))
print('扩散/本文 用时比 %.0f' % (t[4] / t[6]))
