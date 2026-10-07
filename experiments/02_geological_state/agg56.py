# -*- coding: utf-8 -*-
"""第 56 步判决实验汇总：各种 c 来源 对 无地质（训练）/ 对 块实测c，判据同第 49 步（≥5/6 折才计，占优项数 ≥ 对方两倍 ⇒ 优于）。"""
import json, os, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import OUT_ROOT  # noqa: E402
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; D = str(OUT_ROOT) + '/eval56/'
R = {f: json.load(open(D + 'eval56_%s.json' % f)) for f in FOLDS if os.path.exists(D + 'eval56_%s.json' % f)}
MET = [('PSNR', 1), ('SSIM', 1), ('Dice', 1), ('孔隙度误差', -1), ('比表面相对误差', -1), ('欧拉数误差', -1),
       ('弦长相对误差', -1), ('S2距离', -1), ('连通跟踪', 1), ('连通度MAE', -1), ('碎裂度误差', -1)]
ARMS = ['G·块实测c', 'G·样品均值c', 'G·远处10块细扫', 'G·粗扫ĉ', 'G·粗扫ĉ+远处标定', 'G·训练常数c', '无地质(训练)']
print('折数 %d' % len(R))
print('\n一、6 折均值')
print('%-10s' % '指标' + ''.join('%14s' % a for a in ARMS))
for k, s in MET: print('%-10s' % (k + ('↑' if s > 0 else '↓')) + ''.join('%14.4f' % np.mean([R[f][a][k] for f in R]) for a in ARMS))
print('\n  c 与块实测 c 的 R²（IGV, f_pore, mu_inter, f_dense；6 折中位数）')
for a in ARMS[1:5]: print('   %-14s %s' % (a, np.round(np.median([R[f]['c与块实测R2'][a] for f in R], 0), 3).tolist()))


def rule(me, base):
    a = b = 0; det = []
    for k, s in MET:
        w = sum((R[f][me][k] - R[f][base][k]) * s > 0 for f in R); a += w >= 5; b += (len(R) - w) >= 5; det.append('%s %d' % (k, w))
    v = '优于' if a > 0 and a >= 2 * b else ('不如' if b > 0 and b >= 2 * a else '互有胜负')
    return '%d:%d ⇒ %s   [%s]' % (a, b, v, '，'.join(det))


print('\n二、对 无地质（训练）')
for a in ARMS[:6]: print('   %-14s %s' % (a, rule(a, '无地质(训练)')))
print('\n三、对 块实测c（上限）')
for a in ARMS[1:6]: print('   %-14s %s' % (a, rule(a, 'G·块实测c')))
print('\n四、增益保留率 =（无地质 − 该来源）/（无地质 − 块实测），6 折合计误差')
for k in ('孔隙度误差', '欧拉数误差', '弦长相对误差', '连通度MAE', 'S2距离'):
    e = {a: np.mean([R[f][a][k] for f in R]) for a in ARMS}; den = e['无地质(训练)'] - e['G·块实测c']
    print('   %-8s ' % k + '  '.join('%s %.0f%%' % (a[2:], 100 * (e['无地质(训练)'] - e[a]) / den) if abs(den) > 1e-12 else '—' for a in ARMS[1:6]))
print('ALLDONE')
