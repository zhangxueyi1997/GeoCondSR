# -*- coding: utf-8 -*-
"""第 57b 步汇总（诊断）。预先定的主判据：「G·粗扫ĉ+粗扫整体水平」对「无地质（训练）」，规则同第 49 步（≥5/6 折才计，占优项数 ≥ 对方两倍 ⇒ 优于）。"""
import json, os, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import OUT_ROOT  # noqa: E402
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; D = str(OUT_ROOT) + '/eval57b/'
R = {f: json.load(open(D + 'eval57b_%s.json' % f)) for f in FOLDS if os.path.exists(D + 'eval57b_%s.json' % f)}
MET = [('PSNR', 1), ('SSIM', 1), ('Dice', 1), ('孔隙度误差', -1), ('比表面相对误差', -1), ('欧拉数误差', -1),
       ('弦长相对误差', -1), ('S2距离', -1), ('连通跟踪', 1), ('连通度MAE', -1), ('碎裂度误差', -1)]
ARMS = ['G·块实测c', 'A·整体水平4项全真', 'B·孔隙率+孔隙灰度真', 'C·IGV+高密相真', 'D·整体水平全用粗扫', '无地质(训练)']
print('折数 %d' % len(R))
print('\n〇、粗扫推的整体水平 vs 细扫真实（IGV, f_pore, mu_inter, f_dense）')
for f in R:
    for g, v in R[f]['整体水平'].items(): print('   %-4s %s 粗扫推 %s ｜ 真实 %s' % (f, g, np.round(v['粗扫推'], 3), np.round(v['细扫真实'], 3)))
print('\n一、6 折均值')
print('%-10s' % '指标' + ''.join('%16s' % a for a in ARMS))
for k, s in MET: print('%-10s' % (k + ('↑' if s > 0 else '↓')) + ''.join('%16.4f' % np.mean([R[f][a][k] for f in R]) for a in ARMS))


def rule(me, base):
    a = b = 0; det = []
    for k, s in MET:
        w = sum((R[f][me][k] - R[f][base][k]) * s > 0 for f in R); a += w >= 5; b += (len(R) - w) >= 5; det.append('%s %d' % (k, w))
    v = '优于' if a > 0 and a >= 2 * b else ('不如' if b > 0 and b >= 2 * a else '互有胜负')
    return '%d:%d ⇒ %s   [%s]' % (a, b, v, '，'.join(det))


print('\n二、对 无地质（训练）  诊断：哪几项整体水平推准了才有用')
for a in ARMS[:5]: print('   %-18s %s' % (a, rule(a, '无地质(训练)')))
print('\n三、对 块实测c（循环上限）')
for a in ARMS[1:5]: print('   %-18s %s' % (a, rule(a, 'G·块实测c')))
print('\n四、增益保留率 =（无地质 − 该来源）/（无地质 − 块实测），6 折合计')
for k in ('孔隙度误差', '欧拉数误差', '弦长相对误差', '连通度MAE', 'S2距离', '比表面相对误差', '碎裂度误差'):
    e = {a: np.mean([R[f][a][k] for f in R]) for a in ARMS}; den = e['无地质(训练)'] - e['G·块实测c']
    print('   %-8s ' % k + '  '.join('%s %.0f%%' % (a, 100 * (e['无地质(训练)'] - e[a]) / den) for a in ARMS[1:6]))
print('ALLDONE')
