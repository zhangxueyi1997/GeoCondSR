# -*- coding: utf-8 -*-
"""第 54 步终评汇总（测试岩性）。判据沿用建议三条 + 训练出来的有/无地质消融 + 位置指标。"""
import json, os, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import OUT_ROOT  # noqa: E402
FOLDS = ['CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX']; D = os.environ.get('EVAL_OUT', str(OUT_ROOT) + '/eval/')
R = {f: json.load(open(D + 'eval55_%s.json' % f)) for f in FOLDS if os.path.exists(D + 'eval55_%s.json' % f)}
MET = [('PSNR', 1), ('SSIM', 1), ('Dice', 1), ('孔隙度误差', -1), ('比表面相对误差', -1), ('欧拉数误差', -1),
       ('弦长相对误差', -1), ('S2距离', -1), ('连通跟踪', 1), ('连通度MAE', -1), ('碎裂度误差', -1)]
EXT = [('条纹指数', -1), ('细节逐点相关', 1), ('细节位置对准', 1)]
allm = ['三线性', 'EDSR-3D', 'SRGAN-3D', '上一版_rot2', '参照·c_blk版', '仅均值通路', '新·有地质 w=1', '新·有地质 w*', '新·无地质']
print('地质变量来源 C_MODE=%s   折数 %d   各折 w*：%s' % (R[FOLDS[0]]['C_MODE'], len(R), {f: R[f]['wstar'] for f in R}))
print('本方法所用 c 与目标块实测 c_blk 的 R²（IGV, f_pore, mu_inter, f_dense），6 折均值：', np.round(np.mean([R[f]['c与目标块实测c的R2'] for f in R], 0), 3).tolist())
print('\n一、测试岩性指标（6 折均值）')
print('%-12s %2s' % ('指标', '') + ''.join('%13s' % m for m in allm))
for k, s in MET + EXT:
    row = []
    for m in allm:
        fs = [f for f in R if m in R[f]]
        row.append('%13.4f' % np.mean([R[f][m][k] for f in fs]) if len(fs) == len(R) else '%13s' % '—')
    print('%-12s %2s' % (k, '↑' if s > 0 else '↓') + ''.join(row))
print('真值条纹指数 %.4f' % np.mean([R[f]['真值条纹指数'] for f in R]))


def rule(me, base, mets):
    a = b = 0; det = []
    for k, s in mets:
        w = sum((R[f][me][k] - R[f][base][k]) * s > 0 for f in R)
        a += w >= 5; b += (len(R) - w) >= 5; det.append('%s %d/%d' % (k, w, len(R)))
    v = '优于' if a > 0 and a >= 2 * b else ('不如' if b > 0 and b >= 2 * a else '互有胜负')
    return a, b, v, '，'.join(det)


for me in ('新·有地质 w=1', '新·有地质 w*'):
    print('\n二、%s' % me)
    st = sum(R[f][me]['条纹指数'] <= 0.05 for f in R)
    print('① 出图：条纹指数 ≤0.05 的折数 %d/%d ⇒ %s（斑块/边框另由用户目视）' % (st, len(R), '过' if st >= 5 else '不过'))
    for base in ('EDSR-3D', 'SRGAN-3D', '三线性', '上一版_rot2', '参照·c_blk版'):
        if not all(base in R[f] for f in R): continue
        a, b, v, det = rule(me, base, MET)
        print('② 对 %-10s：本方法占优(≥5/6) %d 项，对方 %d 项 ⇒ %s   [%s]' % (base, a, b, v, det))
    tag = 'w1' if me.endswith('w=1') else 'ws'; g = [R[f]['地质'] for f in R]
    fr = sum(x['%s_true' % tag]['frag'] < x['%s_wrong' % tag]['frag'] for x in g)
    ph = sum(x['%s_true' % tag]['phi'] < x['%s_wrong' % tag]['phi'] for x in g)
    cn = sum(x['%s_true' % tag]['conn_trk'] > x['%s_wrong' % tag]['conn_trk'] for x in g)
    kk = sum(v >= 5 for v in (fr, ph, cn))
    print('③ 地质特异性（真c 优于配错c）：碎裂度 %d/6，孔隙度 %d/6，连通跟踪 %d/6 ⇒ 达标 %d 项 ⇒ %s' % (fr, ph, cn, kk, '过' if kk >= 2 else '不过'))
    if all('新·无地质' in R[f] for f in R):
        a, b, v, det = rule(me, '新·无地质', MET)
        print('④ 训练消融 对 无地质版：有地质占优(≥5/6) %d 项，无地质占优 %d 项 ⇒ %s   [%s]' % (a, b, v, det))
    else:
        print('④ 训练消融：无地质对照组尚未训完')
    for base in ('上一版_rot2', '参照·c_blk版', 'EDSR-3D'):
        if not all(base in R[f] for f in R): continue
        w = sum(R[f][me]['细节位置对准'] > R[f][base]['细节位置对准'] for f in R)
        print('⑤ 位置：细节位置对准 胜 %s 的折数 %d/6（%.3f vs %.3f）' % (base, w, np.mean([R[f][me]['细节位置对准'] for f in R]),
              np.mean([R[f][base]['细节位置对准'] for f in R])))
print('ALLDONE')
