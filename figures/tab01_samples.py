# -*- coding: utf-8 -*-
"""表 1：样品与成像。数值全部从原始记录读取，不手抄。
来源：registration.csv（配准结果，registration_2026）；paper1_scale_loss/src/p1/physical.py（实验室孔隙度，平行对）；
manifests/real_data_import_mapping_20260719.csv（组号 → 大圆柱 A 编号）；ana56_data.npz（细扫 2 μm 可分辨孔隙度，逐格 c_blk 的 f_pore 平均）。
小柱取自平行对中的 B 柱（论文一 §2.1），故实验室孔隙度取 B 柱。"""
import csv, re, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402
H = Path(__file__).resolve().parent; D = FIGDATA; ROOT = REPO_ROOT
src = (REPO_ROOT / 'configs' / 'sample_properties.py').read_text(encoding='utf-8')
PHI = {m[0]: float(m[1]) for m in re.findall(r'\("([A-Z]+-\d+)", "[^"]+", [\d.]+, [\d.]+, [\d.]+, ([\d.]+)', src)}
PAIR = dict(re.findall(r'\("([A-Z]+-\d+)", "([A-Z]+-\d+)"\)', src)); PAIR.update({b: a for a, b in list(PAIR.items())})
MAP = {r['group_id']: r for r in csv.DictReader(open(REPO_ROOT / 'configs' / 'sample_mapping.csv', encoding='utf-8-sig'))}
REG = {r['gid']: r for r in csv.DictReader(open(D / 'registration.csv', encoding='utf-8-sig'))}
A56 = np.load(D / 'ana56_data.npz')
CODE = {'G01': 'CQ', 'G19': 'CQ', 'G02': 'GZ', 'G03': 'YN', 'G16': 'YN', 'G17': 'YN', 'G05': 'SC', 'G06': 'SC', 'G07': 'SC',
        'G09': 'SC', 'G11': 'SC', 'G12': 'SD', 'G13': 'SHX', 'G15': 'SX'}
ORIGIN = {'CQ': '重庆', 'GZ': '贵州', 'YN': '云南', 'SC': '四川', 'SD': '山东', 'SHX': '陕西', 'SX': '山西'}
rows = []
for g in sorted(REG):
    r = REG[g]; a = MAP[g]['large_sample_id']; b = PAIR[a]
    reg = r['status'] == 'registered'
    rows.append(dict(
        组号=g, 产地=ORIGIN.get(CODE.get(g, ''), MAP[g]['parent_sample_id'].split('-')[0]), 代号=CODE.get(g, '—'), 小柱=r['sample'],
        B柱=b, 实验室孔隙度_B=PHI[b], 实验室孔隙度_A=PHI[a],
        细扫真实体素_um=float(r['v_measured_um']) if reg else float('nan'),
        配准残差_体素=float(r['align_rms_voxel']) if reg else float('nan'),
        逐层NCC=float(r['perslice_ncc']) if reg else float('nan'),
        可识别波长_um=float(r['identif_depth_um']) if reg else float('nan'),
        有效格数=int(len(A56[g + '_rows'])) if g in CODE else 0,
        细扫可分辨孔隙度=100 * float(np.nanmean(A56[g + '_rows'][:, 40])) if g in CODE else float('nan'),
        用途=('仅训练' if CODE.get(g) == 'SX' else ('测试折 ' + CODE[g])) if g in CODE else '配准失败，未使用'))
with open(H / 'tab01_samples.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r['用途'] != '配准失败，未使用']
print('| 组号 | 产地 | 小柱 | 实验室孔隙度 (%)ᵃ | 细扫体素 (μm)ᵇ | 配准残差 (粗体素) | 逐层相关 | 可识别波长 (μm)ᶜ | 有效格数 | 2 μm 可分辨孔隙度 (%) | 用途 |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for r in ok:
    print('| %s | %s | %s | %.2f | %.3f | %.3f | %.3f | %.1f | %d | %.2f | %s |' % (r['组号'], r['产地'], r['小柱'], r['实验室孔隙度_B'], r['细扫真实体素_um'], r['配准残差_体素'],
          r['逐层NCC'], r['可识别波长_um'], r['有效格数'], r['细扫可分辨孔隙度'], r['用途']))
v = np.array([[r['细扫真实体素_um'], r['配准残差_体素'], r['逐层NCC'], r['可识别波长_um']] for r in ok])
print('\n中位数：体素 %.3f（范围 %.3f–%.3f），残差 %.3f，逐层相关 %.3f，可识别波长 %.1f（范围 %.1f–%.1f）；有效格合计 %d' % (
    np.median(v[:, 0]), v[:, 0].min(), v[:, 0].max(), np.median(v[:, 1]), np.median(v[:, 2]), np.median(v[:, 3]), v[:, 3].min(), v[:, 3].max(), sum(r['有效格数'] for r in ok)))
print('可分辨/实验室孔隙度比：中位 %.2f（范围 %.2f–%.2f）' % tuple(np.percentile([r['细扫可分辨孔隙度'] / r['实验室孔隙度_B'] for r in ok], [50, 0, 100])))
print('配准失败：', ', '.join('%s %s' % (r['组号'], r['小柱']) for r in rows if r['用途'] == '配准失败，未使用'))
