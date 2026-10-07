# -*- coding: utf-8 -*-
"""论文三统一图风格（SCI 期刊版式）：7 pt 无衬线、细线、去上/右框、统一配色、面板标号加粗小写字母。
宽度：单栏 88 mm、1.5 栏 120 mm、双栏 180 mm。输出矢量 PDF + 600 dpi PNG。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing')); from paths import REPO_ROOT, DATA_ROOT, RUNS_ROOT, OUT_ROOT, RAW_ROOT, FIGDATA  # noqa: E402

MM = 1 / 25.4
SINGLE, ONEHALF, DOUBLE = 88, 120, 180
# 方法配色（全文一致）：本文用砖红突出，基线用克制的蓝灰系
C = {'三线性': '#B7B7B7', 'EDSR-3D': '#3E6FA8', 'SRGAN-3D': '#D9A23F', 'SwinIR-3D': '#5A9E5A', '扩散': '#8A6DB0',
     '均值通路': '#8FB8DE', '本文': '#C0392B', '本文+稀疏细扫': '#7B1A14', '细扫': '#222222'}
EN = {'三线性': 'Trilinear', 'EDSR-3D': 'EDSR-3D', 'SRGAN-3D': 'SRGAN-3D', 'SwinIR-3D': 'SwinIR-3D', '扩散': 'Diffusion',
      '均值通路': 'Mean path', '本文': 'Ours', '本文+稀疏细扫': 'Ours + sparse fine', '细扫': 'Fine scan'}
GREY, LIGHT = '#6E6E6E', '#D9D9D9'
FOLD_MK = {'CQ': 'o', 'SC': 's', 'YN': '^', 'GZ': 'D', 'SD': 'v', 'SHX': 'P'}


def apply():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 7,
        'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.0, 'lines.markersize': 3.5, 'legend.frameon': False, 'legend.handlelength': 1.2,
        'axes.labelpad': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2, 'savefig.dpi': 600, 'pdf.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})


def label(ax, s, dx=-0.02, dy=1.0):
    """面板标号：加粗小写字母，放在坐标区左上角外侧。"""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom', ha='right')


def save(fig, path):
    path = Path(path); fig.savefig(path.with_suffix('.pdf')); fig.savefig(path.with_suffix('.png'), dpi=600)
    print('写出', path.with_suffix('.pdf').name, '+ png')
