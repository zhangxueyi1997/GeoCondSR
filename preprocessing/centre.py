"""稳健定心：已知岩石半径，找「覆盖最亮的那个圆盘」的位置。

原来的做法是「四角取背景 + 阈值分割 + 取质心」。这批数据上它会翻车：
细扫里试样占画幅 77%，试样一偏，四角就落进岩石里，背景被高估、门限被抬到
基质之上，于是只框住零星亮矿物，圆心偏出几百像素，裁框被钳到画幅边缘，
试样被切掉一块（G08/G14/G18/G20 都是这样）。

圆盘卷积不依赖任何阈值，也不在乎试样是否贴边或被画幅切掉一角，
只问「把这么大的圆盘放哪里，罩住的总灰度最大」，对纹理、散射晕都免疫。
"""
from __future__ import annotations
import numpy as np
from scipy import signal, ndimage
import fastio


def disc_centre(a, rad_px):
    a = a.astype(np.float32)
    sm = ndimage.uniform_filter(a, size=max(3, int(rad_px / 8)))
    r = int(round(rad_px))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    k = ((yy*yy + xx*xx) <= r*r).astype(np.float32)
    k /= k.sum()
    c = signal.fftconvolve(sm, k, mode="same")
    j = np.unravel_index(int(np.argmax(c)), c.shape)
    return float(j[0]), float(j[1])


def specimen_centre(files, rad_px, n_probe=7):
    idx = np.clip(np.linspace(len(files)*0.3, len(files)*0.7, n_probe).astype(int),
                  0, len(files)-1)
    cy, cx = [], []
    for i in idx:
        y, x = disc_centre(fastio.read(files[i]), rad_px)
        cy.append(y); cx.append(x)
    return float(np.median(cy)), float(np.median(cx))


def physical_bbox(files, voxel_um, half_mm, rad_px):
    cy, cx = specimen_centre(files, rad_px)
    h = int(round(half_mm * 1000.0 / voxel_um))
    H, W = fastio.layout(str(files[len(files)//2]))[1]
    n = 2*h
    if n > H or n > W:
        raise RuntimeError(f"裁框 {n} px 超出画幅 {H}x{W}")
    r0 = max(0, min(int(round(cy)) - h, H - n))
    c0 = max(0, min(int(round(cx)) - h, W - n))
    return (r0, r0+n, c0, c0+n), (cy, cx)
