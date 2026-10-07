"""只读需要的那几行 —— 这批 TIFF 是未压缩连续存储，可以直接 memmap 切片。

粗扫每张 2006x2006 (8 MB)，而试样只占中间 271x271 (147 KB)，98% 是白读。
E: 是 USB 外接机械盘，33 MB/s 封顶，读盘是整条流水线的瓶颈，堵掉这个浪费
比什么都值。行优先存储，所以按行切是一段连续字节，一次寻道就读完。
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import tifffile


@lru_cache(maxsize=64)
def layout(path: str):
    """(数据起始偏移, 形状, dtype)；非连续存储返回 None，调用方退回 imread。"""
    with tifffile.TiffFile(path) as t:
        pg = t.pages[0]
        if not pg.is_contiguous or pg.compression != 1:
            return None
        return int(pg.dataoffsets[0]), tuple(pg.shape), np.dtype(pg.dtype)


def read(path, r0=None, r1=None, c0=None, c1=None):
    """读一张切片的 [r0:r1, c0:c1]；只有这些行的字节会真的过盘。"""
    lay = layout(str(path))
    if lay is None:
        a = tifffile.imread(str(path))
        return a if r0 is None else a[r0:r1, c0:c1]
    off, shape, dt = lay
    if r0 is None:
        r0, r1, c0, c1 = 0, shape[0], 0, shape[1]
    r0 = max(0, r0); r1 = min(shape[0], r1)
    mm = np.memmap(path, dtype=dt, mode="r", offset=off, shape=shape)
    out = np.array(mm[r0:r1, c0:c1])
    del mm
    return out
