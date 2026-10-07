# -*- coding: utf-8 -*-
"""论文一制图的统一样式，按 Elsevier 官方 artwork 规范实现。

规范原文（2026-07-26 抓取自 elsevier.com/about/policies-and-standards/author/
artwork-and-media-instructions/artwork-sizing 与 .../artwork-overview）：

- **目标尺寸**：minimal 30 mm ｜ single column **90 mm** ｜ 1.5 column **140 mm**
  ｜ double column (full width) **190 mm**。300 dpi 下对应 354 / 1063 / 1654 / 2244 px。
- **字号**：*"the lettering on the artwork should have a finished, printed size of
  7 pt for normal text and no smaller than 6 pt for subscript and superscript
  characters. Smaller lettering will yield text that is hardly legible."*
- **字体**：只推荐 Arial (or Helvetica)、Courier、Symbol、Times (or Times New Roman)。
  用其它字体 Elsevier 可能替换，导致缺字符或叠字。**matplotlib 默认的 DejaVu Sans
  不在名单里**，所以这里显式设为 Arial。
- **格式**：矢量图首选 EPS/PDF；位图 halftone 300 dpi、combination 500 dpi、
  line art 1000 dpi。彩图用 RGB。每图单独一个文件。
- 同一篇文章内字号不应相差太多（"does not vary too much in size"）。

**为什么必须管这件事**：改版前全部七张图都画到 289–355 mm 宽，插进 Word 时按
16 cm 缩放，缩放比 0.45–0.55，于是 8 pt 的标注实际印出来只有 3.6–4.4 pt，
远低于 7 pt 底线。本模块把图宽钉在 190 mm，并把最小字号设到 8.5 pt，
使投稿件按 17 cm 版心插入（缩放 0.895）后仍有 7.6 pt。

用法：

    from gse_style import apply_style, figure, save
    apply_style()
    fig, axes = figure(DOUBLE, height_mm=95, ncols=3)
    ...
    save(fig, "fig04_descriptor_loss")   # 同时写 PDF（矢量）与 300 dpi PNG
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

MM = 1.0 / 25.4                      # mm -> inch
MINIMAL, SINGLE, ONE_HALF, DOUBLE = 30.0, 90.0, 140.0, 190.0

# 投稿 Word 里图片按 17 cm 宽插入（图节页边距 2.0 cm，A4 版心 170 mm），
# 故 190 mm 的图缩放比为 170/190 = 0.895。要让成品 ≥7 pt，标称须 ≥7/0.895 = 7.8 pt。
DOCX_SCALE = 170.0 / DOUBLE
MIN_NOMINAL_PT = 7.0 / DOCX_SCALE     # ≈ 7.8

# 分级字号，最小一档仍高于底线，且档差不大（规范要求字号不宜相差过多）
PT_TITLE = 10.0
PT_LABEL = 9.0
PT_TICK = 8.5
PT_NOTE = 8.5
PT_LEGEND = 8.5

# 三级取样层级的固定配色（全文统一）
C_PLUG, C_SLAB, C_MICRO = "#c8102e", "#e8890c", "#1f6fb4"
C_ACCENT, C_GREY = "#2e7d32", "#5a5a5a"
LEVEL_STYLE = {"large_ct": ("Core plug", C_PLUG),
               "slab_ct": ("Slab", C_SLAB),
               "small_ct": ("Micro-column", C_MICRO)}

# 2026-07-27：每个层级再配一个**点形**。只靠颜色区分对色觉缺陷读者和灰度打印都不
# 友好 —— C_PLUG 的红与 C_SLAB 的橙在灰度下亮度接近，红与蓝的亮度也接近。
# 形状是冗余编码，颜色失效时仍能读。
LEVEL_MARKER = {"large_ct": "o", "slab_ct": "s", "small_ct": "^"}


def apply_style() -> None:
    """把 Elsevier 的字体与字号要求装进 rcParams。"""
    plt.rcParams.update({
        # Arial 优先，Helvetica/Times 为回退，全部在 Elsevier 推荐名单内
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": PT_TICK,
        "axes.titlesize": PT_TITLE,
        "axes.labelsize": PT_LABEL,
        "xtick.labelsize": PT_TICK,
        "ytick.labelsize": PT_TICK,
        "legend.fontsize": PT_LEGEND,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        # **不要用 bbox_inches="tight"**。它按实际内容裁边，于是同一批图会输出成
        # 各不相同的宽度（示意图内部空白多，曾被裁到 147 mm，而数据图是 189 mm）。
        # Elsevier 要求 "a uniform look for all artwork contained in a single
        # article"，且图宽必须落在 90/140/190 mm 这几档上，所以这里保留声明的画布
        # 尺寸，由 constrained_layout 负责把内容排进去。
        "savefig.bbox": "standard",
        "pdf.fonttype": 42,          # TrueType，保证字体可嵌入而非转曲/位图
        "ps.fonttype": 42,
    })


def figure(width_mm: float = DOUBLE, height_mm: float = 90.0, **kwargs):
    """按毫米建图。width_mm 应取 SINGLE / ONE_HALF / DOUBLE 之一。"""
    if width_mm not in (MINIMAL, SINGLE, ONE_HALF, DOUBLE):
        raise ValueError("图宽必须是 Elsevier 的目标尺寸之一：30/90/140/190 mm，"
                         f"收到 {width_mm}")
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM), **kwargs)


def panel_label(ax, text: str, *, dx: float = -0.085, dy: float = 1.035) -> None:
    """左上角的 (a)/(b)/(c) 面板标号。图题写在图注里，不占面板空间。"""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=PT_TITLE,
            fontweight="bold", va="bottom", ha="left")


def save(fig, stem: str, *, dpi: int = 300) -> list[Path]:
    """写出 PDF（矢量，投稿用）与 PNG（位图，嵌进 Word 送审用）。

    Elsevier 对线图位图要求 1000 dpi，矢量 PDF 则无此问题，所以投稿以 PDF 为准；
    PNG 只用于把图嵌进送审 Word，300 dpi 足够且不至于把文件撑得过大。
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": dpi})):
        p = FIGDIR / f"{stem}.{ext}"
        fig.savefig(p, facecolor="white", **kw)
        out.append(p)
    plt.close(fig)
    w_mm = fig.get_size_inches()[0] / MM
    print(f"  wrote {stem}.pdf + {stem}.png  ({w_mm:.0f} mm wide, "
          f"min nominal font {PT_TICK} pt -> {PT_TICK * DOCX_SCALE:.1f} pt at 17 cm)")
    return out


def check_min_font() -> None:
    if PT_TICK < MIN_NOMINAL_PT:
        raise AssertionError(
            f"最小字号 {PT_TICK} pt 在 {DOCX_SCALE:.3f} 缩放下只有 "
            f"{PT_TICK * DOCX_SCALE:.1f} pt，低于 Elsevier 的 7 pt 底线")


check_min_font()
