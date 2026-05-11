# -*- coding: utf-8 -*-
"""
快速 tissue mask 模块（替代 CUDA tissue ratio）。

每张 WSI 只生成一次低分辨率 tissue mask，
后续 tile 判断直接从 mask 计算，不再读取原始 tile。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class TissueMask:
    mask: np.ndarray
    level: int
    downsample: float


def build_tissue_mask(slide_reader) -> TissueMask:
    """
    为一张 WSI 构建低分辨率 tissue mask。

    选择最接近 TISSUE_MASK_DOWNSAMPLE_TARGET 的 level，
    整图转 HSV 后按 S/V 阈值二值化。
    """
    slide = slide_reader.slide
    target_downsample = config.TISSUE_MASK_DOWNSAMPLE_TARGET

    # 选择最接近目标的 level（优先级高一点点）
    best_level = 0
    best_diff = float("inf")

    for lvl in range(slide.level_count):
        ds = slide.level_downsamples[lvl]
        diff = abs(ds - target_downsample)
        if diff < best_diff:
            best_diff = diff
            best_level = lvl

    actual_ds = slide.level_downsamples[best_level]
    dims = slide.level_dimensions[best_level]
    w, h = int(dims[0]), int(dims[1])

    # 如果 level 0 的 downsample 就超过目标，直接用 level 0
    if w <= 0 or h <= 0:
        best_level = 0
        actual_ds = slide.level_downsamples[0]
        w, h = slide.dimensions

    img = slide.read_region((0, 0), best_level, (w, h)).convert("RGB")
    arr = np.array(img)

    # RGB -> HSV
    import cv2
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = (sat > config.TISSUE_SAT_THRESHOLD) & (val < config.TISSUE_VAL_THRESHOLD)

    return TissueMask(
        mask=mask,
        level=best_level,
        downsample=float(actual_ds),
    )


def tile_tissue_ratio_from_mask(
    mask: TissueMask,
    x0: int,
    y0: int,
    tile_size: int,
) -> float:
    """
    从低分辨率 mask 计算 tile 的 tissue ratio。

    把 level 0 坐标映射到 mask 坐标，统计 mask 区域内 True 的比例。
    """
    ds = mask.downsample
    mh, mw = mask.mask.shape

    mx0 = max(0, int(round(x0 / ds)))
    my0 = max(0, int(round(y0 / ds)))
    mx1 = min(mw, int(round((x0 + tile_size) / ds)))
    my1 = min(mh, int(round((y0 + tile_size) / ds)))

    if mx0 >= mx1 or my0 >= my1:
        return 0.0

    region = mask.mask[my0:my1, mx0:mx1]
    return float(region.mean())
