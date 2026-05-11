# -*- coding: utf-8 -*-
"""
颜色增强模块。

职责：
1. 对 train 样本生成 orig + 4 种颜色增强版本；
2. val 样本只保留 orig；
3. 所有增强不改变图像尺寸和 bbox 标签；
4. 所有输入/输出均为 RGB uint8 ndarray。
"""

from __future__ import annotations

import random
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import config


def make_color_augmented_images(
    image: Image.Image,
    rng: random.Random,
    split_name: str,
) -> List[Tuple[str, np.ndarray]]:
    """
    根据 split_name 和配置生成颜色增强版本列表。

    train 且 ENABLE_COLOR_AUGMENT=True:
      返回 [(orig, ndarray), (clahe, ndarray), (hsv, ndarray), ...]
    否则:
      只返回 [(orig, ndarray)]
    """
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    results: List[Tuple[str, np.ndarray]] = [("orig", arr)]

    if split_name not in config.COLOR_AUGMENT_SPLITS or not config.ENABLE_COLOR_AUGMENT:
        return results

    results.append(("clahe", apply_clahe(arr, rng)))
    results.append(("hsv", apply_hsv(arr, rng)))
    results.append(("brightness_contrast", apply_brightness_contrast(arr, rng)))
    results.append(("gamma", apply_gamma(arr, rng)))

    return results


def apply_clahe(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    CLAHE 增强。
    RGB -> LAB，只对 L 通道做 CLAHE，LAB -> RGB。
    """
    clip_limit = rng.uniform(*config.CLAHE_CLIP_LIMIT_RANGE)
    tile_grid_size = config.CLAHE_TILE_GRID_SIZE

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def apply_hsv(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    HSV 增强。
    RGB -> HSV，随机偏移 H/S/V，clip 到合法范围，HSV -> RGB。
    """
    hue_shift = rng.randint(*config.HSV_HUE_SHIFT_LIMIT)
    sat_shift = rng.randint(*config.HSV_SAT_SHIFT_LIMIT)
    val_shift = rng.randint(*config.HSV_VAL_SHIFT_LIMIT)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.int16)

    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift, 0, 255)

    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_brightness_contrast(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Brightness/Contrast 增强。
    out = image * (1 + contrast) + brightness * 255
    """
    brightness = rng.uniform(*config.BRIGHTNESS_LIMIT)
    contrast = rng.uniform(*config.CONTRAST_LIMIT)

    out = arr.astype(np.float32) * (1.0 + contrast) + brightness * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Gamma 增强。
    gamma ∈ [0.90, 1.10]，LUT 实现。
    """
    gamma_raw = rng.randint(*config.GAMMA_LIMIT)
    gamma = gamma_raw / 100.0

    lut = np.array(
        [((i / 255.0) ** gamma) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )

    return cv2.LUT(arr, lut)
