# -*- coding: utf-8 -*-
"""
颜色增强模块。

职责：
1. 对 train 样本生成 orig + 4 种颜色增强版本；
2. val 样本只保留 orig；
3. 所有增强不改变图像尺寸和 bbox 标签。
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
) -> List[Tuple[str, Image.Image]]:
    """
    根据 split_name 和配置生成颜色增强版本列表。

    train 且 ENABLE_COLOR_AUGMENT=True:
      返回 orig + clahe + hsv + brightness_contrast + gamma
    否则:
      只返回 orig
    """
    results: List[Tuple[str, Image.Image]] = [("orig", image)]

    if split_name not in config.COLOR_AUGMENT_SPLITS or not config.ENABLE_COLOR_AUGMENT:
        return results

    results.append(("clahe", apply_clahe(image, rng)))
    results.append(("hsv", apply_hsv(image, rng)))
    results.append(("brightness_contrast", apply_brightness_contrast(image, rng)))
    results.append(("gamma", apply_gamma(image, rng)))

    return results


def apply_clahe(image: Image.Image, rng: random.Random) -> Image.Image:
    """
    CLAHE 增强。

    方式：RGB -> LAB，只对 L 通道做 CLAHE，LAB -> RGB。
    """
    clip_limit = rng.uniform(*config.CLAHE_CLIP_LIMIT_RANGE)
    tile_grid_size = config.CLAHE_TILE_GRID_SIZE

    img_np = np.array(image)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return Image.fromarray(rgb_eq)


def apply_hsv(image: Image.Image, rng: random.Random) -> Image.Image:
    """
    HSV 增强。

    方式：RGB -> HSV，随机偏移 H/S/V，clip 到合法范围，HSV -> RGB。
    OpenCV H 范围 [0, 179]。
    """
    hue_shift = rng.randint(*config.HSV_HUE_SHIFT_LIMIT)
    sat_shift = rng.randint(*config.HSV_SAT_SHIFT_LIMIT)
    val_shift = rng.randint(*config.HSV_VAL_SHIFT_LIMIT)

    img_np = np.array(image)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.int16)

    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift, 0, 255)

    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return Image.fromarray(rgb)


def apply_brightness_contrast(image: Image.Image, rng: random.Random) -> Image.Image:
    """
    Brightness/Contrast 增强。

    公式：out = image * (1 + contrast) + brightness * 255
    """
    brightness = rng.uniform(*config.BRIGHTNESS_LIMIT)
    contrast = rng.uniform(*config.CONTRAST_LIMIT)

    img_np = np.array(image).astype(np.float32)
    out = img_np * (1.0 + contrast) + brightness * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


def apply_gamma(image: Image.Image, rng: random.Random) -> Image.Image:
    """
    Gamma 增强。

    参数 gamma ∈ [0.90, 1.10]，使用 LUT 实现。
    注意 config.GAMMA_LIMIT = (90, 110)，实际 gamma 要除以 100。
    """
    gamma_raw = rng.randint(*config.GAMMA_LIMIT)
    gamma = gamma_raw / 100.0

    lut = np.array(
        [((i / 255.0) ** gamma) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )

    img_np = np.array(image)
    out = cv2.LUT(img_np, lut)

    return Image.fromarray(out)
