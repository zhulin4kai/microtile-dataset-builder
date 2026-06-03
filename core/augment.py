# -*- coding: utf-8 -*-
"""颜色增强与 YOLO label 写入。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

import config


def make_color_augmented_images(
    image: Image.Image,
    rng: random.Random,
) -> List[Tuple[str, np.ndarray]]:
    """生成 orig 以及可选的颜色增强版本。"""
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    results: List[Tuple[str, np.ndarray]] = [("orig", arr.copy())]

    if not config.ENABLE_COLOR_AUGMENT:
        return results

    results.append(("clahe", _apply_clahe(arr, rng)))
    results.append(("hsv", _apply_hsv(arr, rng)))
    results.append(("brightness_contrast", _apply_brightness_contrast(arr, rng)))
    results.append(("gamma", _apply_gamma(arr, rng)))

    return results


def save_image_variants(
    arr: np.ndarray,
    stem: str,
    split_name: str,
    rng: random.Random,
) -> List[str]:
    """把 image variants 写入 output_dir/images/<split_name>/。"""
    out_dir = config.OUTPUT_DIR / "images" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 增强逻辑以 PIL Image 为入口，这里统一转换一次。
    img = Image.fromarray(arr)
    variants = make_color_augmented_images(img, rng)
    stems: List[str] = []

    for variant, varr in variants:
        vstem = f"{stem}_{variant}"
        path = out_dir / f"{vstem}{config.IMAGE_EXT}"
        _save_image(varr, path)
        stems.append(vstem)

    return stems


def save_box_label(label_path: Path, boxes: Sequence[Tuple[float, float, float, float]]) -> None:
    """写入 YOLO detect label：class xc yc w h。"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for (xc, yc, w, h) in boxes:
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{config.CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    if not lines:
        if not config.WRITE_EMPTY_LABEL_FOR_NEGATIVE:
            if label_path.exists():
                label_path.unlink()
            return
    label_path.write_text("".join(lines), encoding="utf-8")


# ── internal: color augmentation ──────────────────────────────────────


def get_variant_names() -> List[str]:
    """根据颜色增强开关返回将要写出的 image variant 名称。"""
    if not config.ENABLE_COLOR_AUGMENT:
        return ["orig"]
    return ["orig", "clahe", "hsv", "brightness_contrast", "gamma"]


def _apply_clahe(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    clip_limit = rng.uniform(*config.CLAHE_CLIP_LIMIT_RANGE)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def _apply_hsv(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    hue_shift = rng.randint(*config.HSV_HUE_SHIFT_LIMIT)
    sat_shift = rng.randint(*config.HSV_SAT_SHIFT_LIMIT)
    val_shift = rng.randint(*config.HSV_VAL_SHIFT_LIMIT)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _apply_brightness_contrast(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    brightness = rng.uniform(*config.BRIGHTNESS_LIMIT)
    contrast = rng.uniform(*config.CONTRAST_LIMIT)
    out = arr.astype(np.float32) * (1.0 + contrast) + brightness * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_gamma(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    gamma_raw = rng.randint(*config.GAMMA_LIMIT)
    gamma = gamma_raw / 100.0
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(arr, lut)


# ── internal: image write ──────────────────────────────────────────────


def _save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(config.JPEG_QUALITY)]
        ok, encoded = cv2.imencode(ext, bgr, params)
    elif ext == ".png":
        ok, encoded = cv2.imencode(".png", bgr)
    else:
        raise ValueError(f"不支持的文件扩展名: {ext}")
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    path.write_bytes(encoded.tobytes())
