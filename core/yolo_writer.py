# -*- coding: utf-8 -*-
"""
YOLO 数据写入模块（高性能版）。

- 只支持 train / val。
- train 支持颜色增强多版本输出。
- JPEG 使用 cv2.imencode 加速写入。
- save_image 同时接受 PIL Image 和 ndarray。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import config
from core.color_augment import make_color_augmented_images

YoloBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class WrittenSample:
    image_path: Path
    label_path: Path
    filename_stem: str
    variant: str


def prepare_output_dirs(output_dir: Path) -> None:
    """
    创建 YOLO 数据集目录：

    output_dir/
      images/
        train/
        val/
      labels/
        train/
        val/
    """
    output_dir = Path(output_dir)

    for split_name in ("train", "val"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def build_sample_stem(
    slide_stem: str,
    sample_type: str,
    sample_index: int,
    x0: int,
    y0: int,
    variant: str = "orig",
) -> str:
    return f"{slide_stem}_{sample_type}_{sample_index:06d}_x{x0}_y{y0}_{variant}"


def write_yolo_sample(
    output_dir: Path,
    split_name: str,
    slide_stem: str,
    sample_type: str,
    sample_index: int,
    x0: int,
    y0: int,
    image: Image.Image,
    yolo_boxes: Sequence[YoloBox],
    class_id: int,
    rng: random.Random,
) -> List[WrittenSample]:
    """
    保存一个 YOLO 样本（支持颜色增强多版本）。

    train 且 ENABLE_COLOR_AUGMENT=True：
        保存 orig + clahe + hsv + brightness_contrast + gamma

    val：
        只保存 orig
    """
    output_dir = Path(output_dir)

    if split_name not in ("train", "val"):
        raise ValueError(f"Invalid split_name: {split_name}")

    variants = make_color_augmented_images(image, rng, split_name)

    written: List[WrittenSample] = []

    for variant, variant_arr in variants:
        sample_stem = build_sample_stem(
            slide_stem=slide_stem,
            sample_type=sample_type,
            sample_index=sample_index,
            x0=x0,
            y0=y0,
            variant=variant,
        )

        image_path = (
            output_dir / "images" / split_name / f"{sample_stem}{config.IMAGE_EXT}"
        )
        label_path = output_dir / "labels" / split_name / f"{sample_stem}.txt"

        save_image(variant_arr, image_path)
        save_label(label_path, yolo_boxes, class_id)

        written.append(
            WrittenSample(
                image_path=image_path,
                label_path=label_path,
                filename_stem=sample_stem,
                variant=variant,
            )
        )

    return written


def save_image(image: Union[Image.Image, np.ndarray], image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = image_path.suffix.lower()

    if config.USE_CV2_JPEG_WRITER and suffix in (".jpg", ".jpeg"):
        import cv2

        if isinstance(image, np.ndarray):
            arr = image
        else:
            arr = np.asarray(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
                arr = np.asarray(image)

        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            arr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(config.JPEG_QUALITY)],
        )
        if not ok:
            raise RuntimeError(f"cv2.imencode failed: {image_path}")
        image_path.write_bytes(encoded.tobytes())
        return

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    if suffix in (".jpg", ".jpeg"):
        image.save(image_path, format="JPEG", quality=config.JPEG_QUALITY)
    elif suffix == ".png":
        image.save(image_path, format="PNG")
    else:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")


def save_label(
    label_path: Path,
    yolo_boxes: Sequence[YoloBox],
    class_id: int,
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    for box in yolo_boxes:
        xc, yc, w, h = box

        if not _valid_yolo_box(xc, yc, w, h):
            continue

        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    label_path.write_text("".join(lines), encoding="utf-8")


def _valid_yolo_box(xc: float, yc: float, w: float, h: float) -> bool:
    if w <= 0 or h <= 0:
        return False
    if not (0 <= xc <= 1 and 0 <= yc <= 1):
        return False
    if not (0 < w <= 1 and 0 < h <= 1):
        return False
    return True
