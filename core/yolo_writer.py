# -*- coding: utf-8 -*-
"""
YOLO 数据写入模块。

职责：
1. 创建 images/train、labels/train 等目录；
2. 保存 jpg tile（支持多颜色增强版本）；
3. 保存 YOLO txt；
4. 保证 image 与 label 文件名一一对应。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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
    """
    文件名带来源坐标和 variant，方便回查。
    """
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

    正样本：
        yolo_boxes 非空

    负样本：
        yolo_boxes 为空，写空 txt

    train 且 ENABLE_COLOR_AUGMENT=True：
        保存 orig + 4 种颜色增强版本（每个版本保存独立 image 和 label）

    val：
        只保存 orig
    """
    output_dir = Path(output_dir)

    if split_name not in ("train", "val"):
        raise ValueError(f"Invalid split_name: {split_name}")

    augmented = make_color_augmented_images(image, rng, split_name)

    written: List[WrittenSample] = []

    for variant, variant_image in augmented:
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

        save_image(variant_image, image_path)
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


def save_image(image: Image.Image, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)

    if image.mode != "RGB":
        image = image.convert("RGB")

    suffix = image_path.suffix.lower()

    if suffix in (".jpg", ".jpeg"):
        image.save(
            image_path,
            format="JPEG",
            quality=config.JPEG_QUALITY,
        )
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

        lines.append(f"{class_id} " f"{xc:.6f} " f"{yc:.6f} " f"{w:.6f} " f"{h:.6f}\n")

    label_path.write_text("".join(lines), encoding="utf-8")


def _valid_yolo_box(xc: float, yc: float, w: float, h: float) -> bool:
    if w <= 0 or h <= 0:
        return False

    if not (0 <= xc <= 1 and 0 <= yc <= 1):
        return False

    if not (0 < w <= 1 and 0 < h <= 1):
        return False

    return True
