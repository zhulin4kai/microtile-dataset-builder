"""YOLO detect dataset output adapter."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import config
from core.augment import (
    get_variant_names,
    save_box_label,
    save_image_variants,
)
from core.sampling import TileSample


class YoloDetectFormat:
    name = "yolo_detect"

    def validate_config(self) -> None:
        if config.DATASET_TASK != "detect":
            raise NotImplementedError(
                "当前构建器只支持 YOLO detect 数据集。"
            )
        if not isinstance(config.CLASS_ID, int) or config.CLASS_ID < 0:
            raise ValueError(
                f"CLASS_ID 必须是非负整数，当前值: {config.CLASS_ID}"
            )
        if not config.CLASS_NAME or not isinstance(config.CLASS_NAME, str):
            raise ValueError("CLASS_NAME 必须是非空字符串。")
        ext = config.IMAGE_EXT.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(
                f"IMAGE_EXT 必须是 .jpg / .jpeg / .png，当前值: {config.IMAGE_EXT}"
            )
        if not 1 <= config.JPEG_QUALITY <= 100:
            raise ValueError(
                f"JPEG_QUALITY 必须在 1-100 之间，当前值: {config.JPEG_QUALITY}"
            )

    def prepare_output_dirs(self, output_dir: Path, dry_run: bool) -> None:
        if dry_run:
            return
        if output_dir.exists():
            shutil.rmtree(output_dir)
        for split in ("train", "val"):
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_sample(
        self,
        sample: TileSample,
        split_name: str,
        rng: random.Random,
    ) -> int:
        if config.DRY_RUN:
            return self.variant_count()

        variant_stems = save_image_variants(sample.image, sample.stem, split_name, rng)
        label_dir = config.OUTPUT_DIR / "labels" / split_name
        for variant_stem in variant_stems:
            save_box_label(label_dir / f"{variant_stem}.txt", sample.boxes)
        return len(variant_stems)

    def write_metadata(self, output_dir: Path, dry_run: bool) -> None:
        if dry_run:
            return
        path = output_dir / "dataset.yaml"
        content = (
            f"path: {output_dir.as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n"
            f"  {config.CLASS_ID}: {config.CLASS_NAME}\n"
        )
        path.write_text(content, encoding="utf-8")

    def variant_count(self) -> int:
        return len(get_variant_names())


def get_dataset_format() -> YoloDetectFormat:
    if config.DATASET_TASK == "detect":
        return YoloDetectFormat()
    raise NotImplementedError(
        "当前构建器只支持 YOLO detect 数据集。"
    )

