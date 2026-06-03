from __future__ import annotations

import random

import numpy as np

import config
from formats import YoloDetectFormat, get_dataset_format
from core.sampling import TileSample


def test_yolo_detect_format_writes_sample_and_metadata(tmp_path):
    old_values = {
        "OUTPUT_DIR": config.OUTPUT_DIR,
        "ENABLE_COLOR_AUGMENT": config.ENABLE_COLOR_AUGMENT,
        "CLASS_ID": config.CLASS_ID,
        "CLASS_NAME": config.CLASS_NAME,
        "DRY_RUN": config.DRY_RUN,
    }
    try:
        config.OUTPUT_DIR = tmp_path / "out"
        config.ENABLE_COLOR_AUGMENT = False
        config.CLASS_ID = 2
        config.CLASS_NAME = "lesion"
        config.DRY_RUN = False
        dataset_format = YoloDetectFormat()
        sample = TileSample(
            stem="tile",
            image=np.full((16, 16, 3), (180, 40, 120), dtype=np.uint8),
            boxes=[(0.5, 0.5, 0.25, 0.25)],
        )

        dataset_format.prepare_output_dirs(config.OUTPUT_DIR, config.DRY_RUN)
        written = dataset_format.write_sample(sample, "train", random.Random(1))
        dataset_format.write_metadata(config.OUTPUT_DIR, config.DRY_RUN)

        assert written == 1
        assert (config.OUTPUT_DIR / "images" / "train" / "tile_orig.jpg").is_file()
        assert (config.OUTPUT_DIR / "labels" / "train" / "tile_orig.txt").read_text(
            encoding="utf-8"
        ).startswith("2 0.500000")
        assert "  2: lesion" in (config.OUTPUT_DIR / "dataset.yaml").read_text(
            encoding="utf-8"
        )
    finally:
        for name, value in old_values.items():
            setattr(config, name, value)


def test_get_dataset_format_returns_yolo_detect_adapter():
    old_task = config.DATASET_TASK
    try:
        config.DATASET_TASK = "detect"

        dataset_format = get_dataset_format()

        assert isinstance(dataset_format, YoloDetectFormat)
        assert dataset_format.name == "yolo_detect"
        assert dataset_format.variant_count() >= 1
    finally:
        config.DATASET_TASK = old_task

