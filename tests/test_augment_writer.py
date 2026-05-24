from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
from PIL import Image

import config
from augment_writer import (
    _save_jpeg,
    make_color_augmented_images,
    save_box_label,
    save_image_variants,
)


@pytest.fixture(autouse=True)
def restore_config():
    old_values = {
        "ENABLE_COLOR_AUGMENT": config.ENABLE_COLOR_AUGMENT,
        "OUTPUT_DIR": config.OUTPUT_DIR,
        "IMAGE_EXT": config.IMAGE_EXT,
        "CLASS_ID": config.CLASS_ID,
        "JPEG_QUALITY": config.JPEG_QUALITY,
    }
    yield
    for name, value in old_values.items():
        setattr(config, name, value)


def _rgb_image() -> Image.Image:
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:, :, 0] = 120
    arr[:, :, 1] = np.arange(16, dtype=np.uint8)
    arr[:, :, 2] = 200
    return Image.fromarray(arr)


def test_make_color_augmented_images_can_return_only_original():
    config.ENABLE_COLOR_AUGMENT = False

    variants = make_color_augmented_images(_rgb_image(), random.Random(1))

    assert [name for name, _ in variants] == ["orig"]
    assert variants[0][1].shape == (16, 16, 3)


def test_make_color_augmented_images_returns_expected_variants():
    config.ENABLE_COLOR_AUGMENT = True

    variants = make_color_augmented_images(_rgb_image(), random.Random(1))

    assert [name for name, _ in variants] == [
        "orig",
        "clahe",
        "hsv",
        "brightness_contrast",
        "gamma",
    ]
    assert all(arr.dtype == np.uint8 for _, arr in variants)
    assert all(arr.shape == (16, 16, 3) for _, arr in variants)


def test_save_image_variants_and_box_labels(tmp_path):
    config.ENABLE_COLOR_AUGMENT = False
    config.OUTPUT_DIR = tmp_path / "dataset"

    arr = np.array(_rgb_image(), dtype=np.uint8)
    stems = save_image_variants(arr, "tile", "train", random.Random(1))
    label_path = config.OUTPUT_DIR / "labels" / "train" / "tile_orig.txt"
    save_box_label(label_path, [(0.5, 0.5, 0.25, 0.25), (0.1, 0.1, 0.0, 0.2)])

    assert stems == ["tile_orig"]
    assert (config.OUTPUT_DIR / "images" / "train" / "tile_orig.jpg").is_file()
    assert label_path.read_text(encoding="utf-8") == "0 0.500000 0.500000 0.250000 0.250000\n"


def test_save_jpeg_raises_when_encoder_fails(monkeypatch, tmp_path):
    def fail_imencode(*args, **kwargs):
        return False, None

    monkeypatch.setattr(cv2, "imencode", fail_imencode)

    with pytest.raises(RuntimeError, match="cv2.imencode failed"):
        _save_jpeg(np.zeros((8, 8, 3), dtype=np.uint8), tmp_path / "bad.jpg")
