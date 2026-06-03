"""Runtime config snapshot helpers for spawned workers."""

from __future__ import annotations

from pathlib import Path

import config


def runtime_config_snapshot() -> dict:
    return {
        "OUTPUT_DIR": str(config.OUTPUT_DIR),
        "TILE_SIZE": config.TILE_SIZE,
        "SPLIT_RATIOS": dict(config.SPLIT_RATIOS),
        "ENABLE_COLOR_AUGMENT": config.ENABLE_COLOR_AUGMENT,
        "RANDOM_SEED": config.RANDOM_SEED,
        "CLASS_ID": config.CLASS_ID,
        "CLASS_NAME": config.CLASS_NAME,
        "IMAGE_EXT": config.IMAGE_EXT,
        "JPEG_QUALITY": config.JPEG_QUALITY,
        "WRITE_EMPTY_LABEL_FOR_NEGATIVE": config.WRITE_EMPTY_LABEL_FOR_NEGATIVE,
        "SLIDE_BACKEND": config.SLIDE_BACKEND,
        "CUCIM_DEVICE": config.CUCIM_DEVICE,
        "DRY_RUN": config.DRY_RUN,
        "MAX_NEG_TRIES_PER_POSITIVE": config.MAX_NEG_TRIES_PER_POSITIVE,
        "DISCOVER_RECURSIVE": config.DISCOVER_RECURSIVE,
        "CLAHE_CLIP_LIMIT_RANGE": config.CLAHE_CLIP_LIMIT_RANGE,
        "CLAHE_TILE_GRID_SIZE": config.CLAHE_TILE_GRID_SIZE,
        "HSV_HUE_SHIFT_LIMIT": config.HSV_HUE_SHIFT_LIMIT,
        "HSV_SAT_SHIFT_LIMIT": config.HSV_SAT_SHIFT_LIMIT,
        "HSV_VAL_SHIFT_LIMIT": config.HSV_VAL_SHIFT_LIMIT,
        "BRIGHTNESS_LIMIT": config.BRIGHTNESS_LIMIT,
        "CONTRAST_LIMIT": config.CONTRAST_LIMIT,
        "GAMMA_LIMIT": config.GAMMA_LIMIT,
    }


def apply_runtime_config(snapshot: dict) -> None:
    for name, value in snapshot.items():
        if name == "OUTPUT_DIR":
            setattr(config, name, Path(value))
        else:
            setattr(config, name, value)

