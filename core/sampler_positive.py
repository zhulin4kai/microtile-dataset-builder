# -*- coding: utf-8 -*-
"""
正样本采样模块（确定性快速版本）。

每个 annotation 固定生成 POS_PATCHES_PER_ANNOTATION 个 patch：
1. bbox center 居中
2. 沿 bbox 长边做一个轻微偏移

不重试，不 CUDA。

box 模式：使用 evaluate_positive_tile_np() 做 bbox 可见性判断。
seg 模式：使用 evaluate_seg_positive_tile()  做 polygon 可见性判断。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

import config
from core.fast_geometry import (
    evaluate_positive_tile_np,
)
from core.geojson_parser import Annotation
from core.seg_geometry import evaluate_seg_positive_tile
from core.slide_io import SlideReader
from core.yolo_writer import write_yolo_sample


@dataclass
class PositiveSamplingStats:
    requested: int = 0
    saved: int = 0
    failed: int = 0
    source_not_visible_enough: int = 0
    ambiguous_edge_target: int = 0
    no_valid_label: int = 0
    other_failed: int = 0


def generate_positive_samples_for_slide(
    slide_reader: SlideReader,
    slide_stem: str,
    split_name: str,
    annotations: Sequence[Annotation],
    boxes_np: np.ndarray,
    rng: random.Random,
) -> PositiveSamplingStats:
    """
    为单张 WSI 生成正样本（确定性快速）。

    seg 模式：polygon-area visible ratio 决定标签。
    box 模式：bbox visible ratio 决定标签。
    """
    stats = PositiveSamplingStats()
    sample_index = 1
    is_seg = config.DATASET_TASK == "seg"
    ann_list = list(annotations)

    try:
        for source_index, ann in enumerate(annotations):
            bbox = ann.bbox
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1

            if bw >= bh:
                offsets = [(0, 0), (int(bw * config.POS_OFFSET_RATIO), 0)]
            else:
                offsets = [(0, 0), (0, int(bh * config.POS_OFFSET_RATIO))]

            for dx, dy in offsets:
                stats.requested += 1

                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5

                x0 = int(round(cx - config.TILE_SIZE * 0.5 + dx))
                y0 = int(round(cy - config.TILE_SIZE * 0.5 + dy))

                x0, y0 = slide_reader.clamp_origin(x0, y0)

                # ── tile evaluation ────────────────────────────────
                if is_seg:
                    eval_result = evaluate_seg_positive_tile(
                        annotations=ann_list,
                        source_index=source_index,
                        x0=x0,
                        y0=y0,
                        tile_size=config.TILE_SIZE,
                        source_min_visible_ratio=config.SOURCE_MIN_VISIBLE_RATIO,
                        label_min_visible_ratio=config.LABEL_MIN_VISIBLE_RATIO,
                        ignore_max_visible_ratio=config.IGNORE_MAX_VISIBLE_RATIO,
                    )
                else:
                    eval_result = evaluate_positive_tile_np(
                        boxes=boxes_np,
                        source_index=source_index,
                        x0=x0,
                        y0=y0,
                        tile_size=config.TILE_SIZE,
                        source_min_visible_ratio=config.SOURCE_MIN_VISIBLE_RATIO,
                        label_min_visible_ratio=config.LABEL_MIN_VISIBLE_RATIO,
                        ignore_max_visible_ratio=config.IGNORE_MAX_VISIBLE_RATIO,
                        min_clipped_box_size=config.MIN_CLIPPED_BOX_SIZE,
                    )

                if not eval_result.ok:
                    stats.failed += 1
                    if eval_result.reason == "source_not_visible_enough":
                        stats.source_not_visible_enough += 1
                    elif eval_result.reason == "ambiguous_edge_target":
                        stats.ambiguous_edge_target += 1
                    elif eval_result.reason == "no_valid_label":
                        stats.no_valid_label += 1
                    else:
                        stats.other_failed += 1
                    continue

                tile_result = slide_reader.read_tile(x0, y0)

                write_yolo_sample(
                    output_dir=config.OUTPUT_DIR,
                    split_name=split_name,
                    slide_stem=slide_stem,
                    sample_type="pos",
                    sample_index=sample_index,
                    x0=tile_result.x0,
                    y0=tile_result.y0,
                    image=tile_result.image,
                    yolo_boxes=eval_result.yolo_boxes if hasattr(eval_result, "yolo_boxes") else [],
                    class_id=config.CLASS_ID,
                    rng=rng,
                    yolo_segments=eval_result.yolo_segments if is_seg else None,
                )

                stats.saved += 1
                sample_index += 1

    finally:
        pass

    return stats
