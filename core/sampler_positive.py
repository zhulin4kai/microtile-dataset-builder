# -*- coding: utf-8 -*-
"""
正样本采样模块（确定性快速版本）。

每个 annotation 固定生成 POS_PATCHES_PER_ANNOTATION 个 patch：
1. bbox center 居中
2. 沿 bbox 长边做一个轻微偏移

不重试，不 CUDA。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

import config
from core.fast_geometry import (
    PositiveTileEvalResult,
    evaluate_positive_tile_np,
)
from core.geojson_parser import Annotation
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
    """
    stats = PositiveSamplingStats()
    sample_index = 1

    try:
        for source_index, ann in enumerate(annotations):
            bbox = ann.bbox
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1

            # 决定偏移方向
            if bw >= bh:
                offsets = [(0, 0), (int(bw * config.POS_OFFSET_RATIO), 0)]
            else:
                offsets = [(0, 0), (0, int(bh * config.POS_OFFSET_RATIO))]

            for dx, dy in offsets:
                stats.requested += 1

                # 以 bbox center 为中心计算 tile origin
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5

                x0 = int(round(cx - config.TILE_SIZE * 0.5 + dx))
                y0 = int(round(cy - config.TILE_SIZE * 0.5 + dy))

                x0, y0 = slide_reader.clamp_origin(x0, y0)

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

                if config.DATASET_TASK == "seg":
                    from core.seg_geometry import build_yolo_segments_for_tile

                    segments = build_yolo_segments_for_tile(
                        annotations=list(annotations),
                        label_indices=eval_result.label_indices,
                        x0=x0,
                        y0=y0,
                        tile_size=config.TILE_SIZE,
                    )
                    if not segments:
                        stats.failed += 1
                        stats.no_valid_label += 1
                        continue

                    write_yolo_sample(
                        output_dir=config.OUTPUT_DIR,
                        split_name=split_name,
                        slide_stem=slide_stem,
                        sample_type="pos",
                        sample_index=sample_index,
                        x0=tile_result.x0,
                        y0=tile_result.y0,
                        image=tile_result.image,
                        yolo_boxes=eval_result.yolo_boxes,
                        class_id=config.CLASS_ID,
                        rng=rng,
                        yolo_segments=segments,
                    )
                else:
                    write_yolo_sample(
                        output_dir=config.OUTPUT_DIR,
                        split_name=split_name,
                        slide_stem=slide_stem,
                        sample_type="pos",
                        sample_index=sample_index,
                        x0=tile_result.x0,
                        y0=tile_result.y0,
                        image=tile_result.image,
                        yolo_boxes=eval_result.yolo_boxes,
                        class_id=config.CLASS_ID,
                        rng=rng,
                    )

                stats.saved += 1
                sample_index += 1

    finally:
        pass

    return stats
