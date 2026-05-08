# -*- coding: utf-8 -*-
"""
正样本采样模块。

调用关系：

worker.py
  -> generate_positive_samples_for_slide()
       -> SlideReader.read_tile()
       -> geometry_cuda.evaluate_positive_tile_cuda()
       -> yolo_writer.write_yolo_sample()

每个 annotation 目标生成 3 张正样本：
1. 一张中心 patch；
2. 两张随机 jitter patch。

如果候选 patch 不合格，会重新采样。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, Tuple
import torch

import config
from core.geojson_parser import Annotation
from core.geometry import evaluate_positive_tile_cuda
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
    boxes_cuda: torch.Tensor,
    device: torch.device,
    rng: random.Random,
) -> PositiveSamplingStats:
    """
    为单张 WSI 生成正样本。
    """
    stats = PositiveSamplingStats()

    sample_index = 1

    try:
        for source_index, ann in enumerate(annotations):
            for patch_id in range(config.POS_PATCHES_PER_ANNOTATION):
                stats.requested += 1

                is_center_patch = patch_id == 0

                result = _try_make_one_positive_sample(
                    slide_reader=slide_reader,
                    slide_stem=slide_stem,
                    split_name=split_name,
                    source_index=source_index,
                    source_bbox=ann.bbox,
                    boxes_cuda=boxes_cuda,
                    device=device,
                    rng=rng,
                    sample_index=sample_index,
                    is_center_patch=is_center_patch,
                )

                if result.ok:
                    stats.saved += 1
                    sample_index += 1
                else:
                    stats.failed += 1

                    if result.reason == "source_not_visible_enough":
                        stats.source_not_visible_enough += 1
                    elif result.reason == "ambiguous_edge_target":
                        stats.ambiguous_edge_target += 1
                    elif result.reason == "no_valid_label":
                        stats.no_valid_label += 1
                    else:
                        stats.other_failed += 1

    finally:
        pass
    return stats


@dataclass(frozen=True)
class _PositiveTryResult:
    ok: bool
    reason: str


def _try_make_one_positive_sample(
    slide_reader: SlideReader,
    slide_stem: str,
    split_name: str,
    source_index: int,
    source_bbox: Tuple[float, float, float, float],
    boxes_cuda: torch.Tensor,
    device: torch.device,
    rng: random.Random,
    sample_index: int,
    is_center_patch: bool,
) -> _PositiveTryResult:
    """
    尝试生成一个合格正样本。

    中心 patch：
        先尝试 dx=0, dy=0。
        如果因为边缘目标污染失败，再允许小范围扰动补救。

    jitter patch：
        dx, dy 从 [-JITTER, JITTER] 随机采样。
    """
    if is_center_patch:
        offsets = [(0, 0)]
        offsets.extend(_random_offsets(rng, max_jitter=config.JITTER // 2, count=20))
    else:
        offsets = _random_offsets(rng, max_jitter=config.JITTER, count=40)
        offsets.extend(_random_offsets(rng, max_jitter=config.JITTER // 2, count=20))
        offsets.append((0, 0))

    last_reason = "not_attempted"

    for dx, dy in offsets:
        x0, y0 = _tile_origin_from_bbox_center(
            bbox=source_bbox,
            slide_reader=slide_reader,
            dx=dx,
            dy=dy,
        )

        eval_result = evaluate_positive_tile_cuda(
            boxes=boxes_cuda,
            source_index=source_index,
            x0=x0,
            y0=y0,
            tile_size=config.TILE_SIZE,
            source_min_visible_ratio=config.SOURCE_MIN_VISIBLE_RATIO,
            label_min_visible_ratio=config.LABEL_MIN_VISIBLE_RATIO,
            ignore_max_visible_ratio=config.IGNORE_MAX_VISIBLE_RATIO,
            min_clipped_box_size=config.MIN_CLIPPED_BOX_SIZE,
            device=device,
        )

        if not eval_result.ok:
            last_reason = eval_result.reason
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
            yolo_boxes=eval_result.yolo_boxes,
            class_id=config.CLASS_ID,
        )

        return _PositiveTryResult(ok=True, reason="ok")

    return _PositiveTryResult(ok=False, reason=last_reason)


def _tile_origin_from_bbox_center(
    bbox: Tuple[float, float, float, float],
    slide_reader: SlideReader,
    dx: int,
    dy: int,
) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    x0 = int(round(cx - config.TILE_SIZE * 0.5 + dx))
    y0 = int(round(cy - config.TILE_SIZE * 0.5 + dy))

    return slide_reader.clamp_origin(x0, y0)


def _random_offsets(
    rng: random.Random,
    max_jitter: int,
    count: int,
) -> list[tuple[int, int]]:
    if max_jitter <= 0:
        return [(0, 0)] * count

    return [
        (
            rng.randint(-max_jitter, max_jitter),
            rng.randint(-max_jitter, max_jitter),
        )
        for _ in range(count)
    ]
