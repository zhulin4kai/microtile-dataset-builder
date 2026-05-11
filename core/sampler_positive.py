# -*- coding: utf-8 -*-
"""
正样本采样模块（coverage-driven）。

调用关系：

worker.py
  -> generate_positive_samples_for_slide()
       -> SlideReader.read_tile()
       -> geometry_cuda.evaluate_positive_tile_cuda()
       -> yolo_writer.write_yolo_sample()

coverage-driven 策略：
1. 维护每个 annotation 的 coverage 计数；
2. 优先从 coverage 最低的 annotation 采样；
3. 直到所有 annotation 达到 POS_TARGET_COVERAGE 或达到最大尝试次数。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
    covered_annotations: int = 0
    min_coverage: int = 0
    max_coverage: int = 0
    mean_coverage: float = 0.0
    coverage_failed: int = 0


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
    为单张 WSI 生成正样本（coverage-driven）。
    """
    stats = PositiveSamplingStats()

    if not annotations:
        return stats

    coverage = [0] * len(annotations)
    sample_index = 1

    max_total_tries = len(annotations) * config.POS_MAX_TOTAL_TRIES_FACTOR
    total_tries = 0

    try:
        while min(coverage) < config.POS_TARGET_COVERAGE:
            if total_tries >= max_total_tries:
                break

            total_tries += 1
            stats.requested += 1

            source_index = _choose_lowest_coverage_annotation(coverage, rng)
            source_bbox = annotations[source_index].bbox

            result = _try_make_coverage_positive_sample(
                slide_reader=slide_reader,
                slide_stem=slide_stem,
                split_name=split_name,
                source_index=source_index,
                source_bbox=source_bbox,
                boxes_cuda=boxes_cuda,
                device=device,
                rng=rng,
                sample_index=sample_index,
            )

            if result.ok:
                stats.saved += 1
                sample_index += 1

                for idx in result.label_indices:
                    coverage[idx] += 1
            else:
                stats.failed += 1

                if result.reason == "source_not_visible_enough":
                    stats.source_not_visible_enough += 1
                elif result.reason == "ambiguous_edge_target":
                    stats.ambiguous_edge_target += 1
                elif result.reason == "no_valid_label":
                    stats.no_valid_label += 1
                elif result.reason == "coverage_failed":
                    stats.coverage_failed += 1
                else:
                    stats.other_failed += 1

    finally:
        pass

    # 统计最终 coverage
    if coverage:
        stats.min_coverage = min(coverage)
        stats.max_coverage = max(coverage)
        stats.mean_coverage = sum(coverage) / len(coverage)
        stats.covered_annotations = sum(
            1 for c in coverage if c >= config.POS_TARGET_COVERAGE
        )

    return stats


def _choose_lowest_coverage_annotation(
    coverage: List[int],
    rng: random.Random,
) -> int:
    """选择 coverage 最低的一批 annotation 中的随机一个。"""
    min_cov = min(coverage)
    candidates = [i for i, c in enumerate(coverage) if c == min_cov]
    return rng.choice(candidates)


@dataclass(frozen=True)
class _PositiveTryResult:
    ok: bool
    reason: str
    label_indices: List[int]


def _try_make_coverage_positive_sample(
    slide_reader: SlideReader,
    slide_stem: str,
    split_name: str,
    source_index: int,
    source_bbox: Tuple[float, float, float, float],
    boxes_cuda: torch.Tensor,
    device: torch.device,
    rng: random.Random,
    sample_index: int,
) -> _PositiveTryResult:
    """
    尝试为指定 source annotation 生成一个合格正样本。

    采样策略：
    1. 优先使用 POS_SOURCE_MARGIN 完整包含 source bbox；
    2. 放宽 margin 到 0 作为 fallback；
    3. 仍失败则用 bbox center clamp；
    4. 全部失败记为 coverage_failed。
    """
    # Level 1: 正常 margin
    origin = _sample_origin_containing_bbox(
        bbox=source_bbox,
        slide_reader=slide_reader,
        rng=rng,
        margin=config.POS_SOURCE_MARGIN,
    )

    # Level 2: 放宽 margin 到 0
    if origin is None:
        origin = _sample_origin_containing_bbox(
            bbox=source_bbox,
            slide_reader=slide_reader,
            rng=rng,
            margin=0,
        )

    # Level 3: bbox center clamp（最终 fallback）
    if origin is None:
        origin = _tile_origin_from_bbox_center(
            bbox=source_bbox,
            slide_reader=slide_reader,
            dx=0,
            dy=0,
        )

        # 再尝试带少量 jitter
        if origin is not None:
            for _ in range(30):
                jx = rng.randint(-64, 64)
                jy = rng.randint(-64, 64)
                try_origin = _tile_origin_from_bbox_center(
                    bbox=source_bbox,
                    slide_reader=slide_reader,
                    dx=jx,
                    dy=jy,
                )
                if try_origin is not None:
                    break

    if origin is None:
        return _PositiveTryResult(
            ok=False, reason="coverage_failed", label_indices=[]
        )

    x0, y0 = origin

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
        return _PositiveTryResult(
            ok=False,
            reason=eval_result.reason,
            label_indices=[],
        )

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
        rng=rng,
    )

    return _PositiveTryResult(
        ok=True,
        reason="ok",
        label_indices=eval_result.label_indices,
    )


def _sample_origin_containing_bbox(
    bbox: Tuple[float, float, float, float],
    slide_reader: SlideReader,
    rng: random.Random,
    margin: int,
) -> Tuple[int, int] | None:
    """
    计算能完整包含 source bbox 的 tile 合法范围，从中随机采样。

    返回 None 表示对当前 tile_size/margin 无法完整容纳。
    """
    x1, y1, x2, y2 = bbox
    tile_size = config.TILE_SIZE
    slide_w, slide_h = slide_reader.dimensions

    # 让 bbox 尽量完整落在 tile 内
    # tile 左上角 (x0, y0)，tile 范围 [x0, x0+tile_size-1]
    # bbox 在 tile 内 → x0 <= x1 and x2 <= x0+tile_size
    # → x0 <= x1 and x0 >= x2 - tile_size
    min_x0 = max(0, int(math.ceil(x2 + margin - tile_size)))
    max_x0 = min(slide_w - tile_size, int(math.floor(x1 - margin)))

    min_y0 = max(0, int(math.ceil(y2 + margin - tile_size)))
    max_y0 = min(slide_h - tile_size, int(math.floor(y1 - margin)))

    if min_x0 > max_x0 or min_y0 > max_y0:
        return None

    x0 = rng.randint(min_x0, max_x0)
    y0 = rng.randint(min_y0, max_y0)

    return (x0, y0)


def _tile_origin_from_bbox_center(
    bbox: Tuple[float, float, float, float],
    slide_reader: SlideReader,
    dx: int,
    dy: int,
) -> Tuple[int, int] | None:
    """以 bbox 中心为 tile 中心的备选位置。"""
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    x0 = int(round(cx - config.TILE_SIZE * 0.5 + dx))
    y0 = int(round(cy - config.TILE_SIZE * 0.5 + dy))

    try:
        return slide_reader.clamp_origin(x0, y0)
    except ValueError:
        return None
