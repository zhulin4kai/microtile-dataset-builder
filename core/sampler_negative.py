# -*- coding: utf-8 -*-
"""
负样本采样模块（低分辨率 tissue mask + NumPy bbox 过滤）。

不依赖 CUDA，不依赖 torch。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Set, Tuple

import numpy as np

import config
from core.fast_geometry import filter_candidate_tiles_without_boxes_np
from core.fast_tissue import TissueMask, build_tissue_mask, tile_tissue_ratio_from_mask
from core.slide_io import SlideReader
from core.yolo_writer import write_yolo_sample


@dataclass
class NegativeSamplingStats:
    requested: int = 0
    saved: int = 0
    failed: int = 0
    rejected_by_box: int = 0
    rejected_by_tissue: int = 0
    duplicate_candidate: int = 0
    total_candidates: int = 0
    final_radius_ratio: float = 0.0


def generate_negative_samples_for_slide(
    slide_reader: SlideReader,
    slide_stem: str,
    split_name: str,
    boxes_np: np.ndarray,
    rng: random.Random,
    target_negative_count: int,
) -> NegativeSamplingStats:
    """
    为单张 WSI 生成负样本。

    流程：
    1. 构建低分辨率 tissue mask；
    2. 批量生成候选坐标并用 NumPy 过滤 bbox；
    3. 用 mask 过滤白背景；
    4. 合格后才 read_tile；
    """
    stats = NegativeSamplingStats(requested=target_negative_count)

    if target_negative_count <= 0:
        return stats

    slide_w, slide_h = slide_reader.dimensions
    tile_size = config.TILE_SIZE

    if slide_w < tile_size or slide_h < tile_size:
        stats.failed = target_negative_count
        return stats

    tissue_mask = build_tissue_mask(slide_reader)

    used_origins: Set[Tuple[int, int]] = set()

    radius_ratio = 0.20
    max_radius_ratio = 1.5

    sample_index = 1
    tries = 0

    try:
        while (
            stats.saved < target_negative_count
            and tries < config.NEG_MAX_TRIES_PER_SLIDE
        ):
            batch_size = min(8192, (target_negative_count - stats.saved) * 3)

            candidates = _generate_center_prior_candidates(
                rng=rng,
                slide_w=slide_w,
                slide_h=slide_h,
                tile_size=tile_size,
                radius_ratio=radius_ratio,
                batch_size=batch_size,
            )

            tries += len(candidates)
            stats.total_candidates += len(candidates)

            # 去重
            unique = []
            for xy in candidates:
                if xy not in used_origins:
                    used_origins.add(xy)  # 乐观加入，后面可能有 false positive
                    unique.append(xy)
            candidates_np = np.array(unique, dtype=np.float32)

            if candidates_np.shape[0] == 0:
                radius_ratio = min(max_radius_ratio, radius_ratio + 0.15)
                continue

            # NumPy 过滤 bbox
            valid_mask = filter_candidate_tiles_without_boxes_np(
                boxes=boxes_np,
                candidates_xy=candidates_np,
                tile_size=tile_size,
                margin=config.NEG_SAFE_MARGIN,
            )

            valid = candidates_np[valid_mask]
            rejected_count = (~valid_mask).sum()
            stats.rejected_by_box += int(rejected_count)

            if valid.shape[0] == 0:
                radius_ratio = min(max_radius_ratio, radius_ratio + 0.15)
                continue

            # tissue mask 过滤 + read_tile
            for i in range(valid.shape[0]):
                if stats.saved >= target_negative_count:
                    break

                x0 = int(valid[i, 0])
                y0 = int(valid[i, 1])

                tissue_ratio = tile_tissue_ratio_from_mask(
                    tissue_mask, x0, y0, tile_size
                )

                if tissue_ratio < config.TISSUE_RATIO_THRESHOLD:
                    stats.rejected_by_tissue += 1
                    continue

                tile_result = slide_reader.read_tile(x0, y0)

                write_yolo_sample(
                    output_dir=config.OUTPUT_DIR,
                    split_name=split_name,
                    slide_stem=slide_stem,
                    sample_type="neg",
                    sample_index=sample_index,
                    x0=tile_result.x0,
                    y0=tile_result.y0,
                    image=tile_result.image,
                    yolo_boxes=[],
                    class_id=config.CLASS_ID,
                    rng=rng,
                    yolo_segments=[],
                )

                stats.saved += 1
                sample_index += 1

            if stats.saved < target_negative_count:
                radius_ratio = min(max_radius_ratio, radius_ratio + 0.15)

    finally:
        pass

    stats.failed = max(0, target_negative_count - stats.saved)
    stats.final_radius_ratio = radius_ratio

    return stats


def _generate_center_prior_candidates(
    rng: random.Random,
    slide_w: int,
    slide_h: int,
    tile_size: int,
    radius_ratio: float,
    batch_size: int,
) -> list[tuple[int, int]]:
    max_x = slide_w - tile_size
    max_y = slide_h - tile_size

    center_x = slide_w * 0.5
    center_y = slide_h * 0.5

    base_radius = min(slide_w, slide_h) * radius_ratio

    candidates: list[tuple[int, int]] = []

    for _ in range(batch_size):
        dx = rng.uniform(-base_radius, base_radius)
        dy = rng.uniform(-base_radius, base_radius)

        cx = center_x + dx
        cy = center_y + dy

        x0 = int(round(cx - tile_size * 0.5))
        y0 = int(round(cy - tile_size * 0.5))

        x0 = max(0, min(x0, max_x))
        y0 = max(0, min(y0, max_y))

        candidates.append((x0, y0))

    return candidates
