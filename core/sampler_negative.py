# -*- coding: utf-8 -*-
"""
负样本采样模块。

调用关系：

worker.py
  -> generate_negative_samples_for_slide()
       -> geometry_cuda.filter_candidate_tiles_without_boxes_cuda()
       -> SlideReader.read_tile()
       -> tissue_cuda.tissue_ratio_cuda()
       -> yolo_writer.write_yolo_sample()

负样本规则：
1. 中心优先随机采样；
2. 候选 tile 不得与 expanded annotation bbox 相交；
3. tissue_ratio 必须达到阈值；
4. 保存空 label。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Set, Tuple
import torch

import config
from core.geometry import filter_candidate_tiles_without_boxes_cuda
from core.slide_io import SlideReader
from core.tissue import tissue_ratio_cuda
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
    boxes_cuda: torch.Tensor,
    device: torch.device,
    rng: random.Random,
    target_negative_count: int,
) -> NegativeSamplingStats:
    """
    为单张 WSI 生成负样本。

    负样本数量由 worker 根据正样本保存数量计算：
        target_negative_count = pos_saved * NEG_POS_RATIO
    """
    stats = NegativeSamplingStats(requested=target_negative_count)

    if target_negative_count <= 0:
        return stats

    slide_w, slide_h = slide_reader.dimensions
    tile_size = config.TILE_SIZE

    if slide_w < tile_size or slide_h < tile_size:
        stats.failed = target_negative_count
        return stats

    used_origins: Set[Tuple[int, int]] = set()

    radius_ratio = config.NEG_INITIAL_RADIUS_RATIO
    max_radius_ratio = 1.5

    sample_index = 1
    tries = 0

    try:
        while (
            stats.saved < target_negative_count
            and tries < config.NEG_MAX_TRIES_PER_SLIDE
        ):
            batch_size = _dynamic_batch_size(
                remaining=target_negative_count - stats.saved
            )

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

            candidates = _remove_duplicate_candidates(candidates, used_origins, stats)

            if not candidates:
                radius_ratio = _grow_radius(radius_ratio, max_radius_ratio)
                continue

            candidates_tensor = torch.tensor(
                candidates,
                device=device,
                dtype=torch.float32,
            )

            valid_mask = filter_candidate_tiles_without_boxes_cuda(
                boxes=boxes_cuda,
                candidates_xy=candidates_tensor,
                tile_size=tile_size,
                margin=config.NEG_SAFE_MARGIN,
            )

            valid_candidates = []
            mask_cpu = valid_mask.detach().cpu().tolist()

            for candidate, ok in zip(candidates, mask_cpu):
                if ok:
                    valid_candidates.append(candidate)
                else:
                    stats.rejected_by_box += 1

            if not valid_candidates:
                radius_ratio = _grow_radius(radius_ratio, max_radius_ratio)
                continue

            for x0, y0 in valid_candidates:
                if stats.saved >= target_negative_count:
                    break

                used_origins.add((x0, y0))

                tile_result = slide_reader.read_tile(x0, y0)

                ratio = tissue_ratio_cuda(
                    image=tile_result.image,
                    device=device,
                )

                if ratio < config.TISSUE_RATIO_THRESHOLD:
                    stats.rejected_by_tissue += 1
                    continue

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
                )

                stats.saved += 1
                sample_index += 1

            if stats.saved < target_negative_count:
                radius_ratio = _grow_radius(radius_ratio, max_radius_ratio)

    finally:
        pass

    stats.failed = max(0, target_negative_count - stats.saved)
    stats.final_radius_ratio = radius_ratio

    return stats


def _dynamic_batch_size(remaining: int) -> int:
    """
    控制 CUDA 候选矩阵规模。

    候选数 M 与 bbox 数 N 会形成 [M, N] 的相交矩阵。
    这里不把 batch 设太大，避免 6G 显存上出现没必要的压力。
    """
    if remaining <= 32:
        return 128
    if remaining <= 128:
        return 256
    return 512


def _generate_center_prior_candidates(
    rng: random.Random,
    slide_w: int,
    slide_h: int,
    tile_size: int,
    radius_ratio: float,
    batch_size: int,
) -> list[tuple[int, int]]:
    """
    中心优先随机生成候选 tile 左上角。

    radius_ratio 越大，采样范围越接近全图。
    当 radius 足够大时，候选会自然覆盖到边缘。
    """
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


def _remove_duplicate_candidates(
    candidates: list[tuple[int, int]],
    used_origins: Set[Tuple[int, int]],
    stats: NegativeSamplingStats,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []

    local_seen: Set[Tuple[int, int]] = set()

    for xy in candidates:
        if xy in used_origins or xy in local_seen:
            stats.duplicate_candidate += 1
            continue

        local_seen.add(xy)
        result.append(xy)

    return result


def _grow_radius(radius_ratio: float, max_radius_ratio: float) -> float:
    return min(max_radius_ratio, radius_ratio + config.NEG_RADIUS_GROWTH_RATIO)
