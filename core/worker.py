# -*- coding: utf-8 -*-
"""
单张 WSI 的处理 worker。

去 torch / CUDA 依赖，使用纯 NumPy 几何计算和低分辨率 tissue mask。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import config
from core.discover import SlidePair
from core.fast_geometry import boxes_to_numpy
from core.geojson_parser import (
    annotations_to_bbox_list,
    count_annotations_by_class,
    parse_qupath_geojson,
)
from core.sampler_negative import (
    NegativeSamplingStats,
    generate_negative_samples_for_slide,
)
from core.sampler_positive import (
    PositiveSamplingStats,
    generate_positive_samples_for_slide,
)
from core.slide_io import SlideReader


@dataclass
class SlideProcessStats:
    slide_stem: str
    split_name: str
    annotation_count: int
    class_counts: dict[str, int]
    positive: PositiveSamplingStats
    negative: NegativeSamplingStats


def process_one_slide(
    pair: SlidePair,
    split_name: str,
    worker_seed: int,
) -> SlideProcessStats:
    """
    处理单张 WSI。

    每个进程独立：
    1. 创建随机数生成器；
    2. 打开 OpenSlide；
    3. 解析 GeoJSON；
    4. 建立 NumPy bbox 数组；
    5. 生成正样本；
    6. 按 config.NEG_POS_RATIO 生成负样本。
    """
    rng = random.Random(worker_seed)

    annotations = parse_qupath_geojson(pair.geojson_path)
    class_counts = count_annotations_by_class(annotations)

    if not annotations:
        raise RuntimeError(f"No annotations found: {pair.geojson_path}")

    bbox_list = annotations_to_bbox_list(annotations)
    boxes_np = boxes_to_numpy(bbox_list)

    with SlideReader(
        svs_path=pair.svs_path,
        level=config.LEVEL,
        tile_size=config.TILE_SIZE,
    ) as slide_reader:
        positive_stats = generate_positive_samples_for_slide(
            slide_reader=slide_reader,
            slide_stem=pair.stem,
            split_name=split_name,
            annotations=annotations,
            boxes_np=boxes_np,
            rng=rng,
        )

        target_negative_count = positive_stats.saved * config.NEG_POS_RATIO

        negative_stats = generate_negative_samples_for_slide(
            slide_reader=slide_reader,
            slide_stem=pair.stem,
            split_name=split_name,
            boxes_np=boxes_np,
            rng=rng,
            target_negative_count=target_negative_count,
        )

    return SlideProcessStats(
        slide_stem=pair.stem,
        split_name=split_name,
        annotation_count=len(annotations),
        class_counts=class_counts,
        positive=positive_stats,
        negative=negative_stats,
    )
