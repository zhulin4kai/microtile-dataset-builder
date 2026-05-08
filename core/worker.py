# -*- coding: utf-8 -*-
"""
单张 WSI 的处理 worker。

调用关系：

build_dataset.py
  -> process_one_slide()
       -> parse_qupath_geojson()
       -> boxes_to_cuda()
       -> SlideReader()
       -> generate_positive_samples_for_slide()
       -> generate_negative_samples_for_slide()
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

import config
from core.discover import SlidePair
from core.geojson_parser import (
    annotations_to_bbox_list,
    count_annotations_by_class,
    parse_qupath_geojson,
)
from core.geometry import boxes_to_cuda, get_cuda_device
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

    每个 worker 独立：
    1. 创建随机数生成器；
    2. 打开 OpenSlide；
    3. 解析 GeoJSON；
    4. 建立 CUDA bbox tensor；
    5. 生成正样本；
    6. 按正样本数量生成 2 倍负样本。
    """
    device = get_cuda_device(
        device_name=config.CUDA_DEVICE,
        require_cuda=config.REQUIRE_CUDA,
    )

    rng = random.Random(worker_seed)

    annotations = parse_qupath_geojson(pair.geojson_path)
    class_counts = count_annotations_by_class(annotations)

    if not annotations:
        raise RuntimeError(f"No annotations found: {pair.geojson_path}")

    bbox_list = annotations_to_bbox_list(annotations)
    boxes_cuda = boxes_to_cuda(bbox_list, device=device)

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
            boxes_cuda=boxes_cuda,
            device=device,
            rng=rng,
        )

        target_negative_count = positive_stats.saved * config.NEG_POS_RATIO

        negative_stats = generate_negative_samples_for_slide(
            slide_reader=slide_reader,
            slide_stem=pair.stem,
            split_name=split_name,
            boxes_cuda=boxes_cuda,
            device=device,
            rng=rng,
            target_negative_count=target_negative_count,
        )

    _release_cuda_memory()

    return SlideProcessStats(
        slide_stem=pair.stem,
        split_name=split_name,
        annotation_count=len(annotations),
        class_counts=class_counts,
        positive=positive_stats,
        negative=negative_stats,
    )

def _release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
