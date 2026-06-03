"""Tile sampling and split helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np

import config
from core.annotation import (
    Annotation,
    AnnotationIndex,
    bbox_intersects_tile,
    bbox_to_yolo,
    bbox_visible_ratio,
    clip_bbox_to_tile,
)
from core.reporting import SlideStats
from core.slide_io import SlideReader

MIN_BOX_VISIBLE_RATIO = 0.30
MIN_TISSUE_RATIO = 0.10
NEG_RADIUS_STEP = 0.10
NEG_RADIUS_START = 0.15
NEG_LOG_INTERVAL = 5000


class SampleWriter(Protocol):
    def write_sample(
        self,
        sample: "TileSample",
        split_name: str,
        rng: random.Random,
    ) -> int:
        ...

    def variant_count(self) -> int:
        ...


@dataclass(frozen=True)
class TileSample:
    stem: str
    image: np.ndarray
    boxes: list[tuple[float, float, float, float]]


def annotation_candidates(
    annotations: list[Annotation],
    x0: int,
    y0: int,
    tile_size: int,
    annotation_index: AnnotationIndex | None = None,
) -> list[Annotation]:
    if annotation_index is None:
        return annotations
    return annotation_index.query_tile(x0, y0, tile_size)


def has_large_visible_annotation(
    anns: list[Annotation],
    x0: int,
    y0: int,
    tile_size: int,
    annotation_index: AnnotationIndex | None = None,
) -> bool:
    for a in annotation_candidates(anns, x0, y0, tile_size, annotation_index):
        if not bbox_intersects_tile(a.bbox, x0, y0, tile_size):
            continue

        clipped = clip_bbox_to_tile(a.bbox, x0, y0, tile_size)
        if clipped is None:
            continue

        visible_ratio = bbox_visible_ratio(a.bbox, clipped)
        if visible_ratio >= MIN_BOX_VISIBLE_RATIO:
            return True

    return False


def tissue_ratio(arr: np.ndarray) -> float:
    import cv2

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    tissue_mask = (saturation > 12) & (value < 245)
    return float(tissue_mask.mean())


def sample_center_outward_origin(
    rng: random.Random,
    slide_w: int,
    slide_h: int,
    tile_size: int,
    radius_ratio: float,
) -> tuple[int, int]:
    max_x = max(0, slide_w - tile_size)
    max_y = max(0, slide_h - tile_size)

    cx = slide_w * 0.5
    cy = slide_h * 0.5
    radius = min(slide_w, slide_h) * radius_ratio

    px = cx + rng.uniform(-radius, radius)
    py = cy + rng.uniform(-radius, radius)

    x0 = int(round(px - tile_size * 0.5))
    y0 = int(round(py - tile_size * 0.5))

    x0 = max(0, min(x0, max_x))
    y0 = max(0, min(y0, max_y))

    return x0, y0


def train_slide_count(total_slides: int) -> int:
    if total_slides <= 0:
        return 0
    if total_slides == 1:
        return 1
    train_ratio = config.SPLIT_RATIOS["train"]
    return max(1, min(total_slides - 1, round(total_slides * train_ratio)))


def write_positive_samples(
    reader: SlideReader,
    annotations: list[Annotation],
    annotation_index: AnnotationIndex | None,
    slide_stem: str,
    split_name: str | None,
    rng: random.Random,
    stats: SlideStats,
    writer: SampleWriter,
) -> int:
    pos_index = 0

    for ann in annotations:
        pos_index += 1
        sample = make_positive_sample(
            reader=reader,
            annotations=annotations,
            annotation_index=annotation_index,
            annotation=ann,
            slide_stem=slide_stem,
            pos_index=pos_index,
        )
        write_sample(sample, choose_split(split_name, rng), rng, stats, writer)

    return pos_index


def make_positive_sample(
    reader: SlideReader,
    annotations: list[Annotation],
    annotation_index: AnnotationIndex | None,
    annotation: Annotation,
    slide_stem: str,
    pos_index: int,
) -> TileSample:
    ts = config.TILE_SIZE
    x1, y1, x2, y2 = annotation.bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    x0, y0 = reader.clamp_origin(cx - ts // 2, cy - ts // 2, ts)
    image = np.array(reader.read_tile(x0, y0, ts), dtype=np.uint8)
    stem = f"{slide_stem}_pos_{pos_index:06d}_x{x0}_y{y0}"

    return TileSample(
        stem=stem,
        image=image,
        boxes=visible_yolo_boxes(annotations, x0, y0, ts, annotation_index),
    )


def visible_yolo_boxes(
    annotations: list[Annotation],
    x0: int,
    y0: int,
    tile_size: int,
    annotation_index: AnnotationIndex | None = None,
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []

    for ann in annotation_candidates(annotations, x0, y0, tile_size, annotation_index):
        if not bbox_intersects_tile(ann.bbox, x0, y0, tile_size):
            continue

        clipped = clip_bbox_to_tile(ann.bbox, x0, y0, tile_size)
        if clipped is None:
            continue

        if bbox_visible_ratio(ann.bbox, clipped) < MIN_BOX_VISIBLE_RATIO:
            continue

        box = bbox_to_yolo(clipped, x0, y0, tile_size)
        if box[2] > 0 and box[3] > 0:
            boxes.append(box)

    return boxes


def write_negative_samples(
    reader: SlideReader,
    annotations: list[Annotation],
    annotation_index: AnnotationIndex | None,
    slide_stem: str,
    slide_w: int,
    slide_h: int,
    target_count: int,
    split_name: str | None,
    rng: random.Random,
    stats: SlideStats,
    writer: SampleWriter,
) -> int:
    neg_index = 0
    radius_ratio = NEG_RADIUS_START
    tries = 0
    seen: set[tuple[int, int]] = set()
    max_tries = target_count * config.MAX_NEG_TRIES_PER_POSITIVE

    while neg_index < target_count and tries < max_tries:
        tries += 1

        if tries % NEG_LOG_INTERVAL == 0:
            radius_ratio = min(1.5, radius_ratio + NEG_RADIUS_STEP)
            print(
                f"  [{slide_stem}] 负样本采样: "
                f"{neg_index}/{target_count}, 尝试={tries}, "
                f"radius_ratio={radius_ratio:.2f}"
            )

        sample, reject_reason = try_make_negative_sample(
            reader=reader,
            annotations=annotations,
            annotation_index=annotation_index,
            slide_stem=slide_stem,
            slide_w=slide_w,
            slide_h=slide_h,
            neg_index=neg_index + 1,
            radius_ratio=radius_ratio,
            rng=rng,
            seen=seen,
        )
        if sample is None:
            if reject_reason == "duplicate":
                stats.neg_reject_duplicate += 1
            elif reject_reason == "annotation":
                stats.neg_reject_annotation += 1
            elif reject_reason == "low_tissue":
                stats.neg_reject_low_tissue += 1
            continue

        neg_index += 1
        write_sample(sample, choose_split(split_name, rng), rng, stats, writer)

    if tries >= max_tries and neg_index < target_count:
        gap = target_count - neg_index
        stats.failed_neg += gap
        stats.neg_reject_try_limit += gap

    return neg_index


def try_make_negative_sample(
    reader: SlideReader,
    annotations: list[Annotation],
    slide_stem: str,
    slide_w: int,
    slide_h: int,
    neg_index: int,
    radius_ratio: float,
    rng: random.Random,
    seen: set[tuple[int, int]],
    annotation_index: AnnotationIndex | None = None,
) -> tuple[TileSample | None, str | None]:
    ts = config.TILE_SIZE
    x0, y0 = sample_center_outward_origin(
        rng=rng,
        slide_w=slide_w,
        slide_h=slide_h,
        tile_size=ts,
        radius_ratio=radius_ratio,
    )

    key = (x0, y0)
    if key in seen:
        return None, "duplicate"
    seen.add(key)

    if has_large_visible_annotation(annotations, x0, y0, ts, annotation_index):
        return None, "annotation"

    image = np.array(reader.read_tile(x0, y0, ts), dtype=np.uint8)
    if tissue_ratio(image) < MIN_TISSUE_RATIO:
        return None, "low_tissue"

    return TileSample(
        stem=f"{slide_stem}_neg_{neg_index:06d}_x{x0}_y{y0}",
        image=image,
        boxes=[],
    ), None


def choose_split(split_name: str | None, rng: random.Random) -> str:
    if split_name is not None:
        return split_name
    if rng.random() < config.SPLIT_RATIOS["train"]:
        return "train"
    return "val"


def write_sample(
    sample: TileSample,
    split_name: str,
    rng: random.Random,
    stats: SlideStats,
    writer: SampleWriter,
) -> None:
    written_count = writer.write_sample(sample, split_name, rng)
    if split_name == "train":
        stats.written_train += written_count
    else:
        stats.written_val += written_count


def split_for_slide(slide_index: int, n_train: int) -> str | None:
    if config.DATASET_SPLIT_MODE == "wsi":
        return "train" if slide_index < n_train else "val"
    return None

