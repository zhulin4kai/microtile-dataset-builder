#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO detect dataset builder (multi-process).

This module owns dataset discovery, WSI processing orchestration, and
tile-level positive/negative sample generation. Keep the file count small, but
make each function do one job so the builder remains easy to change.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import shutil
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from annotation import (
    Annotation,
    bbox_intersects_tile,
    bbox_visible_ratio,
    bbox_to_yolo,
    clip_bbox_to_tile,
    load_annotations,
)
from augment_writer import save_box_label, save_image_variants
from slide_io import SlideReader

WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
GT_EXTENSIONS = {".geojson", ".json"}

MIN_BOX_VISIBLE_RATIO = 0.30
MIN_TISSUE_RATIO = 0.10
NEG_RADIUS_STEP = 0.10
NEG_RADIUS_START = 0.15
NEG_LOG_INTERVAL = 5000


@dataclass(frozen=True)
class SlidePair:
    wsi_path: Path
    gt_path: Path


@dataclass
class SlideStats:
    slide_stem: str
    raw_pos: int = 0
    raw_neg: int = 0
    written_train: int = 0
    written_val: int = 0
    failed_neg: int = 0

    def as_dict(self) -> dict:
        return {
            "slide_stem": self.slide_stem,
            "raw_pos": self.raw_pos,
            "raw_neg": self.raw_neg,
            "written_train": self.written_train,
            "written_val": self.written_val,
            "failed_neg": self.failed_neg,
        }


@dataclass
class DatasetTotals:
    raw_pos: int = 0
    raw_neg: int = 0
    written_train: int = 0
    written_val: int = 0
    failed_neg: int = 0

    def add(self, result: dict) -> None:
        self.raw_pos += result["raw_pos"]
        self.raw_neg += result["raw_neg"]
        self.written_train += result["written_train"]
        self.written_val += result["written_val"]
        self.failed_neg += result["failed_neg"]


@dataclass(frozen=True)
class TileSample:
    stem: str
    image: np.ndarray
    boxes: list[tuple[float, float, float, float]]


def _has_large_visible_annotation(anns, x0: int, y0: int, tile_size: int) -> bool:
    for a in anns:
        if not bbox_intersects_tile(a.bbox, x0, y0, tile_size):
            continue

        clipped = clip_bbox_to_tile(a.bbox, x0, y0, tile_size)
        if clipped is None:
            continue

        visible_ratio = bbox_visible_ratio(a.bbox, clipped)
        if visible_ratio >= MIN_BOX_VISIBLE_RATIO:
            return True

    return False


def _tissue_ratio(arr: np.ndarray) -> float:
    import cv2

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    tissue_mask = (saturation > 12) & (value < 245)
    return float(tissue_mask.mean())


def _sample_center_outward_origin(
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


def process_slide_pair(
    wsi_path: str,
    gt_path: str,
    slide_index: int,
    split_name: str | None,
) -> dict:
    """Worker: process one WSI, write images and labels immediately."""
    slide_stem = Path(wsi_path).stem
    stats = SlideStats(slide_stem=slide_stem)
    rng = random.Random(config.RANDOM_SEED + slide_index * 1000003)

    annotations = load_annotations(Path(gt_path))
    if not annotations:
        return stats.as_dict()

    reader = SlideReader(wsi_path)
    try:
        w, h = reader.dimensions
        stats.raw_pos = _write_positive_samples(
            reader=reader,
            annotations=annotations,
            slide_stem=slide_stem,
            split_name=split_name,
            rng=rng,
            stats=stats,
        )
        stats.raw_neg = _write_negative_samples(
            reader=reader,
            annotations=annotations,
            slide_stem=slide_stem,
            slide_w=w,
            slide_h=h,
            target_count=stats.raw_pos,
            split_name=split_name,
            rng=rng,
            stats=stats,
        )

    finally:
        reader.close()

    print(
        f"[done] {slide_stem} raw_pos={stats.raw_pos} raw_neg={stats.raw_neg} "
        f"train={stats.written_train} val={stats.written_val} "
        f"failed_neg={stats.failed_neg}"
    )

    return stats.as_dict()


def _write_positive_samples(
    reader: SlideReader,
    annotations: list[Annotation],
    slide_stem: str,
    split_name: str | None,
    rng: random.Random,
    stats: SlideStats,
) -> int:
    pos_index = 0

    for ann in annotations:
        pos_index += 1
        sample = _make_positive_sample(
            reader=reader,
            annotations=annotations,
            annotation=ann,
            slide_stem=slide_stem,
            pos_index=pos_index,
        )
        _write_sample(sample, _choose_split(split_name, rng), rng, stats)

    return pos_index


def _make_positive_sample(
    reader: SlideReader,
    annotations: list[Annotation],
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
        boxes=_visible_yolo_boxes(annotations, x0, y0, ts),
    )


def _visible_yolo_boxes(
    annotations: list[Annotation],
    x0: int,
    y0: int,
    tile_size: int,
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []

    for ann in annotations:
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


def _write_negative_samples(
    reader: SlideReader,
    annotations: list[Annotation],
    slide_stem: str,
    slide_w: int,
    slide_h: int,
    target_count: int,
    split_name: str | None,
    rng: random.Random,
    stats: SlideStats,
) -> int:
    neg_index = 0
    radius_ratio = NEG_RADIUS_START
    tries = 0
    seen: set[tuple[int, int]] = set()

    while neg_index < target_count:
        tries += 1

        if tries % NEG_LOG_INTERVAL == 0:
            radius_ratio = min(1.5, radius_ratio + NEG_RADIUS_STEP)
            print(
                f"  [{slide_stem}] neg sampling: "
                f"{neg_index}/{target_count}, tries={tries}, "
                f"radius_ratio={radius_ratio:.2f}"
            )

        sample = _try_make_negative_sample(
            reader=reader,
            annotations=annotations,
            slide_stem=slide_stem,
            slide_w=slide_w,
            slide_h=slide_h,
            neg_index=neg_index + 1,
            radius_ratio=radius_ratio,
            rng=rng,
            seen=seen,
        )
        if sample is None:
            continue

        neg_index += 1
        _write_sample(sample, _choose_split(split_name, rng), rng, stats)

    return neg_index


def _try_make_negative_sample(
    reader: SlideReader,
    annotations: list[Annotation],
    slide_stem: str,
    slide_w: int,
    slide_h: int,
    neg_index: int,
    radius_ratio: float,
    rng: random.Random,
    seen: set[tuple[int, int]],
) -> TileSample | None:
    ts = config.TILE_SIZE
    x0, y0 = _sample_center_outward_origin(
        rng=rng,
        slide_w=slide_w,
        slide_h=slide_h,
        tile_size=ts,
        radius_ratio=radius_ratio,
    )

    key = (x0, y0)
    if key in seen:
        return None
    seen.add(key)

    if _has_large_visible_annotation(annotations, x0, y0, ts):
        return None

    image = np.array(reader.read_tile(x0, y0, ts), dtype=np.uint8)
    if _tissue_ratio(image) < MIN_TISSUE_RATIO:
        return None

    return TileSample(
        stem=f"{slide_stem}_neg_{neg_index:06d}_x{x0}_y{y0}",
        image=image,
        boxes=[],
    )


def _choose_split(split_name: str | None, rng: random.Random) -> str:
    if split_name is not None:
        return split_name
    if rng.random() < config.SPLIT_RATIOS["train"]:
        return "train"
    return "val"


def _write_sample(
    sample: TileSample,
    split_name: str,
    rng: random.Random,
    stats: SlideStats,
) -> None:
    variant_stems = save_image_variants(sample.image, sample.stem, split_name, rng)
    label_dir = config.OUTPUT_DIR / "labels" / split_name

    for variant_stem in variant_stems:
        save_box_label(label_dir / f"{variant_stem}.txt", sample.boxes)

    if split_name == "train":
        stats.written_train += len(variant_stems)
    else:
        stats.written_val += len(variant_stems)


def main() -> None:
    if config.DATASET_TASK != "detect":
        raise NotImplementedError(
            "Only detect mode is enabled in this simplified builder.")

    _prepare_output_dirs()
    pairs = _discover_slide_pairs()

    print(f"Found {len(pairs)} slide-annotation pairs")
    print(f"Using {config.NUM_WORKERS} workers")
    print(f"Split mode: {config.DATASET_SPLIT_MODE}")
    print(f"Task: {config.DATASET_TASK}\n")

    if not pairs:
        print("No slide-annotation pairs found. Exiting.")
        return

    totals = _run_slide_pairs(pairs)
    _write_dataset_yaml()
    _print_summary(totals)


def _prepare_output_dirs() -> None:
    if config.OUTPUT_DIR.exists():
        shutil.rmtree(config.OUTPUT_DIR)
    for split in ("train", "val"):
        (config.OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def _discover_slide_pairs() -> list[SlidePair]:
    pairs: list[SlidePair] = []

    for path in sorted(config.TARGET_DIR.iterdir()):
        if path.suffix.lower() in WSI_EXTENSIONS:
            wsi_path = path
            gt_path = _find_gt(wsi_path)
            if gt_path is not None:
                pairs.append(SlidePair(wsi_path=wsi_path, gt_path=gt_path))

    return pairs


def _run_slide_pairs(pairs: list[SlidePair]) -> DatasetTotals:
    rng = random.Random(config.RANDOM_SEED)
    if config.DATASET_SPLIT_MODE == "wsi":
        rng.shuffle(pairs)
        n_train = int(len(pairs) * config.SPLIT_RATIOS["train"])
    else:
        n_train = 0

    ctx = mp.get_context(config.PROCESS_START_METHOD)
    futures: dict[Future, int] = {}
    totals = DatasetTotals()

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS,
        mp_context=ctx,
    ) as executor:
        for i, pair in enumerate(pairs):
            fut = executor.submit(
                process_slide_pair,
                str(pair.wsi_path),
                str(pair.gt_path),
                i,
                _split_for_slide(i, n_train),
            )
            futures[fut] = i

        for fut in as_completed(futures):
            totals.add(fut.result())

    return totals


def _split_for_slide(slide_index: int, n_train: int) -> str | None:
    if config.DATASET_SPLIT_MODE == "wsi":
        return "train" if slide_index < n_train else "val"
    return None


def _print_summary(totals: DatasetTotals) -> None:
    print(f"\nRaw positive: {totals.raw_pos}")
    print(f"Raw negative: {totals.raw_neg}")
    print(f"Written train images: {totals.written_train}")
    print(f"Written val images: {totals.written_val}")
    print(f"Written total images: {totals.written_train + totals.written_val}")
    print(f"Output: {config.OUTPUT_DIR}")
    if totals.failed_neg > 0:
        print(f"Failed negatives (not enough empty space): {totals.failed_neg}")


def _find_gt(wsi_path: Path) -> Path | None:
    stem = wsi_path.stem
    for ext in GT_EXTENSIONS:
        candidate = wsi_path.with_name(stem + ext)
        if candidate.is_file():
            return candidate
        for subext in (".ome",):
            if stem.endswith(subext):
                candidate = wsi_path.with_name(stem[:-len(subext)] + ext)
                if candidate.is_file():
                    return candidate
    return None


def _write_dataset_yaml():
    path = config.OUTPUT_DIR / "dataset.yaml"
    content = (
        f"path: {config.OUTPUT_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: micropapillary\n"
    )
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
