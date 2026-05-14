#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO detect dataset builder (multi-process).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import config
from annotation import (
    bbox_intersects_tile, bbox_to_yolo,
    clip_bbox_to_tile, load_annotations,
)
from augment_writer import save_box_label, save_image_variants
from slide_io import SlideReader

WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
GT_EXTENSIONS = {".geojson", ".json"}


def process_slide_pair(
    wsi_path: str,
    gt_path: str,
    slide_index: int,
    split_name: str | None,
) -> dict:
    """Worker: process one WSI, write images and labels immediately."""
    slide_stem = os.path.splitext(os.path.basename(wsi_path))[0]
    rng = random.Random(config.RANDOM_SEED + slide_index * 1000003)
    ts = config.TILE_SIZE

    annotations = load_annotations(Path(gt_path))
    if not annotations:
        return {
            "slide_stem": slide_stem,
            "raw_pos": 0, "raw_neg": 0,
            "written_train": 0, "written_val": 0,
            "failed_neg": 0,
        }

    reader = SlideReader(wsi_path)
    try:
        w, h = reader.dimensions
        anns = annotations
        pos_index = 0
        neg_index = 0
        written_train = 0
        written_val = 0
        failed_neg = 0

        # ── positive samples ──────────────────────────────────────
        for ann in anns:
            pos_index += 1
            x1, y1, x2, y2 = ann.bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            x0 = cx - ts // 2
            y0 = cy - ts // 2
            x0, y0 = reader.clamp_origin(x0, y0, ts)

            img = reader.read_tile(x0, y0, ts)
            arr = np.array(img, dtype=np.uint8)

            boxes = []
            for a in anns:
                if bbox_intersects_tile(a.bbox, x0, y0, ts):
                    clipped = clip_bbox_to_tile(a.bbox, x0, y0, ts)
                    if clipped is not None:
                        yb = bbox_to_yolo(clipped, x0, y0, ts)
                        if yb[2] > 0 and yb[3] > 0:
                            boxes.append(yb)

            if split_name is not None:
                split = split_name
            else:
                split = "train" if rng.random() < config.SPLIT_RATIOS["train"] else "val"

            stem = f"{slide_stem}_pos_{pos_index:06d}_x{x0}_y{y0}"
            variant_stems = save_image_variants(arr, stem, split, rng)

            label_dir = config.OUTPUT_DIR / "labels" / split
            for vstem in variant_stems:
                label_path = label_dir / f"{vstem}.txt"
                save_box_label(label_path, boxes)

            if split == "train":
                written_train += len(variant_stems)
            else:
                written_val += len(variant_stems)

        raw_pos = pos_index

        # ── negative samples ──────────────────────────────────────
        n_neg = raw_pos
        max_tries = n_neg * 100
        tries = 0
        seen: set[tuple] = set()

        while neg_index < n_neg and tries < max_tries:
            tries += 1
            x0 = rng.randint(0, max(0, w - ts))
            y0 = rng.randint(0, max(0, h - ts))
            key = (x0, y0)
            if key in seen:
                continue
            seen.add(key)
            if any(bbox_intersects_tile(a.bbox, x0, y0, ts) for a in anns):
                continue

            neg_index += 1
            img = reader.read_tile(x0, y0, ts)
            arr = np.array(img, dtype=np.uint8)

            if split_name is not None:
                split = split_name
            else:
                split = "train" if rng.random() < config.SPLIT_RATIOS["train"] else "val"

            stem = f"{slide_stem}_neg_{neg_index:06d}_x{x0}_y{y0}"
            variant_stems = save_image_variants(arr, stem, split, rng)

            label_dir = config.OUTPUT_DIR / "labels" / split
            for vstem in variant_stems:
                label_path = label_dir / f"{vstem}.txt"
                save_box_label(label_path, [])

            if split == "train":
                written_train += len(variant_stems)
            else:
                written_val += len(variant_stems)

        failed_neg = n_neg - neg_index
        if failed_neg > 0:
            print(f"  [{slide_stem}] negative: got {neg_index} / {n_neg} after {tries} tries")

        raw_neg = neg_index

    finally:
        reader.close()

    print(f"[done] {slide_stem} raw_pos={raw_pos} raw_neg={raw_neg} "
          f"train={written_train} val={written_val} failed_neg={failed_neg}")

    return {
        "slide_stem": slide_stem,
        "raw_pos": raw_pos,
        "raw_neg": raw_neg,
        "written_train": written_train,
        "written_val": written_val,
        "failed_neg": failed_neg,
    }


def main() -> None:
    if config.DATASET_TASK != "detect":
        raise NotImplementedError(
            "Only detect mode is enabled in this simplified builder.")

    # 1. rebuild output dir
    if config.OUTPUT_DIR.exists():
        shutil.rmtree(config.OUTPUT_DIR)
    for split in ("train", "val"):
        (config.OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. discover slide + annotation pairs
    pairs: list[tuple[Path, Path]] = []
    for fname in sorted(os.listdir(config.TARGET_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in WSI_EXTENSIONS:
            wsi_path = config.TARGET_DIR / fname
            gt_path = _find_gt(wsi_path)
            if gt_path is not None:
                pairs.append((wsi_path, gt_path))

    print(f"Found {len(pairs)} slide-annotation pairs")
    print(f"Using {config.NUM_WORKERS} workers")
    print(f"Split mode: {config.DATASET_SPLIT_MODE}")
    print(f"Task: {config.DATASET_TASK}\n")

    if not pairs:
        print("No slide-annotation pairs found. Exiting.")
        return

    # 3. determine split per WSI (wsi mode only)
    rng = random.Random(config.RANDOM_SEED)
    if config.DATASET_SPLIT_MODE == "wsi":
        rng.shuffle(pairs)
        n_train = int(len(pairs) * config.SPLIT_RATIOS["train"])
    else:
        n_train = 0

    # 4. submit to process pool
    ctx = mp.get_context(config.PROCESS_START_METHOD)
    futures: dict = {}
    total_pos = 0
    total_neg = 0
    total_train = 0
    total_val = 0
    total_failed = 0

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS,
        mp_context=ctx,
    ) as executor:
        for i, (wsi_path, gt_path) in enumerate(pairs):
            if config.DATASET_SPLIT_MODE == "wsi":
                split_name = "train" if i < n_train else "val"
            else:
                split_name = None
            fut = executor.submit(
                process_slide_pair,
                str(wsi_path),
                str(gt_path),
                i,
                split_name,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            result = fut.result()
            total_pos += result["raw_pos"]
            total_neg += result["raw_neg"]
            total_train += result["written_train"]
            total_val += result["written_val"]
            total_failed += result["failed_neg"]

    # 5. write dataset.yaml + final summary
    _write_dataset_yaml()

    print(f"\nRaw positive: {total_pos}")
    print(f"Raw negative: {total_neg}")
    print(f"Written train images: {total_train}")
    print(f"Written val images: {total_val}")
    print(f"Written total images: {total_train + total_val}")
    print(f"Output: {config.OUTPUT_DIR}")
    if total_failed > 0:
        print(f"Failed negatives (not enough empty space): {total_failed}")


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
