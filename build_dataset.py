#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO detect dataset builder.

Entry point:  python build_dataset.py
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import config
from annotation import (
    Annotation, bbox_intersects_tile, bbox_to_yolo,
    clip_bbox_to_tile, load_annotations,
)
from augment_writer import save_box_label, save_image_variants
from slide_io import SlideReader

WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
GT_EXTENSIONS = {".geojson", ".json"}


def main() -> None:
    rng = random.Random(config.RANDOM_SEED)

    # 1. rebuild output dir
    if config.OUTPUT_DIR.exists():
        shutil.rmtree(config.OUTPUT_DIR)
    for split in ("train", "val"):
        (config.OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. discover slide + annotation pairs
    pairs: list[tuple[Path, Path]] = []
    for fname in sorted(os.listdir(config.TARGET_DIR)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in WSI_EXTENSIONS:
            wsi_path = config.TARGET_DIR / fname
            gt_path = _find_gt(wsi_path)
            if gt_path is not None:
                pairs.append((wsi_path, gt_path))
    print(f"Found {len(pairs)} slide-annotation pairs")

    # 3. generate raw positive patches + negative patches per slide
    pos_groups: list[dict] = []
    neg_groups: list[dict] = []
    total_pos = 0
    total_neg = 0

    for wsi_path, gt_path in pairs:
        slide_stem = os.path.splitext(os.path.basename(wsi_path))[0]
        annotations = load_annotations(gt_path)
        if not annotations:
            continue

        reader = SlideReader(str(wsi_path))
        try:
            w, h = reader.dimensions
            ts = config.TILE_SIZE

            # positive: 1 patch per annotation, centered on bbox
            slide_pos: list[dict] = []
            for ann in annotations:
                x1, y1, x2, y2 = ann.bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                x0 = cx - ts // 2
                y0 = cy - ts // 2
                x0, y0 = reader.clamp_origin(x0, y0, ts)

                img = reader.read_tile(x0, y0, ts)

                # collect labels: all bboxes intersecting this tile
                boxes: list[tuple] = []
                for a in annotations:
                    if bbox_intersects_tile(a.bbox, x0, y0, ts):
                        clipped = clip_bbox_to_tile(a.bbox, x0, y0, ts)
                        if clipped is not None:
                            yb = bbox_to_yolo(clipped, ts)
                            if yb[2] > 0 and yb[3] > 0:
                                boxes.append(yb)

                slide_pos.append({
                    "slide_stem": slide_stem,
                    "index": len(pos_groups) + len(slide_pos) + 1,
                    "x0": x0, "y0": y0,
                    "img": img,
                    "boxes": boxes,
                })

            # negative: random N patches not intersecting any annotation bbox
            n_neg = len(slide_pos)
            slide_neg: list[dict] = []
            max_tries = n_neg * 100
            tries = 0
            seen: set[tuple] = set()

            while len(slide_neg) < n_neg and tries < max_tries:
                tries += 1
                x0 = rng.randint(0, max(0, w - ts))
                y0 = rng.randint(0, max(0, h - ts))
                key = (x0, y0)
                if key in seen:
                    continue
                seen.add(key)
                if any(bbox_intersects_tile(a.bbox, x0, y0, ts) for a in annotations):
                    continue

                img = reader.read_tile(x0, y0, ts)
                slide_neg.append({
                    "slide_stem": slide_stem,
                    "index": len(pos_groups) + len(slide_pos) + len(slide_neg) + 1,
                    "x0": x0, "y0": y0,
                    "img": img,
                    "boxes": [],
                })

            if len(slide_neg) < n_neg:
                print(f"  [{slide_stem}] negative: got {len(slide_neg)} / {n_neg} after {tries} tries")

            pos_groups.extend(slide_pos)
            neg_groups.extend(slide_neg)
            total_pos += len(slide_pos)
            total_neg += len(slide_neg)
            print(f"  [{slide_stem}] pos={len(slide_pos)} neg={len(slide_neg)}")

        finally:
            reader.close()

    print(f"\nRaw: total_pos={total_pos}  total_neg={total_neg}")

    # 4. split
    all_groups = pos_groups + neg_groups
    rng.shuffle(all_groups)

    n_train = int(len(all_groups) * config.SPLIT_RATIOS["train"])
    for group in all_groups[:n_train]:
        group["split"] = "train"
    for group in all_groups[n_train:]:
        group["split"] = "val"

    # 5. write images + labels
    saved_train = saved_val = 0
    label_dir = config.OUTPUT_DIR / "labels"

    for group in all_groups:
        split = group["split"]
        stem = _build_stem(group["slide_stem"], "pos" if group["boxes"] else "neg",
                           group["index"], group["x0"], group["y0"])

        variant_stems = save_image_variants(
            arr=group["img"], stem=stem, split_name=split, rng=rng,
        )

        for vstem in variant_stems:
            label_path = label_dir / split / f"{vstem}.txt"
            save_box_label(label_path, group["boxes"])

        if split == "train":
            saved_train += len(variant_stems)
        else:
            saved_val += len(variant_stems)

    print(f"\nWritten: train={saved_train} val={saved_val} total={saved_train + saved_val}")

    # 6. write dataset.yaml
    _write_dataset_yaml()
    print(f"\nDone. dataset.yaml written.")
    print(f"Output: {config.OUTPUT_DIR}")


def _find_gt(wsi_path: Path) -> Path | None:
    stem = wsi_path.stem
    for ext in GT_EXTENSIONS:
        candidate = wsi_path.with_name(stem + ext)
        if candidate.is_file():
            return candidate
        # also try removing multi-extension (e.g. .ome.tiff -> .geojson)
        for subext in (".ome",):
            if stem.endswith(subext):
                candidate = wsi_path.with_name(stem[:-len(subext)] + ext)
                if candidate.is_file():
                    return candidate
    return None


def _build_stem(slide_stem: str, sample_type: str, idx: int, x0: int, y0: int) -> str:
    return f"{slide_stem}_{sample_type}_{idx:06d}_x{x0}_y{y0}"


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
