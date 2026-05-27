#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO detect 数据集构建器，多进程处理 WSI 与 GeoJSON annotation。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import shutil
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import config
from annotation import (
    Annotation,
    bbox_intersects_tile,
    bbox_visible_ratio,
    bbox_to_yolo,
    clip_bbox_to_tile,
    load_annotations,
    validate_annotation_file,
)
from augment_writer import get_variant_names, save_box_label, save_image_variants
from slide_io import SlideReader

WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
GT_EXTENSIONS = {".geojson", ".json"}

MIN_BOX_VISIBLE_RATIO = 0.30
MIN_TISSUE_RATIO = 0.10
NEG_RADIUS_STEP = 0.10
NEG_RADIUS_START = 0.15
NEG_LOG_INTERVAL = 5000


def _validate_config(
    wsi_path: Path | None = None,
    geojson_path: Path | None = None,
) -> None:
    if wsi_path is None:
        if not config.TARGET_DIR.exists():
            raise FileNotFoundError(f"TARGET_DIR 不存在: {config.TARGET_DIR}")
    elif not wsi_path.exists():
        raise FileNotFoundError(f"WSI 路径不存在: {wsi_path}")
    if geojson_path is not None and not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON 路径不存在: {geojson_path}")
    if config.TILE_SIZE <= 0:
        raise ValueError(f"TILE_SIZE 必须大于 0，当前值: {config.TILE_SIZE}")
    if config.NUM_WORKERS < 1:
        raise ValueError(f"NUM_WORKERS 必须大于等于 1，当前值: {config.NUM_WORKERS}")
    if config.DATASET_TASK != "detect":
        raise NotImplementedError(
            "当前构建器只支持 YOLO detect 数据集。"
        )
    if config.DATASET_SPLIT_MODE not in {"wsi", "patch"}:
        raise ValueError(
            f"DATASET_SPLIT_MODE 必须是 'wsi' 或 'patch'，当前值: {config.DATASET_SPLIT_MODE}"
        )
    split_sum = config.SPLIT_RATIOS.get("train", 0) + config.SPLIT_RATIOS.get("val", 0)
    if abs(split_sum - 1.0) > 0.001:
        raise ValueError(
            f"SPLIT_RATIOS 的 train+val 必须约等于 1.0，当前和: {split_sum}"
        )
    if config.MAX_NEG_TRIES_PER_POSITIVE < 1:
        raise ValueError(
            "MAX_NEG_TRIES_PER_POSITIVE 必须大于等于 1，避免负样本采样无法执行。"
        )


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
    status: str = "ok"
    skip_reason: str = ""
    error: str = ""
    neg_reject_duplicate: int = 0
    neg_reject_annotation: int = 0
    neg_reject_low_tissue: int = 0
    neg_reject_try_limit: int = 0

    def as_dict(self) -> dict:
        return {
            "slide_stem": self.slide_stem,
            "raw_pos": self.raw_pos,
            "raw_neg": self.raw_neg,
            "written_train": self.written_train,
            "written_val": self.written_val,
            "failed_neg": self.failed_neg,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "neg_reject_duplicate": self.neg_reject_duplicate,
            "neg_reject_annotation": self.neg_reject_annotation,
            "neg_reject_low_tissue": self.neg_reject_low_tissue,
            "neg_reject_try_limit": self.neg_reject_try_limit,
        }


@dataclass
class DatasetTotals:
    raw_pos: int = 0
    raw_neg: int = 0
    written_train: int = 0
    written_val: int = 0
    failed_neg: int = 0
    slides_ok: int = 0
    slides_skipped: int = 0
    slides_failed: int = 0
    neg_reject_duplicate: int = 0
    neg_reject_annotation: int = 0
    neg_reject_low_tissue: int = 0
    neg_reject_try_limit: int = 0

    def add(self, result: dict) -> None:
        self.raw_pos += result["raw_pos"]
        self.raw_neg += result["raw_neg"]
        self.written_train += result["written_train"]
        self.written_val += result["written_val"]
        self.failed_neg += result["failed_neg"]
        self.neg_reject_duplicate += result.get("neg_reject_duplicate", 0)
        self.neg_reject_annotation += result.get("neg_reject_annotation", 0)
        self.neg_reject_low_tissue += result.get("neg_reject_low_tissue", 0)
        self.neg_reject_try_limit += result.get("neg_reject_try_limit", 0)
        status = result.get("status", "ok")
        if status == "ok":
            self.slides_ok += 1
        elif status == "skipped":
            self.slides_skipped += 1
        else:
            self.slides_failed += 1


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
    runtime_config: dict | None = None,
) -> dict:
    """处理单张 WSI，并立即写入图片和 YOLO label。"""
    if runtime_config is not None:
        _apply_runtime_config(runtime_config)

    slide_stem = Path(wsi_path).stem
    stats = SlideStats(slide_stem=slide_stem)
    rng = random.Random(config.RANDOM_SEED + slide_index * 1000003)
    reader = None

    try:
        annotations = load_annotations(Path(gt_path))
        if not annotations:
            stats.status = "skipped"
            stats.skip_reason = "empty_annotations"
            print(f"[跳过] {slide_stem} 原因=空 annotation")
            return stats.as_dict()

        reader = SlideReader(wsi_path)
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
        stats.status = "ok"
    except Exception as e:
        stats.status = "failed"
        stats.error = str(e)
    finally:
        if reader is not None:
            reader.close()

    status_text = _status_text(stats.status)
    print(
        f"[{status_text}] {slide_stem} raw_pos={stats.raw_pos} raw_neg={stats.raw_neg} "
        f"train={stats.written_train} val={stats.written_val} "
        f"failed_neg={stats.failed_neg}"
        + (f" 错误={stats.error}" if stats.error else "")
    )

    return stats.as_dict()


def _status_text(status: str) -> str:
    return {
        "ok": "成功",
        "skipped": "跳过",
        "failed": "失败",
    }.get(status, status)


def _runtime_config_snapshot() -> dict:
    return {
        "OUTPUT_DIR": str(config.OUTPUT_DIR),
        "TILE_SIZE": config.TILE_SIZE,
        "SPLIT_RATIOS": dict(config.SPLIT_RATIOS),
        "ENABLE_COLOR_AUGMENT": config.ENABLE_COLOR_AUGMENT,
        "RANDOM_SEED": config.RANDOM_SEED,
        "CLASS_ID": config.CLASS_ID,
        "IMAGE_EXT": config.IMAGE_EXT,
        "JPEG_QUALITY": config.JPEG_QUALITY,
        "WRITE_EMPTY_LABEL_FOR_NEGATIVE": config.WRITE_EMPTY_LABEL_FOR_NEGATIVE,
        "DRY_RUN": config.DRY_RUN,
        "MAX_NEG_TRIES_PER_POSITIVE": config.MAX_NEG_TRIES_PER_POSITIVE,
        "CLAHE_CLIP_LIMIT_RANGE": config.CLAHE_CLIP_LIMIT_RANGE,
        "CLAHE_TILE_GRID_SIZE": config.CLAHE_TILE_GRID_SIZE,
        "HSV_HUE_SHIFT_LIMIT": config.HSV_HUE_SHIFT_LIMIT,
        "HSV_SAT_SHIFT_LIMIT": config.HSV_SAT_SHIFT_LIMIT,
        "HSV_VAL_SHIFT_LIMIT": config.HSV_VAL_SHIFT_LIMIT,
        "BRIGHTNESS_LIMIT": config.BRIGHTNESS_LIMIT,
        "CONTRAST_LIMIT": config.CONTRAST_LIMIT,
        "GAMMA_LIMIT": config.GAMMA_LIMIT,
    }


def _apply_runtime_config(snapshot: dict) -> None:
    for name, value in snapshot.items():
        if name == "OUTPUT_DIR":
            setattr(config, name, Path(value))
        else:
            setattr(config, name, value)


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
    max_tries = target_count * config.MAX_NEG_TRIES_PER_POSITIVE

    # 负样本区域可能很难找到，必须设置上限避免长任务卡死。
    while neg_index < target_count and tries < max_tries:
        tries += 1

        if tries % NEG_LOG_INTERVAL == 0:
            radius_ratio = min(1.5, radius_ratio + NEG_RADIUS_STEP)
            print(
                f"  [{slide_stem}] 负样本采样: "
                f"{neg_index}/{target_count}, 尝试={tries}, "
                f"radius_ratio={radius_ratio:.2f}"
            )

        sample, reject_reason = _try_make_negative_sample(
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
            if reject_reason == "duplicate":
                stats.neg_reject_duplicate += 1
            elif reject_reason == "annotation":
                stats.neg_reject_annotation += 1
            elif reject_reason == "low_tissue":
                stats.neg_reject_low_tissue += 1
            continue

        neg_index += 1
        _write_sample(sample, _choose_split(split_name, rng), rng, stats)

    if tries >= max_tries and neg_index < target_count:
        gap = target_count - neg_index
        stats.failed_neg += gap
        stats.neg_reject_try_limit += gap

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
) -> tuple[TileSample | None, str | None]:
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
        return None, "duplicate"
    seen.add(key)

    if _has_large_visible_annotation(annotations, x0, y0, ts):
        return None, "annotation"

    image = np.array(reader.read_tile(x0, y0, ts), dtype=np.uint8)
    if _tissue_ratio(image) < MIN_TISSUE_RATIO:
        return None, "low_tissue"

    return TileSample(
        stem=f"{slide_stem}_neg_{neg_index:06d}_x{x0}_y{y0}",
        image=image,
        boxes=[],
    ), None


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
    # DRY_RUN 只估算写入数量，不落盘 images/labels。
    if config.DRY_RUN:
        variant_count = len(get_variant_names())
        if split_name == "train":
            stats.written_train += variant_count
        else:
            stats.written_val += variant_count
        return

    variant_stems = save_image_variants(sample.image, sample.stem, split_name, rng)
    label_dir = config.OUTPUT_DIR / "labels" / split_name

    for variant_stem in variant_stems:
        save_box_label(label_dir / f"{variant_stem}.txt", sample.boxes)

    if split_name == "train":
        stats.written_train += len(variant_stems)
    else:
        stats.written_val += len(variant_stems)


def _write_build_report(
    diagnostics: dict,
    totals: DatasetTotals,
    slide_results: list[dict],
) -> None:
    config_snapshot = {
        "TARGET_DIR": str(config.TARGET_DIR),
        "OUTPUT_DIR": str(config.OUTPUT_DIR),
        "TILE_SIZE": config.TILE_SIZE,
        "NUM_WORKERS": config.NUM_WORKERS,
        "SPLIT_RATIOS": config.SPLIT_RATIOS,
        "DATASET_SPLIT_MODE": config.DATASET_SPLIT_MODE,
        "DATASET_TASK": config.DATASET_TASK,
        "ENABLE_COLOR_AUGMENT": config.ENABLE_COLOR_AUGMENT,
        "DRY_RUN": config.DRY_RUN,
        "RANDOM_SEED": config.RANDOM_SEED,
    }
    report = {
        "说明": "YOLO detect 数据集构建报告",
        "config": config_snapshot,
        "discovery": {
            "说明": "构建前数据检查结果",
            **diagnostics,
        },
        "slides": [_with_chinese_status(result) for result in slide_results],
        "totals": {
            "说明": "构建结果汇总",
            "raw_pos": totals.raw_pos,
            "raw_neg": totals.raw_neg,
            "written_train": totals.written_train,
            "written_val": totals.written_val,
            "failed_neg": totals.failed_neg,
            "slides_ok": totals.slides_ok,
            "slides_skipped": totals.slides_skipped,
            "slides_failed": totals.slides_failed,
            "neg_reject_duplicate": totals.neg_reject_duplicate,
            "neg_reject_annotation": totals.neg_reject_annotation,
            "neg_reject_low_tissue": totals.neg_reject_low_tissue,
            "neg_reject_try_limit": totals.neg_reject_try_limit,
        },
        "output_dir": str(config.OUTPUT_DIR),
    }
    if config.DRY_RUN:
        print(f"\n[DRY_RUN] 构建报告预览:\n{json.dumps(report, indent=2, ensure_ascii=False)}")
    elif config.WRITE_BUILD_REPORT:
        report_path = config.OUTPUT_DIR / "build_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n构建报告: {report_path}")


def _with_chinese_status(result: dict) -> dict:
    item = dict(result)
    item["status_text"] = _status_text(item.get("status", ""))
    if item.get("skip_reason") == "empty_annotations":
        item["skip_reason_text"] = "空 annotation"
    return item


def _parse_cli_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 YOLO detect 数据集")
    parser.add_argument(
        "--wsi-path",
        type=Path,
        default=None,
        help="输入切片文件或切片目录；未指定时使用 config.TARGET_DIR",
    )
    parser.add_argument(
        "--geojson-path",
        type=Path,
        default=None,
        help="输入 GeoJSON 文件或目录；未指定时默认与切片路径同目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="YOLO 数据集输出目录；未指定时使用 config.OUTPUT_DIR",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查输入并估算样本数量，不写入 images/labels",
    )
    return parser.parse_args([] if argv is None else list(argv))


def _apply_cli_args(args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        config.OUTPUT_DIR = args.output_dir
    if args.dry_run:
        config.DRY_RUN = True


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_cli_args(argv)
    _apply_cli_args(args)
    _validate_config(wsi_path=args.wsi_path, geojson_path=args.geojson_path)
    _prepare_output_dirs()
    pairs, diagnostics = _discover_slide_pairs(
        wsi_path=args.wsi_path,
        geojson_path=args.geojson_path,
    )

    print(f"找到 {len(pairs)} 个 WSI-annotation 配对")
    print(f"切片路径: {diagnostics['wsi_path']}")
    print(f"GeoJSON 路径: {diagnostics['geojson_path']}")
    print(f"worker 数: {config.NUM_WORKERS}")
    print(f"split 模式: {config.DATASET_SPLIT_MODE}")
    print(f"任务类型: {config.DATASET_TASK}")
    if config.DRY_RUN:
        print("DRY_RUN 已启用：只做检查和估算，不写 images/labels")
    if diagnostics["unmatched_wsi"]:
        print(f"  未配对 WSI 文件数: {diagnostics['unmatched_wsi']}")
    if diagnostics["orphan_annotations"]:
        print(f"  孤立 annotation 文件数: {diagnostics['orphan_annotations']}")
    if diagnostics["invalid_annotations"]:
        print(f"  解析失败 annotation 文件数: {diagnostics['invalid_annotations']}")
    if diagnostics["empty_annotations"]:
        print(f"  空 annotation 文件数: {diagnostics['empty_annotations']}")
    print()

    if not pairs:
        print("没有找到 WSI-annotation 配对，退出。")
        return

    if config.DRY_RUN:
        totals, slide_results = _estimate_dry_run(pairs)
    else:
        totals, slide_results = _run_slide_pairs(pairs)
    _write_dataset_yaml()
    _print_summary(totals)
    _write_build_report(diagnostics, totals, slide_results)


def _prepare_output_dirs() -> None:
    if config.DRY_RUN:
        return
    if config.OUTPUT_DIR.exists():
        shutil.rmtree(config.OUTPUT_DIR)
    for split in ("train", "val"):
        (config.OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def _discover_slide_pairs(
    wsi_path: Path | None = None,
    geojson_path: Path | None = None,
) -> tuple[list[SlidePair], dict]:
    wsi_root = wsi_path or config.TARGET_DIR
    gt_root = geojson_path or (wsi_root.parent if wsi_root.is_file() else wsi_root)
    pairs: list[SlidePair] = []
    diagnostics: dict = {
        "total_files_in_target": 0,
        "wsi_files": 0,
        "unmatched_wsi": 0,
        "orphan_annotations": 0,
        "invalid_annotations": 0,
        "empty_annotations": 0,
        "skipped_features": 0,
        "wsi_path": str(wsi_root),
        "geojson_path": str(gt_root),
    }

    wsi_paths, wsi_scan_count = _collect_wsi_paths(wsi_root)
    gt_paths_set, gt_scan_count = _collect_gt_paths(gt_root)
    diagnostics["wsi_files"] = len(wsi_paths)
    diagnostics["total_files_in_target"] = (
        wsi_scan_count if wsi_root == gt_root else wsi_scan_count + gt_scan_count
    )

    matched_gt_paths: set[Path] = set()
    if wsi_root.is_file() and gt_root.is_file():
        if wsi_paths and gt_paths_set:
            gt_path = next(iter(gt_paths_set))
            pairs.append(SlidePair(wsi_path=wsi_paths[0], gt_path=gt_path))
            matched_gt_paths.add(gt_path)
        elif wsi_paths:
            diagnostics["unmatched_wsi"] += 1
    else:
        for wsi in wsi_paths:
            gt_path = _find_gt(wsi, gt_paths_set)
            if gt_path is not None:
                pairs.append(SlidePair(wsi_path=wsi, gt_path=gt_path))
                matched_gt_paths.add(gt_path)
            else:
                diagnostics["unmatched_wsi"] += 1

    # 构建前顺手检查 annotation 文件质量，避免长任务结束后才发现输入有问题。
    for gt_path in gt_paths_set:
        if gt_path not in matched_gt_paths:
            diagnostics["orphan_annotations"] += 1
        info = validate_annotation_file(gt_path)
        if info["error"]:
            diagnostics["invalid_annotations"] += 1
        elif info["annotation_count"] == 0:
            diagnostics["empty_annotations"] += 1
        diagnostics["skipped_features"] += info["skipped_features"]

    return pairs, diagnostics


def _collect_wsi_paths(path: Path) -> tuple[list[Path], int]:
    if path.is_file():
        return ([path] if path.suffix.lower() in WSI_EXTENSIONS else []), 1

    paths: list[Path] = []
    scan_count = 0
    for item in sorted(path.iterdir()):
        scan_count += 1
        if item.suffix.lower() in WSI_EXTENSIONS:
            paths.append(item)
    return paths, scan_count


def _collect_gt_paths(path: Path) -> tuple[set[Path], int]:
    if path.is_file():
        return ({path} if path.suffix.lower() in GT_EXTENSIONS else set()), 1

    paths: set[Path] = set()
    scan_count = 0
    for item in sorted(path.iterdir()):
        scan_count += 1
        if item.suffix.lower() in GT_EXTENSIONS:
            paths.add(item)
    return paths, scan_count


def _estimate_dry_run(pairs: list[SlidePair]) -> tuple[DatasetTotals, list[dict]]:
    totals = DatasetTotals()
    slide_results: list[dict] = []
    variant_count = len(get_variant_names())
    dry_pairs = list(pairs)
    rng = random.Random(config.RANDOM_SEED)

    if config.DATASET_SPLIT_MODE == "wsi":
        rng.shuffle(dry_pairs)
        n_train = int(len(dry_pairs) * config.SPLIT_RATIOS["train"])
    else:
        n_train = 0

    for i, pair in enumerate(dry_pairs):
        info = validate_annotation_file(pair.gt_path)
        stats = SlideStats(slide_stem=pair.wsi_path.stem)
        if info["error"]:
            stats.status = "failed"
            stats.error = info["error"]
        elif info["annotation_count"] == 0:
            stats.status = "skipped"
            stats.skip_reason = "empty_annotations"
        else:
            stats.raw_pos = info["annotation_count"]
            stats.raw_neg = info["annotation_count"]
            written_total = (stats.raw_pos + stats.raw_neg) * variant_count
            if config.DATASET_SPLIT_MODE == "wsi":
                split_name = _split_for_slide(i, n_train)
                if split_name == "train":
                    stats.written_train = written_total
                else:
                    stats.written_val = written_total
            else:
                stats.written_train = int(round(written_total * config.SPLIT_RATIOS["train"]))
                stats.written_val = written_total - stats.written_train

        result = stats.as_dict()
        totals.add(result)
        slide_results.append(result)

    return totals, slide_results


def _run_slide_pairs(pairs: list[SlidePair]) -> tuple[DatasetTotals, list[dict]]:
    rng = random.Random(config.RANDOM_SEED)
    if config.DATASET_SPLIT_MODE == "wsi":
        rng.shuffle(pairs)
        n_train = int(len(pairs) * config.SPLIT_RATIOS["train"])
    else:
        n_train = 0

    ctx = mp.get_context(config.PROCESS_START_METHOD)
    futures: dict[Future, int] = {}
    totals = DatasetTotals()
    slide_results: list[dict] = []
    runtime_config = _runtime_config_snapshot()

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
                runtime_config,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            result = fut.result()
            totals.add(result)
            slide_results.append(result)

    return totals, slide_results


def _split_for_slide(slide_index: int, n_train: int) -> str | None:
    if config.DATASET_SPLIT_MODE == "wsi":
        return "train" if slide_index < n_train else "val"
    return None


def _print_summary(totals: DatasetTotals) -> None:
    print(f"\nWSI 处理结果：成功={totals.slides_ok} 跳过={totals.slides_skipped} 失败={totals.slides_failed}")
    print(f"原始正样本：{totals.raw_pos}")
    print(f"原始负样本：{totals.raw_neg}")
    print(f"写入 train images：{totals.written_train}")
    print(f"写入 val images：{totals.written_val}")
    print(f"写入 images 总数：{totals.written_train + totals.written_val}")
    print(f"输出目录：{config.OUTPUT_DIR}")
    if totals.failed_neg > 0:
        print(f"负样本不足：{totals.failed_neg}")
    if any([
        totals.neg_reject_duplicate,
        totals.neg_reject_annotation,
        totals.neg_reject_low_tissue,
        totals.neg_reject_try_limit,
    ]):
        print(
            f"负样本拒绝原因: duplicate={totals.neg_reject_duplicate} "
            f"annotation={totals.neg_reject_annotation} "
            f"low_tissue={totals.neg_reject_low_tissue} "
            f"try_limit={totals.neg_reject_try_limit}"
        )


def _find_gt(wsi_path: Path, gt_paths: set[Path] | None = None) -> Path | None:
    if gt_paths is not None:
        match_stems = _gt_match_stems(wsi_path)
        for gt_path in sorted(gt_paths):
            if gt_path.stem in match_stems:
                return gt_path
        return None

    stem = wsi_path.stem
    for ext in sorted(GT_EXTENSIONS):
        candidate = wsi_path.with_name(stem + ext)
        if candidate.is_file():
            return candidate
        for subext in (".ome",):
            if stem.endswith(subext):
                candidate = wsi_path.with_name(stem[:-len(subext)] + ext)
                if candidate.is_file():
                    return candidate
    return None


def _gt_match_stems(wsi_path: Path) -> set[str]:
    stem = wsi_path.stem
    stems = {stem}
    for subext in (".ome",):
        if stem.endswith(subext):
            stems.add(stem[:-len(subext)])
    return stems


def _write_dataset_yaml():
    if config.DRY_RUN:
        return
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
    import sys

    main(sys.argv[1:])
