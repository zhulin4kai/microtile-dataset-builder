"""Dataset build orchestration."""

from __future__ import annotations

import multiprocessing as mp
import random
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import config
from core.annotation import (
    AnnotationIndex,
    load_annotations,
    validate_annotation_file,
)
from core.discovery import SlidePair
from formats import get_dataset_format
from core.reporting import (
    DatasetTotals,
    SlideStats,
    print_summary,
    status_text,
    write_build_report,
)
from core.runtime_config import (
    apply_runtime_config,
    runtime_config_snapshot,
)
from core.sampling import (
    choose_split,
    train_slide_count,
    write_negative_samples,
    write_positive_samples,
    split_for_slide,
)
from core.slide_io import SlideReader


def validate_config(
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
    if config.DATASET_SPLIT_MODE not in {"wsi", "patch"}:
        raise ValueError(
            f"DATASET_SPLIT_MODE 必须是 'wsi' 或 'patch'，当前值: {config.DATASET_SPLIT_MODE}"
        )
    train_ratio = config.SPLIT_RATIOS.get("train", 0)
    val_ratio = config.SPLIT_RATIOS.get("val", 0)
    split_sum = train_ratio + val_ratio
    if abs(split_sum - 1.0) > 0.001:
        raise ValueError(
            f"SPLIT_RATIOS 的 train+val 必须约等于 1.0，当前和: {split_sum}"
        )
    if train_ratio <= 0:
        raise ValueError(
            f"SPLIT_RATIOS['train'] 必须大于 0，当前值: {train_ratio}"
        )
    if val_ratio <= 0:
        raise ValueError(
            f"SPLIT_RATIOS['val'] 必须大于 0，当前值: {val_ratio}"
        )
    if config.MAX_NEG_TRIES_PER_POSITIVE < 1:
        raise ValueError(
            "MAX_NEG_TRIES_PER_POSITIVE 必须大于等于 1，避免负样本采样无法执行。"
        )
    get_dataset_format().validate_config()


def process_slide_pair(
    wsi_path: str,
    gt_path: str,
    slide_index: int,
    split_name: str | None,
    runtime_config: dict | None = None,
) -> dict:
    """处理单张 WSI，并立即写入图片和当前 dataset format 的标签。"""
    if runtime_config is not None:
        apply_runtime_config(runtime_config)

    dataset_format = get_dataset_format()
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
        annotation_index = AnnotationIndex(annotations)

        reader = SlideReader(wsi_path)
        w, h = reader.dimensions
        stats.raw_pos = write_positive_samples(
            reader=reader,
            annotations=annotations,
            annotation_index=annotation_index,
            slide_stem=slide_stem,
            split_name=split_name,
            rng=rng,
            stats=stats,
            writer=dataset_format,
        )
        stats.raw_neg = write_negative_samples(
            reader=reader,
            annotations=annotations,
            annotation_index=annotation_index,
            slide_stem=slide_stem,
            slide_w=w,
            slide_h=h,
            target_count=stats.raw_pos,
            split_name=split_name,
            rng=rng,
            stats=stats,
            writer=dataset_format,
        )
        stats.status = "ok"
    except Exception as e:
        stats.status = "failed"
        stats.error = str(e)
    finally:
        if reader is not None:
            reader.close()

    status = status_text(stats.status)
    print(
        f"[{status}] {slide_stem} raw_pos={stats.raw_pos} raw_neg={stats.raw_neg} "
        f"train={stats.written_train} val={stats.written_val} "
        f"failed_neg={stats.failed_neg}"
        + (f" 错误={stats.error}" if stats.error else "")
    )

    return stats.as_dict()


def prepare_output_dirs() -> None:
    get_dataset_format().prepare_output_dirs(config.OUTPUT_DIR, config.DRY_RUN)


def estimate_dry_run(pairs: list[SlidePair]) -> tuple[DatasetTotals, list[dict]]:
    totals = DatasetTotals()
    slide_results: list[dict] = []
    variant_count = get_dataset_format().variant_count()
    dry_pairs = list(pairs)
    rng = random.Random(config.RANDOM_SEED)

    if config.DATASET_SPLIT_MODE == "wsi":
        rng.shuffle(dry_pairs)
        n_train = train_slide_count(len(dry_pairs))
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
                split_name = split_for_slide(i, n_train)
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


def run_slide_pairs(pairs: list[SlidePair]) -> tuple[DatasetTotals, list[dict]]:
    rng = random.Random(config.RANDOM_SEED)
    if config.DATASET_SPLIT_MODE == "wsi":
        rng.shuffle(pairs)
        n_train = train_slide_count(len(pairs))
    else:
        n_train = 0

    ctx = mp.get_context(config.PROCESS_START_METHOD)
    futures: dict[Future, int] = {}
    totals = DatasetTotals()
    slide_results: list[dict] = []
    worker_config = runtime_config_snapshot()

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
                split_for_slide(i, n_train),
                worker_config,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            result = fut.result()
            totals.add(result)
            slide_results.append(result)

    return totals, slide_results


def write_dataset_metadata() -> None:
    get_dataset_format().write_metadata(config.OUTPUT_DIR, config.DRY_RUN)


def run_pipeline(pairs: list[SlidePair], diagnostics: dict) -> None:
    prepare_output_dirs()
    if config.DRY_RUN:
        totals, slide_results = estimate_dry_run(pairs)
    else:
        totals, slide_results = run_slide_pairs(pairs)
    write_dataset_metadata()
    print_summary(totals)
    write_build_report(diagnostics, totals, slide_results)


def write_sample_for_stats(sample, split_name, rng, stats) -> None:
    from core.sampling import write_sample

    write_sample(sample, split_name, rng, stats, get_dataset_format())

