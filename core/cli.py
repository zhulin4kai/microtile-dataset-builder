"""CLI entrypoint for dataset building."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import config
from core.discovery import discover_slide_pairs
from core.pipeline import run_pipeline, validate_config


def parse_cli_args(argv: Sequence[str] | None) -> argparse.Namespace:
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


def apply_cli_args(args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        config.OUTPUT_DIR = args.output_dir
    if args.dry_run:
        config.DRY_RUN = True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(argv)
    apply_cli_args(args)
    validate_config(wsi_path=args.wsi_path, geojson_path=args.geojson_path)
    pairs, diagnostics = discover_slide_pairs(
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
    if diagnostics["ambiguous_annotations"]:
        print(f"  同名 annotation 歧义数: {diagnostics['ambiguous_annotations']}")
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

    run_pipeline(pairs, diagnostics)

