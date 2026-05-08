# -*- coding: utf-8 -*-
"""
扫描 target-dir，寻找同名 .svs + .geojson 文件对。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SlidePair:
    stem: str
    svs_path: Path
    geojson_path: Path


def discover_slide_pairs(target_dir: Path) -> List[SlidePair]:
    """
    在 target_dir 下寻找同名 .svs 和 .geojson。

    示例：
        2416137-2.svs
        2416137-2.geojson

    会被识别为一组。
    """
    target_dir = Path(target_dir)

    if not target_dir.exists():
        raise FileNotFoundError(f"TARGET_DIR not found: {target_dir}")

    if not target_dir.is_dir():
        raise NotADirectoryError(f"TARGET_DIR is not a directory: {target_dir}")

    svs_map: Dict[str, Path] = {}
    geojson_map: Dict[str, Path] = {}

    for path in target_dir.iterdir():
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".svs":
            svs_map[path.stem] = path

        elif suffix == ".geojson":
            geojson_map[path.stem] = path

    common_stems = sorted(set(svs_map.keys()) & set(geojson_map.keys()))

    pairs = [
        SlidePair(
            stem=stem,
            svs_path=svs_map[stem],
            geojson_path=geojson_map[stem],
        )
        for stem in common_stems
    ]

    missing_geojson = sorted(set(svs_map.keys()) - set(geojson_map.keys()))
    missing_svs = sorted(set(geojson_map.keys()) - set(svs_map.keys()))

    if missing_geojson:
        print("[WARN] 以下 SVS 没有找到同名 GeoJSON，已跳过：")
        for stem in missing_geojson:
            print(f"  - {stem}.svs")

    if missing_svs:
        print("[WARN] 以下 GeoJSON 没有找到同名 SVS，已跳过：")
        for stem in missing_svs:
            print(f"  - {stem}.geojson")

    if not pairs:
        raise RuntimeError(f"No valid .svs + .geojson pairs found in: {target_dir}")

    return pairs


def print_slide_pairs(pairs: List[SlidePair]) -> None:
    print(f"[INFO] 找到 {len(pairs)} 组 WSI / GeoJSON：")

    for pair in pairs:
        print(f"  - {pair.stem}")
