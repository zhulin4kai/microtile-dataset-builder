"""WSI and annotation discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config
from core.annotation import validate_annotation_file

WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
GT_EXTENSIONS = {".geojson", ".json"}


@dataclass(frozen=True)
class SlidePair:
    wsi_path: Path
    gt_path: Path


def discover_slide_pairs(
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
        "ambiguous_annotations": 0,
        "orphan_annotations": 0,
        "invalid_annotations": 0,
        "empty_annotations": 0,
        "skipped_features": 0,
        "recursive_discovery": config.DISCOVER_RECURSIVE,
        "wsi_path": str(wsi_root),
        "geojson_path": str(gt_root),
    }

    wsi_paths, wsi_scan_count = collect_wsi_paths(wsi_root)
    gt_paths_set, gt_scan_count = collect_gt_paths(gt_root)
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
            gt_path = find_gt(wsi, gt_paths_set)
            if has_ambiguous_gt_match(wsi, gt_paths_set):
                diagnostics["ambiguous_annotations"] += 1
            if gt_path is not None:
                pairs.append(SlidePair(wsi_path=wsi, gt_path=gt_path))
                matched_gt_paths.add(gt_path)
            else:
                diagnostics["unmatched_wsi"] += 1

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


def collect_wsi_paths(path: Path) -> tuple[list[Path], int]:
    if path.is_file():
        return ([path] if path.suffix.lower() in WSI_EXTENSIONS else []), 1

    paths: list[Path] = []
    scan_count = 0
    iterator = path.rglob("*") if config.DISCOVER_RECURSIVE else path.iterdir()
    for item in sorted(iterator):
        if not item.is_file():
            continue
        scan_count += 1
        if item.suffix.lower() in WSI_EXTENSIONS:
            paths.append(item)
    return paths, scan_count


def collect_gt_paths(path: Path) -> tuple[set[Path], int]:
    if path.is_file():
        return ({path} if path.suffix.lower() in GT_EXTENSIONS else set()), 1

    paths: set[Path] = set()
    scan_count = 0
    iterator = path.rglob("*") if config.DISCOVER_RECURSIVE else path.iterdir()
    for item in sorted(iterator):
        if not item.is_file():
            continue
        scan_count += 1
        if item.suffix.lower() in GT_EXTENSIONS:
            paths.add(item)
    return paths, scan_count


def find_gt(wsi_path: Path, gt_paths: set[Path] | None = None) -> Path | None:
    if gt_paths is not None:
        same_dir, other_dir = matching_gt_paths(wsi_path, gt_paths)
        candidates = same_dir + other_dir
        return candidates[0] if candidates else None

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


def matching_gt_paths(wsi_path: Path, gt_paths: set[Path]) -> tuple[list[Path], list[Path]]:
    match_stems = gt_match_stems(wsi_path)
    same_dir: list[Path] = []
    other_dir: list[Path] = []
    for gt_path in sorted(gt_paths):
        if gt_path.stem not in match_stems:
            continue
        if gt_path.parent == wsi_path.parent:
            same_dir.append(gt_path)
        else:
            other_dir.append(gt_path)
    return same_dir, other_dir


def has_ambiguous_gt_match(wsi_path: Path, gt_paths: set[Path]) -> bool:
    same_dir, other_dir = matching_gt_paths(wsi_path, gt_paths)
    if same_dir:
        return len(same_dir) > 1
    return len(other_dir) > 1


def gt_match_stems(wsi_path: Path) -> set[str]:
    stem = wsi_path.stem
    stems = {stem}
    for subext in (".ome",):
        if stem.endswith(subext):
            stems.add(stem[:-len(subext)])
    return stems

