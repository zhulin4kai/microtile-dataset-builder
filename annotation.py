# -*- coding: utf-8 -*-
"""GeoJSON annotation 读取与 bbox 工具函数。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Annotation:
    feature_id: str
    polygon: List[Point]
    bbox: BBox


def load_annotations(geojson_path: Path) -> List[Annotation]:
    """读取 QuPath GeoJSON，并转换为可用于切图的 Annotation 列表。"""
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[Annotation] = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        coords = _collect_polygon_coords(geom)
        if not coords or len(coords) < 3:
            continue

        if not all(_is_valid_point(p) for p in coords):
            continue

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        min_x, min_y = float(min(xs)), float(min(ys))
        max_x, max_y = float(max(xs)), float(max(ys))
        bbox = (min_x, min_y, max_x, max_y)

        if (max_x - min_x) <= 0 or (max_y - min_y) <= 0:
            continue

        feature_id = feature.get("id") or feature.get("properties", {}).get("id",
                      f"ann_{len(results)}")

        results.append(Annotation(
            feature_id=str(feature_id),
            polygon=coords,
            bbox=bbox,
        ))
    return results


def validate_annotation_file(path: Path) -> dict:
    """检查 GeoJSON annotation 文件，返回构建前诊断信息。"""
    result: dict = {
        "annotation_count": 0,
        "skipped_features": 0,
        "error": None,
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        result["error"] = str(e)
        return result

    features = data.get("features", [])
    for feature in features:
        geom = feature.get("geometry", {})
        coords = _collect_polygon_coords(geom)
        if not coords or len(coords) < 3:
            result["skipped_features"] += 1
            continue
        if not all(_is_valid_point(p) for p in coords):
            result["skipped_features"] += 1
            continue
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if (max_x - min_x) <= 0 or (max_y - min_y) <= 0:
            result["skipped_features"] += 1
            continue
        result["annotation_count"] += 1

    return result


def _collect_polygon_coords(geom: dict) -> List[Point] | None:
    gtype = geom.get("type", "")
    raw = geom.get("coordinates", [])
    if gtype == "Polygon" and raw:
        ring = raw[0]
        if not ring or len(ring) < 3:
            return None
        return ring
    if gtype == "MultiPolygon":
        all_pts: List[Point] = []
        for polygon in raw:
            if polygon and polygon[0] and len(polygon[0]) >= 3:
                all_pts.extend(polygon[0])
        return all_pts if all_pts else None
    return None


def _is_valid_point(point: object) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return False
    return isinstance(point[0], (int, float)) and isinstance(point[1], (int, float))


def bbox_intersects_tile(bbox: BBox, x0: int, y0: int, tile_size: int) -> bool:
    x1, y1, x2, y2 = bbox
    tx2 = x0 + tile_size
    ty2 = y0 + tile_size
    return not (x2 <= x0 or x1 >= tx2 or y2 <= y0 or y1 >= ty2)


def clip_bbox_to_tile(bbox: BBox, x0: int, y0: int, tile_size: int) -> BBox | None:
    x1, y1, x2, y2 = bbox
    cx1 = max(x1, float(x0))
    cy1 = max(y1, float(y0))
    cx2 = min(x2, float(x0 + tile_size))
    cy2 = min(y2, float(y0 + tile_size))
    if cx1 >= cx2 or cy1 >= cy2:
        return None
    return (cx1, cy1, cx2, cy2)


def bbox_to_yolo(clipped: BBox, x0: int, y0: int, tile_size: int) -> Tuple[float, float, float, float]:
    cx1, cy1, cx2, cy2 = clipped
    xc = ((cx1 + cx2) / 2.0 - x0) / tile_size
    yc = ((cy1 + cy2) / 2.0 - y0) / tile_size
    w = (cx2 - cx1) / tile_size
    h = (cy2 - cy1) / tile_size
    return (xc, yc, w, h)


def bbox_visible_ratio(bbox: BBox, clipped: BBox) -> float:
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = clipped

    full_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    clipped_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)

    if full_area <= 0:
        return 0.0

    return clipped_area / full_area
