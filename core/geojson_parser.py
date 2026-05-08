# -*- coding: utf-8 -*-
"""
解析 QuPath 导出的 GeoJSON 标注。

输出 Annotation 列表，每个 Annotation 保留：
1. feature_id
2. class_name
3. polygon
4. bbox

默认认为 GeoJSON 坐标是 level 0 全局坐标。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Annotation:
    index: int
    feature_id: str
    class_name: str
    polygon: List[Point]
    bbox: BBox


def _extract_class_name(properties: dict, default_class_name: str) -> str:
    """
    兼容 QuPath 常见 GeoJSON classification 字段。
    """
    classification = properties.get("classification")

    if isinstance(classification, dict):
        name = classification.get("name")
        if name:
            return str(name)

    name = properties.get("class_name") or properties.get("name")
    if name:
        return str(name)

    return default_class_name


def _polygon_outer_rings(geometry: dict) -> List[List[Point]]:
    """
    从 GeoJSON geometry 中提取 Polygon / MultiPolygon 的外轮廓 ring。

    Polygon:
        coordinates = [
            [[x, y], [x, y], ...],      # outer ring
            [[x, y], [x, y], ...],      # holes, ignored
        ]

    MultiPolygon:
        coordinates = [
            [
                [[x, y], [x, y], ...],  # outer ring
                ...
            ],
            ...
        ]
    """
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if not coords:
        return []

    rings: List[List[Point]] = []

    if geom_type == "Polygon":
        outer = coords[0] if coords else []
        ring = _clean_ring(outer)
        if ring:
            rings.append(ring)

    elif geom_type == "MultiPolygon":
        for poly in coords:
            if not poly:
                continue
            outer = poly[0]
            ring = _clean_ring(outer)
            if ring:
                rings.append(ring)

    return rings


def _clean_ring(raw_ring: Sequence[Sequence[float]]) -> List[Point]:
    """
    清洗坐标 ring。

    QuPath 导出的 polygon 末尾经常重复第一个点。
    bbox 计算不受影响，但这里顺手去掉重复闭合点。
    """
    points: List[Point] = []

    for p in raw_ring:
        if len(p) < 2:
            continue
        x = float(p[0])
        y = float(p[1])
        points.append((x, y))

    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]

    if len(points) < 3:
        return []

    return points


def bbox_from_polygon(polygon: Sequence[Point]) -> BBox:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def is_valid_bbox(bbox: BBox, min_size: float = 1.0) -> bool:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def parse_qupath_geojson(
    geojson_path: Path,
    default_class_name: str = "micropapillary",
) -> List[Annotation]:
    """
    解析 QuPath GeoJSON 文件。

    Parameters
    ----------
    geojson_path:
        .geojson 文件路径。

    default_class_name:
        如果 GeoJSON 中没有 classification.name，则使用这个类别名。

    Returns
    -------
    List[Annotation]
    """
    geojson_path = Path(geojson_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    with geojson_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError(f"Invalid GeoJSON: missing features list: {geojson_path}")

    annotations: List[Annotation] = []
    ann_index = 0

    for feature_idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue

        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}

        feature_id = str(feature.get("id", f"feature_{feature_idx}"))
        class_name = _extract_class_name(properties, default_class_name)

        rings = _polygon_outer_rings(geometry)

        for ring_idx, polygon in enumerate(rings):
            bbox = bbox_from_polygon(polygon)

            if not is_valid_bbox(bbox):
                continue

            annotations.append(
                Annotation(
                    index=ann_index,
                    feature_id=f"{feature_id}_{ring_idx}",
                    class_name=class_name,
                    polygon=polygon,
                    bbox=bbox,
                )
            )
            ann_index += 1

    return annotations


def annotations_to_bbox_list(annotations: Sequence[Annotation]) -> List[BBox]:
    """
    转成普通 bbox list，后续 geometry_cuda 会再转 CUDA tensor。
    """
    return [ann.bbox for ann in annotations]


def count_annotations_by_class(annotations: Sequence[Annotation]) -> dict[str, int]:
    result: dict[str, int] = {}

    for ann in annotations:
        result[ann.class_name] = result.get(ann.class_name, 0) + 1

    return result
