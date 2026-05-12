# -*- coding: utf-8 -*-
"""
Segmentation polygon clipping and tile evaluation module.

Uses shapely to:
  - evaluate tile quality by polygon-area visible ratio (not bbox).
  - clip annotation polygons to a tile's Level-0 rectangle.
  - output YOLO segmentation label format (normalized [0,1] coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import config
from core.geojson_parser import Annotation
from shapely.geometry import box as shapely_box, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid


@dataclass(frozen=True)
class SegTileEvalResult:
    ok: bool
    reason: str
    source_visible_ratio: float
    label_indices: List[int]
    yolo_segments: List[List[float]]
    ambiguous_count: int


# ── public API ──────────────────────────────────────────────────────────────


def evaluate_seg_positive_tile(
    annotations: List[Annotation],
    source_index: int,
    x0: int,
    y0: int,
    tile_size: int,
    source_min_visible_ratio: float,
    label_min_visible_ratio: float,
    ignore_max_visible_ratio: float,
) -> SegTileEvalResult:
    """Evaluate a tile for seg mode using polygon-area visible ratios.

    Only the source annotation must meet source_min_visible_ratio.
    Other annotations contribute label data when visible >= label_min_visible_ratio.
    Ambiguous annotations (between ignore and label thresholds) cause tile rejection.
    """
    tile_rect = shapely_box(x0, y0, x0 + tile_size, y0 + tile_size)

    visible_ratios: List[float] = []
    clipped_geoms: List = []

    for ann in annotations:
        geom = _make_valid_polygon(ann.polygon)
        if geom is None or geom.is_empty or geom.area <= 0:
            visible_ratios.append(0.0)
            clipped_geoms.append(None)
            continue

        clipped = geom.intersection(tile_rect)
        if clipped.is_empty:
            visible_ratios.append(0.0)
            clipped_geoms.append(None)
            continue

        visible_area = _polygon_area_sum(clipped)
        ratio = visible_area / max(geom.area, 1e-6)
        visible_ratios.append(float(ratio))
        clipped_geoms.append(clipped)

    source_visible = visible_ratios[source_index]

    if source_visible < source_min_visible_ratio:
        return SegTileEvalResult(
            ok=False, reason="source_not_visible_enough",
            source_visible_ratio=source_visible,
            label_indices=[], yolo_segments=[], ambiguous_count=0,
        )

    label_indices: List[int] = []
    segments: List[List[float]] = []
    ambiguous_count = 0

    for idx, ratio in enumerate(visible_ratios):
        if ratio >= label_min_visible_ratio:
            segs = _clipped_geom_to_yolo_segments(
                clipped_geoms[idx], x0, y0, tile_size,
            )
            if segs:
                label_indices.append(idx)
                segments.extend(segs)
        elif ratio >= ignore_max_visible_ratio:
            ambiguous_count += 1

    if ambiguous_count > 0:
        return SegTileEvalResult(
            ok=False, reason="ambiguous_edge_target",
            source_visible_ratio=source_visible,
            label_indices=[], yolo_segments=[], ambiguous_count=ambiguous_count,
        )

    if not segments:
        return SegTileEvalResult(
            ok=False, reason="no_valid_label",
            source_visible_ratio=source_visible,
            label_indices=[], yolo_segments=[], ambiguous_count=0,
        )

    return SegTileEvalResult(
        ok=True, reason="ok",
        source_visible_ratio=source_visible,
        label_indices=label_indices,
        yolo_segments=segments,
        ambiguous_count=0,
    )


def build_yolo_segments_for_tile(
    annotations: List[Annotation],
    label_indices: List[int],
    x0: int,
    y0: int,
    tile_size: int,
) -> List[List[float]]:
    """Clip named Annotation polygons to a tile and return YOLO seg format.

    This is the lightweight variant — no per-annotation visible ratio check.
    Use evaluate_seg_positive_tile() for full tile evaluation.
    """
    tile_rect = shapely_box(x0, y0, x0 + tile_size, y0 + tile_size)
    segments: List[List[float]] = []

    for idx in label_indices:
        ann = annotations[idx]
        geom = _make_valid_polygon(ann.polygon)
        if geom is None:
            continue

        clipped = tile_rect.intersection(geom)
        if clipped.is_empty:
            continue

        segs = _clipped_geom_to_yolo_segments(clipped, x0, y0, tile_size)
        segments.extend(segs)

    return segments


# ── internal helpers ─────────────────────────────────────────────────────────


def _make_valid_polygon(points) -> Polygon | None:
    if len(points) < 3:
        return None
    try:
        geom = Polygon(points)
        if not geom.is_valid:
            geom = make_valid(geom)
        return geom
    except Exception:
        return None


def _polygon_area_sum(geom) -> float:
    if isinstance(geom, Polygon):
        return geom.area
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return sum(g.area for g in geom.geoms if isinstance(g, Polygon))
    return 0.0


def _clipped_geom_to_yolo_segments(
    clipped,
    x0: int,
    y0: int,
    tile_size: int,
) -> List[List[float]]:
    """Convert a clipped shapely geometry into normalized YOLO seg lists."""
    result: List[List[float]] = []
    polys = _to_polygon_list(clipped)

    for poly in polys:
        coords = list(poly.exterior.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < config.SEG_MIN_POLYGON_POINTS:
            continue
        if poly.area < config.SEG_MIN_POLYGON_AREA:
            continue

        flat: List[float] = []
        for (gx, gy) in coords:
            nx = max(0.0, min(1.0, (gx - x0) / tile_size))
            ny = max(0.0, min(1.0, (gy - y0) / tile_size))
            flat.extend([nx, ny])

        if config.SEG_SIMPLIFY_EPSILON > 0:
            pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
            simplified = Polygon(pts).simplify(config.SEG_SIMPLIFY_EPSILON)
            sub_polys = _to_polygon_list(simplified)
            if not sub_polys:
                continue
            flat = []
            for sp in sub_polys:
                cs = list(sp.exterior.coords)
                if len(cs) >= 2 and cs[0] == cs[-1]:
                    cs = cs[:-1]
                if len(cs) < config.SEG_MIN_POLYGON_POINTS:
                    continue
                for (lx, ly) in cs:
                    flat.extend([lx, ly])
            if len(flat) < config.SEG_MIN_POLYGON_POINTS * 2:
                continue

        result.append(flat)

    return result


def _to_polygon_list(geom) -> list:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    return []
