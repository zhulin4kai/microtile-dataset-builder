# -*- coding: utf-8 -*-
"""
Segmentation polygon clipping module.

Uses shapely to clip annotation polygons to a tile's Level-0 rectangle,
then output YOLO segmentation label format (normalized [0,1] coordinates).
"""

from __future__ import annotations

from typing import List

import config
from core.geojson_parser import Annotation
from shapely.geometry import box as shapely_box, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid


def build_yolo_segments_for_tile(
    annotations: List[Annotation],
    label_indices: List[int],
    x0: int,
    y0: int,
    tile_size: int,
) -> List[List[float]]:
    """
    Clip Annotation polygons to a tile rectangle and return YOLO seg format.

    Returns a list of polygon vertex lists.
    Each inner list is [x1, y1, x2, y2, ..., xn, yn] with values in [0, 1].
    """
    tile_rect = shapely_box(x0, y0, x0 + tile_size, y0 + tile_size)
    segments: List[List[float]] = []

    for idx in label_indices:
        ann = annotations[idx]
        raw_poly = ann.polygon
        if len(raw_poly) < 3:
            continue

        try:
            geom = Polygon(raw_poly)
            if not geom.is_valid:
                geom = make_valid(geom)
        except Exception:
            continue

        intersection = tile_rect.intersection(geom)
        if intersection.is_empty:
            continue

        polys = _to_polygon_list(intersection)
        for poly in polys:
            coords = list(poly.exterior.coords)
            # remove closing duplicate
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < config.SEG_MIN_POLYGON_POINTS:
                continue

            flat: List[float] = []
            area = poly.area
            if area < config.SEG_MIN_POLYGON_AREA:
                continue

            for (gx, gy) in coords:
                nx = (gx - x0) / tile_size
                ny = (gy - y0) / tile_size
                flat.append(max(0.0, min(1.0, nx)))
                flat.append(max(0.0, min(1.0, ny)))

            if config.SEG_SIMPLIFY_EPSILON > 0:
                # simplify the polygon in local space
                pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
                simplified = Polygon(pts).simplify(config.SEG_SIMPLIFY_EPSILON)
                # simplify may produce non-Polygon (MultiPolygon, empty, etc.)
                polys = _to_polygon_list(simplified)
                if not polys:
                    continue
                flat = []
                for sp in polys:
                    coords_s = list(sp.exterior.coords)
                    if len(coords_s) >= 2 and coords_s[0] == coords_s[-1]:
                        coords_s = coords_s[:-1]
                    if len(coords_s) < config.SEG_MIN_POLYGON_POINTS:
                        continue
                    for (lx, ly) in coords_s:
                        flat.extend([lx, ly])
                if len(flat) < config.SEG_MIN_POLYGON_POINTS * 2:
                    continue

            segments.append(flat)

    return segments


def _to_polygon_list(geom) -> list:
    """Extract all Polygon parts from a shapely geometry."""
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    return []
