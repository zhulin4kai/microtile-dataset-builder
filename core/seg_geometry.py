# -*- coding: utf-8 -*-
"""
Segmentation polygon clipping and tile evaluation module.

Uses raster-mask + contour extraction instead of Shapely exterior
to avoid brush-annotation degradation (triangles / noise fragments).

Pipeline per annotation per tile:
  1. rasterize GeoJSON polygon to tile-local uint8 mask
  2. clean mask (keep largest component, fill holes, filter small fragments)
  3. findContours -> approxPolyDP -> YOLO seg format
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

import config
from core.geojson_parser import Annotation

# cache annotation original mask area (keyed by ann.index)
_ANN_AREA_CACHE: Dict[int, float] = {}


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
    """Evaluate a tile for seg mode using mask-area visible ratios.

    Rasterizes each annotation polygon into tile-local mask, cleans it,
    then computes visible_ratio = visible_px / original_px.
    """
    ss = max(1, int(config.SEG_MASK_SUPERSAMPLE))
    tw = tile_size * ss

    visible_ratios: List[float] = []
    tile_masks: List[np.ndarray | None] = []

    for ann in annotations:
        clean = _rasterize_and_clean_for_tile(ann, x0, y0, tile_size, tw)

        if clean is None or clean.max() == 0:
            visible_ratios.append(0.0)
            tile_masks.append(None)
            continue

        visible_px = int(np.count_nonzero(clean))
        orig_px = _get_original_area(ann)

        ratio = visible_px / max(orig_px, 1.0)
        visible_ratios.append(float(ratio))

        if ss > 1:
            clean = cv2.resize(clean, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        tile_masks.append(clean)

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
            mask = tile_masks[idx]
            if mask is not None:
                segs = _mask_to_yolo_segments(mask, tile_size)
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
    """Lightweight mask-based segment builder (no ratio check)."""
    result: List[List[float]] = []
    for idx in label_indices:
        ann = annotations[idx]
        clean = _rasterize_and_clean_for_tile(ann, x0, y0, tile_size, tile_size)
        if clean is not None and clean.max() > 0:
            segs = _mask_to_yolo_segments(clean, tile_size)
            result.extend(segs)
    return result


# ── internal: rasterize + clean ──────────────────────────────────────────────


def _rasterize_polygon_to_mask(
    points: List,
    x0: int,
    y0: int,
    tile_size: int,
    target_size: int,
) -> np.ndarray:
    """cv2.fillPoly a list of (gx, gy) points into a target_size x target_size uint8 mask."""
    scale = target_size / tile_size
    local = np.array(
        [(round((px - x0) * scale), round((py - y0) * scale)) for (px, py) in points],
        dtype=np.int32,
    )
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    if local.shape[0] >= 3:
        cv2.fillPoly(mask, [local], 255)
    return mask


def _rasterize_and_clean_for_tile(
    ann: Annotation,
    x0: int,
    y0: int,
    tile_size: int,
    target_size: int,
) -> np.ndarray | None:
    """Rasterize annotation polygon to target_size mask, then clean."""
    pts = ann.polygon
    if len(pts) < 3:
        return None

    mask = _rasterize_polygon_to_mask(pts, x0, y0, tile_size, target_size)
    if mask.max() == 0:
        return None

    return _clean_instance_mask(mask)


def _clean_instance_mask(mask: np.ndarray) -> np.ndarray:
    """Remove small fragments, keep largest component, fill holes."""
    if mask.max() == 0:
        return mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    # sort by area descending
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        return np.zeros_like(mask)

    if config.SEG_KEEP_LARGEST_COMPONENT:
        largest_idx = int(np.argmax(areas))
        largest_area = areas[largest_idx]
        kept = [contours[largest_idx]]
    else:
        largest_area = max(areas)
        kept = []
        for c, a in zip(contours, areas):
            if a >= config.SEG_MIN_COMPONENT_AREA and a >= largest_area * config.SEG_MIN_COMPONENT_AREA_RATIO:
                kept.append(c)

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, kept, -1, 255, thickness=-1)

    if config.SEG_FILL_HOLES:
        contours2, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            largest2 = max(contours2, key=cv2.contourArea)
            clean[:] = 0
            cv2.drawContours(clean, [largest2], -1, 255, thickness=-1)

    return clean


# ── internal: mask -> YOLO segments ──────────────────────────────────────────


def _mask_to_yolo_segments(
    mask: np.ndarray,
    tile_size: int,
) -> List[List[float]]:
    """Extract contours from clean mask and return normalized YOLO seg lists."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result: List[List[float]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < config.SEG_MIN_POLYGON_AREA:
            continue

        approx = cv2.approxPolyDP(c, config.SEG_CONTOUR_APPROX_EPSILON, closed=True)
        pts = approx.squeeze(1)
        if pts.ndim != 2 or pts.shape[0] < config.SEG_MIN_POLYGON_POINTS:
            continue

        flat: List[float] = []
        for (lx, ly) in pts.astype(np.float64):
            nx = max(0.0, min(1.0, lx / tile_size))
            ny = max(0.0, min(1.0, ly / tile_size))
            flat.extend([nx, ny])

        # remove closing duplicate
        if len(flat) >= 4 and flat[0] == flat[-2] and flat[1] == flat[-1]:
            flat = flat[:-2]

        if len(flat) < config.SEG_MIN_POLYGON_POINTS * 2:
            continue

        result.append(flat)

    return result


# ── internal: original area estimation ───────────────────────────────────────


def _estimate_annotation_mask_area(ann: Annotation) -> float:
    """Rasterize polygon in its bbox-local ROI and return clean pixel area."""
    x1, y1, x2, y2 = ann.bbox
    bw = int(x2 - x1) + 1
    bh = int(y2 - y1) + 1

    local_pts = [(px - x1, py - y1) for (px, py) in ann.polygon]
    mask = np.zeros((bh, bw), dtype=np.uint8)
    pts_arr = np.array(local_pts, dtype=np.int32).reshape(-1, 1, 2)
    if pts_arr.shape[0] >= 3:
        cv2.fillPoly(mask, [pts_arr], 255)

    clean = _clean_instance_mask(mask)
    return float(np.count_nonzero(clean))


def _get_original_area(ann: Annotation) -> float:
    key = ann.feature_id
    if key not in _ANN_AREA_CACHE:
        _ANN_AREA_CACHE[key] = _estimate_annotation_mask_area(ann)
    return _ANN_AREA_CACHE[key]
