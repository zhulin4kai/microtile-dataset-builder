# -*- coding: utf-8 -*-
"""
纯 NumPy 几何计算模块（替代 CUDA）。

提供与旧 geometry.py 兼容的 PositiveTileEvalResult，
以及负样本过滤函数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class PositiveTileEvalResult:
    ok: bool
    reason: str
    source_visible_ratio: float
    label_indices: List[int]
    yolo_boxes: List[Tuple[float, float, float, float]]
    ambiguous_count: int


def boxes_to_numpy(boxes):
    return np.asarray(boxes, dtype=np.float32)


def _box_area(boxes: np.ndarray) -> np.ndarray:
    """[N, 4] -> [N]"""
    w = np.maximum(0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return w * h


def _intersection_area(boxes: np.ndarray, tile_box: np.ndarray) -> np.ndarray:
    """N boxes [N, 4] 与 1 个 tile_box [4] 的相交面积。"""
    ix1 = np.maximum(boxes[:, 0], tile_box[0])
    iy1 = np.maximum(boxes[:, 1], tile_box[1])
    ix2 = np.minimum(boxes[:, 2], tile_box[2])
    iy2 = np.minimum(boxes[:, 3], tile_box[3])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    return iw * ih


def _visible_ratios(boxes: np.ndarray, tile_box: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    inter = _intersection_area(boxes, tile_box)
    area = np.maximum(_box_area(boxes), eps)
    return inter / area


def _clip_boxes_to_tile(boxes: np.ndarray, tile_box: np.ndarray) -> np.ndarray:
    """将全局 bbox clip 到 tile 内并转局部坐标。"""
    gx1 = np.maximum(boxes[:, 0], tile_box[0])
    gy1 = np.maximum(boxes[:, 1], tile_box[1])
    gx2 = np.minimum(boxes[:, 2], tile_box[2])
    gy2 = np.minimum(boxes[:, 3], tile_box[3])
    lx1 = gx1 - tile_box[0]
    ly1 = gy1 - tile_box[1]
    lx2 = gx2 - tile_box[0]
    ly2 = gy2 - tile_box[1]
    return np.stack([lx1, ly1, lx2, ly2], axis=1)


def _local_to_yolo(local_boxes: np.ndarray, tile_size: int) -> np.ndarray:
    """局部坐标 [N, 4] -> YOLO 归一化 [N, 4]"""
    x1 = local_boxes[:, 0]
    y1 = local_boxes[:, 1]
    x2 = local_boxes[:, 2]
    y2 = local_boxes[:, 3]
    xc = ((x1 + x2) * 0.5) / tile_size
    yc = ((y1 + y2) * 0.5) / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size
    yolo = np.stack([xc, yc, w, h], axis=1)
    return np.clip(yolo, 0.0, 1.0)


def evaluate_positive_tile_np(
    boxes: np.ndarray,
    source_index: int,
    x0: int,
    y0: int,
    tile_size: int,
    source_min_visible_ratio: float,
    label_min_visible_ratio: float,
    ignore_max_visible_ratio: float,
    min_clipped_box_size: int,
) -> PositiveTileEvalResult:
    """
    判断正样本候选 tile 是否可保存，并生成 YOLO 标签。
    """
    if boxes.shape[0] == 0:
        return PositiveTileEvalResult(
            ok=False,
            reason="empty_boxes",
            source_visible_ratio=0.0,
            label_indices=[],
            yolo_boxes=[],
            ambiguous_count=0,
        )

    tile_box = np.array([x0, y0, x0 + tile_size, y0 + tile_size], dtype=np.float32)

    ratios = _visible_ratios(boxes, tile_box)
    local_boxes = _clip_boxes_to_tile(boxes, tile_box)

    wh = np.maximum(0, local_boxes[:, 2:4] - local_boxes[:, 0:2])
    clipped_valid = (wh[:, 0] >= min_clipped_box_size) & (wh[:, 1] >= min_clipped_box_size)

    source_visible = float(ratios[source_index])

    if source_visible < source_min_visible_ratio:
        return PositiveTileEvalResult(
            ok=False,
            reason="source_not_visible_enough",
            source_visible_ratio=source_visible,
            label_indices=[],
            yolo_boxes=[],
            ambiguous_count=0,
        )

    label_mask = (ratios >= label_min_visible_ratio) & clipped_valid
    ambiguous_mask = (
        (ratios >= ignore_max_visible_ratio)
        & (ratios < label_min_visible_ratio)
        & clipped_valid
    )

    ambiguous_count = int(ambiguous_mask.sum())

    if ambiguous_count > 0:
        return PositiveTileEvalResult(
            ok=False,
            reason="ambiguous_edge_target",
            source_visible_ratio=source_visible,
            label_indices=[],
            yolo_boxes=[],
            ambiguous_count=ambiguous_count,
        )

    label_count = int(label_mask.sum())

    if label_count <= 0:
        return PositiveTileEvalResult(
            ok=False,
            reason="no_valid_label",
            source_visible_ratio=source_visible,
            label_indices=[],
            yolo_boxes=[],
            ambiguous_count=0,
        )

    label_indices = [int(i) for i in np.where(label_mask)[0]]
    selected_local = local_boxes[label_mask]
    selected_yolo = _local_to_yolo(selected_local, tile_size)
    yolo_boxes = [tuple(float(v) for v in row) for row in selected_yolo]

    return PositiveTileEvalResult(
        ok=True,
        reason="ok",
        source_visible_ratio=source_visible,
        label_indices=label_indices,
        yolo_boxes=yolo_boxes,
        ambiguous_count=0,
    )


def _expanded_boxes(boxes: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return boxes
    m = float(margin)
    delta = np.array([-m, -m, m, m], dtype=np.float32)
    return boxes + delta


def filter_candidate_tiles_without_boxes_np(
    boxes: np.ndarray,
    candidates_xy: np.ndarray,
    tile_size: int,
    margin: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    批量过滤候选负样本 tile。

    boxes: [N, 4]
    candidates_xy: [M, 2]
    tile_size: int
    margin: int
    batch_size: 分批大小

    返回 bool mask [M]，True 表示不与任何 expanded bbox 相交。
    """
    M = candidates_xy.shape[0]

    if M == 0:
        return np.empty((0,), dtype=bool)

    if boxes.shape[0] == 0:
        return np.ones((M,), dtype=bool)

    check_boxes = _expanded_boxes(boxes, margin)

    valid_mask = np.ones((M,), dtype=bool)

    for batch_start in range(0, M, batch_size):
        batch_end = min(batch_start + batch_size, M)
        batch_candidates = candidates_xy[batch_start:batch_end]

        bx0 = batch_candidates[:, 0]
        by0 = batch_candidates[:, 1]
        bx1 = bx0 + tile_size
        by1 = by0 + tile_size

        B = batch_candidates.shape[0]
        N = check_boxes.shape[0]

        # [B, N] 相交判断
        ix1 = np.maximum(bx0[:, None], check_boxes[None, :, 0])
        iy1 = np.maximum(by0[:, None], check_boxes[None, :, 1])
        ix2 = np.minimum(bx1[:, None], check_boxes[None, :, 2])
        iy2 = np.minimum(by1[:, None], check_boxes[None, :, 3])

        iw = np.maximum(0, ix2 - ix1)
        ih = np.maximum(0, iy2 - iy1)
        inter = iw * ih

        intersects = np.any(inter > 0, axis=1)
        valid_mask[batch_start:batch_end] = ~intersects

    return valid_mask
