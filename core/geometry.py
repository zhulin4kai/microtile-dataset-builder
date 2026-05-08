"""
CUDA 几何计算模块。

负责：
1. bbox 与 tile 的相交面积；
2. visible_ratio 计算；
3. patch 内目标筛选；
4. bbox clip；
5. YOLO 坐标转换；
6. 负样本 bbox 避让判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PositiveTileEvalResult:
    ok: bool
    reason: str
    source_visible_ratio: float
    label_indices: List[int]
    yolo_boxes: List[Tuple[float, float, float, float]]
    ambiguous_count: int


def get_cuda_device(
    device_name: str = "cuda:0", require_cuda: bool = True
) -> torch.device:
    """
    获取 CUDA device。

    本项目默认强制 CUDA。
    """
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，但当前配置要求必须使用 CUDA。")

    return torch.device(device_name)


def boxes_to_cuda(
    boxes: Sequence[Sequence[float]] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    将 bbox 数据转为 CUDA Tensor。

    boxes 格式：
        [[x_min, y_min, x_max, y_max], ...]
    """
    if isinstance(boxes, torch.Tensor):
        return boxes.to(device=device, dtype=torch.float32, non_blocking=True)

    return torch.tensor(boxes, device=device, dtype=torch.float32)


def make_tile_box_cuda(
    x0: int,
    y0: int,
    tile_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    构造 tile 的全局 bbox：
        [x0, y0, x0 + tile_size, y0 + tile_size]
    """
    return torch.tensor(
        [x0, y0, x0 + tile_size, y0 + tile_size],
        device=device,
        dtype=torch.float32,
    )


def box_area_cuda(boxes: torch.Tensor) -> torch.Tensor:
    """
    计算 bbox 面积。

    boxes:
        shape = [N, 4]
    """
    wh = (boxes[:, 2:4] - boxes[:, 0:2]).clamp_min(0)
    return wh[:, 0] * wh[:, 1]


def intersection_area_cuda(
    boxes: torch.Tensor,
    tile_box: torch.Tensor,
) -> torch.Tensor:
    """
    计算 N 个 bbox 与一个 tile_box 的相交面积。

    boxes:
        shape = [N, 4]

    tile_box:
        shape = [4]
    """
    ix1 = torch.maximum(boxes[:, 0], tile_box[0])
    iy1 = torch.maximum(boxes[:, 1], tile_box[1])
    ix2 = torch.minimum(boxes[:, 2], tile_box[2])
    iy2 = torch.minimum(boxes[:, 3], tile_box[3])

    iw = (ix2 - ix1).clamp_min(0)
    ih = (iy2 - iy1).clamp_min(0)

    return iw * ih


def visible_ratios_cuda(
    boxes: torch.Tensor,
    tile_box: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    计算每个 bbox 在 tile 内的可见比例。

    visible_ratio = intersection_area / bbox_area
    """
    inter = intersection_area_cuda(boxes, tile_box)
    area = box_area_cuda(boxes).clamp_min(eps)
    return inter / area


def clip_boxes_to_tile_cuda(
    boxes: torch.Tensor,
    tile_box: torch.Tensor,
) -> torch.Tensor:
    """
    将全局 bbox clip 到 tile 内，并转换成 tile 局部坐标。

    返回：
        local clipped boxes, shape = [N, 4]
        坐标范围在 [0, tile_size] 内。
    """
    gx1 = torch.maximum(boxes[:, 0], tile_box[0])
    gy1 = torch.maximum(boxes[:, 1], tile_box[1])
    gx2 = torch.minimum(boxes[:, 2], tile_box[2])
    gy2 = torch.minimum(boxes[:, 3], tile_box[3])

    lx1 = gx1 - tile_box[0]
    ly1 = gy1 - tile_box[1]
    lx2 = gx2 - tile_box[0]
    ly2 = gy2 - tile_box[1]

    return torch.stack([lx1, ly1, lx2, ly2], dim=1)


def local_boxes_to_yolo_cuda(
    local_boxes: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """
    将 tile 局部坐标 bbox 转成 YOLO 格式。

    输入：
        local_boxes:
            shape = [N, 4]
            [x_min, y_min, x_max, y_max]

    输出：
        shape = [N, 4]
        [x_center, y_center, width, height]
        坐标归一化到 [0, 1]
    """
    x1 = local_boxes[:, 0]
    y1 = local_boxes[:, 1]
    x2 = local_boxes[:, 2]
    y2 = local_boxes[:, 3]

    xc = ((x1 + x2) * 0.5) / tile_size
    yc = ((y1 + y2) * 0.5) / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size

    yolo = torch.stack([xc, yc, w, h], dim=1)
    return yolo.clamp(0.0, 1.0)


def evaluate_positive_tile_cuda(
    boxes: torch.Tensor,
    source_index: int,
    x0: int,
    y0: int,
    tile_size: int,
    source_min_visible_ratio: float,
    label_min_visible_ratio: float,
    ignore_max_visible_ratio: float,
    min_clipped_box_size: int,
    device: torch.device,
) -> PositiveTileEvalResult:
    """
    判断一个正样本候选 tile 是否可保存，并生成 YOLO 标签。

    规则：
    1. 主目标 visible_ratio 必须 >= source_min_visible_ratio；
    2. patch 内 visible_ratio >= label_min_visible_ratio 的目标写入标签；
    3. visible_ratio 处于 [ignore_max_visible_ratio, label_min_visible_ratio)
       且裁剪后 bbox 仍然可见的目标，会导致该 patch 作废；
    4. visible_ratio < ignore_max_visible_ratio 的极小边缘碎片允许忽略。
    """
    if boxes.numel() == 0:
        return PositiveTileEvalResult(
            ok=False,
            reason="empty_boxes",
            source_visible_ratio=0.0,
            label_indices=[],
            yolo_boxes=[],
            ambiguous_count=0,
        )

    tile_box = make_tile_box_cuda(x0, y0, tile_size, device)

    ratios = visible_ratios_cuda(boxes, tile_box)
    local_boxes = clip_boxes_to_tile_cuda(boxes, tile_box)

    wh = (local_boxes[:, 2:4] - local_boxes[:, 0:2]).clamp_min(0)
    clipped_valid = (wh[:, 0] >= min_clipped_box_size) & (
        wh[:, 1] >= min_clipped_box_size
    )

    source_visible = float(ratios[source_index].detach().cpu().item())

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

    ambiguous_count = int(ambiguous_mask.sum().detach().cpu().item())

    if ambiguous_count > 0:
        return PositiveTileEvalResult(
            ok=False,
            reason="ambiguous_edge_target",
            source_visible_ratio=source_visible,
            label_indices=[],
            yolo_boxes=[],
            ambiguous_count=ambiguous_count,
        )

    label_count = int(label_mask.sum().detach().cpu().item())

    if label_count <= 0:
        return PositiveTileEvalResult(
            ok=False,
            reason="no_valid_label",
            source_visible_ratio=source_visible,
            label_indices=[],
            yolo_boxes=[],
            ambiguous_count=0,
        )

    label_indices_tensor = torch.nonzero(label_mask, as_tuple=False).flatten()
    selected_local_boxes = local_boxes[label_mask]
    selected_yolo_boxes = local_boxes_to_yolo_cuda(selected_local_boxes, tile_size)

    label_indices = [int(v) for v in label_indices_tensor.detach().cpu().tolist()]
    yolo_boxes = [
        (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        for row in selected_yolo_boxes.detach().cpu().tolist()
    ]

    return PositiveTileEvalResult(
        ok=True,
        reason="ok",
        source_visible_ratio=source_visible,
        label_indices=label_indices,
        yolo_boxes=yolo_boxes,
        ambiguous_count=0,
    )


def expanded_boxes_cuda(
    boxes: torch.Tensor,
    margin: int,
) -> torch.Tensor:
    """
    将 bbox 向四周扩大 margin 像素。
    """
    if margin <= 0:
        return boxes

    m = float(margin)
    delta = torch.tensor([-m, -m, m, m], device=boxes.device, dtype=torch.float32)
    return boxes + delta


def tile_intersects_any_box_cuda(
    boxes: torch.Tensor,
    x0: int,
    y0: int,
    tile_size: int,
    device: torch.device,
    margin: int = 0,
) -> bool:
    """
    判断 tile 是否与任意 bbox 相交。

    用于负样本过滤。
    margin > 0 时，会先扩大所有标注框。
    """
    if boxes.numel() == 0:
        return False

    tile_box = make_tile_box_cuda(x0, y0, tile_size, device)
    check_boxes = expanded_boxes_cuda(boxes, margin)

    inter = intersection_area_cuda(check_boxes, tile_box)
    hit = torch.any(inter > 0)

    return bool(hit.detach().cpu().item())


def filter_candidate_tiles_without_boxes_cuda(
    boxes: torch.Tensor,
    candidates_xy: torch.Tensor,
    tile_size: int,
    margin: int,
) -> torch.Tensor:
    """
    批量过滤候选负样本 tile。

    candidates_xy:
        shape = [M, 2]
        每行是 [x0, y0]

    返回：
        valid_mask, shape = [M]
        True 表示该 tile 不与任何 expanded bbox 相交。
    """
    if candidates_xy.numel() == 0:
        return torch.empty((0,), device=candidates_xy.device, dtype=torch.bool)

    if boxes.numel() == 0:
        return torch.ones(
            (candidates_xy.shape[0],), device=candidates_xy.device, dtype=torch.bool
        )

    check_boxes = expanded_boxes_cuda(boxes, margin)

    x0 = candidates_xy[:, 0]
    y0 = candidates_xy[:, 1]
    x1 = x0 + tile_size
    y1 = y0 + tile_size

    tile_boxes = torch.stack([x0, y0, x1, y1], dim=1).to(dtype=torch.float32)

    # tile_boxes: [M, 4]
    # check_boxes: [N, 4]
    # 计算 [M, N] 相交关系
    ix1 = torch.maximum(tile_boxes[:, None, 0], check_boxes[None, :, 0])
    iy1 = torch.maximum(tile_boxes[:, None, 1], check_boxes[None, :, 1])
    ix2 = torch.minimum(tile_boxes[:, None, 2], check_boxes[None, :, 2])
    iy2 = torch.minimum(tile_boxes[:, None, 3], check_boxes[None, :, 3])

    iw = (ix2 - ix1).clamp_min(0)
    ih = (iy2 - iy1).clamp_min(0)
    inter = iw * ih

    intersects = torch.any(inter > 0, dim=1)
    return ~intersects
