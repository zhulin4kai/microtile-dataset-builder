"""
CUDA tissue ratio 判断模块。

输入是 OpenSlide/PIL 读取出的 RGB tile。
转换为 torch CUDA Tensor 后进行组织区域比例判断。

这个模块只做白背景过滤，不做染色归一化。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def pil_rgb_to_cuda_tensor(
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """
    PIL RGB Image -> CUDA Tensor。

    返回：
        uint8 Tensor, shape = [H, W, 3]
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.array(image, dtype=np.uint8, copy=True)

    tensor = torch.from_numpy(arr).to(device=device, non_blocking=True)

    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"Invalid RGB tensor shape: {tuple(tensor.shape)}")

    return tensor


def tissue_ratio_cuda_from_tensor(
    rgb_tensor: torch.Tensor,
    white_brightness_threshold: float = 230.0,
    saturation_threshold: float = 8.0,
    min_channel_threshold: float = 220.0,
) -> float:
    """
    计算 tissue ratio。

    rgb_tensor:
        uint8 / float Tensor
        shape = [H, W, 3]
        device = CUDA

    判断逻辑：
    1. 背景通常接近白色，brightness 高；
    2. H&E 组织区域通常有一定 RGB 通道差异；
    3. 对深染核、粉染胞质均尽量保留。
    """
    if rgb_tensor.ndim != 3 or rgb_tensor.shape[-1] != 3:
        raise ValueError(f"Invalid RGB tensor shape: {tuple(rgb_tensor.shape)}")

    x = rgb_tensor.to(dtype=torch.float32)

    max_c = torch.amax(x, dim=-1)
    min_c = torch.amin(x, dim=-1)
    brightness = torch.mean(x, dim=-1)
    saturation = max_c - min_c

    not_white_by_brightness = brightness < white_brightness_threshold
    not_white_by_channel = min_c < min_channel_threshold
    has_color = saturation > saturation_threshold

    tissue_mask = (not_white_by_brightness | not_white_by_channel) & has_color

    ratio = tissue_mask.to(dtype=torch.float32).mean()
    return float(ratio.detach().cpu().item())


def tissue_ratio_cuda(
    image: Image.Image,
    device: torch.device,
    white_brightness_threshold: float = 230.0,
    saturation_threshold: float = 8.0,
    min_channel_threshold: float = 220.0,
) -> float:
    """
    PIL Image -> CUDA -> tissue ratio。
    """
    tensor = pil_rgb_to_cuda_tensor(image, device=device)

    return tissue_ratio_cuda_from_tensor(
        tensor,
        white_brightness_threshold=white_brightness_threshold,
        saturation_threshold=saturation_threshold,
        min_channel_threshold=min_channel_threshold,
    )


def is_tissue_tile_cuda(
    image: Image.Image,
    device: torch.device,
    min_tissue_ratio: float,
    white_brightness_threshold: float = 230.0,
    saturation_threshold: float = 8.0,
    min_channel_threshold: float = 220.0,
) -> bool:
    """
    判断 tile 是否包含足够组织区域。
    """
    ratio = tissue_ratio_cuda(
        image=image,
        device=device,
        white_brightness_threshold=white_brightness_threshold,
        saturation_threshold=saturation_threshold,
        min_channel_threshold=min_channel_threshold,
    )

    return ratio >= min_tissue_ratio


def batch_tissue_ratio_cuda_from_numpy(
    batch_rgb: np.ndarray,
    device: torch.device,
    white_brightness_threshold: float = 230.0,
    saturation_threshold: float = 8.0,
    min_channel_threshold: float = 220.0,
) -> torch.Tensor:
    """
    批量 tissue ratio。

    batch_rgb:
        np.ndarray
        shape = [B, H, W, 3]
        dtype = uint8

    返回：
        torch.Tensor
        shape = [B]
        device = CUDA

    这个函数给后续批量负样本候选预留。
    """
    if batch_rgb.ndim != 4 or batch_rgb.shape[-1] != 3:
        raise ValueError(f"Invalid batch RGB shape: {batch_rgb.shape}")

    x = torch.as_tensor(batch_rgb, device=device).to(dtype=torch.float32)

    max_c = torch.amax(x, dim=-1)
    min_c = torch.amin(x, dim=-1)
    brightness = torch.mean(x, dim=-1)
    saturation = max_c - min_c

    not_white_by_brightness = brightness < white_brightness_threshold
    not_white_by_channel = min_c < min_channel_threshold
    has_color = saturation > saturation_threshold

    tissue_mask = (not_white_by_brightness | not_white_by_channel) & has_color

    return tissue_mask.to(dtype=torch.float32).mean(dim=(1, 2))
