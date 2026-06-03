# -*- coding: utf-8 -*-
"""WSI tile reader，支持 OpenSlide 和可选 cuCIM 后端。"""

from __future__ import annotations

import numpy as np
from PIL import Image

import config


class OpenSlideBackend:
    name = "openslide"

    def __init__(self, path: str):
        import openslide

        self._slide = openslide.OpenSlide(path)
        self.dimensions = self._slide.dimensions

    def read_region(self, x0: int, y0: int, tile_size: int) -> Image.Image:
        region = self._slide.read_region((x0, y0), 0, (tile_size, tile_size))
        return region.convert("RGB")

    def close(self) -> None:
        self._slide.close()


class CuCIMBackend:
    name = "cucim"

    def __init__(self, path: str):
        try:
            cucim = __import__("cucim", fromlist=["CuImage"])
            CuImage = cucim.CuImage
        except ImportError as exc:
            raise ImportError(
                "cuCIM backend requires the cucim package, which is not installed."
            ) from exc

        self._slide = CuImage(path)
        level_dimensions = self._slide.resolutions["level_dimensions"]
        self.dimensions = tuple(level_dimensions[0])

    def read_region(self, x0: int, y0: int, tile_size: int) -> Image.Image:
        kwargs = {}
        device = getattr(config, "CUCIM_DEVICE", None)
        if device:
            kwargs["device"] = device

        region = self._slide.read_region(
            location=(x0, y0),
            size=(tile_size, tile_size),
            level=0,
            **kwargs,
        )
        if hasattr(region, "get"):
            region = region.get()
        arr = np.asarray(region)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    def close(self) -> None:
        if hasattr(self._slide, "close"):
            self._slide.close()


def _make_backend(path: str):
    backend = getattr(config, "SLIDE_BACKEND", "openslide")
    if backend in {"auto", "cucim"}:
        try:
            return CuCIMBackend(path)
        except Exception:
            # auto 模式用于部署兼容：cuCIM 不可用时退回 OpenSlide。
            if backend == "cucim":
                raise
    return OpenSlideBackend(path)


class SlideReader:
    def __init__(self, path: str):
        self._backend = _make_backend(path)
        self._slide = self._backend._slide
        self._w, self._h = self._backend.dimensions

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._w, self._h

    def clamp_origin(self, x0: int, y0: int, tile_size: int) -> tuple[int, int]:
        return (
            max(0, min(x0, self._w - tile_size)),
            max(0, min(y0, self._h - tile_size)),
        )

    def read_tile(self, x0: int, y0: int, tile_size: int) -> Image.Image:
        """读取 Level-0 tile，返回 PIL RGB 图像。"""
        x0, y0 = self.clamp_origin(x0, y0, tile_size)
        return self._backend.read_region(x0, y0, tile_size)

    def close(self) -> None:
        self._backend.close()
