# -*- coding: utf-8 -*-
"""
OpenSlide-based WSI tile reader.
"""

from __future__ import annotations

import openslide
from PIL import Image


class SlideReader:
    def __init__(self, path: str):
        self._slide = openslide.OpenSlide(path)
        self._w, self._h = self._slide.dimensions

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._w, self._h

    def clamp_origin(self, x0: int, y0: int, tile_size: int) -> tuple[int, int]:
        return (
            max(0, min(x0, self._w - tile_size)),
            max(0, min(y0, self._h - tile_size)),
        )

    def read_tile(self, x0: int, y0: int, tile_size: int) -> Image.Image:
        """Read a tile at Level-0 coordinates, return PIL RGB."""
        x0, y0 = self.clamp_origin(x0, y0, tile_size)
        region = self._slide.read_region((x0, y0), 0, (tile_size, tile_size))
        return region.convert("RGB")

    def close(self):
        self._slide.close()
