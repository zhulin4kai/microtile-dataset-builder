# -*- coding: utf-8 -*-
"""
OpenSlide 读取封装。

职责：
1. 每张 WSI 只打开一次；
2. 固定从 level 0 读取 tile；
3. 保证 tile 左上角不越界；
4. 输出 PIL RGB Image。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import openslide
from PIL import Image


@dataclass(frozen=True)
class TileReadResult:
    image: Image.Image
    x0: int
    y0: int


class SlideReader:
    def __init__(self, svs_path: Path, level: int, tile_size: int):
        self.svs_path = Path(svs_path)
        self.level = int(level)
        self.tile_size = int(tile_size)
        self.slide: openslide.OpenSlide | None = None

    def __enter__(self) -> "SlideReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        if self.slide is not None:
            return

        if not self.svs_path.exists():
            raise FileNotFoundError(f"SVS not found: {self.svs_path}")

        self.slide = openslide.OpenSlide(str(self.svs_path))

        if self.level < 0 or self.level >= self.slide.level_count:
            raise ValueError(
                f"Invalid level={self.level}, "
                f"level_count={self.slide.level_count}, "
                f"svs={self.svs_path}"
            )

    def close(self) -> None:
        if self.slide is not None:
            self.slide.close()
            self.slide = None

    @property
    def dimensions(self) -> Tuple[int, int]:
        self._require_open()
        assert self.slide is not None
        return self.slide.dimensions

    @property
    def width(self) -> int:
        return self.dimensions[0]

    @property
    def height(self) -> int:
        return self.dimensions[1]

    def clamp_origin(self, x0: int, y0: int) -> Tuple[int, int]:
        """
        将 level 0 tile 左上角限制在有效范围内。
        """
        w, h = self.dimensions

        max_x = max(0, w - self.tile_size)
        max_y = max(0, h - self.tile_size)

        x0 = max(0, min(int(x0), max_x))
        y0 = max(0, min(int(y0), max_y))

        return x0, y0

    def read_tile(self, x0: int, y0: int) -> TileReadResult:
        """
        从 WSI 读取一个 tile。

        注意：
        OpenSlide 的 location 坐标永远是 level 0 坐标。
        size 是当前读取 level 下的像素尺寸。
        这里固定 level = 0，所以 size = 768x768。
        """
        self._require_open()
        assert self.slide is not None

        x0, y0 = self.clamp_origin(x0, y0)

        image = self.slide.read_region(
            location=(x0, y0),
            level=self.level,
            size=(self.tile_size, self.tile_size),
        ).convert("RGB")

        return TileReadResult(
            image=image,
            x0=x0,
            y0=y0,
        )

    def print_info(self) -> None:
        self._require_open()
        assert self.slide is not None

        print(f"[INFO] slide: {self.svs_path.name}")
        print(f"[INFO] dimensions: {self.slide.dimensions}")
        print(f"[INFO] level_count: {self.slide.level_count}")
        print(f"[INFO] level_dimensions: {self.slide.level_dimensions}")
        print(f"[INFO] level_downsamples: {self.slide.level_downsamples}")

    def _require_open(self) -> None:
        if self.slide is None:
            raise RuntimeError("SlideReader is not opened.")
