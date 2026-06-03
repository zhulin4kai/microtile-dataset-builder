from __future__ import annotations

import builtins
import types

import numpy as np
import pytest
from PIL import Image

import config
from core.slide_io import SlideReader


def test_slide_reader_clamps_reads_and_closes(monkeypatch):
    class FakeOpenSlide:
        def __init__(self, path: str):
            self.path = path
            self.dimensions = (100, 80)
            self.closed = False

        def read_region(self, location, level, size):
            self.location = location
            self.level = level
            self.size = size
            return Image.new("RGBA", size, (1, 2, 3, 255))

        def close(self):
            self.closed = True

    fake_module = types.SimpleNamespace(OpenSlide=FakeOpenSlide)
    monkeypatch.setitem(__import__("sys").modules, "openslide", fake_module)

    reader = SlideReader("case.svs")
    tile = reader.read_tile(99, -10, 32)

    assert reader.dimensions == (100, 80)
    assert reader.clamp_origin(99, -10, 32) == (68, 0)
    assert tile.mode == "RGB"
    assert tile.size == (32, 32)
    assert reader._slide.location == (68, 0)

    reader.close()
    assert reader._slide.closed


def test_slide_reader_can_use_cucim_backend(monkeypatch):
    class FakeCuImage:
        def __init__(self, path: str):
            self.path = path
            self.resolutions = {"level_dimensions": [(120, 90)]}
            self.closed = False
            self.region_calls = []

        def read_region(self, location, size, level, **kwargs):
            self.region_calls.append((location, size, level, kwargs))
            arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
            arr[:, :] = (10, 20, 30, 255)
            return arr

        def close(self):
            self.closed = True

    fake_module = types.SimpleNamespace(CuImage=FakeCuImage)
    monkeypatch.setitem(__import__("sys").modules, "cucim", fake_module)
    monkeypatch.setattr(config, "SLIDE_BACKEND", "cucim", raising=False)
    monkeypatch.setattr(config, "CUCIM_DEVICE", "cuda", raising=False)

    reader = SlideReader("case.svs")
    tile = reader.read_tile(200, -10, 32)

    assert reader.backend_name == "cucim"
    assert reader.dimensions == (120, 90)
    assert tile.mode == "RGB"
    assert np.array(tile)[0, 0].tolist() == [10, 20, 30]
    assert reader._slide.region_calls == [((88, 0), (32, 32), 0, {"device": "cuda"})]

    reader.close()
    assert reader._slide.closed


def test_slide_reader_cucim_handles_gpu_like_array_and_missing_close(monkeypatch):
    class FakeGpuArray:
        def __init__(self, arr):
            self.arr = arr

        def get(self):
            return self.arr

    class FakeCuImage:
        def __init__(self, path: str):
            self.resolutions = {"level_dimensions": [(64, 64)]}

        def read_region(self, location, size, level, **kwargs):
            arr = np.full((size[1], size[0]), 128, dtype=np.uint8)
            return FakeGpuArray(arr)

    fake_module = types.SimpleNamespace(CuImage=FakeCuImage)
    monkeypatch.setitem(__import__("sys").modules, "cucim", fake_module)
    monkeypatch.setattr(config, "SLIDE_BACKEND", "cucim", raising=False)
    monkeypatch.setattr(config, "CUCIM_DEVICE", "", raising=False)

    reader = SlideReader("case.svs")
    tile = reader.read_tile(0, 0, 16)

    assert tile.mode == "RGB"
    assert np.array(tile)[0, 0].tolist() == [128, 128, 128]
    reader.close()


def test_slide_reader_forced_cucim_requires_cucim(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "cucim":
            raise ImportError("blocked cucim")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setattr(config, "SLIDE_BACKEND", "cucim", raising=False)

    with pytest.raises(ImportError):
        SlideReader("case.svs")


def test_slide_reader_auto_falls_back_when_cucim_fails(monkeypatch):
    class BrokenCuImage:
        def __init__(self, path: str):
            raise RuntimeError("cuCIM 初始化失败")

    class FakeOpenSlide:
        def __init__(self, path: str):
            self.dimensions = (80, 80)
            self.closed = False

        def read_region(self, location, level, size):
            return Image.new("RGBA", size, (1, 2, 3, 255))

        def close(self):
            self.closed = True

    monkeypatch.setitem(__import__("sys").modules, "cucim", types.SimpleNamespace(CuImage=BrokenCuImage))
    monkeypatch.setitem(__import__("sys").modules, "openslide", types.SimpleNamespace(OpenSlide=FakeOpenSlide))
    monkeypatch.setattr(config, "SLIDE_BACKEND", "auto", raising=False)

    reader = SlideReader("case.svs")

    assert reader.backend_name == "openslide"
