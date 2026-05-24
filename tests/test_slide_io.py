from __future__ import annotations

import types

from PIL import Image

from slide_io import SlideReader


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
