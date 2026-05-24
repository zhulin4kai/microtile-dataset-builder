from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def geojson_path(tmp_path: Path) -> Path:
    path = tmp_path / "case.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "poly",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[10, 20], [80, 20], [80, 90], [10, 90], [10, 20]]
                            ],
                        },
                        "properties": {},
                    },
                    {
                        "type": "Feature",
                        "properties": {"id": "multi"},
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": [
                                [[[100, 100], [130, 100], [130, 120], [100, 120]]],
                                [[[140, 140], [170, 140], [170, 170], [140, 170]]],
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": [[0, 0]]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class FakeSlideReader:
    def __init__(self, path: str):
        self.path = path
        self.dimensions = (512, 512)
        self.closed = False

    def clamp_origin(self, x0: int, y0: int, tile_size: int) -> tuple[int, int]:
        return (
            max(0, min(x0, self.dimensions[0] - tile_size)),
            max(0, min(y0, self.dimensions[1] - tile_size)),
        )

    def read_tile(self, x0: int, y0: int, tile_size: int) -> Image.Image:
        arr = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        arr[:, :] = (180, 40, 120)
        return Image.fromarray(arr)

    def close(self) -> None:
        self.closed = True
