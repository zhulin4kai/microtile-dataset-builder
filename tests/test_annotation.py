from __future__ import annotations

import json

import pytest

from annotation import (
    bbox_intersects_tile,
    bbox_to_yolo,
    bbox_visible_ratio,
    clip_bbox_to_tile,
    load_annotations,
)


def test_load_annotations_supports_polygon_and_multipolygon(geojson_path):
    annotations = load_annotations(geojson_path)

    assert [ann.feature_id for ann in annotations] == ["poly", "multi"]
    assert annotations[0].bbox == (10.0, 20.0, 80.0, 90.0)
    assert annotations[1].bbox == (100.0, 100.0, 170.0, 170.0)
    assert len(annotations[1].polygon) == 8


def test_load_annotations_uses_generated_id_for_missing_feature_id(tmp_path):
    path = tmp_path / "missing_id.geojson"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 0]]],
                        },
                        "properties": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    annotations = load_annotations(path)

    assert len(annotations) == 1
    assert annotations[0].feature_id == "ann_0"


@pytest.mark.parametrize(
    ("bbox", "origin", "expected"),
    [
        ((10.0, 10.0, 20.0, 20.0), (0, 0), True),
        ((20.0, 20.0, 30.0, 30.0), (0, 0), False),
        ((-5.0, -5.0, 1.0, 1.0), (0, 0), True),
    ],
)
def test_bbox_intersects_tile_edge_cases(bbox, origin, expected):
    assert bbox_intersects_tile(bbox, origin[0], origin[1], 20) is expected


def test_clip_bbox_and_convert_to_yolo_coordinates():
    clipped = clip_bbox_to_tile((10.0, 20.0, 110.0, 220.0), 0, 0, 128)

    assert clipped == (10.0, 20.0, 110.0, 128.0)
    assert bbox_visible_ratio((10.0, 20.0, 110.0, 220.0), clipped) == pytest.approx(0.54)
    assert bbox_to_yolo(clipped, 0, 0, 128) == pytest.approx(
        (0.46875, 0.578125, 0.78125, 0.84375)
    )


def test_clip_bbox_returns_none_for_non_overlap_and_zero_area_ratio():
    assert clip_bbox_to_tile((100.0, 100.0, 120.0, 120.0), 0, 0, 64) is None
    assert bbox_visible_ratio((1.0, 1.0, 1.0, 5.0), (1.0, 1.0, 1.0, 2.0)) == 0.0
