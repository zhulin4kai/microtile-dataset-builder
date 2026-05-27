from __future__ import annotations

import json
import types

import pytest

from annotation import (
    Annotation,
    AnnotationIndex,
    bbox_intersects_tile,
    bbox_to_yolo,
    bbox_visible_ratio,
    clip_bbox_to_tile,
    load_annotations,
    validate_annotation_file,
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


def test_annotation_index_queries_only_intersecting_candidates():
    annotations = [
        Annotation("a", [(0, 0), (20, 0), (20, 20), (0, 0)], (0.0, 0.0, 20.0, 20.0)),
        Annotation(
            "b",
            [(100, 100), (130, 100), (130, 130), (100, 100)],
            (100.0, 100.0, 130.0, 130.0),
        ),
    ]
    index = AnnotationIndex(annotations)

    assert index.query_tile(0, 0, 64) == [annotations[0]]
    assert index.query_tile(90, 90, 64) == [annotations[1]]
    assert index.query_tile(40, 40, 32) == []


def test_annotation_index_handles_empty_and_geometry_results():
    empty = AnnotationIndex([])
    assert empty.query_tile(0, 0, 64) == []

    annotation = Annotation("a", [(0, 0), (20, 0), (20, 20), (0, 0)], (0.0, 0.0, 20.0, 20.0))
    index = AnnotationIndex([annotation])
    index._tree = types.SimpleNamespace(query=lambda tile: [index._geometries[0]])

    assert index.query_tile(0, 0, 64) == [annotation]


def test_load_annotations_skips_zero_area_bbox(tmp_path):
    path = tmp_path / "zero_area.geojson"
    path.write_text(
        json.dumps({
            "features": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [0, 0], [0, 0], [0, 0]]],
                    },
                    "properties": {},
                },
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 0]]],
                    },
                    "properties": {},
                },
            ]
        }),
        encoding="utf-8",
    )
    annotations = load_annotations(path)
    assert len(annotations) == 1
    assert annotations[0].bbox == (0.0, 0.0, 3.0, 3.0)


def test_load_annotations_skips_non_numeric_coords(tmp_path):
    path = tmp_path / "bad_coords.geojson"
    path.write_text(
        json.dumps({
            "features": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[["a", "b"], [0, 0], [3, 3], [0, 0]]],
                    },
                    "properties": {},
                },
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 0]]],
                    },
                    "properties": {},
                },
            ]
        }),
        encoding="utf-8",
    )
    annotations = load_annotations(path)
    assert len(annotations) == 1


def test_load_annotations_skips_malformed_points(tmp_path):
    path = tmp_path / "malformed_points.geojson"
    path.write_text(
        json.dumps({
            "features": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0], [3, 0], [3, 3], [0, 0]]],
                    },
                    "properties": {},
                },
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 0]]],
                    },
                    "properties": {},
                },
            ]
        }),
        encoding="utf-8",
    )

    annotations = load_annotations(path)

    assert len(annotations) == 1


def test_load_annotations_skips_polygon_with_few_points(tmp_path):
    path = tmp_path / "few_points.geojson"
    path.write_text(
        json.dumps({
            "features": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 1]]],
                    },
                    "properties": {},
                },
            ]
        }),
        encoding="utf-8",
    )
    annotations = load_annotations(path)
    assert len(annotations) == 0


def test_validate_annotation_file_valid(tmp_path):
    path = tmp_path / "valid.geojson"
    path.write_text(
        json.dumps({
            "features": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 0]]],
                    },
                    "properties": {},
                },
            ]
        }),
        encoding="utf-8",
    )
    result = validate_annotation_file(path)
    assert result["annotation_count"] == 1
    assert result["skipped_features"] == 0
    assert result["error"] is None


def test_validate_annotation_file_counts_skipped(tmp_path):
    path = tmp_path / "mixed.geojson"
    path.write_text(
        json.dumps({
            "features": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [0, 0], [0, 0], [0, 0]]],
                    },
                    "properties": {},
                },
                {
                    "geometry": {"type": "LineString", "coordinates": [[0, 0]]},
                    "properties": {},
                },
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 0]]],
                    },
                    "properties": {},
                },
            ]
        }),
        encoding="utf-8",
    )
    result = validate_annotation_file(path)
    assert result["annotation_count"] == 1
    assert result["skipped_features"] == 2


def test_validate_annotation_file_handles_bad_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    result = validate_annotation_file(path)
    assert result["annotation_count"] == 0
    assert result["error"] is not None
