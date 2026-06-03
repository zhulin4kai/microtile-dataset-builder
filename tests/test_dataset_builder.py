from __future__ import annotations

import json
import random

import numpy as np
import pytest

import main
import config
from core.annotation import Annotation
import core.cli as cli
import core.discovery as discovery
import core.pipeline as pipeline
import core.reporting as reporting
import core.runtime_config as runtime_config
import core.sampling as sampling
from formats import get_dataset_format
from core.discovery import SlidePair
from core.reporting import DatasetTotals, SlideStats
from core.sampling import TileSample


@pytest.fixture(autouse=True)
def restore_config():
    old_values = {
        "OUTPUT_DIR": config.OUTPUT_DIR,
        "TARGET_DIR": config.TARGET_DIR,
        "TILE_SIZE": config.TILE_SIZE,
        "ENABLE_COLOR_AUGMENT": config.ENABLE_COLOR_AUGMENT,
        "RANDOM_SEED": config.RANDOM_SEED,
        "DATASET_SPLIT_MODE": config.DATASET_SPLIT_MODE,
        "SPLIT_RATIOS": dict(config.SPLIT_RATIOS),
        "DATASET_TASK": config.DATASET_TASK,
        "NUM_WORKERS": config.NUM_WORKERS,
        "DRY_RUN": config.DRY_RUN,
        "MAX_NEG_TRIES_PER_POSITIVE": config.MAX_NEG_TRIES_PER_POSITIVE,
        "WRITE_BUILD_REPORT": config.WRITE_BUILD_REPORT,
        "WRITE_EMPTY_LABEL_FOR_NEGATIVE": config.WRITE_EMPTY_LABEL_FOR_NEGATIVE,
        "SLIDE_BACKEND": config.SLIDE_BACKEND,
        "CUCIM_DEVICE": config.CUCIM_DEVICE,
        "CLASS_ID": config.CLASS_ID,
        "CLASS_NAME": config.CLASS_NAME,
        "IMAGE_EXT": config.IMAGE_EXT,
        "JPEG_QUALITY": config.JPEG_QUALITY,
        "DISCOVER_RECURSIVE": config.DISCOVER_RECURSIVE,
    }
    yield
    for name, value in old_values.items():
        setattr(config, name, value)


def _annotations() -> list[Annotation]:
    return [
        Annotation(
            feature_id="a",
            polygon=[(20, 20), (60, 20), (60, 60), (20, 60)],
            bbox=(20.0, 20.0, 60.0, 60.0),
        ),
        Annotation(
            feature_id="b",
            polygon=[(180, 180), (240, 180), (240, 240), (180, 240)],
            bbox=(180.0, 180.0, 240.0, 240.0),
        ),
    ]


def test_stats_and_totals_convert_worker_results():
    stats = SlideStats(
        "slide",
        raw_pos=1,
        raw_neg=2,
        written_train=3,
        written_val=4,
        failed_neg=5,
        status="ok",
        neg_reject_duplicate=1,
        neg_reject_annotation=2,
        neg_reject_low_tissue=3,
        neg_reject_try_limit=0,
    )
    totals = DatasetTotals()

    totals.add(stats.as_dict())

    assert stats.as_dict()["slide_stem"] == "slide"
    assert totals.raw_pos == 1
    assert totals.raw_neg == 2
    assert totals.written_train == 3
    assert totals.written_val == 4
    assert totals.failed_neg == 5
    assert totals.slides_ok == 1
    assert totals.neg_reject_duplicate == 1
    assert totals.neg_reject_annotation == 2
    assert totals.neg_reject_low_tissue == 3


def test_prepare_output_dirs_rebuilds_dataset_tree(tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    stale = config.OUTPUT_DIR / "old.txt"
    stale.parent.mkdir()
    stale.write_text("old", encoding="utf-8")

    pipeline.prepare_output_dirs()

    assert not stale.exists()
    for split in ("train", "val"):
        assert (config.OUTPUT_DIR / "images" / split).is_dir()
        assert (config.OUTPUT_DIR / "labels" / split).is_dir()


def test_train_slide_count_keeps_single_slide_in_train():
    assert sampling.train_slide_count(1) == 1

    config.DATASET_SPLIT_MODE = "wsi"
    assert sampling.split_for_slide(0, 1) == "train"


def test_train_slide_count_keeps_train_and_val_for_multiple_slides():
    assert sampling.train_slide_count(2) == 1
    assert sampling.train_slide_count(10) == 8


def test_validate_config_rejects_zero_train_ratio(tmp_path):
    config.TARGET_DIR = tmp_path
    config.SPLIT_RATIOS = {"train": 0, "val": 1.0}
    with pytest.raises(ValueError, match="train"):
        pipeline.validate_config()


def test_validate_config_rejects_zero_val_ratio(tmp_path):
    config.TARGET_DIR = tmp_path
    config.SPLIT_RATIOS = {"train": 1.0, "val": 0}
    with pytest.raises(ValueError, match="val"):
        pipeline.validate_config()


def test_validate_config_rejects_negative_class_id(tmp_path):
    config.TARGET_DIR = tmp_path
    config.CLASS_ID = -1
    with pytest.raises(ValueError, match="CLASS_ID"):
        pipeline.validate_config()


def test_validate_config_rejects_empty_class_name(tmp_path):
    config.TARGET_DIR = tmp_path
    config.CLASS_NAME = ""
    with pytest.raises(ValueError, match="CLASS_NAME"):
        pipeline.validate_config()


def test_validate_config_rejects_bad_image_ext(tmp_path):
    config.TARGET_DIR = tmp_path
    config.IMAGE_EXT = ".bmp"
    with pytest.raises(ValueError, match="IMAGE_EXT"):
        pipeline.validate_config()


def test_validate_config_rejects_image_ext_without_dot(tmp_path):
    config.TARGET_DIR = tmp_path
    config.IMAGE_EXT = "jpg"
    with pytest.raises(ValueError, match="IMAGE_EXT"):
        pipeline.validate_config()


def test_validate_config_rejects_bad_jpeg_quality(tmp_path):
    config.TARGET_DIR = tmp_path
    config.JPEG_QUALITY = 101
    with pytest.raises(ValueError, match="JPEG_QUALITY"):
        pipeline.validate_config()


def test_runtime_config_snapshot_contains_class_name_and_discover_recursive():
    snapshot = runtime_config.runtime_config_snapshot()

    assert "CLASS_NAME" in snapshot
    assert "DISCOVER_RECURSIVE" in snapshot
    assert snapshot["CLASS_NAME"] == config.CLASS_NAME


def test_discover_slide_pairs_recurses_by_default(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "case.svs").touch()
    (sub / "case.geojson").write_text('{"features": []}', encoding="utf-8")
    config.TARGET_DIR = tmp_path
    config.DISCOVER_RECURSIVE = True

    pairs, diag = discovery.discover_slide_pairs()

    assert len(pairs) == 1
    assert pairs[0].wsi_path.parent == sub
    assert diag["recursive_discovery"] is True


def test_discover_slide_pairs_can_disable_recursive_discovery(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "case.svs").touch()
    (sub / "case.geojson").write_text('{"features": []}', encoding="utf-8")
    config.TARGET_DIR = tmp_path
    config.DISCOVER_RECURSIVE = False

    pairs, _ = discovery.discover_slide_pairs()

    assert len(pairs) == 0


def test_find_gt_prefers_annotation_in_same_directory(tmp_path):
    same_dir = tmp_path / "wsi"
    other_dir = tmp_path / "other"
    same_dir.mkdir()
    other_dir.mkdir()
    wsi = same_dir / "case.svs"
    wsi.touch()
    same_gt = same_dir / "case.geojson"
    same_gt.touch()
    other_gt = other_dir / "case.geojson"
    other_gt.touch()

    result = discovery.find_gt(wsi, {same_gt, other_gt})

    assert result == same_gt


def test_discover_slide_pairs_reports_ambiguous_annotations(tmp_path):
    wsi_dir = tmp_path / "wsi"
    gt_a_dir = tmp_path / "gt_a"
    gt_b_dir = tmp_path / "gt_b"
    wsi_dir.mkdir()
    gt_a_dir.mkdir()
    gt_b_dir.mkdir()
    (wsi_dir / "case.svs").touch()
    (gt_a_dir / "case.geojson").write_text('{"features": []}', encoding="utf-8")
    (gt_b_dir / "case.geojson").write_text('{"features": []}', encoding="utf-8")
    config.DISCOVER_RECURSIVE = True

    pairs, diagnostics = discovery.discover_slide_pairs(
        wsi_path=tmp_path,
        geojson_path=tmp_path,
    )

    assert len(pairs) == 1
    assert diagnostics["ambiguous_annotations"] == 1


def test_find_gt_supports_regular_and_ome_names(tmp_path):
    regular_wsi = tmp_path / "a.svs"
    regular_gt = tmp_path / "a.geojson"
    ome_wsi = tmp_path / "b.ome.tif"
    ome_gt = tmp_path / "b.json"
    regular_wsi.touch()
    regular_gt.touch()
    ome_wsi.touch()
    ome_gt.touch()

    assert discovery.find_gt(regular_wsi) == regular_gt
    assert discovery.find_gt(ome_wsi) == ome_gt
    assert discovery.find_gt(tmp_path / "missing.svs") is None


def test_discover_slide_pairs_defaults_geojson_to_wsi_path(tmp_path):
    wsi_dir = tmp_path / "wsi"
    wsi_dir.mkdir()
    (wsi_dir / "case.svs").touch()
    (wsi_dir / "case.geojson").write_text('{"features": []}', encoding="utf-8")

    pairs, diagnostics = discovery.discover_slide_pairs(wsi_path=wsi_dir)

    assert pairs == [SlidePair(wsi_dir / "case.svs", wsi_dir / "case.geojson")]
    assert diagnostics["wsi_files"] == 1


def test_discover_slide_pairs_uses_separate_geojson_path(tmp_path):
    wsi_dir = tmp_path / "wsi"
    geojson_dir = tmp_path / "geojson"
    wsi_dir.mkdir()
    geojson_dir.mkdir()
    (wsi_dir / "case.svs").touch()
    (geojson_dir / "case.geojson").write_text('{"features": []}', encoding="utf-8")

    pairs, diagnostics = discovery.discover_slide_pairs(
        wsi_path=wsi_dir,
        geojson_path=geojson_dir,
    )

    assert pairs == [SlidePair(wsi_dir / "case.svs", geojson_dir / "case.geojson")]
    assert diagnostics["unmatched_wsi"] == 0


def test_discover_slide_pairs_accepts_explicit_file_pair(tmp_path):
    wsi_path = tmp_path / "slide-a.svs"
    gt_path = tmp_path / "manual.geojson"
    wsi_path.touch()
    gt_path.write_text('{"features": []}', encoding="utf-8")

    pairs, diagnostics = discovery.discover_slide_pairs(
        wsi_path=wsi_path,
        geojson_path=gt_path,
    )

    assert pairs == [SlidePair(wsi_path, gt_path)]
    assert diagnostics["wsi_files"] == 1
    assert diagnostics["orphan_annotations"] == 0


def test_discover_slide_pairs_ignores_unmatched_and_non_wsi_files(tmp_path):
    config.TARGET_DIR = tmp_path
    (tmp_path / "case.svs").touch()
    (tmp_path / "case.geojson").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "orphan.tif").touch()

    pairs, _ = discovery.discover_slide_pairs()

    assert pairs == [SlidePair(tmp_path / "case.svs", tmp_path / "case.geojson")]


def test_choose_split_uses_fixed_or_random_split():
    assert sampling.choose_split("val", random.Random(1)) == "val"
    assert sampling.choose_split(None, random.Random(1)) == "train"
    assert sampling.choose_split(None, random.Random(2)) == "val"


def test_tile_visibility_and_center_sampling():
    annotations = _annotations()

    assert sampling.has_large_visible_annotation(annotations, 0, 0, 64)
    assert not sampling.has_large_visible_annotation(annotations, 70, 70, 64)

    boxes = sampling.visible_yolo_boxes(annotations, 0, 0, 64)
    assert boxes == pytest.approx([(0.625, 0.625, 0.625, 0.625)])

    x0, y0 = sampling.sample_center_outward_origin(
        random.Random(3), slide_w=128, slide_h=128, tile_size=64, radius_ratio=0.2
    )
    assert 0 <= x0 <= 64
    assert 0 <= y0 <= 64


def test_tile_annotation_queries_can_use_spatial_index():
    annotations = _annotations()

    class SpyIndex:
        def __init__(self):
            self.calls = []

        def query_tile(self, x0, y0, tile_size):
            self.calls.append((x0, y0, tile_size))
            return annotations[:1]

    spy = SpyIndex()

    boxes = sampling.visible_yolo_boxes(
        annotations,
        180,
        180,
        64,
        annotation_index=spy,
    )
    has_annotation = sampling.has_large_visible_annotation(
        annotations,
        180,
        180,
        64,
        annotation_index=spy,
    )

    assert boxes == []
    assert has_annotation is False
    assert spy.calls == [(180, 180, 64), (180, 180, 64)]


def test_tissue_ratio_detects_colored_tissue_and_blank_background():
    tissue = np.zeros((16, 16, 3), dtype=np.uint8)
    tissue[:, :] = (180, 40, 120)
    blank = np.full((16, 16, 3), 255, dtype=np.uint8)

    assert sampling.tissue_ratio(tissue) > 0.9
    assert sampling.tissue_ratio(blank) == 0.0


def test_make_positive_and_negative_samples(monkeypatch):
    from tests.conftest import FakeSlideReader

    config.TILE_SIZE = 64
    reader = FakeSlideReader("slide.svs")
    annotations = _annotations()
    rng = random.Random(1)

    positive = sampling.make_positive_sample(
        reader, annotations, None, annotations[0], "slide", 1
    )
    negative, neg_reason = sampling.try_make_negative_sample(
        reader=reader,
        annotations=annotations,
        slide_stem="slide",
        slide_w=512,
        slide_h=512,
        neg_index=1,
        radius_ratio=0.1,
        rng=rng,
        seen=set(),
    )
    duplicate, dup_reason = sampling.try_make_negative_sample(
        reader=reader,
        annotations=[],
        slide_stem="slide",
        slide_w=512,
        slide_h=512,
        neg_index=1,
        radius_ratio=0.1,
        rng=random.Random(1),
        seen={(187, 260)},
    )
    # 负样本不能覆盖病灶区域
    overlap, overlap_reason = sampling.try_make_negative_sample(
        reader=reader,
        annotations=_annotations()[:1],
        slide_stem="slide",
        slide_w=128,
        slide_h=128,
        neg_index=1,
        radius_ratio=0.0,
        rng=random.Random(1),
        seen=set(),
    )

    assert positive.stem == "slide_pos_000001_x8_y8"
    assert positive.boxes
    assert negative is not None
    assert negative.boxes == []
    assert neg_reason is None
    assert duplicate is None
    assert dup_reason == "duplicate"
    assert overlap is None
    assert overlap_reason == "annotation"


def test_negative_sample_rejects_annotation_overlap_and_blank_tissue(monkeypatch):
    from tests.conftest import FakeSlideReader

    class BlankSlideReader(FakeSlideReader):
        def read_tile(self, x0: int, y0: int, tile_size: int):
            from PIL import Image

            return Image.new("RGB", (tile_size, tile_size), "white")

    config.TILE_SIZE = 64
    overlapped, overlap_reason = sampling.try_make_negative_sample(
        reader=FakeSlideReader("slide.svs"),
        annotations=_annotations()[:1],
        slide_stem="slide",
        slide_w=128,
        slide_h=128,
        neg_index=1,
        radius_ratio=0.0,
        rng=random.Random(1),
        seen=set(),
    )
    blank, blank_reason = sampling.try_make_negative_sample(
        reader=BlankSlideReader("slide.svs"),
        annotations=[],
        slide_stem="slide",
        slide_w=128,
        slide_h=128,
        neg_index=1,
        radius_ratio=0.0,
        rng=random.Random(1),
        seen=set(),
    )

    assert overlapped is None
    assert overlap_reason == "annotation"
    assert blank is None
    assert blank_reason == "low_tissue"


def test_write_sample_creates_image_and_label(tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    config.ENABLE_COLOR_AUGMENT = False
    stats = SlideStats("slide")
    sample = TileSample(
        stem="tile",
        image=np.full((16, 16, 3), (180, 40, 120), dtype=np.uint8),
        boxes=[(0.5, 0.5, 0.25, 0.25)],
    )

    pipeline.write_sample_for_stats(sample, "train", random.Random(1), stats)

    assert stats.written_train == 1
    assert (config.OUTPUT_DIR / "images" / "train" / "tile_orig.jpg").is_file()
    assert (config.OUTPUT_DIR / "labels" / "train" / "tile_orig.txt").read_text(
        encoding="utf-8"
    ).startswith("0 0.500000")


def test_write_positive_and_negative_sample_counts(tmp_path):
    from tests.conftest import FakeSlideReader

    config.OUTPUT_DIR = tmp_path / "out"
    config.TILE_SIZE = 64
    config.ENABLE_COLOR_AUGMENT = False
    reader = FakeSlideReader("slide.svs")
    stats = SlideStats("slide")

    raw_pos = sampling.write_positive_samples(
        reader=reader,
        annotations=_annotations()[:1],
        annotation_index=None,
        slide_stem="slide",
        split_name="train",
        rng=random.Random(1),
        stats=stats,
        writer=get_dataset_format(),
    )
    raw_neg = sampling.write_negative_samples(
        reader=reader,
        annotations=[],
        annotation_index=None,
        slide_stem="slide",
        slide_w=512,
        slide_h=512,
        target_count=1,
        split_name="val",
        rng=random.Random(2),
        stats=stats,
        writer=get_dataset_format(),
    )

    assert raw_pos == 1
    assert raw_neg == 1
    assert stats.written_train == 1
    assert stats.written_val == 1


def test_process_slide_pair_builds_positive_and_negative_tiles(monkeypatch, tmp_path, geojson_path):
    from tests.conftest import FakeSlideReader

    config.OUTPUT_DIR = tmp_path / "out"
    config.TILE_SIZE = 64
    config.ENABLE_COLOR_AUGMENT = False
    config.RANDOM_SEED = 7
    monkeypatch.setattr(pipeline, "SlideReader", FakeSlideReader)

    result = pipeline.process_slide_pair(str(tmp_path / "case.svs"), str(geojson_path), 0, "train")

    assert result["raw_pos"] == 3
    assert result["raw_neg"] == 3
    assert result["written_train"] == 6
    assert len(list((config.OUTPUT_DIR / "images" / "train").glob("*.jpg"))) == 6
    labels = list((config.OUTPUT_DIR / "labels" / "train").glob("*.txt"))
    assert len(labels) == 6
    assert any(path.read_text(encoding="utf-8").strip() for path in labels)
    assert any(not path.read_text(encoding="utf-8").strip() for path in labels)


def test_process_slide_pair_returns_zero_stats_for_empty_annotations(tmp_path):
    empty = tmp_path / "empty.geojson"
    empty.write_text('{"features": []}', encoding="utf-8")

    result = pipeline.process_slide_pair(str(tmp_path / "case.svs"), str(empty), 0, "train")

    assert result["slide_stem"] == "case"
    assert result["raw_pos"] == 0
    assert result["raw_neg"] == 0
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "empty_annotations"


def test_split_for_slide_and_summary_output(capsys, tmp_path):
    config.DATASET_SPLIT_MODE = "wsi"
    config.OUTPUT_DIR = tmp_path / "out"

    assert sampling.split_for_slide(0, 1) == "train"
    assert sampling.split_for_slide(1, 1) == "val"
    totals = DatasetTotals(
        raw_pos=1, raw_neg=1, written_train=2, failed_neg=3,
        slides_ok=1, slides_skipped=0, slides_failed=0,
    )
    reporting.print_summary(totals)

    captured = capsys.readouterr().out
    assert "WSI 处理结果：成功=1 跳过=0 失败=0" in captured
    assert "原始正样本：1" in captured
    assert "负样本不足" in captured


def test_run_slide_pairs_uses_executor_and_accumulates(monkeypatch, tmp_path):
    submitted_configs = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, wsi_path, gt_path, index, split_name, runtime_config):
            self.calls.append((fn, wsi_path, gt_path, index, split_name))
            submitted_configs.append(runtime_config)
            return FakeFuture(
                {
                    "slide_stem": f"slide{index}",
                    "raw_pos": 1,
                    "raw_neg": 1,
                    "written_train": 2 if split_name == "train" else 0,
                    "written_val": 2 if split_name == "val" else 0,
                    "failed_neg": 0,
                    "status": "ok",
                    "skip_reason": "",
                    "error": "",
                    "neg_reject_duplicate": 0,
                    "neg_reject_annotation": 0,
                    "neg_reject_low_tissue": 0,
                    "neg_reject_try_limit": 0,
                }
            )

    pair_a = SlidePair(tmp_path / "a.svs", tmp_path / "a.geojson")
    pair_b = SlidePair(tmp_path / "b.svs", tmp_path / "b.geojson")
    config.DATASET_SPLIT_MODE = "wsi"
    config.NUM_WORKERS = 2
    config.OUTPUT_DIR = tmp_path / "custom-out"
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(pipeline, "as_completed", lambda futures: list(futures))

    totals, slide_results = pipeline.run_slide_pairs([pair_a, pair_b])

    assert totals.raw_pos == 2
    assert totals.raw_neg == 2
    assert totals.written_train + totals.written_val == 4
    assert len(slide_results) == 2
    assert submitted_configs
    assert all(item["OUTPUT_DIR"] == str(config.OUTPUT_DIR) for item in submitted_configs)


def test_main_handles_invalid_task_and_empty_dataset(monkeypatch, tmp_path, capsys):
    config.DATASET_TASK = "seg"
    config.TARGET_DIR = tmp_path
    with pytest.raises(NotImplementedError):
        pipeline.validate_config()

    config.DATASET_TASK = "detect"
    config.TARGET_DIR = tmp_path / "input"
    config.OUTPUT_DIR = tmp_path / "out"
    config.TARGET_DIR.mkdir()

    cli.main()

    assert "没有找到 WSI-annotation 配对" in capsys.readouterr().out
    assert not (config.OUTPUT_DIR / "images" / "train").exists()


def test_main_empty_dataset_does_not_clear_existing_output(tmp_path, capsys):
    config.TARGET_DIR = tmp_path / "input"
    config.OUTPUT_DIR = tmp_path / "out"
    config.TARGET_DIR.mkdir()
    config.OUTPUT_DIR.mkdir()
    keep_file = config.OUTPUT_DIR / "keep.txt"
    keep_file.write_text("do not delete", encoding="utf-8")

    cli.main()

    assert "没有找到 WSI-annotation 配对" in capsys.readouterr().out
    assert keep_file.is_file()
    assert not (config.OUTPUT_DIR / "images" / "train").exists()


def test_main_cli_overrides_input_paths_and_dry_run(tmp_path, capsys):
    wsi_dir = tmp_path / "wsi"
    geojson_dir = tmp_path / "geojson"
    output_dir = tmp_path / "out"
    wsi_dir.mkdir()
    geojson_dir.mkdir()
    (wsi_dir / "case.svs").touch()
    (geojson_dir / "case.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[10, 10], [30, 10], [30, 30], [10, 30], [10, 10]]
                            ],
                        },
                        "properties": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config.TARGET_DIR = tmp_path / "missing"

    cli.main([
        "--wsi-path",
        str(wsi_dir),
        "--geojson-path",
        str(geojson_dir),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ])

    output = capsys.readouterr().out
    assert "找到 1 个 WSI-annotation 配对" in output
    assert "DRY_RUN 已启用" in output
    assert config.OUTPUT_DIR == output_dir


def test_build_dataset_entrypoint_exports_builder_main():
    assert main.main is cli.main


def test_write_dataset_yaml_uses_default_class_name(tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    config.OUTPUT_DIR.mkdir()

    pipeline.write_dataset_metadata()

    content = (config.OUTPUT_DIR / "dataset.yaml").read_text(encoding="utf-8")
    assert f"  {config.CLASS_ID}: {config.CLASS_NAME}" in content


def test_write_dataset_yaml_uses_configured_class_id_and_name(tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    config.OUTPUT_DIR.mkdir()
    config.CLASS_ID = 3
    config.CLASS_NAME = "lesion"

    pipeline.write_dataset_metadata()

    content = (config.OUTPUT_DIR / "dataset.yaml").read_text(encoding="utf-8")
    assert "  3: lesion" in content
    assert "  0: micropapillary" not in content


def test_validate_config_rejects_missing_target_dir(tmp_path):
    config.TARGET_DIR = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="TARGET_DIR"):
        pipeline.validate_config()


def test_validate_config_rejects_bad_tile_size(tmp_path):
    config.TARGET_DIR = tmp_path
    config.TILE_SIZE = 0
    with pytest.raises(ValueError, match="TILE_SIZE"):
        pipeline.validate_config()


def test_validate_config_rejects_bad_split_ratios(tmp_path):
    config.TARGET_DIR = tmp_path
    config.SPLIT_RATIOS = {"train": 0.5, "val": 0.3}
    with pytest.raises(ValueError, match="SPLIT_RATIOS"):
        pipeline.validate_config()


def test_validate_config_rejects_bad_split_mode(tmp_path):
    config.TARGET_DIR = tmp_path
    config.DATASET_SPLIT_MODE = "slide"
    with pytest.raises(ValueError, match="DATASET_SPLIT_MODE"):
        pipeline.validate_config()


def test_validate_config_passes_for_good_config(tmp_path):
    config.TARGET_DIR = tmp_path
    config.TILE_SIZE = 512
    config.NUM_WORKERS = 1
    config.SPLIT_RATIOS = {"train": 0.75, "val": 0.25}
    config.DATASET_SPLIT_MODE = "patch"
    config.DATASET_TASK = "detect"
    pipeline.validate_config()


def test_dry_run_does_not_write_files(tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    config.ENABLE_COLOR_AUGMENT = False
    config.DRY_RUN = True
    stats = SlideStats("slide")
    sample = TileSample(
        stem="tile",
        image=np.full((16, 16, 3), (180, 40, 120), dtype=np.uint8),
        boxes=[(0.5, 0.5, 0.25, 0.25)],
    )

    pipeline.write_sample_for_stats(sample, "train", random.Random(1), stats)

    assert stats.written_train == 1
    assert not (config.OUTPUT_DIR / "images").exists()
    assert not (config.OUTPUT_DIR / "labels").exists()


def test_dry_run_skips_output_dir_prep(tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    config.DRY_RUN = True
    config.OUTPUT_DIR.mkdir()

    pipeline.prepare_output_dirs()

    assert not (config.OUTPUT_DIR / "images").exists()


def test_main_dry_run_uses_preflight_without_opening_slide(monkeypatch, tmp_path, geojson_path, capsys):
    config.TARGET_DIR = tmp_path
    config.OUTPUT_DIR = tmp_path / "out"
    config.DRY_RUN = True
    (tmp_path / "case.svs").touch()

    class FailingSlideReader:
        def __init__(self, path):
            raise AssertionError("DRY_RUN 不应打开 WSI")

    monkeypatch.setattr(pipeline, "SlideReader", FailingSlideReader)

    cli.main()

    output = capsys.readouterr().out
    assert "DRY_RUN 已启用" in output
    assert "原始正样本：3" in output
    assert not (config.OUTPUT_DIR / "images").exists()


def test_negative_sample_try_limit_stops_and_records(monkeypatch, tmp_path):
    from tests.conftest import FakeSlideReader

    config.OUTPUT_DIR = tmp_path / "out"
    config.TILE_SIZE = 64
    config.ENABLE_COLOR_AUGMENT = False
    config.MAX_NEG_TRIES_PER_POSITIVE = 2

    reader = FakeSlideReader("slide.svs")
    stats = SlideStats("slide")

    raw_neg = sampling.write_negative_samples(
        reader=reader,
        annotations=_annotations(),
        annotation_index=None,
        slide_stem="slide",
        slide_w=64,
        slide_h=64,
        target_count=10,
        split_name="train",
        rng=random.Random(1),
        stats=stats,
        writer=get_dataset_format(),
    )

    assert raw_neg < 10
    assert stats.failed_neg > 0
    assert stats.neg_reject_try_limit > 0


def test_process_slide_pair_handles_exception(monkeypatch, tmp_path, geojson_path):
    class BrokenReader:
        def __init__(self, path):
            self.dimensions = (512, 512)

        def clamp_origin(self, x0, y0, ts):
            return x0, y0

        def read_tile(self, x0, y0, ts):
            raise RuntimeError("simulated crash")

        def close(self):
            pass

    config.OUTPUT_DIR = tmp_path / "out"
    config.TILE_SIZE = 64
    monkeypatch.setattr(pipeline, "SlideReader", BrokenReader)

    result = pipeline.process_slide_pair(str(tmp_path / "case.svs"), str(geojson_path), 0, "train")

    assert result["status"] == "failed"
    assert "simulated crash" in result["error"]


def test_write_build_report_writes_json(monkeypatch, tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    config.OUTPUT_DIR.mkdir()
    config.WRITE_BUILD_REPORT = True
    config.DRY_RUN = False

    diagnostics = {"total_files_in_target": 5, "wsi_files": 2, "unmatched_wsi": 0, "orphan_annotations": 0}
    totals = DatasetTotals(raw_pos=2, raw_neg=2, written_train=2, slides_ok=1)
    slide_results = [{
        "slide_stem": "case", "raw_pos": 2, "raw_neg": 2,
        "written_train": 2, "written_val": 0, "failed_neg": 0,
        "status": "ok", "skip_reason": "", "error": "",
        "neg_reject_duplicate": 0, "neg_reject_annotation": 0,
        "neg_reject_low_tissue": 0, "neg_reject_try_limit": 0,
    }]

    reporting.write_build_report(diagnostics, totals, slide_results)

    report_path = config.OUTPUT_DIR / "build_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["totals"]["raw_pos"] == 2
    assert report["slides"][0]["slide_stem"] == "case"
    assert "config" in report
    assert report["说明"] == "YOLO detect 数据集构建报告"


def test_discover_slide_pairs_diagnostics(tmp_path):
    config.TARGET_DIR = tmp_path

    (tmp_path / "a.svs").touch()
    (tmp_path / "a.geojson").touch()
    (tmp_path / "b.tif").touch()
    (tmp_path / "orphan.geojson").touch()
    (tmp_path / "notes.txt").touch()

    pairs, diag = discovery.discover_slide_pairs()

    assert len(pairs) == 1
    assert pairs[0].wsi_path.name == "a.svs"
    assert diag["unmatched_wsi"] == 1
    assert diag["orphan_annotations"] == 1


def test_process_slide_pair_logs_failed_status(monkeypatch, tmp_path, geojson_path, capsys):
    class BrokenReader:
        def __init__(self, path):
            self.dimensions = (512, 512)

        def clamp_origin(self, x0, y0, ts):
            return x0, y0

        def read_tile(self, x0, y0, ts):
            raise RuntimeError("simulated crash")

        def close(self):
            pass

    config.OUTPUT_DIR = tmp_path / "out"
    config.TILE_SIZE = 64
    monkeypatch.setattr(pipeline, "SlideReader", BrokenReader)

    pipeline.process_slide_pair(str(tmp_path / "case.svs"), str(geojson_path), 0, "train")

    captured = capsys.readouterr().out
    assert "[失败]" in captured
    assert "simulated crash" in captured
