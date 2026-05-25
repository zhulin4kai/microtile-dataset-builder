from __future__ import annotations

import json
import random

import numpy as np
import pytest

import build_dataset
import config
import dataset_builder
from annotation import Annotation
from dataset_builder import DatasetTotals, SlidePair, SlideStats, TileSample


@pytest.fixture(autouse=True)
def restore_config():
    old_values = {
        "OUTPUT_DIR": config.OUTPUT_DIR,
        "TARGET_DIR": config.TARGET_DIR,
        "TILE_SIZE": config.TILE_SIZE,
        "ENABLE_COLOR_AUGMENT": config.ENABLE_COLOR_AUGMENT,
        "RANDOM_SEED": config.RANDOM_SEED,
        "DATASET_SPLIT_MODE": config.DATASET_SPLIT_MODE,
        "DATASET_TASK": config.DATASET_TASK,
        "NUM_WORKERS": config.NUM_WORKERS,
        "DRY_RUN": config.DRY_RUN,
        "MAX_NEG_TRIES_PER_POSITIVE": config.MAX_NEG_TRIES_PER_POSITIVE,
        "WRITE_BUILD_REPORT": config.WRITE_BUILD_REPORT,
        "WRITE_EMPTY_LABEL_FOR_NEGATIVE": config.WRITE_EMPTY_LABEL_FOR_NEGATIVE,
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

    dataset_builder._prepare_output_dirs()

    assert not stale.exists()
    for split in ("train", "val"):
        assert (config.OUTPUT_DIR / "images" / split).is_dir()
        assert (config.OUTPUT_DIR / "labels" / split).is_dir()


def test_find_gt_supports_regular_and_ome_names(tmp_path):
    regular_wsi = tmp_path / "a.svs"
    regular_gt = tmp_path / "a.geojson"
    ome_wsi = tmp_path / "b.ome.tif"
    ome_gt = tmp_path / "b.json"
    regular_wsi.touch()
    regular_gt.touch()
    ome_wsi.touch()
    ome_gt.touch()

    assert dataset_builder._find_gt(regular_wsi) == regular_gt
    assert dataset_builder._find_gt(ome_wsi) == ome_gt
    assert dataset_builder._find_gt(tmp_path / "missing.svs") is None


def test_discover_slide_pairs_ignores_unmatched_and_non_wsi_files(tmp_path):
    config.TARGET_DIR = tmp_path
    (tmp_path / "case.svs").touch()
    (tmp_path / "case.geojson").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "orphan.tif").touch()

    pairs, _ = dataset_builder._discover_slide_pairs()

    assert pairs == [SlidePair(tmp_path / "case.svs", tmp_path / "case.geojson")]


def test_choose_split_uses_fixed_or_random_split():
    assert dataset_builder._choose_split("val", random.Random(1)) == "val"
    assert dataset_builder._choose_split(None, random.Random(1)) == "train"
    assert dataset_builder._choose_split(None, random.Random(2)) == "val"


def test_tile_visibility_and_center_sampling():
    annotations = _annotations()

    assert dataset_builder._has_large_visible_annotation(annotations, 0, 0, 64)
    assert not dataset_builder._has_large_visible_annotation(annotations, 70, 70, 64)

    boxes = dataset_builder._visible_yolo_boxes(annotations, 0, 0, 64)
    assert boxes == pytest.approx([(0.625, 0.625, 0.625, 0.625)])

    x0, y0 = dataset_builder._sample_center_outward_origin(
        random.Random(3), slide_w=128, slide_h=128, tile_size=64, radius_ratio=0.2
    )
    assert 0 <= x0 <= 64
    assert 0 <= y0 <= 64


def test_tissue_ratio_detects_colored_tissue_and_blank_background():
    tissue = np.zeros((16, 16, 3), dtype=np.uint8)
    tissue[:, :] = (180, 40, 120)
    blank = np.full((16, 16, 3), 255, dtype=np.uint8)

    assert dataset_builder._tissue_ratio(tissue) > 0.9
    assert dataset_builder._tissue_ratio(blank) == 0.0


def test_make_positive_and_negative_samples(monkeypatch):
    from tests.conftest import FakeSlideReader

    config.TILE_SIZE = 64
    reader = FakeSlideReader("slide.svs")
    annotations = _annotations()
    rng = random.Random(1)

    positive = dataset_builder._make_positive_sample(reader, annotations, annotations[0], "slide", 1)
    negative, neg_reason = dataset_builder._try_make_negative_sample(
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
    duplicate, dup_reason = dataset_builder._try_make_negative_sample(
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
    overlap, overlap_reason = dataset_builder._try_make_negative_sample(
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
    overlapped, overlap_reason = dataset_builder._try_make_negative_sample(
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
    blank, blank_reason = dataset_builder._try_make_negative_sample(
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

    dataset_builder._write_sample(sample, "train", random.Random(1), stats)

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

    raw_pos = dataset_builder._write_positive_samples(
        reader=reader,
        annotations=_annotations()[:1],
        slide_stem="slide",
        split_name="train",
        rng=random.Random(1),
        stats=stats,
    )
    raw_neg = dataset_builder._write_negative_samples(
        reader=reader,
        annotations=[],
        slide_stem="slide",
        slide_w=512,
        slide_h=512,
        target_count=1,
        split_name="val",
        rng=random.Random(2),
        stats=stats,
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
    monkeypatch.setattr(dataset_builder, "SlideReader", FakeSlideReader)

    result = dataset_builder.process_slide_pair(str(tmp_path / "case.svs"), str(geojson_path), 0, "train")

    assert result["raw_pos"] == 2
    assert result["raw_neg"] == 2
    assert result["written_train"] == 4
    assert len(list((config.OUTPUT_DIR / "images" / "train").glob("*.jpg"))) == 4
    labels = list((config.OUTPUT_DIR / "labels" / "train").glob("*.txt"))
    assert len(labels) == 4
    assert any(path.read_text(encoding="utf-8").strip() for path in labels)
    assert any(not path.read_text(encoding="utf-8").strip() for path in labels)


def test_process_slide_pair_returns_zero_stats_for_empty_annotations(tmp_path):
    empty = tmp_path / "empty.geojson"
    empty.write_text('{"features": []}', encoding="utf-8")

    result = dataset_builder.process_slide_pair(str(tmp_path / "case.svs"), str(empty), 0, "train")

    assert result["slide_stem"] == "case"
    assert result["raw_pos"] == 0
    assert result["raw_neg"] == 0
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "empty_annotations"


def test_split_for_slide_and_summary_output(capsys, tmp_path):
    config.DATASET_SPLIT_MODE = "wsi"
    config.OUTPUT_DIR = tmp_path / "out"

    assert dataset_builder._split_for_slide(0, 1) == "train"
    assert dataset_builder._split_for_slide(1, 1) == "val"
    totals = DatasetTotals(
        raw_pos=1, raw_neg=1, written_train=2, failed_neg=3,
        slides_ok=1, slides_skipped=0, slides_failed=0,
    )
    dataset_builder._print_summary(totals)

    captured = capsys.readouterr().out
    assert "WSI 处理结果：成功=1 跳过=0 失败=0" in captured
    assert "原始正样本：1" in captured
    assert "负样本不足" in captured


def test_run_slide_pairs_uses_executor_and_accumulates(monkeypatch, tmp_path):
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

        def submit(self, fn, wsi_path, gt_path, index, split_name):
            self.calls.append((fn, wsi_path, gt_path, index, split_name))
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
    monkeypatch.setattr(dataset_builder, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(dataset_builder, "as_completed", lambda futures: list(futures))

    totals, slide_results = dataset_builder._run_slide_pairs([pair_a, pair_b])

    assert totals.raw_pos == 2
    assert totals.raw_neg == 2
    assert totals.written_train + totals.written_val == 4
    assert len(slide_results) == 2


def test_main_handles_invalid_task_and_empty_dataset(monkeypatch, tmp_path, capsys):
    config.DATASET_TASK = "seg"
    config.TARGET_DIR = tmp_path
    with pytest.raises(NotImplementedError):
        dataset_builder._validate_config()

    config.DATASET_TASK = "detect"
    config.TARGET_DIR = tmp_path / "input"
    config.OUTPUT_DIR = tmp_path / "out"
    config.TARGET_DIR.mkdir()

    dataset_builder.main()

    assert "没有找到 WSI-annotation 配对" in capsys.readouterr().out
    assert (config.OUTPUT_DIR / "images" / "train").is_dir()


def test_build_dataset_entrypoint_exports_builder_main():
    assert build_dataset.main is dataset_builder.main


def test_write_dataset_yaml_uses_default_class_name(tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    config.OUTPUT_DIR.mkdir()

    dataset_builder._write_dataset_yaml()

    content = (config.OUTPUT_DIR / "dataset.yaml").read_text(encoding="utf-8")
    assert "  0: micropapillary" in content


def test_validate_config_rejects_missing_target_dir(tmp_path):
    config.TARGET_DIR = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="TARGET_DIR"):
        dataset_builder._validate_config()


def test_validate_config_rejects_bad_tile_size(tmp_path):
    config.TARGET_DIR = tmp_path
    config.TILE_SIZE = 0
    with pytest.raises(ValueError, match="TILE_SIZE"):
        dataset_builder._validate_config()


def test_validate_config_rejects_bad_split_ratios(tmp_path):
    config.TARGET_DIR = tmp_path
    config.SPLIT_RATIOS = {"train": 0.5, "val": 0.3}
    with pytest.raises(ValueError, match="SPLIT_RATIOS"):
        dataset_builder._validate_config()


def test_validate_config_rejects_bad_split_mode(tmp_path):
    config.TARGET_DIR = tmp_path
    config.DATASET_SPLIT_MODE = "slide"
    with pytest.raises(ValueError, match="DATASET_SPLIT_MODE"):
        dataset_builder._validate_config()


def test_validate_config_passes_for_good_config(tmp_path):
    config.TARGET_DIR = tmp_path
    config.TILE_SIZE = 512
    config.NUM_WORKERS = 1
    config.SPLIT_RATIOS = {"train": 0.75, "val": 0.25}
    config.DATASET_SPLIT_MODE = "patch"
    config.DATASET_TASK = "detect"
    dataset_builder._validate_config()


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

    dataset_builder._write_sample(sample, "train", random.Random(1), stats)

    assert stats.written_train == 1
    assert not (config.OUTPUT_DIR / "images").exists()
    assert not (config.OUTPUT_DIR / "labels").exists()


def test_dry_run_skips_output_dir_prep(tmp_path):
    config.OUTPUT_DIR = tmp_path / "out"
    config.DRY_RUN = True
    config.OUTPUT_DIR.mkdir()

    dataset_builder._prepare_output_dirs()

    assert not (config.OUTPUT_DIR / "images").exists()


def test_main_dry_run_uses_preflight_without_opening_slide(monkeypatch, tmp_path, geojson_path, capsys):
    config.TARGET_DIR = tmp_path
    config.OUTPUT_DIR = tmp_path / "out"
    config.DRY_RUN = True
    (tmp_path / "case.svs").touch()

    class FailingSlideReader:
        def __init__(self, path):
            raise AssertionError("DRY_RUN 不应打开 WSI")

    monkeypatch.setattr(dataset_builder, "SlideReader", FailingSlideReader)

    dataset_builder.main()

    output = capsys.readouterr().out
    assert "DRY_RUN 已启用" in output
    assert "原始正样本：2" in output
    assert not (config.OUTPUT_DIR / "images").exists()


def test_negative_sample_try_limit_stops_and_records(monkeypatch, tmp_path):
    from tests.conftest import FakeSlideReader

    config.OUTPUT_DIR = tmp_path / "out"
    config.TILE_SIZE = 64
    config.ENABLE_COLOR_AUGMENT = False
    config.MAX_NEG_TRIES_PER_POSITIVE = 2

    reader = FakeSlideReader("slide.svs")
    stats = SlideStats("slide")

    raw_neg = dataset_builder._write_negative_samples(
        reader=reader,
        annotations=_annotations(),
        slide_stem="slide",
        slide_w=64,
        slide_h=64,
        target_count=10,
        split_name="train",
        rng=random.Random(1),
        stats=stats,
    )

    assert raw_neg < 10
    assert stats.failed_neg > 0
    assert stats.neg_reject_try_limit > 0


def test_process_slide_pair_handles_exception(monkeypatch, tmp_path, geojson_path):
    import dataset_builder as db

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
    monkeypatch.setattr(db, "SlideReader", BrokenReader)

    result = db.process_slide_pair(str(tmp_path / "case.svs"), str(geojson_path), 0, "train")

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

    dataset_builder._write_build_report(diagnostics, totals, slide_results)

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

    pairs, diag = dataset_builder._discover_slide_pairs()

    assert len(pairs) == 1
    assert pairs[0].wsi_path.name == "a.svs"
    assert diag["unmatched_wsi"] == 1
    assert diag["orphan_annotations"] == 1


def test_process_slide_pair_logs_failed_status(monkeypatch, tmp_path, geojson_path, capsys):
    import dataset_builder as db

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
    monkeypatch.setattr(db, "SlideReader", BrokenReader)

    db.process_slide_pair(str(tmp_path / "case.svs"), str(geojson_path), 0, "train")

    captured = capsys.readouterr().out
    assert "[失败]" in captured
    assert "simulated crash" in captured
