from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from tools import check_yolo_dataset, render_seg_labels


def test_check_yolo_dataset_pair_collection_and_label_parsing(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_dir / "a.jpg")
    Image.new("RGB", (8, 8), "white").save(image_dir / "b.png")
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.25 0.25\nbad line\n", encoding="utf-8")

    pairs = check_yolo_dataset.collect_image_label_pairs(image_dir, label_dir)
    labels = check_yolo_dataset.read_yolo_label(label_dir / "a.txt")

    assert pairs == [(image_dir / "a.jpg", label_dir / "a.txt")]
    assert labels == [(0, 0.5, 0.5, 0.25, 0.25)]


def test_check_yolo_dataset_sampling_and_drawing(tmp_path):
    pairs = [(tmp_path / f"{idx}.jpg", tmp_path / f"{idx}.txt") for idx in range(4)]

    assert check_yolo_dataset.sample_pairs(pairs, 10, random.Random(1)) == pairs
    assert len(check_yolo_dataset.sample_pairs(pairs, 2, random.Random(1))) == 2
    assert check_yolo_dataset.clamp(10, 0, 5) == 5

    image = Image.new("RGB", (64, 64), "white")
    pos = check_yolo_dataset.draw_labels(image, [(0, 0.5, 0.5, 0.25, 0.25)], "a.jpg", "pos")
    neg = check_yolo_dataset.draw_labels(image, [], "b.jpg", "neg")

    assert pos.size == image.size
    assert neg.size == image.size
    assert np.any(np.array(pos) != np.array(neg))


def test_save_checked_samples_writes_rendered_examples(monkeypatch, tmp_path):
    image_path = tmp_path / "tile.jpg"
    label_path = tmp_path / "tile.txt"
    Image.new("RGB", (32, 32), "white").save(image_path)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    monkeypatch.setattr(check_yolo_dataset, "CHECK_OUTPUT_DIR", tmp_path / "checked")

    check_yolo_dataset.save_checked_samples([(image_path, label_path)], "train", "pos")

    assert (tmp_path / "checked" / "train" / "pos" / "0001_tile.jpg").is_file()


def test_check_yolo_dataset_main_reports_integrity_and_writes_samples(monkeypatch, tmp_path, capsys):
    dataset_dir = tmp_path / "dataset"
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)

    Image.new("RGB", (32, 32), "white").save(train_images / "pos.jpg")
    Image.new("RGB", (32, 32), "white").save(train_images / "neg.jpg")
    Image.new("RGB", (32, 32), "white").save(train_images / "missing_label.jpg")
    (train_labels / "pos.txt").write_text("0 1.2 0.5 0.25 0.25\n", encoding="utf-8")
    (train_labels / "neg.txt").write_text("", encoding="utf-8")
    (train_labels / "missing_image.txt").write_text("", encoding="utf-8")

    monkeypatch.setattr(check_yolo_dataset, "DATASET_DIR", dataset_dir)
    monkeypatch.setattr(check_yolo_dataset, "CHECK_OUTPUT_DIR", tmp_path / "checked")
    monkeypatch.setattr(check_yolo_dataset, "SAMPLES_PER_SPLIT_POS", 10)
    monkeypatch.setattr(check_yolo_dataset, "SAMPLES_PER_SPLIT_NEG", 10)

    check_yolo_dataset.main()

    output = capsys.readouterr().out
    assert "images without label" in output
    assert "labels without image" in output
    assert "invalid YOLO coordinates" in output
    assert "image dir not found" in output
    assert (tmp_path / "checked" / "train" / "pos" / "0001_pos.jpg").is_file()
    assert (tmp_path / "checked" / "train" / "neg" / "0001_neg.jpg").is_file()


def test_render_seg_label_validation(tmp_path):
    label_path = tmp_path / "seg.txt"
    label_path.write_text("1 0.1 0.2 0.3 0.2 0.1 0.4\n", encoding="utf-8")

    segments = render_seg_labels.read_yolo_seg_label(label_path)

    assert segments[0][0] == 1
    assert segments[0][1].shape == (3, 2)

    for text, message in [
        ("1 0.1 0.2", "too few"),
        ("1 0.1 0.2 0.3 0.4 0.5 0.6 0.7", "odd number"),
        ("1 0.1 0.2 nan 0.4 0.5 0.6", "non-finite"),
        ("1 0.1 0.2 1.2 0.4 0.5 0.6", "out of"),
    ]:
        bad = tmp_path / f"{message}.txt"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            render_seg_labels.read_yolo_seg_label(bad)


def test_render_seg_image_lookup_base_loading_and_render(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), "white").save(image_dir / "tile.jpg")
    monkeypatch.setattr(render_seg_labels, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(render_seg_labels, "CANVAS_SIZE", 32)
    monkeypatch.setattr(render_seg_labels, "OVERLAY_ON_IMAGE", True)

    assert render_seg_labels.find_image_path("tile") == image_dir / "tile.jpg"
    assert render_seg_labels.find_image_path("missing") is None
    assert render_seg_labels.load_base_image("tile").size == (32, 32)

    rendered = render_seg_labels.render_segments(
        Image.new("RGB", (32, 32), "white"),
        [(0, np.array([[0.1, 0.1], [0.8, 0.1], [0.1, 0.8]], dtype=np.float32))],
    )

    assert rendered.size == (32, 32)
    assert np.any(np.array(rendered) != 255)


def test_render_seg_main_writes_outputs_and_counts_bad_files(monkeypatch, tmp_path, capsys):
    label_dir = tmp_path / "labels"
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "renders"
    label_dir.mkdir()
    image_dir.mkdir()
    Image.new("RGB", (16, 16), "white").save(image_dir / "valid.jpg")
    (label_dir / "valid.txt").write_text("0 0.1 0.1 0.8 0.1 0.1 0.8\n", encoding="utf-8")
    (label_dir / "empty.txt").write_text("", encoding="utf-8")
    (label_dir / "bad.txt").write_text("0 0.1 0.2\n", encoding="utf-8")

    monkeypatch.setattr(render_seg_labels, "LABEL_DIR", label_dir)
    monkeypatch.setattr(render_seg_labels, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(render_seg_labels, "RENDER_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(render_seg_labels, "CANVAS_SIZE", 32)
    monkeypatch.setattr(render_seg_labels, "MAX_FILES", None)
    monkeypatch.setattr(render_seg_labels, "OVERLAY_ON_IMAGE", True)

    render_seg_labels.main()

    output = capsys.readouterr().out
    assert "empty label files: 1" in output
    assert "bad files: 1" in output
    assert (output_dir / "valid_render.jpg").is_file()
    assert (output_dir / "empty_render.jpg").is_file()
