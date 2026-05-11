"""
YOLO 数据集抽查脚本。

运行方式：
    python tools/check_yolo_dataset.py

功能：
1. 从 images/train、images/val 随机抽正样本；
2. 从空 label 中随机抽负样本；
3. 正样本画 YOLO bbox；
4. 统计 train 中各 variant 数量（颜色增强检查）；
5. 输出到 CHECK_OUTPUT_DIR，方便人工查看。
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# =========================
# 配置
# =========================

DATASET_DIR = Path(r"E:/dataset")
CHECK_OUTPUT_DIR = Path(r"E:/dataset-analysis-results")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SAMPLES_PER_SPLIT_POS = 100
SAMPLES_PER_SPLIT_NEG = 100

RANDOM_SEED = 42

CLASS_NAMES = {
    0: "micropapillary",
}

BOX_WIDTH = 3


# =========================
# 主流程
# =========================


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    for split_name in ("train", "val"):
        print(f"[INFO] checking split: {split_name}")

        image_dir = DATASET_DIR / "images" / split_name
        label_dir = DATASET_DIR / "labels" / split_name

        if not image_dir.exists():
            print(f"[WARN] image dir not found: {image_dir}")
            continue

        if not label_dir.exists():
            print(f"[WARN] label dir not found: {label_dir}")
            continue

        pairs = collect_image_label_pairs(image_dir, label_dir)

        pos_pairs = []
        neg_pairs = []

        for image_path, label_path in pairs:
            labels = read_yolo_label(label_path)
            if labels:
                pos_pairs.append((image_path, label_path))
            else:
                neg_pairs.append((image_path, label_path))

        print(
            f"  total={len(pairs)}, "
            f"pos={len(pos_pairs)}, "
            f"neg={len(neg_pairs)}"
        )

        # 统计 variant 数量
        variant_counts = _count_variants(image_dir)
        if variant_counts:
            vc_str = " ".join(f"{k}={v}" for k, v in sorted(variant_counts.items()))
            print(f"  variants: {vc_str}")

        sampled_pos = sample_pairs(pos_pairs, SAMPLES_PER_SPLIT_POS, rng)
        sampled_neg = sample_pairs(neg_pairs, SAMPLES_PER_SPLIT_NEG, rng)

        save_checked_samples(
            pairs=sampled_pos,
            split_name=split_name,
            sample_type="pos",
        )

        save_checked_samples(
            pairs=sampled_neg,
            split_name=split_name,
            sample_type="neg",
        )

    print(f"[DONE] check results saved to: {CHECK_OUTPUT_DIR}")


def _count_variants(image_dir: Path) -> Dict[str, int]:
    """统计目录中各 variant 的文件数量。"""
    variant_counter: Counter[str] = Counter()

    for p in image_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = p.stem
        # 期望格式: slide_pos_000001_x123_y456_variant
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in {
            "orig",
            "clahe",
            "hsv",
            "brightness_contrast",
            "gamma",
        }:
            variant_counter[parts[1]] += 1

    return dict(variant_counter)


def collect_image_label_pairs(
    image_dir: Path,
    label_dir: Path,
) -> List[Tuple[Path, Path]]:
    image_paths = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    pairs: List[Tuple[Path, Path]] = []

    for image_path in sorted(image_paths):
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[WARN] missing label: {label_path}")
            continue

        pairs.append((image_path, label_path))

    return pairs


def sample_pairs(
    pairs: List[Tuple[Path, Path]],
    sample_count: int,
    rng: random.Random,
) -> List[Tuple[Path, Path]]:
    if len(pairs) <= sample_count:
        return list(pairs)

    return rng.sample(pairs, sample_count)


def save_checked_samples(
    pairs: List[Tuple[Path, Path]],
    split_name: str,
    sample_type: str,
) -> None:
    out_dir = CHECK_OUTPUT_DIR / split_name / sample_type
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (image_path, label_path) in enumerate(pairs, start=1):
        image = Image.open(image_path).convert("RGB")
        labels = read_yolo_label(label_path)

        checked = draw_labels(
            image=image,
            labels=labels,
            image_name=image_path.name,
            sample_type=sample_type,
        )

        out_path = out_dir / f"{idx:04d}_{image_path.name}"
        checked.save(out_path, quality=95)


def read_yolo_label(
    label_path: Path,
) -> List[Tuple[int, float, float, float, float]]:
    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    labels: List[Tuple[int, float, float, float, float]] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            print(
                f"[WARN] invalid label line: {label_path}, line={line_no}, text={line}"
            )
            continue

        cls = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        labels.append((cls, xc, yc, w, h))

    return labels


def draw_labels(
    image: Image.Image,
    labels: List[Tuple[int, float, float, float, float]],
    image_name: str,
    sample_type: str,
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    width, height = img.size

    if labels:
        for cls, xc, yc, bw, bh in labels:
            x1 = int(round((xc - bw / 2) * width))
            y1 = int(round((yc - bh / 2) * height))
            x2 = int(round((xc + bw / 2) * width))
            y2 = int(round((yc + bh / 2) * height))

            x1 = clamp(x1, 0, width - 1)
            y1 = clamp(y1, 0, height - 1)
            x2 = clamp(x2, 0, width - 1)
            y2 = clamp(y2, 0, height - 1)

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(255, 0, 0),
                width=BOX_WIDTH,
            )

            class_name = CLASS_NAMES.get(cls, str(cls))
            text = f"{class_name}"

            text_y = max(0, y1 - 16)
            draw.rectangle(
                [x1, text_y, x1 + 150, text_y + 16],
                fill=(255, 255, 255),
            )
            draw.text(
                (x1 + 2, text_y + 1),
                text,
                fill=(255, 0, 0),
            )

        header = f"POS | boxes={len(labels)} | {image_name}"

    else:
        header = f"NEGATIVE | {image_name}"
        draw.rectangle(
            [0, 0, min(width - 1, 330), 28],
            fill=(255, 255, 255),
        )
        draw.text(
            (8, 8),
            "NEGATIVE SAMPLE",
            fill=(0, 0, 255),
        )

    draw.rectangle(
        [0, height - 24, min(width - 1, 620), height - 1],
        fill=(255, 255, 255),
    )
    draw.text(
        (8, height - 20),
        header,
        fill=(0, 0, 0),
    )

    return img


def clamp(v: int, low: int, high: int) -> int:
    return max(low, min(high, v))


if __name__ == "__main__":
    main()
