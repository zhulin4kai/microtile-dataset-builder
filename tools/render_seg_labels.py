"""
YOLO segmentation label 可视化工具。

**注意：当前 dataset builder 只生成 YOLO detect label（bbox）。**
**本脚本仅用于可视化已有外部 YOLO segmentation label。**
**不应被用于检查本项目默认构建输出。**

功能：
1. 读取 labels/*.txt
2. 如果有同名 image，则叠加到原图上
3. 如果没有 image，则画到白底图上
4. 输出到 RENDER_OUTPUT_DIR

YOLO-seg label 格式：
class x1 y1 x2 y2 x3 y3 ...

运行方式：
    python tools/render_seg_labels.py
    python tools/render_seg_labels.py --label-dir /path/to/labels --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

# =========================
# 配置（默认值，可被 CLI 覆盖）
# =========================

IMAGE_DIR = Path(r"/data/dataset_yolo/images/train")
LABEL_DIR = Path(r"/data/dataset_yolo/labels/train")
RENDER_OUTPUT_DIR = Path(r"/data/dataset_yolo_seg_render/train")

CANVAS_SIZE = 1024

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

MAX_FILES = 10

OVERLAY_ON_IMAGE = True

ALPHA = 0.35


# =========================
# 主流程
# =========================


def main(argv: Sequence[str] | None = None) -> None:
    global IMAGE_DIR, LABEL_DIR, RENDER_OUTPUT_DIR, CANVAS_SIZE, MAX_FILES, OVERLAY_ON_IMAGE

    parser = argparse.ArgumentParser(
        description="YOLO segmentation label 可视化工具"
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--canvas-size", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--overlay-on-image",
        dest="overlay_on_image",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-overlay",
        dest="overlay_on_image",
        action="store_false",
    )
    args = parser.parse_args([] if argv is None else list(argv))

    if args.image_dir is not None:
        IMAGE_DIR = args.image_dir
    if args.label_dir is not None:
        LABEL_DIR = args.label_dir
    if args.output_dir is not None:
        RENDER_OUTPUT_DIR = args.output_dir
    if args.canvas_size is not None:
        CANVAS_SIZE = args.canvas_size
    if args.max_files is not None:
        MAX_FILES = args.max_files
    if args.overlay_on_image is not None:
        OVERLAY_ON_IMAGE = args.overlay_on_image

    RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(LABEL_DIR.glob("*.txt"))

    if MAX_FILES is not None:
        label_paths = label_paths[:MAX_FILES]

    print(f"[INFO] labels: {len(label_paths)}")

    bad_files = 0
    empty_files = 0

    for idx, label_path in enumerate(label_paths, start=1):
        try:
            segments = read_yolo_seg_label(label_path)

            if not segments:
                empty_files += 1

            image = load_base_image(label_path.stem)
            rendered = render_segments(image, segments)

            out_path = RENDER_OUTPUT_DIR / f"{label_path.stem}_render.jpg"
            rendered.save(out_path, quality=95)

        except Exception as e:
            bad_files += 1
            print(f"[BAD] {label_path}: {e}")

        if idx % 100 == 0:
            print(f"[INFO] rendered {idx}/{len(label_paths)}")

    print("[DONE]")
    print(f"empty label files: {empty_files}")
    print(f"bad files: {bad_files}")
    print(f"output: {RENDER_OUTPUT_DIR}")


# =========================
# 读取 label
# =========================


def read_yolo_seg_label(label_path: Path) -> List[Tuple[int, np.ndarray]]:
    """
    返回：
        [
          (class_id, points_xy_norm[N, 2]),
          ...
        ]
    """
    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    result: List[Tuple[int, np.ndarray]] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()

        if len(parts) < 7:
            raise ValueError(f"line {line_no}: too few columns: {len(parts)}")

        class_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]

        if len(coords) % 2 != 0:
            raise ValueError(f"line {line_no}: odd number of coords")

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)

        if pts.shape[0] < 3:
            raise ValueError(f"line {line_no}: less than 3 points")

        if np.any(~np.isfinite(pts)):
            raise ValueError(f"line {line_no}: non-finite coords")

        if np.any(pts < 0.0) or np.any(pts > 1.0):
            raise ValueError(f"line {line_no}: coords out of [0, 1]")

        result.append((class_id, pts))

    return result


# =========================
# 图像加载
# =========================


def load_base_image(stem: str) -> Image.Image:
    if OVERLAY_ON_IMAGE:
        image_path = find_image_path(stem)
        if image_path is not None:
            img = Image.open(image_path).convert("RGB")
            if img.size != (CANVAS_SIZE, CANVAS_SIZE):
                img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.BILINEAR)
            return img

    return Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))


def find_image_path(stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = IMAGE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# =========================
# 渲染
# =========================


def render_segments(
    image: Image.Image,
    segments: List[Tuple[int, np.ndarray]],
) -> Image.Image:
    base = np.array(image.convert("RGB"))

    overlay = base.copy()
    outline = base.copy()

    for class_id, pts_norm in segments:
        pts = np.round(pts_norm * CANVAS_SIZE).astype(np.int32)

        pts[:, 0] = np.clip(pts[:, 0], 0, CANVAS_SIZE - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, CANVAS_SIZE - 1)

        pts_cv = pts.reshape(-1, 1, 2)

        cv2.fillPoly(
            overlay,
            [pts_cv],
            color=(255, 255, 0),
        )

        cv2.polylines(
            outline,
            [pts_cv],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        x_text = int(pts[0, 0])
        y_text = int(pts[0, 1])
        cv2.putText(
            outline,
            str(class_id),
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    blended = cv2.addWeighted(overlay, ALPHA, base, 1.0 - ALPHA, 0)
    result = cv2.addWeighted(outline, 1.0, blended, 0.0, 0)

    return Image.fromarray(result)


if __name__ == "__main__":
    main()
