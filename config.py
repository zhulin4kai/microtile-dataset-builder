from pathlib import Path
import os

# 路径配置
TARGET_DIR = Path("/data/wsi/annotations")
OUTPUT_DIR = Path("/data/dataset_yolo")

LEVEL = 0
TILE_SIZE = 768

SPLIT_RATIOS = {
    "train": 0.75,
    "val": 0.25,
}

SPLIT_MIN_SLIDES = {
    "train": 1,
    "val": 1,
}

SPLIT_ANN_WEIGHT = 1.0
SPLIT_SLIDE_WEIGHT = 0.15

MANUAL_SPLIT = {
    "train": [],
    "val": [],
}

RANDOM_SEED = 42
CLASS_ID = 0

# 正样本：确定性快速采样
POS_PATCHES_PER_ANNOTATION = 2
POS_OFFSET_RATIO = 0.18
SOURCE_MIN_VISIBLE_RATIO = 0.70
LABEL_MIN_VISIBLE_RATIO = 0.15
IGNORE_MAX_VISIBLE_RATIO = 0.03
MIN_CLIPPED_BOX_SIZE = 8

# 负样本：raw 正负比例 1:1
NEG_POS_RATIO = 1
NEG_SAFE_MARGIN = 128
NEG_MAX_TRIES_PER_SLIDE = 200_000
TISSUE_RATIO_THRESHOLD = 0.20

# 低分辨率 tissue mask
TISSUE_MASK_LEVEL = -1
TISSUE_MASK_DOWNSAMPLE_TARGET = 64
TISSUE_SAT_THRESHOLD = 20
TISSUE_VAL_THRESHOLD = 220

# 服务器并行
NUM_WORKERS = max(1, min(12, os.cpu_count() or 4))
PROCESS_START_METHOD = "spawn"

# 图像写入
IMAGE_EXT = ".jpg"
JPEG_QUALITY = 90
USE_CV2_JPEG_WRITER = True
WRITE_EMPTY_LABEL_FOR_NEGATIVE = True

# 颜色增强（仅 train 生效，val 只写 orig）
ENABLE_COLOR_AUGMENT = True

COLOR_AUGMENT_SPLITS = {"train"}

COLOR_AUGMENT_VARIANTS = [
    "orig",
    "clahe",
    "hsv",
    "brightness_contrast",
    "gamma",
]

CLAHE_CLIP_LIMIT_RANGE = (1.5, 2.5)
CLAHE_TILE_GRID_SIZE = (8, 8)

HSV_HUE_SHIFT_LIMIT = (-4, 4)
HSV_SAT_SHIFT_LIMIT = (-12, 12)
HSV_VAL_SHIFT_LIMIT = (-10, 10)

BRIGHTNESS_LIMIT = (-0.08, 0.08)
CONTRAST_LIMIT = (-0.10, 0.10)

GAMMA_LIMIT = (90, 110)

# 禁用训练阶段才需要的功能
REQUIRE_CUDA = False

# ── 数据集任务类型 ──────────────────────────────────────────────────
# "box": YOLO detect bbox 标签 (class xc yc w h)
# "seg": YOLO segmentation 标签 (class x1 y1 x2 y2 ...)
DATASET_TASK = "box"

# segmentation 参数（仅 DATASET_TASK == "seg" 时生效）
SEG_MIN_POLYGON_POINTS = 3
SEG_MIN_POLYGON_AREA = 4.0
SEG_SIMPLIFY_EPSILON = 0.0
