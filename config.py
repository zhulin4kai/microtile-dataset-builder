from pathlib import Path

TARGET_DIR = Path("/home/yuzhoukai/projects/WSIs")
OUTPUT_DIR = Path("/home/yuzhoukai/projects/ultralytics/dataset")

TILE_SIZE = 1024

SPLIT_RATIOS = {
    "train": 0.75,
    "val": 0.25,
}

# "wsi"：WSI 级别划分 train/val
# "patch"：patch group 级别划分 train/val
DATASET_SPLIT_MODE = "patch"

# "detect"：YOLO bbox detect 标签
# "seg"：保留字段，本次默认不用
DATASET_TASK = "detect"

ENABLE_COLOR_AUGMENT = True
RANDOM_SEED = 42
CLASS_ID = 0
IMAGE_EXT = ".jpg"
JPEG_QUALITY = 90
WRITE_EMPTY_LABEL_FOR_NEGATIVE = True

CLAHE_CLIP_LIMIT_RANGE = (1.5, 2.5)
CLAHE_TILE_GRID_SIZE = (8, 8)
HSV_HUE_SHIFT_LIMIT = (-4, 4)
HSV_SAT_SHIFT_LIMIT = (-12, 12)
HSV_VAL_SHIFT_LIMIT = (-10, 10)
BRIGHTNESS_LIMIT = (-0.08, 0.08)
CONTRAST_LIMIT = (-0.10, 0.10)
GAMMA_LIMIT = (90, 110)
