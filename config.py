from pathlib import Path

# 路径配置
TARGET_DIR = Path(r"E:/slices/lung3/5.4标注")
OUTPUT_DIR = Path(r"E:/dataset")


# 基础切图配置
LEVEL = 0
TILE_SIZE = 768

# 正样本 coverage-driven sampling
POS_TARGET_COVERAGE = 2
POS_SOURCE_MARGIN = 32
POS_MAX_TRIES_PER_ANN = 80
POS_MAX_TOTAL_TRIES_FACTOR = 120

# 正负样本比例：
NEG_POS_RATIO = 1


# 数据集划分
SPLIT_RATIOS = {
    "train": 0.75,
    "val": 0.25,
}

SPLIT_MIN_SLIDES = {
    "train": 1,
    "val": 1,
}

# ann 数量平衡是主目标，slide 数量平衡是辅助目标
SPLIT_ANN_WEIGHT = 1.0
SPLIT_SLIDE_WEIGHT = 0.15

MANUAL_SPLIT = {
    "train": [],
    "val": [],
}

RANDOM_SEED = 42


# 颜色增强配置
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


# 标签与样本过滤配置
CLASS_ID = 0

# 主目标可见比例。低于这个值，正样本候选作废并重新采样。
SOURCE_MIN_VISIBLE_RATIO = 0.85

# patch 内其他目标达到这个可见比例，就写入 YOLO 标签。
LABEL_MIN_VISIBLE_RATIO = 0.15

# 低于这个可见比例，认为只是极小边缘碎片，允许忽略。
IGNORE_MAX_VISIBLE_RATIO = 0.03

# 如果目标处于 IGNORE_MAX_VISIBLE_RATIO 与 LABEL_MIN_VISIBLE_RATIO 之间，
# 且裁剪后的框仍大于 MIN_CLIPPED_BOX_SIZE，则认为是 ambiguous patch，直接丢弃重采。
MIN_CLIPPED_BOX_SIZE = 8


# 负样本配置
# 负样本避开标注框时，额外扩大的安全边界，单位 level 0 像素。
NEG_SAFE_MARGIN = 128

# 负样本中心优先随机采样时的初始半径比例。
# 例如 0.20 表示先在 slide 宽高较小边的 20% 半径范围内采样。
NEG_INITIAL_RADIUS_RATIO = 0.20

# 采样失败后，每轮扩大的半径比例。
NEG_RADIUS_GROWTH_RATIO = 0.15

# 负样本候选最大重试次数。
NEG_MAX_TRIES_PER_SLIDE = 200_000

# CUDA tissue ratio 判断阈值。
# 低于这个值的 tile 认为白背景过多，不保存为负样本。
TISSUE_RATIO_THRESHOLD = 0.20


# 图像写入配置
IMAGE_EXT = ".jpg"
JPEG_QUALITY = 95


# 并行与 CUDA 配置
NUM_WORKERS = 4

CUDA_DEVICE = "cuda:0"
REQUIRE_CUDA = True

TORCH_FLOAT_DTYPE = "float32"
