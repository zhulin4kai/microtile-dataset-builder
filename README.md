# WSI Dataset Builder

从 Whole Slide Image (WSI) 和 GeoJSON 标注中切割图像块，构建可用于视觉模型训练的数据集。

这个项目的目标不是绑定某一种具体模型，而是把 WSI 数据集构建过程拆成稳定的通用流水线：

- 读取 WSI
- 解析病理标注
- 生成正负样本 tile
- 做训练/验证划分
- 写出图像、标签和数据集元信息
- 为不同视觉模型的数据格式保留适配层

## 工作流

输入：

- WSI 文件：`.svs`、`.tif`、`.tiff`、`.ndpi`、`.mrxs`
- 标注文件：QuPath 风格 GeoJSON / JSON

输出：

- `images/train`
- `images/val`
- `labels/train`
- `labels/val`
- `dataset.yaml`
- `build_report.json`

构建流程：

1. 扫描 WSI 和 GeoJSON，按文件名配对。
2. 校验 annotation 是否可解析、是否为空、是否存在孤立标注。
3. 从每个 annotation 中心附近切正样本 tile。
4. 从 WSI 中心向外采样负样本 tile，并过滤空白背景和 annotation 重叠区域。
5. 按 WSI 或 patch 级别划分 train / val。
6. 写出图像、标签和构建报告。

## 目录结构

```text
.
├── main.py                   # 命令行入口
├── config.py                 # 用户配置
├── core/                     # WSI 数据集构建主干
│   ├── annotation.py         # GeoJSON 解析与 bbox 工具
│   ├── slide_io.py           # WSI 读取后端
│   ├── discovery.py          # WSI / annotation 配对发现
│   ├── sampling.py           # 正负样本采样与 split
│   ├── augment.py            # 图像增强与编码
│   ├── pipeline.py           # 构建流程编排
│   ├── reporting.py          # 统计与构建报告
│   └── runtime_config.py     # 多进程 worker 配置同步
├── formats/                  # 视觉模型数据格式适配层
│   ├── base.py               # DatasetFormat 协议
│   └── yolo_detect.py        # 检测数据集格式适配器
├── tools/                    # 数据检查和辅助脚本
└── tests/                    # 测试
```

## 安装

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖包括：

- OpenSlide Python
- Pillow
- NumPy
- OpenCV
- Shapely
- 可选 cuCIM 后端

如果使用 OpenSlide，请确保系统中已安装 OpenSlide 动态库。

## 配置

主要配置在 [config.py](config.py)：

```python
TARGET_DIR = Path("/path/to/WSIs")
OUTPUT_DIR = Path("/path/to/output/dataset")

TILE_SIZE = 1024
NUM_WORKERS = 8

SPLIT_RATIOS = {
    "train": 0.75,
    "val": 0.25,
}

DATASET_SPLIT_MODE = "patch"  # "patch" 或 "wsi"
ENABLE_COLOR_AUGMENT = True
RANDOM_SEED = 42
```

常用配置说明：

- `TARGET_DIR`：WSI 和 annotation 所在目录。
- `OUTPUT_DIR`：构建后的数据集输出目录。
- `TILE_SIZE`：切图尺寸。
- `DATASET_SPLIT_MODE`：
  - `patch`：每个 tile 独立划分 train / val。
  - `wsi`：按整张 WSI 划分 train / val，避免同一张 WSI 同时出现在两个 split。
- `DISCOVER_RECURSIVE`：是否递归扫描输入目录。
- `SLIDE_BACKEND`：`auto`、`openslide` 或 `cucim`。
- `DRY_RUN`：只做输入检查和样本数量估算，不写图像和标签。

## 运行

使用配置文件中的路径：

```bash
python main.py
```

临时指定输入和输出：

```bash
python main.py \
  --wsi-path /path/to/wsi-or-dir \
  --geojson-path /path/to/geojson-or-dir \
  --output-dir /path/to/output
```

只做预检查：

```bash
python main.py \
  --wsi-path /path/to/wsi-or-dir \
  --geojson-path /path/to/geojson-or-dir \
  --dry-run
```

## 标注约定

项目读取 GeoJSON 中的 `Polygon` 和 `MultiPolygon`：

- `Polygon` 会生成一个 annotation。
- `MultiPolygon` 中的每个 polygon 会拆成独立 annotation。
- 当前主流程使用 annotation 的 bbox 参与切图、可见性判断和标签写出。
- 无效 polygon、空 annotation、无法解析的文件会进入构建诊断信息。

WSI 和 annotation 默认按 stem 匹配：

```text
case_001.svs      <-> case_001.geojson
case_002.ome.tif  <-> case_002.geojson
```

当递归扫描中出现多个同名 annotation 时，构建报告会记录歧义计数。

## 输出格式适配层

`formats/` 是视觉模型数据格式的适配层。

主干流水线只负责生成 `TileSample`，不直接关心最终标签文件该怎么写。具体输出格式由 `DatasetFormat` 负责：

```python
class DatasetFormat(Protocol):
    name: str

    def validate_config(self) -> None: ...
    def prepare_output_dirs(self, output_dir: Path, dry_run: bool) -> None: ...
    def write_sample(self, sample: TileSample, split_name: str, rng: random.Random) -> int: ...
    def write_metadata(self, output_dir: Path, dry_run: bool) -> None: ...
    def variant_count(self) -> int: ...
```

如果后续接入新的视觉模型数据格式，应优先新增一个 `formats/<format_name>.py`：

1. 实现 `DatasetFormat`。
2. 定义输出目录结构。
3. 定义单个 `TileSample` 如何写出标签或 metadata。
4. 在 `formats/__init__.py` 或 selector 中注册。

这样可以保持 WSI 读取、annotation 解析、tile 采样、split、并发处理等主流程不变。

## 辅助工具

检查构建后的数据集：

```bash
python tools/check_yolo_dataset.py \
  --dataset-dir /path/to/output/dataset \
  --output-dir /path/to/check-result
```

统计 GeoJSON annotation 数量：

```bash
python tools/count_geojson_annotations.py /path/to/WSIs -r
```

渲染已有 segmentation label：

```bash
python tools/render_seg_labels.py \
  --image-dir /path/to/images \
  --label-dir /path/to/labels \
  --output-dir /path/to/rendered
```

## 测试

```bash
.venv/bin/python -m pytest -q
```

测试覆盖：

- GeoJSON annotation 解析
- WSI reader 后端
- 正负样本采样
- split 规则
- 输出写入
- dry-run
- 构建报告
- 数据格式适配层
- 工具脚本
