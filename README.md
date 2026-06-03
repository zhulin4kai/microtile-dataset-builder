# WSI Dataset Builder

WSI Dataset Builder 用于从全切片图像（Whole Slide Image, WSI）和 GeoJSON 标注中切割图像块，并构建可用于视觉模型训练的数据集。

项目将 WSI 处理流程与数据集输出格式分离。`core/` 负责 WSI 发现、标注解析、tile 采样、数据划分、多进程处理和构建报告；`formats/` 负责具体数据集格式的输出适配。

## 输入与输出

支持的输入文件：

- WSI 文件：`.svs`、`.tif`、`.tiff`、`.ndpi`、`.mrxs`
- 标注文件：`.geojson`、`.json`

默认输出结构：

```text
OUTPUT_DIR/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── dataset.yaml
└── build_report.json
```

`build_report.json` 记录输入发现结果、每张 WSI 的处理状态、输出样本数量和负样本采样拒绝统计。

## 目录结构

```text
.
├── main.py
├── config.py
├── core/
│   ├── annotation.py
│   ├── augment.py
│   ├── cli.py
│   ├── discovery.py
│   ├── pipeline.py
│   ├── reporting.py
│   ├── runtime_config.py
│   ├── sampling.py
│   └── slide_io.py
├── formats/
│   ├── base.py
│   └── yolo_detect.py
├── tools/
└── tests/
```

模块职责：

- `core.annotation`：GeoJSON 解析、标注校验和 bbox 工具函数。
- `core.slide_io`：WSI 读取后端。
- `core.discovery`：WSI 与标注文件发现、配对和诊断。
- `core.sampling`：正负样本采样和 train / val 划分。
- `core.augment`：图像增强和图像编码。
- `core.pipeline`：构建流程编排和多进程调度。
- `core.reporting`：汇总输出和构建报告。
- `core.runtime_config`：多进程 worker 配置同步。
- `formats`：数据集输出格式适配层。
- `tools`：数据检查和标注统计脚本。

## 安装

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 OpenSlide 后端，需要额外安装 OpenSlide 系统库。cuCIM 后端为可选项，通过配置启用。

## 配置

运行配置位于 `config.py`。

路径配置：

```python
TARGET_DIR = Path("/path/to/wsi-directory")
OUTPUT_DIR = Path("/path/to/output-directory")
```

构建配置：

```python
TILE_SIZE = 1024
NUM_WORKERS = 8
PROCESS_START_METHOD = "spawn"

SPLIT_RATIOS = {
    "train": 0.75,
    "val": 0.25,
}
DATASET_SPLIT_MODE = "patch"  # "patch" 或 "wsi"

ENABLE_COLOR_AUGMENT = True
RANDOM_SEED = 42
DISCOVER_RECURSIVE = True
DRY_RUN = False
```

WSI 后端配置：

```python
SLIDE_BACKEND = "auto"  # "auto"、"openslide" 或 "cucim"
CUCIM_DEVICE = "cuda"
```

## 运行

使用 `config.py` 中的路径配置运行：

```bash
python main.py
```

通过命令行临时指定输入和输出路径：

```bash
python main.py \
  --wsi-path /path/to/wsi-or-directory \
  --geojson-path /path/to/geojson-or-directory \
  --output-dir /path/to/output-directory
```

只执行输入发现和样本数量估算，不写入图像和标签：

```bash
python main.py \
  --wsi-path /path/to/wsi-or-directory \
  --geojson-path /path/to/geojson-or-directory \
  --dry-run
```

## 标注处理

标注解析器读取 GeoJSON 中的 `Polygon` 和 `MultiPolygon`。

解析规则：

- 一个 `Polygon` 生成一个 annotation。
- `MultiPolygon` 中的每个 polygon 会拆分为独立 annotation。
- 无效几何会被跳过，并计入诊断信息。
- annotation 的 bbox 由 polygon 外环坐标计算得到。
- GeoJSON feature ID 会在存在时保留；缺失时使用自动生成的 ID。

WSI 和标注文件按文件名 stem 配对：

```text
case_001.svs      -> case_001.geojson
case_002.ome.tif  -> case_002.geojson
```

递归发现时，如果多个标注文件与同一 WSI stem 匹配，歧义数量会写入诊断信息。

## 样本采样

正样本围绕 annotation 中心生成。负样本从 WSI 中采样，并在以下情况下被拒绝：

- tile origin 与已选负样本重复；
- tile 中包含达到可见比例阈值的 annotation；
- tile 的组织区域比例低于阈值；
- 超过最大采样尝试次数。

数据划分由 `DATASET_SPLIT_MODE` 控制：

- `patch`：每个 tile 独立划分到 train 或 val。
- `wsi`：同一张 WSI 生成的全部 tile 进入同一个 split。

## 输出格式适配层

数据集输出逻辑集中在 `formats/`。

适配器接口定义在 `formats/base.py`：

```python
class DatasetFormat(Protocol):
    name: str

    def validate_config(self) -> None: ...
    def prepare_output_dirs(self, output_dir: Path, dry_run: bool) -> None: ...
    def write_sample(self, sample: TileSample, split_name: str, rng: random.Random) -> int: ...
    def write_metadata(self, output_dir: Path, dry_run: bool) -> None: ...
    def variant_count(self) -> int: ...
```

新增输出格式时，应按以下步骤处理：

1. 在 `formats/` 下新增适配器模块。
2. 实现 `DatasetFormat` 接口。
3. 定义输出目录结构和元信息写入方式。
4. 将 `TileSample` 转换为目标格式所需的标签表示。
5. 在格式选择逻辑中注册该适配器。

`core/` 中不应包含具体输出格式的标签写入逻辑。

## 工具脚本

检查构建后的数据集：

```bash
python tools/check_yolo_dataset.py \
  --dataset-dir /path/to/dataset \
  --output-dir /path/to/check-output
```

统计 GeoJSON 标注数量：

```bash
python tools/count_geojson_annotations.py /path/to/input -r
```

渲染已有 segmentation label：

```bash
python tools/render_seg_labels.py \
  --image-dir /path/to/images \
  --label-dir /path/to/labels \
  --output-dir /path/to/render-output
```

## 测试

运行测试：

```bash
.venv/bin/python -m pytest -q
```

