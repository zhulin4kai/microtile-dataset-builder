# WSI Dataset Builder

WSI Dataset Builder is a dataset construction tool for generating vision model training datasets from whole-slide images (WSI) and GeoJSON annotations.

The project separates WSI-specific processing from dataset-format-specific output. The core pipeline handles slide discovery, annotation parsing, tile sampling, split assignment, multiprocessing, and reporting. Output formats are implemented as adapters under `formats/`.

## Inputs and Outputs

Supported input files:

- WSI: `.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs`
- Annotation: `.geojson`, `.json`

Default output layout:

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

`build_report.json` records input discovery diagnostics, per-slide processing status, output counts, and negative-sampling rejection statistics.

## Repository Layout

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

Module responsibilities:

- `core.annotation`: GeoJSON parsing, annotation validation, bbox utilities.
- `core.slide_io`: WSI reading backends.
- `core.discovery`: WSI and annotation file discovery and pairing.
- `core.sampling`: positive/negative tile sampling and split assignment.
- `core.augment`: image augmentation and image encoding.
- `core.pipeline`: build orchestration and multiprocessing.
- `core.reporting`: summary and build report generation.
- `core.runtime_config`: runtime configuration snapshot for spawned workers.
- `formats`: dataset output format adapters.
- `tools`: standalone dataset inspection and annotation utility scripts.

## Installation

Create a virtual environment and install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The OpenSlide backend requires the OpenSlide system library in addition to the Python package. The cuCIM backend is optional and is selected through configuration.

## Configuration

Runtime configuration is defined in `config.py`.

Required path settings:

```python
TARGET_DIR = Path("/path/to/wsi-directory")
OUTPUT_DIR = Path("/path/to/output-directory")
```

Common build settings:

```python
TILE_SIZE = 1024
NUM_WORKERS = 8
PROCESS_START_METHOD = "spawn"

SPLIT_RATIOS = {
    "train": 0.75,
    "val": 0.25,
}
DATASET_SPLIT_MODE = "patch"  # "patch" or "wsi"

ENABLE_COLOR_AUGMENT = True
RANDOM_SEED = 42
DISCOVER_RECURSIVE = True
DRY_RUN = False
```

Slide backend settings:

```python
SLIDE_BACKEND = "auto"  # "auto", "openslide", or "cucim"
CUCIM_DEVICE = "cuda"
```

## Usage

Run with values from `config.py`:

```bash
python main.py
```

Override input and output paths:

```bash
python main.py \
  --wsi-path /path/to/wsi-or-directory \
  --geojson-path /path/to/geojson-or-directory \
  --output-dir /path/to/output-directory
```

Run input discovery and count estimation without writing images or labels:

```bash
python main.py \
  --wsi-path /path/to/wsi-or-directory \
  --geojson-path /path/to/geojson-or-directory \
  --dry-run
```

## Annotation Handling

The annotation parser reads GeoJSON `Polygon` and `MultiPolygon` geometries.

Parsing rules:

- A `Polygon` produces one annotation.
- Each polygon inside a `MultiPolygon` is converted into an independent annotation.
- Invalid geometries are skipped and counted in diagnostics.
- Annotation bbox values are computed from polygon exterior coordinates.
- GeoJSON feature IDs are preserved when present; generated IDs are used otherwise.

WSI and annotation pairing is based on filename stem. For example:

```text
case_001.svs      -> case_001.geojson
case_002.ome.tif  -> case_002.geojson
```

When recursive discovery finds multiple annotation candidates with the same stem, the ambiguity is recorded in diagnostics.

## Sampling

Positive samples are generated around annotation centers. Negative samples are sampled from the slide and rejected when they:

- duplicate a previously selected negative tile origin,
- contain a sufficiently visible annotation,
- fail the minimum tissue-ratio threshold,
- exceed the configured retry limit.

Split assignment is controlled by `DATASET_SPLIT_MODE`:

- `patch`: each generated tile is assigned independently.
- `wsi`: all tiles from the same WSI are assigned to the same split.

## Output Format Adapters

Dataset output logic is isolated under `formats/`.

The adapter interface is defined in `formats/base.py`:

```python
class DatasetFormat(Protocol):
    name: str

    def validate_config(self) -> None: ...
    def prepare_output_dirs(self, output_dir: Path, dry_run: bool) -> None: ...
    def write_sample(self, sample: TileSample, split_name: str, rng: random.Random) -> int: ...
    def write_metadata(self, output_dir: Path, dry_run: bool) -> None: ...
    def variant_count(self) -> int: ...
```

To add a new output format:

1. Add a new adapter module under `formats/`.
2. Implement `DatasetFormat`.
3. Define the output directory structure and metadata writer.
4. Convert `TileSample` objects into the target label representation.
5. Register the adapter in the format selector.

The core pipeline should not contain format-specific label-writing logic.

## Tools

Inspect generated datasets:

```bash
python tools/check_yolo_dataset.py \
  --dataset-dir /path/to/dataset \
  --output-dir /path/to/check-output
```

Count GeoJSON annotations:

```bash
python tools/count_geojson_annotations.py /path/to/input -r
```

Render existing segmentation labels:

```bash
python tools/render_seg_labels.py \
  --image-dir /path/to/images \
  --label-dir /path/to/labels \
  --output-dir /path/to/render-output
```

## Testing

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

