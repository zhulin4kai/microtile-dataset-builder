#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count annotations in all GeoJSON files under a directory.

Default counting rule:
- A QuPath GeoJSON annotation is treated as one item in FeatureCollection["features"].
- For a plain Feature object, count = 1.
- For a plain geometry object, count = 1.

Examples:
    python count_geojson_annotations.py /home/yuzhoukai/projects/WSIs
    python count_geojson_annotations.py /home/yuzhoukai/projects/WSIs -r -w 12
    python count_geojson_annotations.py /home/yuzhoukai/projects/WSIs --show-files
    python count_geojson_annotations.py /home/yuzhoukai/projects/WSIs --output geojson_count.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None


DEFAULT_EXTENSIONS = (".geojson", ".json")


@dataclass(frozen=True)
class CountResult:
    path: str
    count: int
    ok: bool
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count GeoJSON annotations under a directory with multiprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing GeoJSON files.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=max(1, min(12, os.cpu_count() or 4)),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions to include, e.g. --ext .geojson .json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path for per-file counts.",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Print per-file annotation counts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any file fails to parse.",
    )
    parser.add_argument(
        "--sort",
        choices=("path", "count", "none"),
        default="path",
        help="Sort per-file output.",
    )
    return parser.parse_args()


def normalize_exts(exts: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for ext in exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        result.append(ext)
    return tuple(dict.fromkeys(result))


def discover_files(directory: Path, recursive: bool, exts: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def load_json_bytes(path: Path):
    data = path.read_bytes()
    if orjson is not None:
        return orjson.loads(data)
    return json.loads(data.decode("utf-8"))


def count_geojson_object(obj) -> int:
    """Return annotation count from a GeoJSON-like object."""
    if not isinstance(obj, dict):
        return 0

    geo_type = obj.get("type")

    if geo_type == "FeatureCollection":
        features = obj.get("features", [])
        return len(features) if isinstance(features, list) else 0

    if geo_type == "Feature":
        return 1

    # Plain geometry file: Polygon, MultiPolygon, etc.
    if geo_type in {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    }:
        return 1

    # Some exported JSON may omit type but still contain a features list.
    features = obj.get("features")
    if isinstance(features, list):
        return len(features)

    return 0


def count_one_file(path_str: str) -> CountResult:
    path = Path(path_str)
    try:
        obj = load_json_bytes(path)
        count = count_geojson_object(obj)
        return CountResult(path=str(path), count=count, ok=True)
    except Exception as exc:
        return CountResult(path=str(path), count=0, ok=False, error=f"{type(exc).__name__}: {exc}")


def sort_results(results: list[CountResult], mode: str) -> list[CountResult]:
    if mode == "path":
        return sorted(results, key=lambda r: r.path)
    if mode == "count":
        return sorted(results, key=lambda r: r.count, reverse=True)
    return results


def write_csv(path: Path, results: list[CountResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "count", "ok", "error"])
        for r in results:
            writer.writerow([r.path, r.count, int(r.ok), r.error])


def main() -> int:
    args = parse_args()
    exts = normalize_exts(args.ext)
    start = time.perf_counter()

    try:
        files = discover_files(args.directory, args.recursive, exts)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not files:
        print("No GeoJSON files found.")
        return 0

    workers = max(1, min(args.workers, len(files)))
    print(f"Found files: {len(files)}")
    print(f"Workers: {workers}")
    print(f"Parser: {'orjson' if orjson is not None else 'json'}")

    results: list[CountResult] = []

    if workers == 1:
        for p in files:
            results.append(count_one_file(str(p)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(count_one_file, str(p)): p for p in files}
            for future in as_completed(future_map):
                results.append(future.result())

    results = sort_results(results, args.sort)

    total = sum(r.count for r in results if r.ok)
    failed = [r for r in results if not r.ok]

    if args.show_files:
        for r in results:
            if r.ok:
                print(f"{r.count:8d}  {r.path}")
            else:
                print(f"FAILED    {r.path}  {r.error}")

    if args.output is not None:
        write_csv(args.output, results)
        print(f"CSV written: {args.output}")

    elapsed = time.perf_counter() - start
    print("-" * 60)
    print(f"Total annotations: {total}")
    print(f"Parsed files:       {len(results) - len(failed)}")
    print(f"Failed files:       {len(failed)}")
    print(f"Elapsed:            {elapsed:.3f}s")

    if failed:
        print("\nFailed files:", file=sys.stderr)
        for r in failed:
            print(f"  {r.path}: {r.error}", file=sys.stderr)
        return 1 if args.strict else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
