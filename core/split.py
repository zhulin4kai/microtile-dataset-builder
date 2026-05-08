# -*- coding: utf-8 -*-
"""
按 WSI 级别划分 train / val / test。

自动划分逻辑：
1. 先读取每张 WSI 的 annotation 数量；
2. 搜索 train / val / test 的所有分配方式；
3. 找到 annotation 数量比例最接近 SPLIT_RATIOS 的方案；
4. slide 数量比例作为辅助约束。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence

from core.discover import SlidePair
from core.geojson_parser import parse_qupath_geojson


@dataclass(frozen=True)
class SlideInfo:
    pair: SlidePair
    ann_count: int


@dataclass(frozen=True)
class SplitResult:
    train: List[SlidePair]
    val: List[SlidePair]
    test: List[SlidePair]
    ann_counts: Dict[str, int]
    slide_counts: Dict[str, int]
    score: float

    def as_dict(self) -> Dict[str, List[SlidePair]]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }


def has_manual_split(manual_split: Dict[str, Sequence[str]]) -> bool:
    return any(bool(manual_split.get(k)) for k in ("train", "val", "test"))


def split_slide_pairs(
    pairs: Sequence[SlidePair],
    split_ratios: Dict[str, float],
    split_min_slides: Dict[str, int],
    ann_weight: float,
    slide_weight: float,
    manual_split: Dict[str, Sequence[str]] | None = None,
) -> SplitResult:
    pairs = list(pairs)

    slide_infos = build_slide_infos(pairs)

    if manual_split and has_manual_split(manual_split):
        return _manual_split(slide_infos, manual_split)

    return _auto_balance_split(
        slide_infos=slide_infos,
        split_ratios=split_ratios,
        split_min_slides=split_min_slides,
        ann_weight=ann_weight,
        slide_weight=slide_weight,
    )


def build_slide_infos(pairs: Sequence[SlidePair]) -> List[SlideInfo]:
    slide_infos: List[SlideInfo] = []

    print("[INFO] 开始统计每张 WSI 的 annotation 数量：")

    for pair in sorted(pairs, key=lambda p: p.stem):
        anns = parse_qupath_geojson(pair.geojson_path)
        ann_count = len(anns)

        slide_infos.append(
            SlideInfo(
                pair=pair,
                ann_count=ann_count,
            )
        )

        print(f"  - {pair.stem}: ann={ann_count}")

    total_ann = sum(item.ann_count for item in slide_infos)
    print(f"[INFO] annotation 总数：{total_ann}")

    return slide_infos


def _auto_balance_split(
    slide_infos: Sequence[SlideInfo],
    split_ratios: Dict[str, float],
    split_min_slides: Dict[str, int],
    ann_weight: float,
    slide_weight: float,
) -> SplitResult:
    slide_infos = list(slide_infos)

    n = len(slide_infos)
    if n < 3:
        raise ValueError("至少需要 3 张 WSI 才能划分 train / val / test。")

    _validate_split_ratios(split_ratios)

    split_names = ("train", "val", "test")
    total_ann = sum(item.ann_count for item in slide_infos)

    if total_ann <= 0:
        raise ValueError("annotation 总数为 0，无法进行按 ann 数量平衡的划分。")

    best_assignment = None
    best_score = float("inf")
    best_ann_counts: Dict[str, int] | None = None
    best_slide_counts: Dict[str, int] | None = None

    # assignment 中每个元素取值 0/1/2，分别表示 train/val/test
    for assignment in itertools.product(range(3), repeat=n):
        slide_counts = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        ann_counts = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        for idx, split_id in enumerate(assignment):
            split_name = split_names[split_id]
            slide_counts[split_name] += 1
            ann_counts[split_name] += slide_infos[idx].ann_count

        if not _satisfy_min_slides(slide_counts, split_min_slides):
            continue

        score = _split_score(
            ann_counts=ann_counts,
            slide_counts=slide_counts,
            total_ann=total_ann,
            total_slides=n,
            split_ratios=split_ratios,
            ann_weight=ann_weight,
            slide_weight=slide_weight,
        )

        if score < best_score:
            best_score = score
            best_assignment = assignment
            best_ann_counts = ann_counts
            best_slide_counts = slide_counts

    if best_assignment is None or best_ann_counts is None or best_slide_counts is None:
        raise RuntimeError("没有找到满足条件的 train / val / test 划分。")

    split_items = {
        "train": [],
        "val": [],
        "test": [],
    }

    for idx, split_id in enumerate(best_assignment):
        split_name = split_names[split_id]
        split_items[split_name].append(slide_infos[idx].pair)

    return SplitResult(
        train=split_items["train"],
        val=split_items["val"],
        test=split_items["test"],
        ann_counts=best_ann_counts,
        slide_counts=best_slide_counts,
        score=best_score,
    )


def _split_score(
    ann_counts: Dict[str, int],
    slide_counts: Dict[str, int],
    total_ann: int,
    total_slides: int,
    split_ratios: Dict[str, float],
    ann_weight: float,
    slide_weight: float,
) -> float:
    score = 0.0

    for split_name in ("train", "val", "test"):
        target_ratio = split_ratios[split_name]

        ann_ratio = ann_counts[split_name] / total_ann
        slide_ratio = slide_counts[split_name] / total_slides

        ann_error = abs(ann_ratio - target_ratio)
        slide_error = abs(slide_ratio - target_ratio)

        score += ann_weight * ann_error
        score += slide_weight * slide_error

    return score


def _satisfy_min_slides(
    slide_counts: Dict[str, int],
    split_min_slides: Dict[str, int],
) -> bool:
    for split_name in ("train", "val", "test"):
        min_count = int(split_min_slides.get(split_name, 1))
        if slide_counts[split_name] < min_count:
            return False

    return True


def _validate_split_ratios(split_ratios: Dict[str, float]) -> None:
    required = {"train", "val", "test"}

    if set(split_ratios.keys()) != required:
        raise ValueError("SPLIT_RATIOS 必须包含 train / val / test。")

    total = sum(float(v) for v in split_ratios.values())

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"SPLIT_RATIOS 总和必须为 1.0，当前为 {total}")


def _manual_split(
    slide_infos: Sequence[SlideInfo],
    manual_split: Dict[str, Sequence[str]],
) -> SplitResult:
    pair_map = {item.pair.stem: item.pair for item in slide_infos}
    ann_map = {item.pair.stem: item.ann_count for item in slide_infos}

    train_stems = list(manual_split.get("train", []))
    val_stems = list(manual_split.get("val", []))
    test_stems = list(manual_split.get("test", []))

    all_stems = train_stems + val_stems + test_stems

    if len(all_stems) != len(set(all_stems)):
        raise ValueError("MANUAL_SPLIT 中存在重复 slide stem。")

    unknown = sorted(set(all_stems) - set(pair_map.keys()))
    if unknown:
        raise ValueError(f"MANUAL_SPLIT 中存在找不到的 slide stem: {unknown}")

    missing = sorted(set(pair_map.keys()) - set(all_stems))
    if missing:
        raise ValueError(f"MANUAL_SPLIT 没有覆盖以下 slide stem: {missing}")

    train = [pair_map[stem] for stem in train_stems]
    val = [pair_map[stem] for stem in val_stems]
    test = [pair_map[stem] for stem in test_stems]

    ann_counts = {
        "train": sum(ann_map[stem] for stem in train_stems),
        "val": sum(ann_map[stem] for stem in val_stems),
        "test": sum(ann_map[stem] for stem in test_stems),
    }

    slide_counts = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }

    return SplitResult(
        train=train,
        val=val,
        test=test,
        ann_counts=ann_counts,
        slide_counts=slide_counts,
        score=0.0,
    )


def print_split_result(split_result: SplitResult) -> None:
    split_dict = split_result.as_dict()

    print("[INFO] WSI 数据集划分：")

    total_ann = sum(split_result.ann_counts.values())
    total_slides = sum(split_result.slide_counts.values())

    for split_name in ("train", "val", "test"):
        pairs = split_dict[split_name]
        ann_count = split_result.ann_counts[split_name]
        slide_count = split_result.slide_counts[split_name]

        ann_ratio = ann_count / total_ann if total_ann > 0 else 0.0
        slide_ratio = slide_count / total_slides if total_slides > 0 else 0.0

        print(
            f"  {split_name}: "
            f"slides={slide_count}, "
            f"ann={ann_count}, "
            f"ann_ratio={ann_ratio:.3f}, "
            f"slide_ratio={slide_ratio:.3f}"
        )

        for pair in pairs:
            print(f"    - {pair.stem}")

    print(f"[INFO] split score: {split_result.score:.6f}")