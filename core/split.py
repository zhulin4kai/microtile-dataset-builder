# -*- coding: utf-8 -*-
"""
按 WSI 级别划分 train / val / test。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from core.discover import SlidePair


@dataclass(frozen=True)
class SplitResult:
    train: List[SlidePair]
    val: List[SlidePair]
    test: List[SlidePair]

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
    split_counts: Dict[str, int],
    random_seed: int,
    manual_split: Dict[str, Sequence[str]] | None = None,
) -> SplitResult:
    """
    划分 WSI。

    如果 manual_split 不为空，则按手动指定的 stem 划分。
    否则按固定随机种子打乱后，根据 split_counts 划分。
    """
    pairs = list(pairs)

    if manual_split and has_manual_split(manual_split):
        return _manual_split(pairs, manual_split)

    return _random_split(
        pairs=pairs,
        split_counts=split_counts,
        random_seed=random_seed,
    )


def _random_split(
    pairs: List[SlidePair],
    split_counts: Dict[str, int],
    random_seed: int,
) -> SplitResult:
    train_count = int(split_counts.get("train", 0))
    val_count = int(split_counts.get("val", 0))
    test_count = int(split_counts.get("test", 0))

    expected_total = train_count + val_count + test_count

    if expected_total != len(pairs):
        raise ValueError(
            "SPLIT_COUNTS 与实际 WSI 数量不一致："
            f"train={train_count}, val={val_count}, test={test_count}, "
            f"sum={expected_total}, actual={len(pairs)}"
        )

    shuffled = sorted(pairs, key=lambda p: p.stem)

    rng = random.Random(random_seed)
    rng.shuffle(shuffled)

    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]

    return SplitResult(train=train, val=val, test=test)


def _manual_split(
    pairs: List[SlidePair],
    manual_split: Dict[str, Sequence[str]],
) -> SplitResult:
    pair_map = {pair.stem: pair for pair in pairs}

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

    return SplitResult(train=train, val=val, test=test)


def print_split_result(split_result: SplitResult) -> None:
    split_dict = split_result.as_dict()

    print("[INFO] WSI 数据集划分：")

    for split_name in ("train", "val", "test"):
        pairs = split_dict[split_name]
        print(f"  {split_name}: {len(pairs)} 张")

        for pair in pairs:
            print(f"    - {pair.stem}")
