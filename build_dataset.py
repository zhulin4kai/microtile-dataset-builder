from __future__ import annotations

import concurrent.futures
import traceback
from tqdm import tqdm
from typing import Dict, List, Tuple

import config
from core.discover import SlidePair, discover_slide_pairs, print_slide_pairs
from core.split import SplitResult, print_split_result, split_slide_pairs
from core.worker import SlideProcessStats, process_one_slide
from core.yolo_writer import prepare_output_dirs


def main() -> None:
    _check_config()

    prepare_output_dirs(config.OUTPUT_DIR)

    pairs = discover_slide_pairs(config.TARGET_DIR)
    print_slide_pairs(pairs)

    split_result = split_slide_pairs(
        pairs=pairs,
        split_counts=config.SPLIT_COUNTS,
        random_seed=config.RANDOM_SEED,
        manual_split=config.MANUAL_SPLIT,
    )
    print_split_result(split_result)

    tasks = _build_tasks(split_result)

    print(f"[INFO] 准备开始切图，任务数：{len(tasks)}，workers={config.NUM_WORKERS}")

    stats_list: List[SlideProcessStats] = []
    failures: List[Tuple[str, str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        future_map = {}

        for task_index, (split_name, pair) in enumerate(tasks):
            worker_seed = config.RANDOM_SEED + task_index * 1009

            future = executor.submit(
                process_one_slide,
                pair,
                split_name,
                worker_seed,
            )
            future_map[future] = (split_name, pair)

        with tqdm(
            total=len(future_map),
            desc="Generating WSI dataset",
            unit="slide",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(future_map):
                split_name, pair = future_map[future]

                try:
                    stats = future.result()
                    stats_list.append(stats)

                    pbar.set_postfix_str(
                        f"{split_name}/{pair.stem} "
                        f"pos={stats.positive.saved} "
                        f"neg={stats.negative.saved}"
                    )

                except Exception as e:
                    tb = traceback.format_exc()
                    failures.append((split_name, pair.stem, str(e)))

                    pbar.write(f"[ERROR] {split_name} | {pair.stem} | {e}")
                    pbar.write(tb)

                finally:
                    pbar.update(1)

    _print_summary(stats_list, failures)

    if failures:
        raise RuntimeError(f"部分 WSI 处理失败，失败数量：{len(failures)}")

    print("[ALL DONE] 数据集生成完成。")

def _check_config() -> None:
    if config.TILE_SIZE <= 0:
        raise ValueError("TILE_SIZE must be positive.")

    if config.LEVEL != 0:
        raise ValueError("当前工程方案固定使用 level 0。")

    if config.POS_PATCHES_PER_ANNOTATION <= 0:
        raise ValueError("POS_PATCHES_PER_ANNOTATION must be positive.")

    if config.NEG_POS_RATIO < 0:
        raise ValueError("NEG_POS_RATIO must be >= 0.")

    if config.NUM_WORKERS <= 0:
        raise ValueError("NUM_WORKERS must be positive.")

    if config.REQUIRE_CUDA:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "config.REQUIRE_CUDA=True，但当前环境 torch.cuda.is_available() 为 False。"
            )


def _build_tasks(split_result: SplitResult) -> List[Tuple[str, SlidePair]]:
    tasks: List[Tuple[str, SlidePair]] = []

    split_dict = split_result.as_dict()

    for split_name in ("train", "val", "test"):
        for pair in split_dict[split_name]:
            tasks.append((split_name, pair))

    return tasks


def _print_summary(
    stats_list: List[SlideProcessStats],
    failures: List[Tuple[str, str, str]],
) -> None:
    print("\n========== SUMMARY ==========")

    total_pos_requested = 0
    total_pos_saved = 0
    total_neg_requested = 0
    total_neg_saved = 0

    split_summary: Dict[str, Dict[str, int]] = {
        "train": {"pos": 0, "neg": 0},
        "val": {"pos": 0, "neg": 0},
        "test": {"pos": 0, "neg": 0},
    }

    for stats in sorted(stats_list, key=lambda s: (s.split_name, s.slide_stem)):
        pos = stats.positive
        neg = stats.negative

        total_pos_requested += pos.requested
        total_pos_saved += pos.saved
        total_neg_requested += neg.requested
        total_neg_saved += neg.saved

        split_summary[stats.split_name]["pos"] += pos.saved
        split_summary[stats.split_name]["neg"] += neg.saved

        print(
            f"{stats.split_name:5s} | {stats.slide_stem:30s} | "
            f"ann={stats.annotation_count:4d} | "
            f"pos={pos.saved:5d}/{pos.requested:5d} | "
            f"neg={neg.saved:5d}/{neg.requested:5d} | "
            f"ambiguous={pos.ambiguous_edge_target:4d} | "
            f"neg_box_rej={neg.rejected_by_box:6d} | "
            f"neg_tissue_rej={neg.rejected_by_tissue:6d}"
        )

    print("\n[Split]")
    for split_name in ("train", "val", "test"):
        item = split_summary[split_name]
        print(f"  {split_name}: pos={item['pos']}, neg={item['neg']}")

    print("\n[Total]")
    print(f"  positive: {total_pos_saved}/{total_pos_requested}")
    print(f"  negative: {total_neg_saved}/{total_neg_requested}")

    if failures:
        print("\n[Failures]")
        for split_name, stem, reason in failures:
            print(f"  {split_name} | {stem} | {reason}")

    print("=============================\n")


if __name__ == "__main__":
    main()
