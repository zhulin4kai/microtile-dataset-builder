"""Build statistics and reporting helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

import config


@dataclass
class SlideStats:
    slide_stem: str
    raw_pos: int = 0
    raw_neg: int = 0
    written_train: int = 0
    written_val: int = 0
    failed_neg: int = 0
    status: str = "ok"
    skip_reason: str = ""
    error: str = ""
    neg_reject_duplicate: int = 0
    neg_reject_annotation: int = 0
    neg_reject_low_tissue: int = 0
    neg_reject_try_limit: int = 0

    def as_dict(self) -> dict:
        return {
            "slide_stem": self.slide_stem,
            "raw_pos": self.raw_pos,
            "raw_neg": self.raw_neg,
            "written_train": self.written_train,
            "written_val": self.written_val,
            "failed_neg": self.failed_neg,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "neg_reject_duplicate": self.neg_reject_duplicate,
            "neg_reject_annotation": self.neg_reject_annotation,
            "neg_reject_low_tissue": self.neg_reject_low_tissue,
            "neg_reject_try_limit": self.neg_reject_try_limit,
        }


@dataclass
class DatasetTotals:
    raw_pos: int = 0
    raw_neg: int = 0
    written_train: int = 0
    written_val: int = 0
    failed_neg: int = 0
    slides_ok: int = 0
    slides_skipped: int = 0
    slides_failed: int = 0
    neg_reject_duplicate: int = 0
    neg_reject_annotation: int = 0
    neg_reject_low_tissue: int = 0
    neg_reject_try_limit: int = 0

    def add(self, result: dict) -> None:
        self.raw_pos += result["raw_pos"]
        self.raw_neg += result["raw_neg"]
        self.written_train += result["written_train"]
        self.written_val += result["written_val"]
        self.failed_neg += result["failed_neg"]
        self.neg_reject_duplicate += result.get("neg_reject_duplicate", 0)
        self.neg_reject_annotation += result.get("neg_reject_annotation", 0)
        self.neg_reject_low_tissue += result.get("neg_reject_low_tissue", 0)
        self.neg_reject_try_limit += result.get("neg_reject_try_limit", 0)
        status = result.get("status", "ok")
        if status == "ok":
            self.slides_ok += 1
        elif status == "skipped":
            self.slides_skipped += 1
        else:
            self.slides_failed += 1


def status_text(status: str) -> str:
    return {
        "ok": "成功",
        "skipped": "跳过",
        "failed": "失败",
    }.get(status, status)


def print_summary(totals: DatasetTotals) -> None:
    print(f"\nWSI 处理结果：成功={totals.slides_ok} 跳过={totals.slides_skipped} 失败={totals.slides_failed}")
    print(f"原始正样本：{totals.raw_pos}")
    print(f"原始负样本：{totals.raw_neg}")
    print(f"写入 train images：{totals.written_train}")
    print(f"写入 val images：{totals.written_val}")
    print(f"写入 images 总数：{totals.written_train + totals.written_val}")
    print(f"输出目录：{config.OUTPUT_DIR}")
    if totals.failed_neg > 0:
        print(f"负样本不足：{totals.failed_neg}")
    if any([
        totals.neg_reject_duplicate,
        totals.neg_reject_annotation,
        totals.neg_reject_low_tissue,
        totals.neg_reject_try_limit,
    ]):
        print(
            f"负样本拒绝原因: duplicate={totals.neg_reject_duplicate} "
            f"annotation={totals.neg_reject_annotation} "
            f"low_tissue={totals.neg_reject_low_tissue} "
            f"try_limit={totals.neg_reject_try_limit}"
        )


def write_build_report(
    diagnostics: dict,
    totals: DatasetTotals,
    slide_results: list[dict],
) -> None:
    config_snapshot = {
        "TARGET_DIR": str(config.TARGET_DIR),
        "OUTPUT_DIR": str(config.OUTPUT_DIR),
        "TILE_SIZE": config.TILE_SIZE,
        "NUM_WORKERS": config.NUM_WORKERS,
        "SPLIT_RATIOS": config.SPLIT_RATIOS,
        "DATASET_SPLIT_MODE": config.DATASET_SPLIT_MODE,
        "DATASET_TASK": config.DATASET_TASK,
        "ENABLE_COLOR_AUGMENT": config.ENABLE_COLOR_AUGMENT,
        "DRY_RUN": config.DRY_RUN,
        "RANDOM_SEED": config.RANDOM_SEED,
        "SLIDE_BACKEND": config.SLIDE_BACKEND,
        "CUCIM_DEVICE": config.CUCIM_DEVICE,
    }
    report = {
        "说明": "YOLO detect 数据集构建报告",
        "config": config_snapshot,
        "discovery": {
            "说明": "构建前数据检查结果",
            **diagnostics,
        },
        "slides": [with_chinese_status(result) for result in slide_results],
        "totals": {
            "说明": "构建结果汇总",
            "raw_pos": totals.raw_pos,
            "raw_neg": totals.raw_neg,
            "written_train": totals.written_train,
            "written_val": totals.written_val,
            "failed_neg": totals.failed_neg,
            "slides_ok": totals.slides_ok,
            "slides_skipped": totals.slides_skipped,
            "slides_failed": totals.slides_failed,
            "neg_reject_duplicate": totals.neg_reject_duplicate,
            "neg_reject_annotation": totals.neg_reject_annotation,
            "neg_reject_low_tissue": totals.neg_reject_low_tissue,
            "neg_reject_try_limit": totals.neg_reject_try_limit,
        },
        "output_dir": str(config.OUTPUT_DIR),
    }
    if config.DRY_RUN:
        print(f"\n[DRY_RUN] 构建报告预览:\n{json.dumps(report, indent=2, ensure_ascii=False)}")
    elif config.WRITE_BUILD_REPORT:
        report_path = config.OUTPUT_DIR / "build_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n构建报告: {report_path}")


def with_chinese_status(result: dict) -> dict:
    item = dict(result)
    item["status_text"] = status_text(item.get("status", ""))
    if item.get("skip_reason") == "empty_annotations":
        item["skip_reason_text"] = "空 annotation"
    return item

