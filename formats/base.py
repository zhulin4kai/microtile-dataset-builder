"""Dataset output format protocol."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

from core.sampling import TileSample


class DatasetFormat(Protocol):
    name: str

    def validate_config(self) -> None:
        ...

    def prepare_output_dirs(self, output_dir: Path, dry_run: bool) -> None:
        ...

    def write_sample(
        self,
        sample: TileSample,
        split_name: str,
        rng: random.Random,
    ) -> int:
        ...

    def write_metadata(self, output_dir: Path, dry_run: bool) -> None:
        ...

    def variant_count(self) -> int:
        ...

