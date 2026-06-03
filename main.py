#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI entrypoint for building the YOLO detection dataset."""

from __future__ import annotations

import sys

from core.cli import main


if __name__ == "__main__":
    main(sys.argv[1:])
