from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, TypeVar

from .errors import MissingInputError

T = TypeVar("T")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_root", required=True, help="Root containing TCGA cancer folders and cross-cohort/CPTAC-* folders.")
    parser.add_argument("--resource_root", required=True, help="Root containing prepared downstream resources and clinical tables.")
    parser.add_argument("--results_root", required=True, help="Directory where downstream result CSVs are written.")


def task_dir(results_root: str | Path, task: str) -> Path:
    out = Path(results_root) / task
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_cli_task(
    *,
    task: str,
    args: argparse.Namespace,
    run_fn: Callable[[], T],
    save_fn: Callable[[T, Path], None],
) -> None:
    try:
        result = run_fn()
    except MissingInputError as exc:
        raise SystemExit(str(exc)) from exc
    save_fn(result, task_dir(args.results_root, task))
