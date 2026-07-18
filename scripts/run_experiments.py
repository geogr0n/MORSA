from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_config import ExperimentSpec  # noqa: E402


DEFAULT_CANCERS = (
    "BRCA",
    "LUAD",
    "LUSC",
    "COAD",
    "KIRC",
    "KIRP",
    "GBM",
    "HNSC",
    "LIHC",
    "PAAD",
    "PRAD",
    "SKCM",
    "STAD",
    "THCA",
    "UCEC",
    "BLCA",
)


@dataclass(frozen=True)
class PlannedRun:
    groups: tuple[str, ...]
    spec: ExperimentSpec
    fixed_basis_experiment: str | None = None


def _load_plan(path: Path) -> tuple[dict, list[PlannedRun]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"Unsupported experiment-matrix version in {path}.")
    defaults = dict(payload.get("defaults") or {})
    runs: list[PlannedRun] = []
    for entry in payload.get("experiments") or []:
        groups = tuple(str(group) for group in entry.get("groups", [entry.get("group")]) if group)
        if not groups:
            raise ValueError("Every experiment must belong to at least one group.")
        ranks = entry.get("ranks", [entry.get("rank_k", 16)])
        training_seed = int(entry.get("training_seed", defaults.get("training_seed", 29)))
        for rank_k in ranks:
            runs.append(
                PlannedRun(
                    groups=groups,
                    spec=ExperimentSpec(
                        model_type=str(entry["model_type"]),
                        head_type=str(entry["head_type"]),
                        rank_k=int(rank_k),
                        training_seed=training_seed,
                        split_seed=int(defaults.get("split_seed", 0)),
                        basis_seed=int(defaults.get("basis_seed", 29)),
                    ),
                    fixed_basis_experiment=entry.get("fixed_basis_experiment"),
                )
            )
    names = [run.spec.name for run in runs]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate experiment names in experiment matrix: {duplicates}")
    return defaults, runs


def _split_values(values: list[str] | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return fallback
    parsed = []
    for value in values:
        parsed.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(parsed)


def _python_command(
    python: str,
    *,
    run: PlannedRun,
    cancer: str,
    data_root: Path,
    defaults: dict,
) -> list[str]:
    cancer_dir = data_root / cancer
    command = [
        python,
        str(SRC_DIR / "main.py"),
        "--ref_file",
        str(cancer_dir / "ref_file.csv"),
        "--feature_path",
        str(cancer_dir / "features"),
        "--save_dir",
        str(cancer_dir / "output"),
        "--cohort",
        str(defaults.get("cohort", "TCGA")),
        "--cancer",
        cancer,
        "--model_type",
        run.spec.model_type,
        "--head_type",
        run.spec.head_type,
        "--rank_k",
        str(run.spec.rank_k),
        "--seed",
        str(run.spec.training_seed),
        "--split_seed",
        str(run.spec.split_seed),
        "--basis_seed",
        str(run.spec.basis_seed),
        "--folds",
        str(defaults.get("folds", 5)),
        "--batch_size",
        str(defaults.get("batch_size", 16)),
        "--lr",
        str(defaults.get("learning_rate", 0.001)),
        "--num_epochs",
        str(defaults.get("num_epochs", 200)),
        "--patience",
        str(defaults.get("patience", 20)),
        "--morsa_r",
        str(defaults.get("morsa_r", 96)),
    ]
    if run.fixed_basis_experiment is not None:
        command.extend(
            [
                "--fixed_basis_dir",
                str(
                    cancer_dir
                    / "output"
                    / str(defaults.get("cohort", "TCGA"))
                    / run.fixed_basis_experiment
                ),
            ]
        )
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MORSA experiment matrix.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "experiment_matrix.yaml")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("MORSA_DATA_ROOT", PROJECT_DIR.parent / "data")))
    parser.add_argument("--cancers", nargs="*", help="Cancer names or comma-separated groups; defaults to all 16.")
    parser.add_argument("--groups", nargs="*", help="Experiment groups to run; defaults to every group.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Experiment names to run after group filtering; defaults to every selected experiment.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Run commands. Without this flag, print a dry run only.")
    parser.add_argument("--no-evaluate", action="store_true", help="Do not evaluate each completed experiment.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    defaults, runs = _load_plan(args.config)
    cancers = _split_values(args.cancers, DEFAULT_CANCERS)
    known_groups = {group for run in runs for group in run.groups}
    groups = set(_split_values(args.groups, tuple(sorted(known_groups))))
    unknown_groups = groups - known_groups
    if unknown_groups:
        raise ValueError(f"Unknown groups: {sorted(unknown_groups)}")
    selected_runs = [run for run in runs if groups.intersection(run.groups)]
    if args.experiments:
        requested_experiments = set(_split_values(args.experiments, ()))
        known_experiments = {run.spec.name for run in runs}
        unknown_experiments = requested_experiments - known_experiments
        if unknown_experiments:
            raise ValueError(f"Unknown experiments: {sorted(unknown_experiments)}")
        selected_runs = [run for run in selected_runs if run.spec.name in requested_experiments]
    if not selected_runs:
        raise ValueError("Experiment selection is empty.")
    unknown_cancers = set(cancers) - set(DEFAULT_CANCERS)
    if unknown_cancers:
        raise ValueError(f"Unknown cancers: {sorted(unknown_cancers)}")

    print(f"Selected experiment configurations: {len(selected_runs)}")
    print(f"Cancer cohorts: {len(cancers)}")
    print(f"Cancer-level tasks: {len(selected_runs) * len(cancers)}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print()

    for cancer in cancers:
        cancer_dir = args.data_root / cancer
        if args.execute:
            for required in (cancer_dir / "ref_file.csv", cancer_dir / "features"):
                if not required.exists():
                    raise FileNotFoundError(f"Missing required input: {required}")
        for run in selected_runs:
            command = _python_command(
                args.python,
                run=run,
                cancer=cancer,
                data_root=args.data_root,
                defaults=defaults,
            )
            print(f"[{cancer}] [{','.join(run.groups)}] {run.spec.name}")
            print(subprocess.list2cmdline(command))
            if not args.execute:
                continue
            subprocess.run(command, check=True, cwd=PROJECT_DIR)
            if not args.no_evaluate:
                evaluation_command = [
                    args.python,
                    str(PROJECT_DIR / "evaluation" / "evaluate_model.py"),
                    "--experiment",
                    run.spec.name,
                    "--model_dir",
                    str(cancer_dir / "output" / str(defaults.get("cohort", "TCGA"))),
                    "--folds",
                    str(defaults.get("folds", 5)),
                ]
                subprocess.run(evaluation_command, check=True, cwd=PROJECT_DIR)


if __name__ == "__main__":
    main()
