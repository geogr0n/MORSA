from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_config import ExperimentSpec  # noqa: E402


CANCERS = (
    "BLCA",
    "BRCA",
    "COAD",
    "GBM",
    "HNSC",
    "KIRC",
    "KIRP",
    "LIHC",
    "LUAD",
    "LUSC",
    "PAAD",
    "PRAD",
    "SKCM",
    "STAD",
    "THCA",
    "UCEC",
)
TRAINING_SEED = 29
K_VALUES = (8, 16, 32, 48, 64)
HEADS = (
    ("Linear", "linear"),
    ("CovNull-SPEX-16", "covnull_spex"),
    ("LearnedRank-16", "learned_rank"),
    ("SPEX-16", "spex"),
)


def _experiment_dir(data_root: Path, cancer: str, experiment: str) -> Path:
    return data_root / cancer / "output" / "TCGA" / experiment


def _result_summary(data_root: Path, cancer: str, experiment: str) -> Path:
    return data_root / cancer / "output" / "TCGA" / "results" / experiment / "summary_row.csv"


def _read_summary(data_root: Path, cancer: str, experiment: str) -> dict[str, object] | None:
    path = _result_summary(data_root, cancer, experiment)
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one summary row in {path}, got {len(frame)}.")
    row = frame.iloc[0].to_dict()
    row.update({"cancer": cancer, "experiment": experiment, "summary_path": str(path)})
    return row


def _macro_summary(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[value].to_numpy(dtype=float)
        n = len(values)
        sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n else np.nan
        critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan
        row = dict(zip(groups, keys))
        row.update(
            {
                "n_cancers": n,
                "macro_mean": float(np.mean(values)) if n else np.nan,
                "sd_across_cancers": sd,
                "se_across_cancers": se,
                "ci95_low": float(np.mean(values) - critical * se) if n > 1 else np.nan,
                "ci95_high": float(np.mean(values) + critical * se) if n > 1 else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(values, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min((total - rank) * values[index], 1.0)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _paired_contrast(
    frame: pd.DataFrame,
    *,
    family: str,
    metric: str,
    method_column: str,
    left: str,
    right: str,
) -> dict[str, object]:
    pivot = frame.pivot(index="cancer", columns=method_column, values=metric)
    paired = pivot[[left, right]].dropna()
    differences = (paired[left] - paired[right]).to_numpy(dtype=float)
    n = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else np.nan
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan
    p_value = float(stats.ttest_1samp(differences, 0.0).pvalue) if n > 1 else np.nan
    return {
        "family": family,
        "metric": metric,
        "left": left,
        "right": right,
        "effect_definition": "left_minus_right",
        "n_cancers": n,
        "mean_paired_difference": mean,
        "ci95_low": mean - critical * se if n > 1 else np.nan,
        "ci95_high": mean + critical * se if n > 1 else np.nan,
        "paired_t_p": p_value,
    }


def build_basis_stability(data_root: Path, cancers: tuple[str, ...], missing: list[str]) -> pd.DataFrame:
    rows = []
    for cancer in cancers:
        bases = []
        for fold in range(5):
            path = _experiment_dir(data_root, cancer, "morsa") / f"basis_fold_{fold}.npz"
            if not path.is_file():
                missing.append(f"basis:{cancer}:fold{fold}")
                continue
            with np.load(path, allow_pickle=False) as payload:
                basis = np.asarray(payload["U_k"], dtype=np.float64)
                source = (
                    str(payload["basis_source"].item())
                    if "basis_source" in payload
                    else "training_cache"
                )
                if "genes" in payload:
                    genes = [str(value) for value in payload["genes"].tolist()]
                    digest = hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()
                    if "gene_order_sha256" in payload and digest != str(payload["gene_order_sha256"].item()):
                        raise ValueError(f"Gene-order hash mismatch in {path}.")
            bases.append((fold, basis, source))
        if len(bases) != 5:
            continue
        for (fold_i, left, source_i), (fold_j, right, source_j) in itertools.combinations(bases, 2):
            if left.shape != right.shape:
                raise ValueError(f"Basis shape mismatch for {cancer}: {left.shape} vs {right.shape}.")
            left, _ = np.linalg.qr(left, mode="reduced")
            right, _ = np.linalg.qr(right, mode="reduced")
            singular_values = np.clip(np.linalg.svd(left.T @ right, compute_uv=False), 0.0, 1.0)
            rows.append(
                {
                    "cancer": cancer,
                    "fold_i": fold_i,
                    "fold_j": fold_j,
                    "basis_source_i": source_i,
                    "basis_source_j": source_j,
                    "projection_similarity": float(np.mean(singular_values**2)),
                    "mean_cosine_principal_vectors": float(np.mean(singular_values)),
                    "max_principal_angle_deg": float(
                        np.degrees(np.max(np.arccos(singular_values)))
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_head_attribution(data_root: Path, cancers: tuple[str, ...], missing: list[str]):
    rows = []
    for cancer in cancers:
        for order, (label, head) in enumerate(HEADS):
            experiment = ExperimentSpec(
                "morsa_enc", head, 16, TRAINING_SEED, 0, 29
            ).name
            summary = _read_summary(data_root, cancer, experiment)
            if summary is None:
                missing.append(f"summary:{cancer}:{experiment}")
                continue
            rows.append(
                {
                    "cancer": cancer,
                    "head": label,
                    "head_order": order,
                    "training_seed": TRAINING_SEED,
                    "experiment": experiment,
                    "all_gene_pcc": float(summary["all_mean_pcc"]),
                    "top1000_pcc": float(summary["top1000_mean_pcc"]),
                    "recovered_genes": int(summary["num_sig_genes"]),
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw
    raw = raw.sort_values(["head_order", "cancer"]).reset_index(drop=True)
    summary = _macro_summary(raw, ["head", "head_order"], "all_gene_pcc")
    return raw, summary


def build_rank_sensitivity(data_root: Path, cancers: tuple[str, ...], missing: list[str]):
    rows = []
    for cancer in cancers:
        for rank_k in K_VALUES:
            experiment = ExperimentSpec(
                "morsa_enc", "spex", rank_k, TRAINING_SEED, 0, 29
            ).name
            path = _experiment_dir(data_root, cancer, experiment) / "validation_metrics.csv"
            if not path.is_file():
                missing.append(f"validation:{cancer}:{experiment}")
                continue
            metrics = pd.read_csv(path)
            if len(metrics) != 5 or set(metrics["fold"].astype(int)) != set(range(5)):
                raise ValueError(f"Expected five validation folds in {path}.")
            values = metrics["all_gene_pcc"].to_numpy(dtype=float)
            rows.append(
                {
                    "cancer": cancer,
                    "rank_k": rank_k,
                    "experiment": experiment,
                    "training_seed": TRAINING_SEED,
                    "validation_all_gene_pcc": float(np.mean(values)),
                    "fold_sd": float(np.std(values, ddof=1)),
                    "n_folds": len(values),
                }
            )
    raw = pd.DataFrame(rows)
    summary = (
        _macro_summary(raw, ["rank_k"], "validation_all_gene_pcc")
        if not raw.empty
        else raw
    )
    return raw, summary


def build_diag_comparison(data_root: Path, cancers: tuple[str, ...], missing: list[str]):
    rows = []
    methods = (("MORSA", "morsa_enc"), ("DiagSPD", "diag_spd"))
    for cancer in cancers:
        for method_order, (method, model_type) in enumerate(methods):
            experiment = ExperimentSpec(
                model_type, "spex", 16, TRAINING_SEED, 0, 29
            ).name
            summary = _read_summary(data_root, cancer, experiment)
            if summary is None:
                missing.append(f"summary:{cancer}:{experiment}")
                continue
            rows.append(
                {
                    "cancer": cancer,
                    "method": method,
                    "method_order": method_order,
                    "training_seed": TRAINING_SEED,
                    "experiment": experiment,
                    "all_gene_pcc": float(summary["all_mean_pcc"]),
                    "top1000_pcc": float(summary["top1000_mean_pcc"]),
                    "recovered_genes": int(summary["num_sig_genes"]),
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw
    raw = raw.sort_values(["method_order", "cancer"]).reset_index(drop=True)
    summaries = []
    for metric in ("all_gene_pcc", "top1000_pcc", "recovered_genes"):
        part = _macro_summary(raw, ["method", "method_order"], metric)
        part.insert(0, "metric", metric)
        summaries.append(part)
    return raw, pd.concat(summaries, ignore_index=True)


def write_data_dictionary(path: Path) -> None:
    path.write_text(
        """# Experiment summary outputs

- `basis_stability_pairs.csv`: ten checkpoint-basis fold pairs per cancer; projection similarity is the mean squared cosine of principal angles.
- `head_attribution_by_cancer.csv`: one test-set evaluator row per cancer and output head.
- `head_attribution_summary.csv`: macro mean and 95% t interval across cancers.
- `rank_validation_by_cancer.csv`: five-fold validation all-gene PCC averaged within each cancer; test metrics are not used for K selection.
- `rank_validation_summary.csv`: macro validation summary across cancers.
- `k_selection.json`: the validation decision, choosing the K with the highest across-cancer macro mean.
- `morphology_ablation_by_cancer.csv`: one evaluator row per cancer and morphology method.
- `morphology_ablation_summary.csv`: macro summaries for all-gene PCC, top-1000 PCC and recovered genes.
- `planned_contrasts.csv`: paired cancer-level effects; positive values favor the named left method. Holm correction is applied across the listed contrasts.
""",
        encoding="utf-8",
    )


def summarize_experiments(args) -> None:
    cancers = tuple(args.cancers)
    missing: list[str] = []

    basis = build_basis_stability(args.data_root, cancers, missing)
    head_raw, head_summary = build_head_attribution(args.data_root, cancers, missing)
    k_raw, k_summary = build_rank_sensitivity(args.data_root, cancers, missing)
    diag_raw, diag_summary = build_diag_comparison(args.data_root, cancers, missing)

    unique_missing = sorted(set(missing))
    if unique_missing:
        preview = "\n".join(unique_missing[:20])
        raise FileNotFoundError(
            f"Experiment inputs are incomplete ({len(unique_missing)} missing). First entries:\n{preview}"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    basis.to_csv(args.output / "basis_stability_pairs.csv", index=False)
    head_raw.to_csv(args.output / "head_attribution_by_cancer.csv", index=False)
    head_summary.to_csv(args.output / "head_attribution_summary.csv", index=False)
    k_raw.to_csv(args.output / "rank_validation_by_cancer.csv", index=False)
    k_summary.to_csv(args.output / "rank_validation_summary.csv", index=False)
    diag_raw.to_csv(args.output / "morphology_ablation_by_cancer.csv", index=False)
    diag_summary.to_csv(args.output / "morphology_ablation_summary.csv", index=False)

    contrasts = []
    if not head_raw.empty:
        for right in ("LearnedRank-16", "CovNull-SPEX-16", "Linear"):
            if {"SPEX-16", right}.issubset(set(head_raw["head"])):
                contrasts.append(
                    _paired_contrast(
                        head_raw,
                        family="head_attribution",
                        metric="all_gene_pcc",
                        method_column="head",
                        left="SPEX-16",
                        right=right,
                    )
                )
    if not diag_raw.empty and {"MORSA", "DiagSPD"}.issubset(set(diag_raw["method"])):
        for metric in ("all_gene_pcc", "top1000_pcc", "recovered_genes"):
            contrasts.append(
                _paired_contrast(
                    diag_raw,
                    family="morphology_covariance",
                    metric=metric,
                    method_column="method",
                    left="MORSA",
                    right="DiagSPD",
                )
            )
    contrast_frame = pd.DataFrame(contrasts)
    if not contrast_frame.empty:
        contrast_frame["holm_p"] = _holm_adjust(contrast_frame["paired_t_p"].tolist())
    contrast_frame.to_csv(args.output / "planned_contrasts.csv", index=False)

    selection = {"status": "incomplete"}
    if len(k_raw) == len(cancers) * len(K_VALUES):
        best_row = k_summary.loc[k_summary["macro_mean"].idxmax()]
        selection = {
            "status": "complete",
            "rule": "K with the highest across-cancer macro validation mean",
            "best_k": int(best_row["rank_k"]),
            "best_macro_mean": float(best_row["macro_mean"]),
            "best_standard_error": float(best_row["se_across_cancers"]),
            "selected_k": int(best_row["rank_k"]),
        }
    (args.output / "k_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_data_dictionary(args.output / "data_dictionary.md")

    print(f"[summary] cancers={len(cancers)} output={args.output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize MORSA structural and ablation experiments.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancers", nargs="+", default=list(CANCERS))
    return parser


def main(argv: list[str] | None = None) -> None:
    summarize_experiments(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
